#!/usr/bin/env python3
"""
02_build_training_jsonl.py

Builds train/val/test JSONL splits for OpenAI fine-tuning from:
  - text2typeql/dataset/all_queries.csv     (13,939 query examples)
  - finetune/data/schema_descriptions.json  (9 NL descriptions × 15 domains)
  - finetune/data/mutation_examples.jsonl   (45 mutation examples; optional)

Outputs:
  - finetune/data/train.jsonl
  - finetune/data/val.jsonl
  - finetune/data/test.jsonl    (northwind only)
  - finetune/data/.build-manifest

Usage:
  python finetune/02_build_training_jsonl.py
  python finetune/02_build_training_jsonl.py --validate-only
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent.resolve()
DATASET_DIR = REPO_ROOT / "text2typeql" / "dataset"
FINETUNE_DIR = REPO_ROOT / "finetune"
DATA_DIR = FINETUNE_DIR / "data"

SYSTEM_PROMPT_PATH = FINETUNE_DIR / "system_prompt.txt"
ALL_QUERIES_CSV = DATASET_DIR / "all_queries.csv"
SCHEMA_DESCRIPTIONS_JSON = DATA_DIR / "schema_descriptions.json"
MUTATION_EXAMPLES_JSONL = DATA_DIR / "mutation_examples.jsonl"

OUTPUT_TRAIN = DATA_DIR / "train.jsonl"
OUTPUT_VAL = DATA_DIR / "val.jsonl"
OUTPUT_TEST = DATA_DIR / "test.jsonl"
BUILD_MANIFEST = DATA_DIR / ".build-manifest"

RANDOM_SEED = 42
VAL_FRACTION = 0.2
TEST_DOMAIN = "northwind"
INTERLEAVE_INTERVAL = 100  # 1 non-query example per N query examples

DOMAINS = [
    "bluesky",
    "buzzoverflow",
    "companies",
    "fincen",
    "gameofthrones",
    "grandstack",
    "movies",
    "neoflix",
    "network",
    "northwind",
    "offshoreleaks",
    "recommendations",
    "stackoverflow2",
    "twitch",
    "twitter",
]

# ---------------------------------------------------------------------------
# Schema loading (cached)
# ---------------------------------------------------------------------------

_schema_cache: dict[tuple[str, str], str] = {}


def load_schema(source: str, domain: str) -> str:
    """Load schema TQL for the given source and domain. Caches results."""
    key = (source, domain)
    if key not in _schema_cache:
        schema_path = DATASET_DIR / source / domain / "schema.tql"
        if not schema_path.exists():
            raise FileNotFoundError(f"Schema not found: {schema_path}")
        _schema_cache[key] = schema_path.read_text(encoding="utf-8").strip()
    return _schema_cache[key]


# ---------------------------------------------------------------------------
# Schema explanation generation (deterministic, no LLM)
# ---------------------------------------------------------------------------

def _extract_schema_stats(schema_tql: str) -> dict[str, list[str]]:
    """Extract entity/relation/attribute names from a schema TQL."""
    entities = re.findall(r"\bentity\s+(\w+)", schema_tql)
    relations = re.findall(r"\brelation\s+(\w+)", schema_tql)
    attributes = re.findall(r"\battribute\s+(\w+)", schema_tql)
    return {"entities": entities, "relations": relations, "attributes": attributes}


def _generate_schema_explanation(domain: str, schema_tql: str) -> str:
    """Generate a brief 1-2 sentence schema explanation without calling an LLM."""
    stats = _extract_schema_stats(schema_tql)
    entities = stats["entities"]
    relations = stats["relations"]

    n_ent = len(entities)
    n_rel = len(relations)
    ent_sample = ", ".join(entities[:3])
    rel_sample = ", ".join(relations[:3])

    if n_ent >= 2 and n_rel >= 1:
        return (
            f"This schema models the {domain} domain with {n_ent} entity type(s) "
            f"(e.g. {ent_sample}) and {n_rel} relation type(s) (e.g. {rel_sample}). "
            f"Each entity owns typed attributes and participates in named roles within relations."
        )
    elif n_ent >= 1:
        return (
            f"This schema defines the {domain} domain with key entity types: {ent_sample}. "
            f"Entities own typed attributes for structured data modelling."
        )
    else:
        return (
            f"This schema defines the {domain} domain using TypeQL 3.0 entity and relation types "
            f"with typed attributes and explicit role participation."
        )


# ---------------------------------------------------------------------------
# Example builders
# ---------------------------------------------------------------------------

def _make_query_example(
    system_prompt: str, schema_tql: str, question: str, typeql: str
) -> dict:
    """
    Query training example: model sees existing schema in system prompt and
    must produce only queries (empty schema, single query, empty explanation).
    """
    system_content = f"{system_prompt}\n\nSchema:\n{schema_tql}"
    assistant_payload = json.dumps(
        {"schema": "", "queries": [typeql], "explanation": ""},
        ensure_ascii=False,
    )
    return {
        "_kind": "query",
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": question},
            {"role": "assistant", "content": assistant_payload},
        ],
    }


def _make_schema_example(
    system_prompt: str, description: str, schema_tql: str, domain: str
) -> dict:
    """
    Schema training example: model generates a schema from a NL description.
    No schema in the system prompt — model learns to create from scratch.
    """
    explanation = _generate_schema_explanation(domain, schema_tql)
    assistant_payload = json.dumps(
        {"schema": schema_tql, "queries": [], "explanation": explanation},
        ensure_ascii=False,
    )
    return {
        "_kind": "schema",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": description},
            {"role": "assistant", "content": assistant_payload},
        ],
    }


def _make_combined_example(
    system_prompt: str,
    description: str,
    schema_tql: str,
    queries: list[str],
    domain: str,
) -> dict:
    """
    Combined training example: model generates schema + queries together.
    Full agent output — both schema and queries are present.
    """
    explanation = _generate_schema_explanation(domain, schema_tql)
    assistant_payload = json.dumps(
        {"schema": schema_tql, "queries": queries, "explanation": explanation},
        ensure_ascii=False,
    )
    return {
        "_kind": "combined",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": description},
            {"role": "assistant", "content": assistant_payload},
        ],
    }


def _strip_kind(example: dict) -> dict:
    """Remove internal _kind tag before writing to JSONL."""
    return {k: v for k, v in example.items() if k != "_kind"}


# ---------------------------------------------------------------------------
# Interleaving
# ---------------------------------------------------------------------------

def _interleave(
    query_examples: list[dict],
    other_examples: list[dict],
    interval: int,
) -> list[dict]:
    """
    Insert one item from other_examples after every `interval` query_examples.
    Remaining other_examples are appended at the end.
    Preserves insertion order of other_examples (round-robin).
    """
    result: list[dict] = []
    other_iter = iter(other_examples)
    count = 0

    for ex in query_examples:
        result.append(ex)
        count += 1
        if count % interval == 0:
            try:
                result.append(next(other_iter))
            except StopIteration:
                pass

    # Append any remaining non-query examples
    for ex in other_iter:
        result.append(ex)

    return result


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

REQUIRED_ASSISTANT_KEYS = ("schema", "queries", "explanation")


def _validate_line(line: str, lineno: int) -> str | None:
    """
    Validate a single JSONL line.
    Returns an error message string, or None if valid.
    """
    try:
        obj = json.loads(line)
    except json.JSONDecodeError as exc:
        return f"Line {lineno}: JSON parse error: {exc}"

    messages = obj.get("messages")
    if not isinstance(messages, list) or not messages:
        return f"Line {lineno}: missing or empty 'messages' array"

    assistant_msg = next(
        (m for m in messages if isinstance(m, dict) and m.get("role") == "assistant"),
        None,
    )
    if not assistant_msg:
        return f"Line {lineno}: no assistant message found"

    content = assistant_msg.get("content", "")
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        return f"Line {lineno}: assistant content is not valid JSON: {exc}"

    for key in REQUIRED_ASSISTANT_KEYS:
        if key not in parsed:
            return f"Line {lineno}: assistant JSON missing key '{key}'"

    if not isinstance(parsed["queries"], list):
        return f"Line {lineno}: 'queries' must be an array"

    return None


def validate_jsonl_file(path: Path) -> tuple[int, list[str]]:
    """
    Validate every line in a JSONL file.
    Returns (valid_count, list_of_error_messages).
    """
    valid = 0
    errors: list[str] = []

    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            err = _validate_line(line, lineno)
            if err:
                errors.append(err)
            else:
                valid += 1

    return valid, errors


# ---------------------------------------------------------------------------
# Validate-only mode
# ---------------------------------------------------------------------------

def validate_only_mode() -> int:
    """Validate existing JSONL output files without rebuilding."""
    print("=== VALIDATE ONLY ===\n")
    all_ok = True

    for label, path in [
        ("train", OUTPUT_TRAIN),
        ("val", OUTPUT_VAL),
        ("test", OUTPUT_TEST),
    ]:
        if not path.exists():
            print(f"[MISSING] {label}.jsonl: {path}")
            all_ok = False
            continue

        valid, errors = validate_jsonl_file(path)
        if errors:
            status = "ERRORS"
            all_ok = False
            for e in errors[:10]:  # show first 10 errors
                print(f"  {e}")
            if len(errors) > 10:
                print(f"  ... and {len(errors) - 10} more errors")
        else:
            status = "OK"
        print(f"[{status:6s}] {label}.jsonl — {valid:6d} valid, {len(errors)} errors")

    print()
    return 0 if all_ok else 1


# ---------------------------------------------------------------------------
# Build mode
# ---------------------------------------------------------------------------

def build_mode() -> int:
    """Build the JSONL training files from source data."""
    print("=== BUILD MODE ===\n")
    rng = random.Random(RANDOM_SEED)

    # ------------------------------------------------------------------
    # 1. Load system prompt
    # ------------------------------------------------------------------
    if not SYSTEM_PROMPT_PATH.exists():
        print(f"[ERROR] System prompt not found: {SYSTEM_PROMPT_PATH}")
        return 1

    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
    system_prompt_sha256 = hashlib.sha256(system_prompt.encode()).hexdigest()
    print(
        f"[INFO] System prompt loaded "
        f"({len(system_prompt)} chars, sha256={system_prompt_sha256[:16]}...)"
    )

    # ------------------------------------------------------------------
    # 2. Read CSV → query examples + test examples
    # ------------------------------------------------------------------
    print(f"[INFO] Reading {ALL_QUERIES_CSV} ...")

    test_examples: list[dict] = []
    query_examples: list[dict] = []
    # Collect typeql queries per domain for combined examples (top-5 in CSV order)
    domain_typeql: dict[str, list[str]] = {d: [] for d in DOMAINS}

    schema_miss = 0
    row_count = 0

    with ALL_QUERIES_CSV.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_count += 1
            source = row["source"]
            domain = row["domain"]
            question = row["question"].strip()
            typeql = row["typeql"].strip()

            # Accumulate per-domain queries (CSV order = original_index order)
            domain_typeql.setdefault(domain, []).append(typeql)

            try:
                schema_tql = load_schema(source, domain)
            except FileNotFoundError as exc:
                if schema_miss < 5:
                    print(f"  [WARN] {exc}")
                elif schema_miss == 5:
                    print("  [WARN] (further schema-missing warnings suppressed)")
                schema_miss += 1
                continue

            ex = _make_query_example(system_prompt, schema_tql, question, typeql)

            if domain == TEST_DOMAIN:
                test_examples.append(ex)
            else:
                query_examples.append(ex)

            if row_count % 2000 == 0:
                print(f"  ... {row_count} rows processed")

    print(
        f"[INFO] CSV rows: {row_count}  |  "
        f"test ({TEST_DOMAIN}): {len(test_examples)}  |  "
        f"non-test query: {len(query_examples)}"
    )
    if schema_miss:
        print(f"[WARN] {schema_miss} rows skipped due to missing schema files")

    # ------------------------------------------------------------------
    # 3. Schema descriptions → schema examples + combined examples
    # ------------------------------------------------------------------
    schema_examples: list[dict] = []
    combined_examples: list[dict] = []

    if not SCHEMA_DESCRIPTIONS_JSON.exists():
        print(f"\n[WARN] schema_descriptions.json not found: {SCHEMA_DESCRIPTIONS_JSON}")
        print("[WARN] Skipping schema and combined examples — run 01_generate_schema_descriptions.py first")
    else:
        print(f"\n[INFO] Loading schema descriptions from {SCHEMA_DESCRIPTIONS_JSON} ...")
        with SCHEMA_DESCRIPTIONS_JSON.open("r", encoding="utf-8") as f:
            descriptions: dict[str, list[str]] = json.load(f)

        for domain in DOMAINS:
            domain_descs = descriptions.get(domain, [])
            if not domain_descs:
                print(f"  [WARN] No descriptions for domain '{domain}' — skipping")
                continue

            try:
                schema_tql = load_schema("synthetic-2", domain)
            except FileNotFoundError as exc:
                print(f"  [WARN] {exc} — skipping domain '{domain}'")
                continue

            # Schema examples: one per description (9 per domain = ~135 total)
            for desc in domain_descs:
                schema_examples.append(
                    _make_schema_example(system_prompt, desc, schema_tql, domain)
                )

            # Combined examples: 3 per domain (pick evenly spaced descriptions)
            top_queries = domain_typeql.get(domain, [])[:5]
            if not top_queries:
                print(f"  [WARN] No queries found for domain '{domain}' — skipping combined examples")
                continue

            n = len(domain_descs)
            # Pick 3 indices evenly spaced across the descriptions
            if n >= 3:
                indices = [0, n // 3, (2 * n) // 3]
            else:
                indices = list(range(n))
            # Deduplicate while preserving order
            seen: set[int] = set()
            deduped_indices = []
            for idx in indices:
                if idx not in seen:
                    seen.add(idx)
                    deduped_indices.append(idx)

            for idx in deduped_indices[:3]:
                desc = domain_descs[idx]
                combined_examples.append(
                    _make_combined_example(
                        system_prompt, desc, schema_tql, top_queries, domain
                    )
                )

        print(
            f"[INFO] Schema examples:   {len(schema_examples):4d}  "
            f"(~{len(DOMAINS)} domains × 9 descriptions)"
        )
        print(
            f"[INFO] Combined examples: {len(combined_examples):4d}  "
            f"(~{len(DOMAINS)} domains × 3)"
        )

    # ------------------------------------------------------------------
    # 4. Mutation examples (optional, merged as-is)
    # ------------------------------------------------------------------
    mutation_count = 0

    if not MUTATION_EXAMPLES_JSONL.exists():
        print(f"\n[WARN] mutation_examples.jsonl not found: {MUTATION_EXAMPLES_JSONL}")
        print("[WARN] Skipping mutation combined examples — run 02b_generate_mutation_queries.py first")
    else:
        print(f"\n[INFO] Loading mutation examples from {MUTATION_EXAMPLES_JSONL} ...")
        bad_lines = 0
        with MUTATION_EXAMPLES_JSONL.open("r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    ex = json.loads(line)
                    # Tag as combined so it is interleaved correctly
                    ex["_kind"] = "combined"
                    combined_examples.append(ex)
                    mutation_count += 1
                except json.JSONDecodeError as exc:
                    bad_lines += 1
                    if bad_lines <= 3:
                        print(f"  [WARN] Line {lineno}: bad JSON: {exc}")

        if bad_lines:
            print(f"  [WARN] {bad_lines} malformed lines skipped in mutation_examples.jsonl")
        print(f"[INFO] Mutation examples loaded: {mutation_count}")

    # ------------------------------------------------------------------
    # 5. Pool and split (query + schema + combined shuffled together, 80/20)
    # ------------------------------------------------------------------
    print("\n[INFO] Building train/val split ...")

    all_non_test: list[dict] = query_examples + schema_examples + combined_examples
    rng.shuffle(all_non_test)

    n_total = len(all_non_test)
    n_val = int(n_total * VAL_FRACTION)
    n_train = n_total - n_val

    val_pool = all_non_test[:n_val]
    train_pool = all_non_test[n_val:]

    # ------------------------------------------------------------------
    # 6. Interleave non-query examples in the train set
    #    (prevents all schema examples clustering at one point)
    # ------------------------------------------------------------------
    train_queries = [ex for ex in train_pool if ex.get("_kind") == "query"]
    train_other = [ex for ex in train_pool if ex.get("_kind") != "query"]

    train_examples = _interleave(train_queries, train_other, INTERLEAVE_INTERVAL)

    # Val: leave in shuffled order (already well-mixed)
    val_examples = val_pool

    print(
        f"[INFO] Split — train: {len(train_examples)}, "
        f"val: {len(val_examples)}, "
        f"test: {len(test_examples)}"
    )

    # ------------------------------------------------------------------
    # 7. Write output files
    # ------------------------------------------------------------------
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for label, path, examples in [
        ("train", OUTPUT_TRAIN, train_examples),
        ("val", OUTPUT_VAL, val_examples),
        ("test", OUTPUT_TEST, test_examples),
    ]:
        print(f"[INFO] Writing {path} ({len(examples)} lines) ...")
        with path.open("w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(_strip_kind(ex), ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------
    # 8. Write build manifest
    # ------------------------------------------------------------------
    manifest = {
        "system_prompt_sha256": system_prompt_sha256,
        "random_seed": RANDOM_SEED,
        "train_count": len(train_examples),
        "val_count": len(val_examples),
        "test_count": len(test_examples),
        "total_count": len(train_examples) + len(val_examples) + len(test_examples),
        "query_examples_non_test": len(query_examples),
        "schema_examples": len(schema_examples),
        "combined_examples_fresh": len(combined_examples) - mutation_count,
        "mutation_examples_merged": mutation_count,
    }
    with BUILD_MANIFEST.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Build manifest written to {BUILD_MANIFEST}")

    # ------------------------------------------------------------------
    # 9. Validate all output files
    # ------------------------------------------------------------------
    print("\n[INFO] Validating output files ...")
    total_valid = 0
    total_errors = 0

    for label, path in [
        ("train", OUTPUT_TRAIN),
        ("val", OUTPUT_VAL),
        ("test", OUTPUT_TEST),
    ]:
        valid, errors = validate_jsonl_file(path)
        total_valid += valid
        total_errors += len(errors)
        status = "OK" if not errors else "ERRORS"
        print(f"  [{status:6s}] {label}.jsonl — {valid} valid, {len(errors)} errors")
        for e in errors[:5]:
            print(f"    {e}")
        if len(errors) > 5:
            print(f"    ... and {len(errors) - 5} more")

    # ------------------------------------------------------------------
    # 10. Summary
    # ------------------------------------------------------------------
    print("\n=== SUMMARY ===")
    print(f"  train.jsonl : {len(train_examples):7d} lines")
    print(f"  val.jsonl   : {len(val_examples):7d} lines")
    print(f"  test.jsonl  : {len(test_examples):7d} lines")
    print(f"  total       : {total_valid:7d} validated")
    print(f"  errors      : {total_errors}")
    print()

    if total_errors > 0:
        print("[FAIL] Build completed with validation errors.")
        return 1

    print("[OK] Build complete — all output files valid.")
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build train/val/test JSONL splits for OpenAI fine-tuning."
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate existing output files without rebuilding.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.validate_only:
        return validate_only_mode()
    return build_mode()


if __name__ == "__main__":
    sys.exit(main())
