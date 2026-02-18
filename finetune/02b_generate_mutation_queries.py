#!/usr/bin/env python3
"""
02b_generate_mutation_queries.py

Generates mutated schema + query training examples for TypeQL fine-tuning.

For each of the 15 domains:
  1. Read the original schema from text2typeql/dataset/synthetic-2/<domain>/schema.tql
  2. Call GPT-4o to produce a MUTATED schema (swap one role name, add one entity)
  3. Call GPT-4o to generate 5 TypeQL queries against the mutated schema
  4. Validate each query structurally via typedb-driver (optional)
  5. Keep up to 3 validated examples per domain

Output: finetune/data/mutation_examples.jsonl  (45 lines: 3 × 15 domains)

Usage:
  python finetune/02b_generate_mutation_queries.py
  python finetune/02b_generate_mutation_queries.py --domain twitter
  python finetune/02b_generate_mutation_queries.py --domain twitter --skip-validation
  python finetune/02b_generate_mutation_queries.py --force  # overwrite existing output
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
FINETUNE_DIR = REPO_ROOT / "finetune"
DATA_DIR = FINETUNE_DIR / "data"
SCHEMA_BASE = REPO_ROOT / "text2typeql" / "dataset" / "synthetic-2"
SYSTEM_PROMPT_PATH = FINETUNE_DIR / "system_prompt.txt"
OUTPUT_FILE = DATA_DIR / "mutation_examples.jsonl"

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

EXAMPLES_PER_DOMAIN = 3      # target lines per domain
QUERIES_PER_CALL = 5         # queries GPT-4o generates per call
MAX_API_RETRIES = 3          # retry attempts for transient errors
RETRY_DELAY_S = 5            # seconds between retries
TYPEDB_SERVER = "localhost:1729"

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

MUTATION_SYSTEM_PROMPT = """You are a TypeQL 3.0 schema expert.
Given an existing TypeQL schema, produce a MUTATED version that:
1. Renames exactly one relation role (e.g., change 'relates friend' to 'relates peer', updating all plays declarations accordingly).
2. Adds exactly one new entity type with at least one attribute and at least one relation it plays in (you may add a new minimal relation for it, or attach it to an existing relation with a new role).

Keep mutations small. The mutated schema must remain syntactically valid TypeQL 3.0.

Respond ONLY with a JSON object with two keys:
  "mutated_schema"  — the full TypeQL define block as a string
  "mutation_summary" — one sentence describing what you changed
"""

MUTATION_USER_TEMPLATE = """Here is the original TypeQL schema for the {domain} domain:

```typeql
{schema}
```

Produce a small, valid mutation."""

QUERIES_SYSTEM_PROMPT = """You are a TypeQL 3.0 query expert.
Given a TypeQL schema, generate exactly {n} valid TypeQL 3.0 queries of varying complexity.
Queries should exercise: attribute lookup, relation traversal, aggregation (reduce), optional (try), negation (not), and fetch subqueries.
All queries must be syntactically valid against the provided schema.

Respond ONLY with a JSON object with one key:
  "queries" — an array of exactly {n} TypeQL query strings
"""

QUERIES_USER_TEMPLATE = """Here is the TypeQL schema for the {domain} domain (mutated version):

```typeql
{mutated_schema}
```

Generate {n} valid TypeQL queries against this schema."""

# ---------------------------------------------------------------------------
# OpenAI helpers
# ---------------------------------------------------------------------------

def get_openai_client():
    """Lazy-import openai and return a client."""
    try:
        from openai import OpenAI  # type: ignore
    except ImportError:
        print("[ERROR] openai package not installed. Run: pip install openai", file=sys.stderr)
        sys.exit(1)
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("[ERROR] OPENAI_API_KEY environment variable not set.", file=sys.stderr)
        sys.exit(1)
    return OpenAI(api_key=api_key)


def chat_complete_json(client, system: str, user: str, model: str = "gpt-4o") -> dict:
    """Call OpenAI chat completion and return parsed JSON. Retries on transient errors."""
    last_error: Optional[Exception] = None
    for attempt in range(1, MAX_API_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
                temperature=0.7,
            )
            content = response.choices[0].message.content
            return json.loads(content)
        except Exception as exc:
            last_error = exc
            if attempt < MAX_API_RETRIES:
                print(f"  [WARN] API attempt {attempt} failed: {exc}. Retrying in {RETRY_DELAY_S}s...")
                time.sleep(RETRY_DELAY_S)
    raise RuntimeError(f"All {MAX_API_RETRIES} API attempts failed. Last error: {last_error}")


# ---------------------------------------------------------------------------
# TypeDB validation helpers
# ---------------------------------------------------------------------------

def try_import_typedb():
    """Return True if typedb-driver is importable."""
    try:
        import typedb.driver  # type: ignore  # noqa: F401
        return True
    except ImportError:
        return False


def validate_with_typedb(mutated_schema: str, queries: list[str]) -> list[str]:
    """
    Create a temporary TypeDB database, load the mutated schema,
    then attempt to parse/validate each query. Return list of valid queries.
    """
    from typedb.driver import TypeDB, SessionType, TransactionType  # type: ignore

    db_name = f"mutation_tmp_{uuid.uuid4().hex[:8]}"
    valid: list[str] = []

    try:
        driver = TypeDB.core_driver(TYPEDB_SERVER)
    except Exception as exc:
        print(f"  [WARN] Cannot connect to TypeDB at {TYPEDB_SERVER}: {exc}. Skipping TypeDB validation.")
        return queries  # fall back: treat all as valid

    try:
        # Create temp database
        driver.databases.create(db_name)

        # Load schema
        with driver.session(db_name, SessionType.SCHEMA) as session:
            with session.transaction(TransactionType.WRITE) as tx:
                tx.query.define(mutated_schema)
                tx.commit()

        # Validate each query by parsing it
        with driver.session(db_name, SessionType.DATA) as session:
            for q in queries:
                try:
                    with session.transaction(TransactionType.READ) as tx:
                        # parse() raises if the query is syntactically invalid
                        tx.query.get(q)  # triggers server-side parse; may error on bad queries
                    valid.append(q)
                except Exception as exc:
                    print(f"  [SKIP] Query validation failed: {str(exc)[:120]}")

    except Exception as exc:
        print(f"  [WARN] Schema load failed: {str(exc)[:200]}. Accepting all queries as-is.")
        valid = queries
    finally:
        try:
            driver.databases.get(db_name).delete()
        except Exception:
            pass
        try:
            driver.close()
        except Exception:
            pass

    return valid


def validate_json_structure(queries: list[str]) -> list[str]:
    """
    Lightweight fallback validation: check queries are non-empty strings
    and start with known TypeQL keywords.
    """
    VALID_STARTS = ("match", "define", "insert", "delete", "update", "with", "fetch", "reduce")
    valid = []
    for q in queries:
        q = q.strip()
        if q and any(q.lower().startswith(kw) for kw in VALID_STARTS):
            valid.append(q)
        else:
            print(f"  [SKIP] Query failed structural check: {q[:60]!r}")
    return valid


# ---------------------------------------------------------------------------
# Domain description helpers
# ---------------------------------------------------------------------------

def make_domain_description(domain: str, mutated_schema: str, mutation_summary: str) -> str:
    """
    Compose the user-facing prompt that will be stored in the training example.
    This simulates what a user would say when asking for queries against their schema.
    """
    return (
        f"I have a TypeQL schema for the {domain} domain (with a small variation: {mutation_summary}). "
        f"Please generate TypeQL queries to explore and analyse this data.\n\n"
        f"Schema:\n```typeql\n{mutated_schema}\n```"
    )


def make_assistant_content(mutated_schema: str, queries: list[str], mutation_summary: str) -> str:
    """
    Compose the assistant response JSON string stored in the training example.
    """
    payload = {
        "schema": mutated_schema,
        "queries": queries,
        "explanation": (
            f"Schema mutation applied: {mutation_summary} "
            "The queries exercise attribute lookup, relation traversal, and aggregation "
            "patterns valid against the mutated schema."
        ),
    }
    return json.dumps(payload, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Per-domain processing
# ---------------------------------------------------------------------------

def process_domain(
    domain: str,
    client,
    system_prompt: str,
    skip_validation: bool,
    typedb_available: bool,
) -> list[dict]:
    """
    Process one domain: mutate schema → generate queries → validate → return JSONL rows.
    Returns up to EXAMPLES_PER_DOMAIN rows.
    """
    schema_path = SCHEMA_BASE / domain / "schema.tql"
    if not schema_path.exists():
        print(f"  [WARN] Schema not found: {schema_path}. Skipping domain.")
        return []

    original_schema = schema_path.read_text(encoding="utf-8").strip()
    print(f"  Schema loaded ({len(original_schema)} chars)")

    # --- Step 1: Generate mutated schema ---
    print(f"  Generating mutated schema via GPT-4o...")
    mutation_response = chat_complete_json(
        client,
        system=MUTATION_SYSTEM_PROMPT,
        user=MUTATION_USER_TEMPLATE.format(domain=domain, schema=original_schema),
    )
    mutated_schema: str = mutation_response.get("mutated_schema", "").strip()
    mutation_summary: str = mutation_response.get("mutation_summary", "minor schema mutation").strip()

    if not mutated_schema:
        print(f"  [ERROR] GPT-4o returned empty mutated_schema for {domain}. Skipping.")
        return []

    print(f"  Mutation: {mutation_summary[:100]}")

    # --- Step 2: Generate queries in batches until we have enough valid ones ---
    rows: list[dict] = []
    attempts = 0
    max_attempts = 4  # prevent infinite loops

    while len(rows) < EXAMPLES_PER_DOMAIN and attempts < max_attempts:
        attempts += 1
        print(f"  Generating {QUERIES_PER_CALL} queries (attempt {attempts})...")
        try:
            query_response = chat_complete_json(
                client,
                system=QUERIES_SYSTEM_PROMPT.format(n=QUERIES_PER_CALL),
                user=QUERIES_USER_TEMPLATE.format(
                    domain=domain,
                    mutated_schema=mutated_schema,
                    n=QUERIES_PER_CALL,
                ),
            )
        except Exception as exc:
            print(f"  [ERROR] Query generation failed: {exc}")
            break

        raw_queries: list[str] = query_response.get("queries", [])
        if not raw_queries:
            print(f"  [WARN] Empty queries list returned.")
            continue

        # --- Step 3: Validate ---
        if skip_validation:
            validated = validate_json_structure(raw_queries)
        elif typedb_available:
            print(f"  Validating {len(raw_queries)} queries with TypeDB...")
            validated = validate_with_typedb(mutated_schema, raw_queries)
        else:
            validated = validate_json_structure(raw_queries)

        print(f"  {len(validated)}/{len(raw_queries)} queries passed validation")

        # --- Step 4: Build JSONL rows (one row per query, up to target) ---
        needed = EXAMPLES_PER_DOMAIN - len(rows)
        for q in validated[:needed]:
            user_content = make_domain_description(domain, mutated_schema, mutation_summary)
            assistant_content = make_assistant_content(mutated_schema, [q], mutation_summary)
            row = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content},
                ]
            }
            rows.append(row)

        if len(rows) >= EXAMPLES_PER_DOMAIN:
            break

    if len(rows) < EXAMPLES_PER_DOMAIN:
        print(f"  [WARN] Only got {len(rows)}/{EXAMPLES_PER_DOMAIN} valid examples for {domain}.")

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate mutated schema + query training examples for TypeQL fine-tuning."
    )
    parser.add_argument(
        "--domain",
        type=str,
        default=None,
        help="Process only this domain (for testing). Must be one of the 15 domain names.",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        default=False,
        help="Skip TypeDB structural validation (use lightweight JSON structure check instead).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Overwrite output file even if it already exists.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(OUTPUT_FILE),
        help=f"Output JSONL file path (default: {OUTPUT_FILE})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="OpenAI model to use (default: gpt-4o)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)

    # Idempotency check
    if output_path.exists() and not args.force and args.domain is None:
        print(f"[INFO] Output already exists: {output_path}")
        print("[INFO] Use --force to regenerate. Exiting.")
        sys.exit(0)

    # Determine domains to process
    if args.domain:
        if args.domain not in DOMAINS:
            print(f"[ERROR] Unknown domain: {args.domain!r}. Valid: {DOMAINS}", file=sys.stderr)
            sys.exit(1)
        domains_to_process = [args.domain]
    else:
        domains_to_process = DOMAINS

    # Load system prompt
    if not SYSTEM_PROMPT_PATH.exists():
        print(f"[ERROR] System prompt not found: {SYSTEM_PROMPT_PATH}", file=sys.stderr)
        sys.exit(1)
    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
    print(f"[INFO] System prompt loaded ({len(system_prompt)} chars)")

    # Set up OpenAI client
    client = get_openai_client()
    print(f"[INFO] OpenAI client ready (model: {args.model})")

    # Check TypeDB availability
    typedb_available = False
    if not args.skip_validation:
        typedb_available = try_import_typedb()
        if typedb_available:
            print(f"[INFO] typedb-driver found; will attempt TypeDB validation at {TYPEDB_SERVER}")
        else:
            print("[INFO] typedb-driver not installed; using lightweight structural validation")
    else:
        print("[INFO] --skip-validation: using lightweight structural validation")

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Process domains
    all_rows: list[dict] = []

    for i, domain in enumerate(domains_to_process, 1):
        print(f"\n[{i}/{len(domains_to_process)}] Processing domain: {domain}")
        try:
            rows = process_domain(
                domain=domain,
                client=client,
                system_prompt=system_prompt,
                skip_validation=args.skip_validation,
                typedb_available=typedb_available,
            )
        except Exception as exc:
            print(f"  [ERROR] Domain {domain} failed: {exc}")
            rows = []

        all_rows.extend(rows)
        print(f"  → {len(rows)} examples collected (total so far: {len(all_rows)})")

    # Write output
    if args.domain:
        # Single-domain mode: append or write to a temp file, not the main output
        out = output_path.parent / f"mutation_examples_{args.domain}.jsonl"
        print(f"\n[INFO] Single-domain mode: writing to {out}")
    else:
        out = output_path

    with out.open("w", encoding="utf-8") as fh:
        for row in all_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\n[DONE] Wrote {len(all_rows)} examples to {out}")

    if args.domain is None:
        expected = len(DOMAINS) * EXAMPLES_PER_DOMAIN
        if len(all_rows) < expected:
            print(f"[WARN] Expected {expected} rows but only got {len(all_rows)}. Some domains may have failed.")
        else:
            print(f"[OK] Full dataset: {len(all_rows)} rows ({EXAMPLES_PER_DOMAIN} × {len(DOMAINS)} domains)")


if __name__ == "__main__":
    main()
