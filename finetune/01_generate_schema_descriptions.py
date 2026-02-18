#!/usr/bin/env python3
"""
01_generate_schema_descriptions.py

Generates 9 natural-language descriptions per domain schema using GPT-4o.
Reads neo4j_schema.json (property-graph vocabulary) for each of the 15 domains
and writes output to finetune/data/schema_descriptions.json.

Usage:
  python finetune/01_generate_schema_descriptions.py
  python finetune/01_generate_schema_descriptions.py --dry-run
  python finetune/01_generate_schema_descriptions.py --domain twitter
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent.resolve()
DATASET_DIR = REPO_ROOT / "text2typeql" / "dataset" / "synthetic-2"
OUTPUT_FILE = REPO_ROOT / "finetune" / "data" / "schema_descriptions.json"

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
# Prompt construction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are helping build a dataset for fine-tuning a text-to-TypeQL model.

Your job is to generate natural-language descriptions of a graph database domain.
Given a Neo4j property-graph schema (node labels, relationship types, and key properties),
write exactly 9 distinct descriptions that a developer might type when describing the
domain they want to model in a graph database.

Rules:
1. Return a JSON array of exactly 9 strings — nothing else, no markdown fences.
2. Do NOT mention TypeQL, type hierarchies, roles, or any TypeQL-specific vocabulary.
3. Do NOT use the raw schema names verbatim (e.g. don't say "FOLLOWS relationship" or
   "User node") — rephrase naturally.
4. Each description should sound like something a developer would actually type as a
   database request or requirement, not a formal specification.
5. Vary the style across the 9 descriptions:
   - At least 2 imperative ("Build a ...", "Create a database for ...")
   - At least 2 noun-phrase ("A social network ...", "An e-commerce platform ...")
   - At least 1 very brief (≤ 1 sentence)
   - At least 1 verbose (2-3 sentences with specifics)
   - At least 1 technical (mentions graph concepts like nodes, edges, properties)
   - At least 1 casual / conversational
   - The remaining can be any style
6. Each description must clearly convey the same domain so a reader could infer
   what data is stored and the main entities involved.

Output format — raw JSON array only:
["description 1", "description 2", ..., "description 9"]
"""


def build_user_prompt(domain: str, schema: dict) -> str:
    """Summarise the neo4j schema into a compact prompt for GPT-4o."""
    nodes = list(schema.get("node_props", {}).keys())
    rels = schema.get("relationships", [])
    # Deduplicate relationship types
    rel_types = sorted({r["type"] for r in rels})

    # Build a compact edge list: SRC -[TYPE]-> DST (deduplicated)
    edge_set: set[tuple[str, str, str]] = set()
    for r in rels:
        edge_set.add((r["start"], r["type"], r["end"]))
    edges = [f"{s} -[{t}]-> {e}" for s, t, e in sorted(edge_set)]

    # Sample a few property names per node (to give flavour, not overwhelm)
    node_summary_parts = []
    for label, props in schema.get("node_props", {}).items():
        prop_names = [p["property"] for p in props[:5]]  # first 5 props max
        node_summary_parts.append(f"  {label}: {', '.join(prop_names)}")

    node_summary = "\n".join(node_summary_parts)
    edge_summary = "\n".join(f"  {e}" for e in edges[:30])  # cap at 30

    return (
        f"Domain: {domain}\n\n"
        f"Node labels and sample properties:\n{node_summary}\n\n"
        f"Relationship types:\n  {', '.join(rel_types)}\n\n"
        f"Graph edges (node -[REL]-> node):\n{edge_summary}\n\n"
        f"Generate 9 natural-language descriptions for this domain."
    )


# ---------------------------------------------------------------------------
# API call with retry
# ---------------------------------------------------------------------------

def call_openai(client, domain: str, schema: dict, max_retries: int = 3) -> list[str]:
    """Call GPT-4o and return a list of exactly 9 description strings."""
    user_prompt = build_user_prompt(domain, schema)

    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.9,
                max_tokens=1500,
            )
            raw = response.choices[0].message.content.strip()

            # Strip optional markdown fences
            if raw.startswith("```"):
                lines = raw.splitlines()
                # Remove first and last fence lines
                raw = "\n".join(
                    line for line in lines
                    if not line.strip().startswith("```")
                ).strip()

            parsed = json.loads(raw)

            if not isinstance(parsed, list):
                raise ValueError(f"Expected JSON array, got {type(parsed).__name__}")
            if len(parsed) != 9:
                raise ValueError(f"Expected 9 descriptions, got {len(parsed)}")
            for i, item in enumerate(parsed):
                if not isinstance(item, str):
                    raise ValueError(f"Item {i} is not a string: {item!r}")
                if not item.strip():
                    raise ValueError(f"Item {i} is empty")

            return parsed

        except Exception as exc:
            last_error = exc
            if attempt < max_retries:
                wait = 2 ** attempt
                print(
                    f"  [WARN] Attempt {attempt}/{max_retries} failed for '{domain}': {exc}. "
                    f"Retrying in {wait}s..."
                )
                time.sleep(wait)
            else:
                print(
                    f"  [ERROR] All {max_retries} attempts failed for '{domain}': {exc}"
                )

    raise RuntimeError(
        f"Failed to generate descriptions for domain '{domain}' "
        f"after {max_retries} attempts"
    ) from last_error


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_schema(domain: str) -> dict:
    schema_path = DATASET_DIR / domain / "neo4j_schema.json"
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema not found: {schema_path}")
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)


def dry_run(domains: list[str]) -> None:
    print("=== DRY RUN ===")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Domains to process ({len(domains)}):")
    for domain in domains:
        schema_path = DATASET_DIR / domain / "neo4j_schema.json"
        exists = "✓" if schema_path.exists() else "✗ MISSING"
        try:
            schema = load_schema(domain)
            nodes = list(schema.get("node_props", {}).keys())
            rels = sorted({r["type"] for r in schema.get("relationships", [])})
            print(
                f"  {exists}  {domain:20s}  "
                f"nodes={nodes}  rels={rels[:5]}{'...' if len(rels) > 5 else ''}"
            )
        except FileNotFoundError:
            print(f"  {exists}  {domain}")

    print()
    print("Would call gpt-4o once per domain (9 descriptions each).")
    print(f"Total API calls: {len(domains)}")
    print(f"Total descriptions: {len(domains)} × 9 = {len(domains) * 9}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate 9 NL descriptions per schema domain using GPT-4o."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen without calling the API.",
    )
    parser.add_argument(
        "--domain",
        metavar="NAME",
        help="Process a single domain only (for testing).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-generate even if output file already exists.",
    )
    args = parser.parse_args()

    # Determine domains to process
    if args.domain:
        if args.domain not in DOMAINS:
            print(
                f"[ERROR] Unknown domain '{args.domain}'. "
                f"Valid domains: {', '.join(DOMAINS)}"
            )
            return 1
        domains_to_process = [args.domain]
    else:
        domains_to_process = list(DOMAINS)

    # Dry run — no API calls
    if args.dry_run:
        dry_run(domains_to_process)
        return 0

    # Idempotency check (only for full runs, not single-domain)
    if not args.domain and not args.force and OUTPUT_FILE.exists():
        print(
            f"[INFO] Output file already exists: {OUTPUT_FILE}\n"
            "       Use --force to regenerate. Exiting."
        )
        return 0

    # Load existing output for incremental updates (single-domain mode or partial runs)
    existing: dict[str, list[str]] = {}
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            existing = json.load(f)

    # Check API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("[ERROR] OPENAI_API_KEY environment variable is not set.")
        return 1

    # Import openai (lazy, so --dry-run works without it)
    try:
        from openai import OpenAI  # type: ignore[import]
    except ImportError:
        print("[ERROR] 'openai' package is not installed. Run: pip install openai")
        return 1

    client = OpenAI(api_key=api_key)

    # Create output directory
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    results: dict[str, list[str]] = dict(existing)
    errors: list[str] = []

    total = len(domains_to_process)
    for idx, domain in enumerate(domains_to_process, start=1):
        print(f"[{idx:2d}/{total}] Processing domain: {domain} ...")

        try:
            schema = load_schema(domain)
        except FileNotFoundError as exc:
            print(f"  [SKIP] {exc}")
            errors.append(domain)
            continue

        try:
            descriptions = call_openai(client, domain, schema)
            results[domain] = descriptions
            print(f"  [OK] Generated {len(descriptions)} descriptions.")

            # Write incrementally after each domain so progress isn't lost
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

        except RuntimeError as exc:
            print(f"  [FAIL] {exc}")
            errors.append(domain)

        # Small polite delay between API calls
        if idx < total:
            time.sleep(0.5)

    # Final summary
    print()
    print("=== SUMMARY ===")
    print(f"Domains processed: {len(results)}")
    print(f"Total descriptions: {sum(len(v) for v in results.values())}")
    print(f"Output written to: {OUTPUT_FILE}")

    if errors:
        print(f"[WARN] Failed domains ({len(errors)}): {', '.join(errors)}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
