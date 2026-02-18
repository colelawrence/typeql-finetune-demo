#!/usr/bin/env python3
"""
syntax_check.py — Fine-tuned model quality gate.

Evaluates a fine-tuned OpenAI model on held-out northwind test queries,
validating structural validity (syntax + schema correctness) via TypeDB.
Exits 0 if structural validity ≥ 90%, exits 1 otherwise.

Usage:
    python finetune/eval/syntax_check.py \\
        --model ft:gpt-4o-mini:org:name:id \\
        --test finetune/data/test.jsonl \\
        [--typedb-addr localhost:1729] \\
        [--limit N] \\
        [--skip-typedb] \\
        [--novel finetune/eval/novel_domain_prompts.txt]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import openai

# ---------------------------------------------------------------------------
# Paths (relative to repo root when run as: python finetune/eval/syntax_check.py)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
SYSTEM_PROMPT_PATH = REPO_ROOT / "finetune" / "system_prompt.txt"
NORTHWIND_SCHEMA_PATH = (
    REPO_ROOT / "text2typeql" / "dataset" / "synthetic-2" / "northwind" / "schema.tql"
)
EVAL_DB = "eval_northwind"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_system_prompt() -> str:
    return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")


def load_northwind_schema() -> str:
    return NORTHWIND_SCHEMA_PATH.read_text(encoding="utf-8")


def build_system_message(system_prompt: str, schema: str) -> str:
    """Concatenate base system prompt + schema section (matches training format)."""
    return f"{system_prompt}\n\nSchema:\n{schema}"


def load_test_examples(path: str, limit: int | None = None) -> list[dict]:
    """Load test.jsonl and extract user questions + expected structure."""
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            msgs = record["messages"]
            # messages: [system, user, assistant]
            user_msg = next((m["content"] for m in msgs if m["role"] == "user"), None)
            if user_msg is None:
                continue
            examples.append({"user": user_msg})
            if limit and len(examples) >= limit:
                break
    return examples


def call_model(client: openai.OpenAI, model: str, system_msg: str, user_msg: str) -> str:
    """Call the fine-tuned model and return raw response text."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0,
    )
    return response.choices[0].message.content or ""


def parse_response(raw: str) -> dict | None:
    """Parse model JSON response. Returns None on failure."""
    raw = raw.strip()
    # Strip markdown code fences if present
    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(
            l for l in lines if not l.startswith("```")
        ).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# TypeDB validation
# ---------------------------------------------------------------------------


def setup_typedb(addr: str, schema_tql: str):
    """Connect to TypeDB, create eval database, load northwind schema.
    Returns (driver, session) tuple, or raises on failure."""
    from typedb.driver import TypeDB, TransactionType, SessionType

    driver = TypeDB.core_driver(addr)

    # Drop existing eval db if it exists (clean slate)
    if driver.databases.contains(EVAL_DB):
        driver.databases.get(EVAL_DB).delete()

    driver.databases.create(EVAL_DB)

    # Load schema
    schema_session = driver.session(EVAL_DB, SessionType.SCHEMA)
    try:
        tx = schema_session.transaction(TransactionType.WRITE)
        try:
            tx.query(schema_tql)
            tx.commit()
        except Exception:
            tx.close()
            raise
    finally:
        schema_session.close()

    return driver


def validate_query_typedb(driver, query: str) -> tuple[bool, str]:
    """Validate a single TypeQL query against the northwind schema.
    Returns (is_valid, error_message)."""
    from typedb.driver import TransactionType, SessionType

    session = driver.session(EVAL_DB, SessionType.DATA)
    tx = None
    try:
        tx = session.transaction(TransactionType.READ)
        result = tx.query(query)
        # Consume at least first result to trigger evaluation / validation
        try:
            if hasattr(result, "__iter__"):
                for _ in result:
                    break
        except StopIteration:
            pass
        return True, ""
    except Exception as e:
        return False, str(e)
    finally:
        if tx is not None:
            try:
                tx.close()
            except Exception:
                pass
        session.close()


def teardown_typedb(driver):
    """Drop eval database and close driver."""
    try:
        if driver.databases.contains(EVAL_DB):
            driver.databases.get(EVAL_DB).delete()
    except Exception:
        pass
    try:
        driver.close()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Main eval flows
# ---------------------------------------------------------------------------


def run_northwind_eval(
    client: openai.OpenAI,
    model: str,
    test_path: str,
    typedb_addr: str,
    skip_typedb: bool,
    limit: int | None,
) -> int:
    """Evaluate model on test.jsonl. Returns exit code (0=pass, 1=fail)."""
    system_prompt = load_system_prompt()
    schema_tql = load_northwind_schema()
    system_msg = build_system_message(system_prompt, schema_tql)

    examples = load_test_examples(test_path, limit)
    print(f"Loaded {len(examples)} test examples from {test_path}")

    driver = None
    if not skip_typedb:
        print(f"Setting up TypeDB at {typedb_addr} with northwind schema…")
        try:
            driver = setup_typedb(typedb_addr, schema_tql)
            print("TypeDB ready.")
        except Exception as e:
            print(f"ERROR: Could not connect to TypeDB: {e}", file=sys.stderr)
            print("  → Run: docker run -d --name typedb -p 1729:1729 vaticle/typedb:3.3.0 --development-mode.enabled=true")
            print("  → Or pass --skip-typedb to validate structure without TypeDB.")
            return 1
    else:
        print("Skipping TypeDB validation (--skip-typedb).")

    total_queries = 0
    valid_queries = 0
    invalid_examples: list[dict] = []
    parse_failures = 0

    print(f"\nRunning eval on {len(examples)} examples…\n")

    for i, ex in enumerate(examples, 1):
        user_msg = ex["user"]
        prefix = f"[{i:4d}/{len(examples)}]"

        try:
            raw = call_model(client, model, system_msg, user_msg)
        except Exception as e:
            print(f"{prefix} API error: {e}")
            parse_failures += 1
            time.sleep(0.5)
            continue

        parsed = parse_response(raw)
        if parsed is None:
            print(f"{prefix} JSON parse failure")
            parse_failures += 1
            time.sleep(0.5)
            continue

        queries = parsed.get("queries", [])
        if not isinstance(queries, list):
            queries = []

        if not queries:
            # Schema-only response — count as valid structural output
            print(f"{prefix} OK (schema-only, no queries)")
            time.sleep(0.5)
            continue

        for q_idx, q in enumerate(queries):
            total_queries += 1
            if skip_typedb:
                # Structural check: non-empty query string
                if q and isinstance(q, str) and q.strip():
                    valid_queries += 1
                    status = "OK"
                else:
                    status = "EMPTY"
                    invalid_examples.append(
                        {"example": i, "query_idx": q_idx, "user": user_msg, "error": "empty query string"}
                    )
            else:
                ok, err = validate_query_typedb(driver, q)
                if ok:
                    valid_queries += 1
                    status = "OK"
                else:
                    status = f"INVALID: {err[:80]}"
                    invalid_examples.append(
                        {
                            "example": i,
                            "query_idx": q_idx,
                            "user": user_msg,
                            "query": q[:200],
                            "error": err,
                        }
                    )

        queries_label = f"{len(queries)} quer{'y' if len(queries)==1 else 'ies'}"
        print(f"{prefix} {queries_label} — {status}")

        time.sleep(0.5)  # rate limiting

    # Teardown
    if driver is not None:
        teardown_typedb(driver)

    # Summary
    print("\n" + "=" * 60)
    print("NORTHWIND EVAL SUMMARY")
    print("=" * 60)
    print(f"  Examples evaluated : {len(examples)}")
    print(f"  Parse failures     : {parse_failures}")
    print(f"  Total queries      : {total_queries}")
    print(f"  Valid queries      : {valid_queries}")
    print(f"  Invalid queries    : {total_queries - valid_queries}")

    if total_queries == 0:
        print("\nNo queries to evaluate — check model output format.")
        return 1

    pct = valid_queries / total_queries * 100
    threshold = 90.0
    print(f"\n  Structural validity: {pct:.1f}% (≥{threshold:.0f}% required)")

    if invalid_examples:
        print("\nFirst 5 invalid examples:")
        for ex in invalid_examples[:5]:
            print(f"  Example {ex['example']}, query {ex.get('query_idx', 0)}: {ex['error'][:120]}")

    if pct >= threshold:
        print("\n✅ PASS — structural validity meets threshold.")
        return 0
    else:
        print("\n❌ FAIL — structural validity below threshold.")
        return 1


def run_novel_eval(
    client: openai.OpenAI,
    model: str,
    prompts_path: str,
) -> None:
    """Qualitative eval on novel domain prompts. No TypeDB, no exit code."""
    system_prompt = load_system_prompt()
    # Novel domain: model must generate its own schema — no schema in context
    system_msg = system_prompt

    prompts = []
    with open(prompts_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(line)

    print(f"Loaded {len(prompts)} novel domain prompts from {prompts_path}\n")

    results = {"schema_generated": 0, "queries_generated": 0, "parse_ok": 0, "parse_fail": 0}

    for i, prompt in enumerate(prompts, 1):
        print(f"[{i:2d}/{len(prompts)}] {prompt[:70]}…")
        try:
            raw = call_model(client, model, system_msg, prompt)
        except Exception as e:
            print(f"         API error: {e}\n")
            results["parse_fail"] += 1
            time.sleep(0.5)
            continue

        parsed = parse_response(raw)
        if parsed is None:
            print(f"         ⚠ JSON parse failure\n")
            results["parse_fail"] += 1
            time.sleep(0.5)
            continue

        results["parse_ok"] += 1
        schema = parsed.get("schema", "")
        queries = parsed.get("queries", [])
        explanation = parsed.get("explanation", "")

        has_schema = bool(schema and schema.strip())
        has_queries = bool(queries)

        if has_schema:
            results["schema_generated"] += 1
        if has_queries:
            results["queries_generated"] += 1

        schema_status = f"schema({len(schema)} chars)" if has_schema else "no schema"
        queries_status = f"{len(queries)} quer{'y' if len(queries)==1 else 'ies'}" if has_queries else "no queries"
        expl_status = f"explanation({len(explanation)} chars)" if explanation else "no explanation"
        print(f"         → {schema_status} | {queries_status} | {expl_status}")
        if has_schema:
            print(f"           Schema preview: {schema[:100].strip()!r}")
        print()

        time.sleep(0.5)

    print("=" * 60)
    print("NOVEL DOMAIN EVAL SUMMARY (qualitative)")
    print("=" * 60)
    print(f"  Prompts evaluated    : {len(prompts)}")
    print(f"  Parse OK             : {results['parse_ok']}")
    print(f"  Parse failures       : {results['parse_fail']}")
    print(f"  Schemas generated    : {results['schema_generated']}")
    print(f"  Queries generated    : {results['queries_generated']}")
    print("\n(Novel domain results are qualitative — no pass/fail threshold.)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Eval fine-tuned TypeQL model on northwind test set + novel domains."
    )
    p.add_argument(
        "--model",
        required=True,
        help="Fine-tuned model ID, e.g. ft:gpt-4o-mini:org:name:id",
    )
    p.add_argument(
        "--test",
        default=str(REPO_ROOT / "finetune" / "data" / "test.jsonl"),
        help="Path to test.jsonl (default: finetune/data/test.jsonl)",
    )
    p.add_argument(
        "--typedb-addr",
        default="localhost:1729",
        help="TypeDB server address (default: localhost:1729)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit evaluation to first N test examples",
    )
    p.add_argument(
        "--skip-typedb",
        action="store_true",
        help="Skip TypeDB validation; only check JSON parse + non-empty queries",
    )
    p.add_argument(
        "--novel",
        metavar="FILE",
        default=None,
        help="Path to novel domain prompts file; run qualitative eval after main eval",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set.", file=sys.stderr)
        return 1

    client = openai.OpenAI(api_key=api_key)

    # Main northwind eval
    exit_code = run_northwind_eval(
        client=client,
        model=args.model,
        test_path=args.test,
        typedb_addr=args.typedb_addr,
        skip_typedb=args.skip_typedb,
        limit=args.limit,
    )

    # Optional: novel domain eval
    if args.novel:
        print("\n" + "=" * 60)
        print("NOVEL DOMAIN EVAL")
        print("=" * 60 + "\n")
        run_novel_eval(client=client, model=args.model, prompts_path=args.novel)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
