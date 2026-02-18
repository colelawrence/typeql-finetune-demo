#!/usr/bin/env python3
"""
04_build_vector_index.py

Embeds all 13,939 questions from all_queries.csv using OpenAI
text-embedding-3-small (512 dims) and upserts them to the Cloudflare
Vectorize `typeql-examples` index.

Each vector:
  - ID:       {source}_{domain}_{original_index}  e.g. synthetic-1_twitter_0
  - Values:   512-dimensional float vector
  - Metadata: { domain, source, question, typeql }

Usage:
  python finetune/04_build_vector_index.py
  python finetune/04_build_vector_index.py --dry-run
  python finetune/04_build_vector_index.py --limit 100

ENV vars required (unless --dry-run):
  OPENAI_API_KEY   — OpenAI API key
  CF_ACCOUNT_ID    — Cloudflare account ID
  CF_API_TOKEN     — Cloudflare API token (Vectorize write permission)
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent.resolve()
CSV_PATH = REPO_ROOT / "text2typeql" / "dataset" / "all_queries.csv"

INDEX_NAME = "typeql-examples"
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIMS = 512
BATCH_SIZE = 100  # vectors per Vectorize upsert call
EMBED_BATCH_SIZE = 100  # texts per OpenAI embedding call


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def load_rows(csv_path: Path, limit: int | None = None) -> list[dict]:
    """Load rows from all_queries.csv."""
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for i, row in enumerate(reader):
            if limit is not None and i >= limit:
                break
            rows.append(row)
    return rows


def make_vector_id(row: dict) -> str:
    """Build a stable, filesystem-safe vector ID from row fields."""
    source = row["source"].replace("/", "-")
    domain = row["domain"].replace("/", "-")
    idx = row["original_index"].strip()
    return f"{source}_{domain}_{idx}"


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def embed_batch(client, texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts (max 100) using text-embedding-3-small."""
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=texts,
        dimensions=EMBED_DIMS,
    )
    # Sort by index to preserve order (API may not guarantee it)
    sorted_data = sorted(response.data, key=lambda d: d.index)
    return [d.embedding for d in sorted_data]


def embed_all(client, rows: list[dict], dry_run: bool = False) -> list[list[float]]:
    """Embed all questions in batches; returns list of embedding vectors."""
    if dry_run:
        print(
            f"  [DRY RUN] Would embed {len(rows)} questions in "
            f"{math.ceil(len(rows) / EMBED_BATCH_SIZE)} OpenAI API calls "
            f"(batch size {EMBED_BATCH_SIZE})."
        )
        return []

    questions = [row["question"] for row in rows]
    embeddings: list[list[float]] = []
    total_batches = math.ceil(len(questions) / EMBED_BATCH_SIZE)
    t_start = time.time()

    for batch_num, start in enumerate(range(0, len(questions), EMBED_BATCH_SIZE), 1):
        batch_texts = questions[start : start + EMBED_BATCH_SIZE]

        # Retry with exponential back-off
        for attempt in range(1, 4):
            try:
                vecs = embed_batch(client, batch_texts)
                embeddings.extend(vecs)
                break
            except Exception as exc:
                if attempt == 3:
                    raise
                wait = 2 ** attempt
                print(
                    f"  [WARN] Embedding batch {batch_num}/{total_batches} failed "
                    f"(attempt {attempt}/3): {exc}. Retrying in {wait}s..."
                )
                time.sleep(wait)

        elapsed = time.time() - t_start
        done = batch_num * EMBED_BATCH_SIZE
        if total_batches > 0:
            eta = (elapsed / batch_num) * (total_batches - batch_num)
        else:
            eta = 0
        print(
            f"  Embedded batch {batch_num}/{total_batches} "
            f"({min(done, len(questions))}/{len(questions)} questions) "
            f"ETA: {eta:.0f}s"
        )

    return embeddings


# ---------------------------------------------------------------------------
# Cloudflare Vectorize upsert
# ---------------------------------------------------------------------------

def cf_upsert_batch(
    account_id: str,
    api_token: str,
    index_name: str,
    vectors: list[dict],
) -> None:
    """
    Upsert a batch of vectors to Cloudflare Vectorize using the v2 HTTP API.

    vectors: list of { id, values, metadata }
    """
    import urllib.request
    import urllib.error

    url = (
        f"https://api.cloudflare.com/client/v4/accounts/{account_id}"
        f"/vectorize/v2/indexes/{index_name}/upsert"
    )
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/x-ndjson",
    }

    # Build NDJSON payload
    ndjson_lines = [json.dumps(v) for v in vectors]
    body = "\n".join(ndjson_lines).encode("utf-8")

    req = urllib.request.Request(url, data=body, headers=headers, method="POST")

    for attempt in range(1, 4):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw = resp.read().decode("utf-8")
                result = json.loads(raw)
                if not result.get("success", False):
                    errors = result.get("errors", [])
                    raise RuntimeError(f"Vectorize upsert failed: {errors}")
            return
        except urllib.error.HTTPError as exc:
            body_text = exc.read().decode("utf-8", errors="replace")
            if attempt == 3:
                raise RuntimeError(
                    f"HTTP {exc.code} from Vectorize: {body_text}"
                ) from exc
            wait = 2 ** attempt
            print(
                f"  [WARN] Upsert attempt {attempt}/3 failed (HTTP {exc.code}). "
                f"Retrying in {wait}s..."
            )
            time.sleep(wait)
        except Exception as exc:
            if attempt == 3:
                raise
            wait = 2 ** attempt
            print(
                f"  [WARN] Upsert attempt {attempt}/3 failed: {exc}. "
                f"Retrying in {wait}s..."
            )
            time.sleep(wait)


def upsert_all(
    account_id: str,
    api_token: str,
    rows: list[dict],
    embeddings: list[list[float]],
    dry_run: bool = False,
) -> int:
    """Build vector dicts and upsert to Vectorize in batches."""
    assert len(rows) == len(embeddings), "Row/embedding count mismatch"

    vectors = []
    for row, vec in zip(rows, embeddings):
        vectors.append(
            {
                "id": make_vector_id(row),
                "values": vec,
                "metadata": {
                    "domain": row["domain"],
                    "source": row["source"],
                    "question": row["question"],
                    "typeql": row["typeql"],
                },
            }
        )

    total_batches = math.ceil(len(vectors) / BATCH_SIZE)
    upserted = 0
    t_start = time.time()

    for batch_num, start in enumerate(range(0, len(vectors), BATCH_SIZE), 1):
        batch = vectors[start : start + BATCH_SIZE]

        if dry_run:
            upserted += len(batch)
            continue

        cf_upsert_batch(account_id, api_token, INDEX_NAME, batch)
        upserted += len(batch)

        elapsed = time.time() - t_start
        eta = (elapsed / batch_num) * (total_batches - batch_num) if total_batches > 0 else 0
        print(
            f"  Upserted batch {batch_num}/{total_batches} "
            f"({upserted}/{len(vectors)} vectors) ETA: {eta:.0f}s"
        )

    return upserted


# ---------------------------------------------------------------------------
# Dry-run summary
# ---------------------------------------------------------------------------

def dry_run_summary(rows: list[dict]) -> None:
    print("=== DRY RUN ===")
    print(f"CSV path:    {CSV_PATH}")
    print(f"Rows loaded: {len(rows)}")
    print(f"Index:       {INDEX_NAME}")
    print(f"Model:       {EMBED_MODEL} ({EMBED_DIMS} dims)")
    print()

    # Show domain breakdown
    domains: dict[str, int] = {}
    for row in rows:
        domains[row["domain"]] = domains.get(row["domain"], 0) + 1
    print(f"Domain breakdown ({len(domains)} domains):")
    for domain, count in sorted(domains.items()):
        print(f"  {domain:30s} {count:6d} rows")

    print()
    embed_calls = math.ceil(len(rows) / EMBED_BATCH_SIZE)
    upsert_calls = math.ceil(len(rows) / BATCH_SIZE)
    print(f"Embedding batches: {embed_calls}  (size {EMBED_BATCH_SIZE})")
    print(f"Upsert batches:    {upsert_calls}  (size {BATCH_SIZE})")
    print()
    print("Sample vector IDs:")
    for row in rows[:3]:
        print(f"  {make_vector_id(row)}")
    print()
    print("[DRY RUN] No embeddings computed, no upserts performed.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Embed all_queries.csv questions and upsert to Cloudflare Vectorize."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen without calling any API.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        metavar="N",
        default=None,
        help="Process first N rows only (for testing).",
    )
    args = parser.parse_args()

    # Load CSV
    if not CSV_PATH.exists():
        print(f"[ERROR] CSV not found: {CSV_PATH}", file=sys.stderr)
        return 1

    print(f"Loading rows from {CSV_PATH} ...")
    rows = load_rows(CSV_PATH, limit=args.limit)
    print(f"Loaded {len(rows)} rows.")

    if args.dry_run:
        dry_run_summary(rows)
        return 0

    # Validate env vars
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    cf_account = os.environ.get("CF_ACCOUNT_ID", "")
    cf_token = os.environ.get("CF_API_TOKEN", "")
    missing = [
        name
        for name, val in [
            ("OPENAI_API_KEY", openai_key),
            ("CF_ACCOUNT_ID", cf_account),
            ("CF_API_TOKEN", cf_token),
        ]
        if not val
    ]
    if missing:
        print(f"[ERROR] Missing env vars: {', '.join(missing)}", file=sys.stderr)
        return 1

    try:
        from openai import OpenAI
    except ImportError:
        print(
            "[ERROR] openai package not installed. Run: pip install openai",
            file=sys.stderr,
        )
        return 1

    client = OpenAI(api_key=openai_key)

    # Embed
    print(f"\nEmbedding {len(rows)} questions using {EMBED_MODEL} ({EMBED_DIMS} dims)...")
    t0 = time.time()
    embeddings = embed_all(client, rows)
    print(f"Embedding complete in {time.time() - t0:.1f}s.")

    # Upsert
    print(f"\nUpserting {len(rows)} vectors to '{INDEX_NAME}'...")
    t0 = time.time()
    upserted = upsert_all(cf_account, cf_token, rows, embeddings)
    print(f"Upsert complete in {time.time() - t0:.1f}s.")

    print(f"\nUpserted {upserted} vectors to {INDEX_NAME}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
