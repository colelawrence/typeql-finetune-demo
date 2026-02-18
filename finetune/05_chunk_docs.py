#!/usr/bin/env python3
"""
05_chunk_docs.py

Clones the typedb-docs repository at a pinned tag, chunks all .md / .adoc
files into ~500-token segments (~2000 chars) with ~50-token overlap (~200 chars),
embeds each chunk using OpenAI text-embedding-3-small (512 dims), and upserts
them to the Cloudflare Vectorize `typeql-docs` index.

Writes the commit hash used to finetune/data/docs_version.txt.

Usage:
  python finetune/05_chunk_docs.py
  python finetune/05_chunk_docs.py --dry-run
  python finetune/05_chunk_docs.py --docs-dir /path/to/existing/typedb-docs
  python finetune/05_chunk_docs.py --limit 50

ENV vars required (unless --dry-run):
  OPENAI_API_KEY   — OpenAI API key
  CF_ACCOUNT_ID    — Cloudflare account ID
  CF_API_TOKEN     — Cloudflare API token (Vectorize write permission)
"""

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent.resolve()
DATA_DIR = REPO_ROOT / "finetune" / "data"
DOCS_VERSION_FILE = DATA_DIR / "docs_version.txt"

DOCS_REPO_URL = "https://github.com/typedb/typedb-docs"
# Pinned to the latest stable tag as of 2025-01
DOCS_PINNED_TAG = "3.3.0"

INDEX_NAME = "typeql-docs"
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIMS = 512

# Chunking parameters (character-based approximation of tokens)
CHARS_PER_TOKEN = 4          # rough average for English/code text
CHUNK_TOKENS = 500           # target chunk size in tokens
OVERLAP_TOKENS = 50          # overlap between consecutive chunks in tokens
CHUNK_SIZE_CHARS = CHUNK_TOKENS * CHARS_PER_TOKEN      # 2000
OVERLAP_CHARS = OVERLAP_TOKENS * CHARS_PER_TOKEN       # 200

EMBED_BATCH_SIZE = 100       # texts per OpenAI embedding call
UPSERT_BATCH_SIZE = 100      # vectors per Vectorize upsert call

DOC_EXTENSIONS = {".md", ".adoc"}


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def clone_docs(target_dir: Path, tag: str) -> str:
    """
    Clone typedb-docs at the given tag into target_dir.
    Returns the resolved commit hash.
    """
    print(f"Cloning {DOCS_REPO_URL} @ tag={tag} into {target_dir} ...")
    subprocess.run(
        [
            "git", "clone",
            "--depth", "1",
            "--branch", tag,
            DOCS_REPO_URL,
            str(target_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = subprocess.run(
        ["git", "-C", str(target_dir), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    commit_hash = result.stdout.strip()
    print(f"Cloned at commit {commit_hash}")
    return commit_hash


def resolve_commit(docs_dir: Path) -> str:
    """Return HEAD commit hash for an existing checkout."""
    result = subprocess.run(
        ["git", "-C", str(docs_dir), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return result.stdout.strip()
    # Not a git repo — use the directory path as fallback
    return f"unknown-dir:{docs_dir}"


# ---------------------------------------------------------------------------
# File discovery & chunking
# ---------------------------------------------------------------------------

def find_doc_files(docs_dir: Path) -> list[Path]:
    """Find all .md and .adoc files under docs_dir."""
    files = []
    for ext in DOC_EXTENSIONS:
        files.extend(docs_dir.rglob(f"*{ext}"))
    return sorted(files)


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Split text into overlapping chunks of ~chunk_size characters.
    Returns list of chunk strings.
    """
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        if end >= len(text):
            break
        start = end - overlap  # step back by overlap for next chunk

    return chunks


def collect_chunks(docs_dir: Path) -> list[dict]:
    """
    Walk all doc files and return a list of chunk dicts:
      { file_path (relative), chunk_index, text, text_preview }
    """
    files = find_doc_files(docs_dir)
    chunks: list[dict] = []

    for file_path in files:
        try:
            text = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            print(f"  [WARN] Could not read {file_path}: {exc}", file=sys.stderr)
            continue

        file_chunks = chunk_text(text, CHUNK_SIZE_CHARS, OVERLAP_CHARS)
        rel_path = str(file_path.relative_to(docs_dir))

        for i, chunk in enumerate(file_chunks):
            chunks.append(
                {
                    "file_path": rel_path,
                    "chunk_index": i,
                    "text": chunk,
                    "text_preview": chunk[:100].replace("\n", " "),
                }
            )

    return chunks


def make_chunk_id(chunk: dict) -> str:
    """Build a stable vector ID from file path and chunk index."""
    safe_path = (
        chunk["file_path"]
        .replace("/", "__")
        .replace(".", "_")
        .replace(" ", "-")
    )
    return f"doc__{safe_path}__{chunk['chunk_index']}"


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def embed_batch(client, texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts using text-embedding-3-small."""
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=texts,
        dimensions=EMBED_DIMS,
    )
    sorted_data = sorted(response.data, key=lambda d: d.index)
    return [d.embedding for d in sorted_data]


def embed_all_chunks(client, chunks: list[dict]) -> list[list[float]]:
    """Embed all chunks in batches; returns list of embedding vectors."""
    texts = [c["text"] for c in chunks]
    embeddings: list[list[float]] = []
    total_batches = math.ceil(len(texts) / EMBED_BATCH_SIZE)
    t_start = time.time()

    for batch_num, start in enumerate(range(0, len(texts), EMBED_BATCH_SIZE), 1):
        batch_texts = texts[start : start + EMBED_BATCH_SIZE]

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
        done = min(batch_num * EMBED_BATCH_SIZE, len(texts))
        eta = (elapsed / batch_num) * (total_batches - batch_num) if total_batches > 0 else 0
        print(
            f"  Embedded batch {batch_num}/{total_batches} "
            f"({done}/{len(texts)} chunks) ETA: {eta:.0f}s"
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
    """Upsert a batch of vectors to Cloudflare Vectorize (v2 API, NDJSON)."""
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


def upsert_all_chunks(
    account_id: str,
    api_token: str,
    chunks: list[dict],
    embeddings: list[list[float]],
) -> int:
    """Build vector dicts and upsert to Vectorize in batches."""
    assert len(chunks) == len(embeddings), "Chunk/embedding count mismatch"

    vectors = []
    for chunk, vec in zip(chunks, embeddings):
        vectors.append(
            {
                "id": make_chunk_id(chunk),
                "values": vec,
                "metadata": {
                    "file_path": chunk["file_path"],
                    "chunk_index": chunk["chunk_index"],
                    "text_preview": chunk["text_preview"],
                },
            }
        )

    total_batches = math.ceil(len(vectors) / UPSERT_BATCH_SIZE)
    upserted = 0
    t_start = time.time()

    for batch_num, start in enumerate(range(0, len(vectors), UPSERT_BATCH_SIZE), 1):
        batch = vectors[start : start + UPSERT_BATCH_SIZE]
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

def dry_run_summary(chunks: list[dict], commit_hash: str, docs_dir: Path) -> None:
    print("=== DRY RUN ===")
    print(f"Docs dir:    {docs_dir}")
    print(f"Commit hash: {commit_hash}")
    print(f"Index:       {INDEX_NAME}")
    print(f"Model:       {EMBED_MODEL} ({EMBED_DIMS} dims)")
    print(f"Chunk size:  ~{CHUNK_TOKENS} tokens (~{CHUNK_SIZE_CHARS} chars)")
    print(f"Overlap:     ~{OVERLAP_TOKENS} tokens (~{OVERLAP_CHARS} chars)")
    print()

    # File-type breakdown
    md_count = sum(1 for c in chunks if c["file_path"].endswith(".md"))
    adoc_count = sum(1 for c in chunks if c["file_path"].endswith(".adoc"))
    unique_files = len({c["file_path"] for c in chunks})
    print(f"Total chunks:  {len(chunks)}")
    print(f"Unique files:  {unique_files}  ({md_count} .md chunks, {adoc_count} .adoc chunks)")

    embed_calls = math.ceil(len(chunks) / EMBED_BATCH_SIZE)
    upsert_calls = math.ceil(len(chunks) / UPSERT_BATCH_SIZE)
    print(f"Embedding batches: {embed_calls}  (size {EMBED_BATCH_SIZE})")
    print(f"Upsert batches:    {upsert_calls}  (size {UPSERT_BATCH_SIZE})")
    print()
    print("Sample chunk IDs:")
    for c in chunks[:3]:
        print(f"  {make_chunk_id(c)}")
    print()
    print("[DRY RUN] No embeddings computed, no upserts performed.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Chunk TypeQL docs and upsert to Cloudflare Vectorize."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count chunks without embedding or upserting.",
    )
    parser.add_argument(
        "--docs-dir",
        metavar="PATH",
        default=None,
        help="Use an existing typedb-docs checkout instead of cloning.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        metavar="N",
        default=None,
        help="Process first N chunks only (for testing).",
    )
    args = parser.parse_args()

    # Resolve docs directory
    tmp_dir = None
    if args.docs_dir:
        docs_dir = Path(args.docs_dir).expanduser().resolve()
        if not docs_dir.is_dir():
            print(f"[ERROR] --docs-dir not found: {docs_dir}", file=sys.stderr)
            return 1
        commit_hash = resolve_commit(docs_dir)
        print(f"Using existing docs checkout: {docs_dir} (commit {commit_hash})")
    else:
        tmp_dir = tempfile.mkdtemp(prefix="typedb-docs-")
        docs_dir = Path(tmp_dir)
        try:
            commit_hash = clone_docs(docs_dir, DOCS_PINNED_TAG)
        except subprocess.CalledProcessError as exc:
            print(
                f"[ERROR] Failed to clone {DOCS_REPO_URL}: {exc.stderr}",
                file=sys.stderr,
            )
            return 1

    # Collect and optionally limit chunks
    print(f"\nScanning docs for {', '.join(sorted(DOC_EXTENSIONS))} files...")
    all_chunks = collect_chunks(docs_dir)
    print(f"Found {len(all_chunks)} chunks across {len({c['file_path'] for c in all_chunks})} files.")

    chunks = all_chunks if args.limit is None else all_chunks[: args.limit]
    if args.limit is not None:
        print(f"Limiting to first {args.limit} chunks.")

    if args.dry_run:
        dry_run_summary(chunks, commit_hash, docs_dir)
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
    print(f"\nEmbedding {len(chunks)} chunks using {EMBED_MODEL} ({EMBED_DIMS} dims)...")
    t0 = time.time()
    embeddings = embed_all_chunks(client, chunks)
    print(f"Embedding complete in {time.time() - t0:.1f}s.")

    # Upsert
    print(f"\nUpserting {len(chunks)} vectors to '{INDEX_NAME}'...")
    t0 = time.time()
    upserted = upsert_all_chunks(cf_account, cf_token, chunks, embeddings)
    print(f"Upsert complete in {time.time() - t0:.1f}s.")

    # Write version file
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_VERSION_FILE.write_text(commit_hash + "\n", encoding="utf-8")
    print(f"\nWrote docs version to {DOCS_VERSION_FILE}")

    print(f"\nUpserted {upserted} vectors to {INDEX_NAME}")
    print(f"Docs version: {commit_hash}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
