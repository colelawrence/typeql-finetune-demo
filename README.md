# TypeQL Agent

> Natural-language → TypeQL 3.0 schema + queries, powered by a fine-tuned GPT-4o-mini + RAG on Cloudflare Workers.

[![Live Demo](https://img.shields.io/badge/demo-live-brightgreen)](https://typeql-agent.pages.dev)

---

## What it does

TypeQL Agent lets developers describe their data model or query intent in plain English and immediately receive a TypeQL 3.0 schema, example queries, and a plain-English explanation. Under the hood it fine-tunes GPT-4o-mini on 13,939 validated TypeQL query pairs across 15 domains, augments generation with retrieval from a Vectorize index of those same examples plus TypeQL 3.0 documentation, and serves everything from a single Cloudflare Worker with a zero-dependency HTML UI.

---

## Live Demo

[https://typeql-agent.pages.dev](https://typeql-agent.pages.dev)

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Browser  →  Single-page HTML UI (Cloudflare Pages)     │
└───────────────────────┬─────────────────────────────────┘
                        │  POST /generate  { prompt }
                        ▼
┌─────────────────────────────────────────────────────────┐
│  Cloudflare Worker  (worker/src/index.ts)               │
│                                                         │
│  1. Embed prompt with text-embedding-3-small            │
│  2. Query Vectorize "typeql-examples" index             │
│     (13,939 question embeddings)                        │
│  3. Query Vectorize "typeql-docs" index                 │
│     (TypeQL 3.0 doc chunks)                             │
│  4. Call fine-tuned GPT-4o-mini                         │
│     (JSON schema response format)                       │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
          { schema, queries, explanation }
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│  3-panel UI: Schema │ Queries │ Explanation             │
└─────────────────────────────────────────────────────────┘
```

---

## Quick Start (local development)

### Prerequisites

| Tool | Version |
|------|---------|
| Bun | ≥ 1.0 |
| Python | ≥ 3.10 |
| OpenAI API key | — |
| Wrangler CLI | ≥ 3.0 (`bun add -g wrangler`) |

### 1. Clone + submodule init

```bash
git clone https://github.com/your-org/typeql-finetune.git
cd typeql-finetune
git submodule update --init --recursive
```

### 2. Set environment variables

```bash
export OPENAI_API_KEY=sk-...
```

### 3. Install Python dependencies

```bash
pip install openai tiktoken numpy tqdm
```

### 4. Run the pipeline (in order)

See the **Pipeline Reference** table below for full details.

```bash
# Step 1 — generate NL schema descriptions (GPT-4o, ~$2)
python finetune/01_generate_schema_descriptions.py

# Step 2a — build train / val / test JSONL splits
python finetune/02_build_training_jsonl.py

# Step 2b — add mutated-schema training examples
python finetune/02b_generate_mutation_queries.py

# Step 3 — upload to OpenAI and start fine-tune job
bun finetune/03_upload_and_train.ts

# Step 4 — embed questions into Cloudflare Vectorize
python finetune/04_build_vector_index.py

# Step 5 — chunk TypeQL docs into Vectorize
python finetune/05_chunk_docs.py
```

### 5. Deploy the Worker

```bash
cd worker
bun install
# Set your fine-tuned model ID in wrangler.toml → [vars].OPENAI_MODEL
wrangler deploy
```

### 6. Deploy the UI

```bash
# The UI is a single static file — deploy via Cloudflare Pages dashboard
# or wrangler pages deploy ui/
wrangler pages deploy ui/
```

---

## Pipeline Reference

| Script | Purpose |
|--------|---------|
| `finetune/system_prompt.txt` | Shared system prompt for training and inference |
| `finetune/01_generate_schema_descriptions.py` | Generates 9 natural-language descriptions per domain via GPT-4o |
| `finetune/02_build_training_jsonl.py` | Builds train / val / test JSONL splits from the dataset |
| `finetune/02b_generate_mutation_queries.py` | Generates mutated-schema examples for training diversity |
| `finetune/03_upload_and_train.ts` | Uploads JSONL to OpenAI Files API and polls the fine-tune job |
| `finetune/04_build_vector_index.py` | Embeds all 13,939 questions and upserts into Cloudflare Vectorize |
| `finetune/05_chunk_docs.py` | Chunks TypeQL 3.0 documentation and upserts into Vectorize |
| `finetune/eval/syntax_check.py` | Structural validity eval against TypeDB (northwind test set) |

---

## Eval

The structural validity eval runs generated queries against a live TypeDB instance. **Pass condition: ≥ 90% structural validity on the 807-example northwind held-out test set.**

### Start TypeDB

```bash
docker run -d \
  --name typedb \
  -p 1729:1729 \
  vaticle/typedb:3.3.0 \
  --development-mode.enabled=true
```

### Full eval

```bash
python finetune/eval/syntax_check.py \
  --model ft:gpt-4o-mini:YOUR_ORG:YOUR_NAME:YOUR_ID \
  --test finetune/data/test.jsonl \
  --typedb-addr localhost:1729
```

### Quick smoke (no TypeDB)

```bash
python finetune/eval/syntax_check.py \
  --model ft:gpt-4o-mini:YOUR_ORG:YOUR_NAME:YOUR_ID \
  --test finetune/data/test.jsonl \
  --skip-typedb \
  --limit 10
```

### Novel domain eval (qualitative)

```bash
python finetune/eval/syntax_check.py \
  --model ft:gpt-4o-mini:YOUR_ORG:YOUR_NAME:YOUR_ID \
  --test finetune/data/test.jsonl \
  --novel finetune/eval/novel_domain_prompts.txt
```

See [`finetune/eval/README.md`](finetune/eval/README.md) for full CLI reference and expected output.

---

## Dataset

The training data comes from [text2typeql](https://github.com/typedb-osi/text2typeql) — 13,939 validated natural-language → TypeQL query pairs spanning 15 domains:

| Source | Pairs |
|--------|-------|
| synthetic-1 | 4,733 |
| synthetic-2 | 9,206 |
| **Total** | **13,939** |

The submodule lives at `text2typeql/`. After `git submodule update --init --recursive`, the dataset is available at `text2typeql/dataset/`.

---

## Project Structure

```
typeql-finetune/
├── finetune/                  # Pipeline scripts
│   ├── system_prompt.txt      # Shared system prompt
│   ├── 01_generate_schema_descriptions.py
│   ├── 02_build_training_jsonl.py
│   ├── 02b_generate_mutation_queries.py
│   ├── 03_upload_and_train.ts
│   ├── 04_build_vector_index.py
│   ├── 05_chunk_docs.py
│   ├── data/                  # Generated JSONL splits (gitignored)
│   └── eval/
│       ├── syntax_check.py    # TypeDB structural validity eval
│       └── novel_domain_prompts.txt
├── worker/                    # Cloudflare Worker
│   ├── wrangler.toml
│   └── src/index.ts
├── ui/
│   └── index.html             # Single-page demo UI
└── text2typeql/               # Dataset submodule
```

---

## License

MIT
