# TypeQL Fine-tune Eval

Quality gate for fine-tuned TypeQL models. Validates structural validity of
generated queries against the northwind schema via TypeDB before Cloudflare
deployment.

## Requirements

**Pass condition:** Structural validity ≥ 90% on held-out northwind test set (807 queries).

---

## Setup

### 1. Start TypeDB

```bash
docker run -d \
  --name typedb \
  -p 1729:1729 \
  vaticle/typedb:3.3.0 \
  --development-mode.enabled=true
```

Wait ~10 seconds for TypeDB to start, then verify:

```bash
docker logs typedb | tail -5
```

### 2. Install Python dependencies

```bash
pip install typedb-driver openai
```

### 3. Set OpenAI API key

```bash
export OPENAI_API_KEY=sk-...
```

---

## Running the Eval

### Full eval (807 northwind test queries + TypeDB validation)

```bash
python finetune/eval/syntax_check.py \
  --model ft:gpt-4o-mini:YOUR_ORG:YOUR_NAME:YOUR_ID \
  --test finetune/data/test.jsonl \
  --typedb-addr localhost:1729
```

### Quick smoke test (first 10 examples, no TypeDB)

```bash
python finetune/eval/syntax_check.py \
  --model ft:gpt-4o-mini:YOUR_ORG:YOUR_NAME:YOUR_ID \
  --test finetune/data/test.jsonl \
  --skip-typedb \
  --limit 10
```

### With novel domain prompts

```bash
python finetune/eval/syntax_check.py \
  --model ft:gpt-4o-mini:YOUR_ORG:YOUR_NAME:YOUR_ID \
  --test finetune/data/test.jsonl \
  --novel finetune/eval/novel_domain_prompts.txt
```

---

## Expected Output

```
Loaded 807 test examples from finetune/data/test.jsonl
Setting up TypeDB at localhost:1729 with northwind schema…
TypeDB ready.

Running eval on 807 examples…

[   1/ 807] 2 queries — OK
[   2/ 807] 1 query  — OK
...

============================================================
NORTHWIND EVAL SUMMARY
============================================================
  Examples evaluated : 807
  Parse failures     : 3
  Total queries      : 950
  Valid queries      : 860
  Invalid queries    : 90

  Structural validity: 90.5% (≥90% required)

✅ PASS — structural validity meets threshold.
```

Exit code `0` = pass (≥90%), exit code `1` = fail (<90%).

---

## Novel Domain Eval Output

Novel domain results are qualitative only — no pass/fail threshold.

```
============================================================
NOVEL DOMAIN EVAL
============================================================

[ 1/20] I need a schema for tracking clinical trials…
         → schema(842 chars) | 2 queries | explanation(310 chars)
           Schema preview: 'define\n  attribute trial_id value string;...'
...

============================================================
NOVEL DOMAIN EVAL SUMMARY (qualitative)
============================================================
  Prompts evaluated    : 20
  Parse OK             : 20
  Parse failures       : 0
  Schemas generated    : 17
  Queries generated    : 19
```

---

## CLI Reference

| Flag | Description |
|------|-------------|
| `--model` | Fine-tuned model ID (required) |
| `--test` | Path to test.jsonl (default: `finetune/data/test.jsonl`) |
| `--typedb-addr` | TypeDB address (default: `localhost:1729`) |
| `--limit N` | Evaluate only first N examples |
| `--skip-typedb` | Skip TypeDB; check only JSON parse + non-empty strings |
| `--novel FILE` | Also run qualitative eval on novel domain prompts |

---

## Troubleshooting

**`Could not connect to TypeDB`**  
→ Check `docker ps` and ensure TypeDB container is running and port 1729 is exposed.

**`OPENAI_API_KEY environment variable not set`**  
→ `export OPENAI_API_KEY=sk-...`

**Rate limit errors from OpenAI**  
→ The script adds a 0.5s delay between calls. For smaller-tier accounts, consider using `--limit` to test a subset first.

**TypeDB schema errors**  
→ Schema is loaded from `text2typeql/dataset/synthetic-2/northwind/schema.tql`. If the schema file is missing, the eval will fail at setup.
