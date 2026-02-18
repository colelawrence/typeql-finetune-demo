#!/usr/bin/env bun
import { readFileSync, existsSync } from "fs";
import { join, dirname } from "path";
import { createHash } from "crypto";

const scriptDir = dirname(import.meta.url.replace("file://", ""));
const lockPath = join(scriptDir, "../../finetune/data/prompt_lock.json");
const promptPath = join(scriptDir, "../../finetune/system_prompt.txt");

if (!existsSync(lockPath)) {
  console.warn(
    "⚠ prompt_lock.json not found — skipping drift check (pre-training state, this is OK)"
  );
  process.exit(0);
}

const lock = JSON.parse(readFileSync(lockPath, "utf-8")) as { prompt_sha256?: string };
const expectedSha = lock.prompt_sha256;

if (!expectedSha) {
  console.warn("⚠ prompt_lock.json has no prompt_sha256 field — skipping drift check");
  process.exit(0);
}

const promptContent = readFileSync(promptPath, "utf-8").trim();
const actualSha = createHash("sha256").update(promptContent, "utf-8").digest("hex");

if (actualSha === expectedSha) {
  console.log(`✓ system_prompt.txt matches prompt_lock.json (sha256: ${actualSha.slice(0, 16)}…)`);
  process.exit(0);
} else {
  console.error("✗ PROMPT DRIFT DETECTED");
  console.error(`  Expected: ${expectedSha}`);
  console.error(`  Actual:   ${actualSha}`);
  console.error(
    "  The system prompt has changed since training. Re-run fine-tuning or update prompt_lock.json."
  );
  process.exit(1);
}
