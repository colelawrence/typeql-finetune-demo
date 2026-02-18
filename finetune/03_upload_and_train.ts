#!/usr/bin/env bun
/**
 * 03_upload_and_train.ts
 *
 * Uploads training/validation JSONL to OpenAI and polls until the fine-tune
 * job completes, then writes finetune/data/prompt_lock.json.
 *
 * Usage:
 *   bun finetune/03_upload_and_train.ts [options]
 *
 * Options:
 *   --dry-run          Validate files and manifest, print what would happen, don't upload
 *   --model <model>    Override base model (default: gpt-4o-mini-2024-07-18)
 *   --resume <job-id>  Skip upload, resume polling an existing fine-tune job
 */

import OpenAI from "openai"
import type { FineTuningJobEvent } from "openai/resources/fine-tuning"
import * as path from "path"

// ── Paths ─────────────────────────────────────────────────────────────────────

const ROOT = path.resolve(import.meta.dir, "..")
const DATA_DIR = path.join(ROOT, "finetune", "data")
const TRAIN_JSONL = path.join(DATA_DIR, "train.jsonl")
const VAL_JSONL = path.join(DATA_DIR, "val.jsonl")
const MUTATION_JSONL = path.join(DATA_DIR, "mutation_examples.jsonl")
const BUILD_MANIFEST = path.join(DATA_DIR, ".build-manifest")
const SYSTEM_PROMPT = path.join(ROOT, "finetune", "system_prompt.txt")
const PROMPT_LOCK = path.join(DATA_DIR, "prompt_lock.json")

const DEFAULT_MODEL = "gpt-4o-mini-2024-07-18"

// ── CLI arg parsing ────────────────────────────────────────────────────────────

function parseArgs(argv: string[]): {
  dryRun: boolean
  model: string
  resumeJobId: string | null
} {
  const args = argv.slice(2) // strip bun + script path
  let dryRun = false
  let model = DEFAULT_MODEL
  let resumeJobId: string | null = null

  for (let i = 0; i < args.length; i++) {
    if (args[i] === "--dry-run") {
      dryRun = true
    } else if (args[i] === "--model" && args[i + 1]) {
      model = args[++i]
    } else if (args[i] === "--resume" && args[i + 1]) {
      resumeJobId = args[++i]
    } else if (args[i].startsWith("--model=")) {
      model = args[i].slice("--model=".length)
    } else if (args[i].startsWith("--resume=")) {
      resumeJobId = args[i].slice("--resume=".length)
    }
  }

  return { dryRun, model, resumeJobId }
}

// ── Utilities ──────────────────────────────────────────────────────────────────

function log(msg: string) {
  console.log(`[${new Date().toISOString()}] ${msg}`)
}

async function fileExists(filePath: string): Promise<boolean> {
  return Bun.file(filePath).exists()
}

async function sha256File(filePath: string): Promise<string> {
  const file = Bun.file(filePath)
  const text = (await file.text()).trim()
  const hasher = new Bun.CryptoHasher("sha256")
  hasher.update(text)
  return hasher.digest("hex")
}

async function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

// ── Pre-upload checks ──────────────────────────────────────────────────────────

async function runPreChecks(): Promise<{
  trainPath: string
  sha256: string
  hasMutations: boolean
}> {
  // 1. Verify required JSONL files exist
  const trainExists = await fileExists(TRAIN_JSONL)
  const valExists = await fileExists(VAL_JSONL)

  if (!trainExists) {
    console.error(`ERROR: Training file not found: ${TRAIN_JSONL}`)
    console.error("  Run 02_build_training_jsonl.py first.")
    process.exit(1)
  }
  if (!valExists) {
    console.error(`ERROR: Validation file not found: ${VAL_JSONL}`)
    console.error("  Run 02_build_training_jsonl.py first.")
    process.exit(1)
  }

  log(`✓ train.jsonl exists`)
  log(`✓ val.jsonl exists`)

  // 2. Verify build manifest + system prompt SHA256
  const manifestExists = await fileExists(BUILD_MANIFEST)
  if (!manifestExists) {
    console.error(`ERROR: Build manifest not found: ${BUILD_MANIFEST}`)
    console.error("  Run 02_build_training_jsonl.py first.")
    process.exit(1)
  }

  const manifest = JSON.parse(await Bun.file(BUILD_MANIFEST).text()) as {
    system_prompt_sha256: string
    [key: string]: unknown
  }
  const expectedSha = manifest.system_prompt_sha256

  const promptExists = await fileExists(SYSTEM_PROMPT)
  if (!promptExists) {
    console.error(`ERROR: System prompt not found: ${SYSTEM_PROMPT}`)
    process.exit(1)
  }

  const actualSha = await sha256File(SYSTEM_PROMPT)

  if (actualSha !== expectedSha) {
    console.error(`ERROR: Prompt drift detected!`)
    console.error(`  Expected SHA256: ${expectedSha}`)
    console.error(`  Actual SHA256:   ${actualSha}`)
    console.error(
      `  The system_prompt.txt has changed since the training data was built.`
    )
    console.error(`  Re-run 02_build_training_jsonl.py to rebuild the dataset.`)
    process.exit(1)
  }

  log(`✓ system_prompt.txt SHA256 matches manifest (${actualSha.slice(0, 16)}…)`)

  // 3. Check for mutation examples
  const hasMutations = await fileExists(MUTATION_JSONL)
  if (hasMutations) {
    log(`✓ mutation_examples.jsonl found — will merge into training data`)
  }

  return { trainPath: TRAIN_JSONL, sha256: actualSha, hasMutations }
}

// ── Merge mutation examples into training data ─────────────────────────────────

async function buildUploadTrainFile(hasMutations: boolean): Promise<{
  content: Uint8Array
  lineCount: number
}> {
  const trainContent = await Bun.file(TRAIN_JSONL).text()

  let combined = trainContent.trimEnd()
  let lineCount = combined.split("\n").length

  if (hasMutations) {
    const mutContent = await Bun.file(MUTATION_JSONL).text()
    const mutLines = mutContent
      .split("\n")
      .filter((l) => l.trim().length > 0)
    combined += "\n" + mutLines.join("\n")
    lineCount += mutLines.length
    log(
      `  Merged ${mutLines.length} mutation examples (total: ${lineCount} lines)`
    )
  }

  const encoder = new TextEncoder()
  return { content: encoder.encode(combined + "\n"), lineCount }
}

// ── Upload a JSONL buffer to OpenAI ───────────────────────────────────────────

async function uploadJsonl(
  client: OpenAI,
  content: Uint8Array,
  filename: string,
  label: string
): Promise<string> {
  log(`Uploading ${label} (${(content.byteLength / 1024).toFixed(1)} KB)…`)

  const fileForUpload = new File([content], filename, {
    type: "application/jsonl",
  })

  const uploaded = await client.files.create({
    file: fileForUpload,
    purpose: "fine-tune",
  })

  log(`  ${label} uploaded → ID: ${uploaded.id}`)
  return uploaded.id
}

// ── Poll a file until processed ────────────────────────────────────────────────

async function pollFileProcessed(
  client: OpenAI,
  fileId: string,
  label: string
): Promise<void> {
  log(`Waiting for ${label} (${fileId}) to be processed…`)
  while (true) {
    const file = await client.files.retrieve(fileId)
    if (file.status === "processed") {
      log(`  ${label} processed ✓`)
      return
    }
    if (file.status === "error") {
      console.error(`ERROR: File ${fileId} (${label}) failed processing`)
      process.exit(1)
    }
    log(`  ${label} status: ${file.status} — waiting 2s…`)
    await sleep(2000)
  }
}

// ── Poll fine-tune job until terminal state ────────────────────────────────────

async function pollJob(
  client: OpenAI,
  jobId: string
): Promise<OpenAI.FineTuning.FineTuningJob> {
  log(`Polling fine-tune job ${jobId}…`)

  const seenEvents = new Set<string>()

  // Give user a way to recover if interrupted
  process.on("SIGINT", () => {
    console.error(`\n\nInterrupted. To resume polling, run:`)
    console.error(`  bun finetune/03_upload_and_train.ts --resume ${jobId}`)
    process.exit(130)
  })

  while (true) {
    const job = await client.fineTuning.jobs.retrieve(jobId)
    log(`  Job status: ${job.status}`)

    // Fetch and print new events (API returns newest first, so reverse)
    const { data: events } = await client.fineTuning.jobs.listEvents(jobId, {
      limit: 100,
    })
    for (const event of (events as FineTuningJobEvent[]).reverse()) {
      if (seenEvents.has(event.id)) continue
      seenEvents.add(event.id)
      const ts = new Date(event.created_at * 1000).toLocaleTimeString()
      log(`  [event] ${ts}: ${event.message}`)
    }

    // Terminal states
    if (job.status === "succeeded") return job
    if (job.status === "failed" || job.status === "cancelled") {
      console.error(
        `\nERROR: Fine-tune job ${jobId} ended with status: ${job.status}`
      )
      if (job.error) {
        console.error(`  Error code: ${job.error.code}`)
        console.error(`  Error message: ${job.error.message}`)
      }
      process.exit(1)
    }

    // Still running or queued — wait 30s
    log(`  Waiting 30s before next check…`)
    await sleep(30_000)
  }
}

// ── Write prompt_lock.json ─────────────────────────────────────────────────────

async function writePromptLock(opts: {
  modelId: string
  jobId: string
  promptSha256: string
}): Promise<void> {
  const lock = {
    model_id: opts.modelId,
    job_id: opts.jobId,
    prompt_sha256: opts.promptSha256,
    created_at: new Date().toISOString(),
  }

  await Bun.write(PROMPT_LOCK, JSON.stringify(lock, null, 2) + "\n")
  log(`✓ Wrote prompt_lock.json:`)
  log(`    model_id:     ${lock.model_id}`)
  log(`    job_id:       ${lock.job_id}`)
  log(`    prompt_sha256: ${lock.prompt_sha256.slice(0, 16)}…`)
  log(`    created_at:   ${lock.created_at}`)
}

// ── Main ───────────────────────────────────────────────────────────────────────

async function main() {
  const { dryRun, model, resumeJobId } = parseArgs(process.argv)

  log(`=== TypeQL Fine-Tune Upload & Train ===`)
  log(`Mode:  ${dryRun ? "DRY RUN" : resumeJobId ? `RESUME ${resumeJobId}` : "LIVE"}`)
  log(`Model: ${model}`)

  // ── Validate API key (unless dry-run) ─────────────────────────────────────
  if (!dryRun && !resumeJobId && !process.env.OPENAI_API_KEY) {
    console.error("ERROR: OPENAI_API_KEY environment variable is not set")
    process.exit(1)
  }
  if (resumeJobId && !process.env.OPENAI_API_KEY) {
    console.error("ERROR: OPENAI_API_KEY environment variable is not set")
    process.exit(1)
  }

  // ── Pre-checks ─────────────────────────────────────────────────────────────
  const { sha256, hasMutations } = await runPreChecks()

  if (dryRun) {
    log(`\n── DRY RUN SUMMARY ──────────────────────────────────────────`)
    log(`  Training file:    ${TRAIN_JSONL}`)
    log(`  Validation file:  ${VAL_JSONL}`)
    if (hasMutations) {
      log(`  Mutation examples: ${MUTATION_JSONL} (will be merged)`)
    }
    log(`  System prompt SHA256: ${sha256}`)
    log(`  Base model:       ${model}`)
    log(`  Fine-tune output: ${PROMPT_LOCK}`)
    log(`\n  All checks passed. Exiting (dry run).`)
    process.exit(0)
  }

  const client = new OpenAI() // reads OPENAI_API_KEY automatically

  // ── Resume mode: skip upload, go straight to polling ──────────────────────
  if (resumeJobId) {
    log(`Resuming poll for job: ${resumeJobId}`)
    const job = await pollJob(client, resumeJobId)

    const finishedModelId = job.fine_tuned_model
    if (!finishedModelId) {
      console.error("ERROR: Job succeeded but fine_tuned_model is null")
      process.exit(1)
    }

    await writePromptLock({
      modelId: finishedModelId,
      jobId: job.id,
      promptSha256: sha256,
    })
    log(`\n✓ Done! Fine-tuned model: ${finishedModelId}`)
    process.exit(0)
  }

  // ── Build upload training content (merge mutations if present) ─────────────
  const { content: trainContent, lineCount: trainLineCount } =
    await buildUploadTrainFile(hasMutations)
  log(`Training data ready: ${trainLineCount} examples`)

  const valContent = await Bun.file(VAL_JSONL).arrayBuffer()
  const valBytes = new Uint8Array(valContent)

  // ── Upload both files ──────────────────────────────────────────────────────
  const [trainFileId, valFileId] = await Promise.all([
    uploadJsonl(client, trainContent, "train.jsonl", "training file"),
    uploadJsonl(client, valBytes, "val.jsonl", "validation file"),
  ])

  // ── Poll until both files are processed ───────────────────────────────────
  await Promise.all([
    pollFileProcessed(client, trainFileId, "training file"),
    pollFileProcessed(client, valFileId, "validation file"),
  ])

  // ── Create fine-tune job ───────────────────────────────────────────────────
  log(`Creating fine-tune job…`)
  const job = await client.fineTuning.jobs.create({
    model,
    training_file: trainFileId,
    validation_file: valFileId,
    hyperparameters: { n_epochs: 3 },
  })

  log(`Fine-tune job created → ID: ${job.id}`)
  log(`To resume polling if interrupted:`)
  log(`  bun finetune/03_upload_and_train.ts --resume ${job.id}`)

  // Register SIGINT handler now that we have a job ID
  // (pollJob will override this with the same message)
  process.on("SIGINT", () => {
    console.error(`\n\nInterrupted. To resume polling, run:`)
    console.error(`  bun finetune/03_upload_and_train.ts --resume ${job.id}`)
    process.exit(130)
  })

  // ── Poll job until done ────────────────────────────────────────────────────
  const finishedJob = await pollJob(client, job.id)

  const finishedModelId = finishedJob.fine_tuned_model
  if (!finishedModelId) {
    console.error("ERROR: Job succeeded but fine_tuned_model is null")
    process.exit(1)
  }

  // ── Write prompt_lock.json ─────────────────────────────────────────────────
  await writePromptLock({
    modelId: finishedModelId,
    jobId: finishedJob.id,
    promptSha256: sha256,
  })

  log(`\n✓ Fine-tuning complete! Model: ${finishedModelId}`)
}

main().catch((err) => {
  console.error("Unhandled error:", err)
  process.exit(1)
})
