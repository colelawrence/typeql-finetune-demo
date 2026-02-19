import OpenAI from "openai";
import { SYSTEM_PROMPT } from "./generated/system-prompt";

export interface Env {
  OPENAI_API_KEY: string;
  OPENAI_MODEL: string;
  EXAMPLES_INDEX: VectorizeIndex;
  DOCS_INDEX: VectorizeIndex;
}

const CORS_HEADERS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type, Authorization",
};

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: {
      "Content-Type": "application/json",
      ...CORS_HEADERS,
    },
  });
}

async function handleHealth(env: Env): Promise<Response> {
  return jsonResponse({ ok: true, timestamp: new Date().toISOString(), model: env.OPENAI_MODEL });
}

/** Embed user prompt via OpenAI text-embedding-3-small (512 dims) */
async function embedPrompt(client: OpenAI, text: string): Promise<number[]> {
  const resp = await client.embeddings.create({
    model: "text-embedding-3-small",
    input: text,
    dimensions: 512,
  });
  return resp.data[0].embedding;
}

/** Build few-shot examples string from Vectorize matches */
function buildFewShotExamples(
  matches: VectorizeMatch[]
): string {
  if (matches.length === 0) return "";
  const examples = matches
    .filter((m) => m.metadata?.question && m.metadata?.typeql)
    .slice(0, 8)
    .map(
      (m) =>
        `Q: "${m.metadata!.question}"\nA: ${JSON.stringify({ schema: "", queries: [m.metadata!.typeql as string], explanation: "" })}`
    )
    .join("\n\n");
  return examples
    ? `\nSimilar examples from the TypeQL dataset:\n${examples}\n\n---\n`
    : "";
}

/** Build doc context string from Vectorize doc matches */
function buildDocContext(matches: VectorizeMatch[]): string {
  if (matches.length === 0) return "";
  const chunks = matches
    .filter((m) => m.metadata?.text_preview)
    .slice(0, 2)
    .map((m) => String(m.metadata!.text_preview))
    .join("\n\n");
  return chunks ? `\nTypeQL Reference:\n${chunks}\n` : "";
}

async function handleGenerate(request: Request, env: Env): Promise<Response> {
  let body: { prompt?: string; schema?: string };
  try {
    body = await request.json<{ prompt?: string; schema?: string }>();
  } catch {
    return jsonResponse({ error: "Invalid JSON body" }, 400);
  }

  if (!body.prompt || typeof body.prompt !== "string") {
    return jsonResponse({ error: "Missing required field: prompt" }, 400);
  }

  const client = new OpenAI({ apiKey: env.OPENAI_API_KEY });
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 25_000);

  try {
    // Step 1: Embed user prompt + parallel Vectorize queries (if indexes are populated)
    let fewShotSection = "";
    let docSection = "";

    try {
      const embedding = await embedPrompt(client, body.prompt);

      const [examplesResult, docsResult] = await Promise.all([
        env.EXAMPLES_INDEX
          ? env.EXAMPLES_INDEX.query(embedding, { topK: 8, returnMetadata: "all" })
          : Promise.resolve({ matches: [] }),
        env.DOCS_INDEX
          ? env.DOCS_INDEX.query(embedding, { topK: 2, returnMetadata: "all" })
          : Promise.resolve({ matches: [] }),
      ]);

      fewShotSection = buildFewShotExamples(examplesResult.matches);
      docSection = buildDocContext(docsResult.matches);
    } catch {
      // RAG failure is non-fatal — proceed without retrieval
    }

    // Step 2: Build prompt and call model
    // If a schema is provided (e.g. from a training sample), inject it into the
    // system message — matching the training format: "...\n\nSchema:\n<tql>"
    const schemaSection = body.schema ? `\n\nSchema:\n${body.schema}` : "";
    const modeInstruction = body.schema
      ? `\n\nIMPORTANT: A TypeQL schema has been provided above. Do NOT generate a new schema. Write TypeQL queries against the provided schema only. Set the "schema" field to an empty string in your JSON response.`
      : `\n\nNo schema has been provided. You must define a TypeQL schema as part of your response in the "schema" field.`;
    const systemContent = SYSTEM_PROMPT + schemaSection + modeInstruction + docSection;
    const userContent = fewShotSection
      ? `${fewShotSection}User request: ${body.prompt}`
      : body.prompt;

    const completion = await client.chat.completions.create(
      {
        model: env.OPENAI_MODEL || "gpt-4.1-mini-2025-04-14",
        messages: [
          { role: "system", content: systemContent },
          { role: "user", content: userContent },
        ],
        response_format: {
          type: "json_schema",
          json_schema: {
            name: "typeql_response",
            strict: true,
            schema: {
              type: "object",
              properties: {
                schema: { type: "string" },
                queries: { type: "array", items: { type: "string" } },
                explanation: { type: "string" },
              },
              required: ["schema", "queries", "explanation"],
              additionalProperties: false,
            },
          },
        },
        max_completion_tokens: 1200,
        stream: false,
      },
      { signal: controller.signal }
    );

    clearTimeout(timeout);

    const content = completion.choices[0]?.message?.content;
    if (!content) {
      return jsonResponse({ error: "Empty response from model" }, 500);
    }

    let parsed: { schema?: string; queries?: string[]; explanation?: string };
    try {
      parsed = JSON.parse(content);
    } catch {
      return jsonResponse({ error: "Model returned non-JSON content" }, 500);
    }

    if (
      typeof parsed.schema === "undefined" ||
      typeof parsed.queries === "undefined" ||
      typeof parsed.explanation === "undefined"
    ) {
      return jsonResponse(
        { error: "Model response missing required fields (schema, queries, explanation)" },
        500
      );
    }

    return jsonResponse({
      schema: parsed.schema,
      queries: parsed.queries,
      explanation: parsed.explanation,
    });
  } catch (err: unknown) {
    clearTimeout(timeout);

    if (err instanceof Error && err.name === "AbortError") {
      return jsonResponse({ error: "Request timed out after 25 seconds" }, 504);
    }

    const message = err instanceof Error ? err.message : "Unknown error";
    return jsonResponse({ error: `OpenAI API error: ${message}` }, 500);
  }
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);
    const method = request.method.toUpperCase();

    // Handle CORS preflight
    if (method === "OPTIONS") {
      return new Response(null, { status: 204, headers: CORS_HEADERS });
    }

    if (url.pathname === "/health" && method === "GET") {
      return handleHealth(env);
    }

    if (url.pathname === "/generate" && method === "POST") {
      return handleGenerate(request, env);
    }

    return jsonResponse({ error: "Not found" }, 404);
  },
} satisfies ExportedHandler<Env>;
