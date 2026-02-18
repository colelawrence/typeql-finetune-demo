import OpenAI from "openai";
import { SYSTEM_PROMPT } from "./generated/system-prompt";

export interface Env {
  OPENAI_API_KEY: string;
  OPENAI_MODEL: string;
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

async function handleHealth(): Promise<Response> {
  return jsonResponse({ ok: true, timestamp: new Date().toISOString() });
}

async function handleGenerate(request: Request, env: Env): Promise<Response> {
  let body: { prompt?: string };
  try {
    body = await request.json<{ prompt?: string }>();
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
    const completion = await client.chat.completions.create(
      {
        model: env.OPENAI_MODEL || "gpt-4o-mini-2024-07-18",
        messages: [
          { role: "system", content: SYSTEM_PROMPT },
          { role: "user", content: body.prompt },
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
      return handleHealth();
    }

    if (url.pathname === "/generate" && method === "POST") {
      return handleGenerate(request, env);
    }

    return jsonResponse({ error: "Not found" }, 404);
  },
} satisfies ExportedHandler<Env>;
