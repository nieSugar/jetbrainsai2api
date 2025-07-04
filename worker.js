// worker.js

// Cloudflare Worker that provides an OpenAI-compatible API backed by JetBrains AI
// Converted from the original Python FastAPI implementation.
//
// Environment Variables expected (can be configured in Wrangler or the Cloudflare dashboard):
//   CLIENT_API_KEYS  - comma separated list of allowed client API keys (Bearer tokens)
//   JETBRAINS_JWTS   - comma separated list of JetBrains AI service JWTs
//   MODELS           - comma separated list of model identifiers to expose (e.g. "anthropic-claude-3.5-sonnet")
//
// If you define these variables in wrangler.toml they will be available via `env` parameter.

export default {
  /**
   * Entry point for Cloudflare Worker fetch events.
   * @param {Request} request
   * @param {import('@cloudflare/workers-types').Env} env
   * @param {ExecutionContext} ctx
   * @returns {Promise<Response>}
   */
  async fetch(request, env, ctx) {
    try {
      const url = new URL(request.url);
      const pathname = url.pathname;

      // Simple router
      if (pathname === "/v1/models" && request.method === "GET") {
        return await handleListModels(request, env);
      }

      if (pathname === "/v1/chat/completions" && request.method === "POST") {
        return await handleChatCompletions(request, env);
      }

      return new Response("Not Found", { status: 404 });
    } catch (err) {
      console.error("Unhandled worker error", err);
      return jsonResponse({ error: { message: "Internal Server Error" } }, 500);
    }
  },
};

/* -------------------------------------------------------------------------- */
/*                              Helper Methods                                */
/* -------------------------------------------------------------------------- */

function jsonResponse(data, status = 200, extraHeaders = {}) {
  return new Response(JSON.stringify(data), {
    status,
    headers: {
      "Content-Type": "application/json; charset=utf-8",
      ...extraHeaders,
    },
  });
}

function unauthorized(message = "Unauthorized") {
  return jsonResponse({ error: { message } }, 401, {
    "WWW-Authenticate": "Bearer",
  });
}

/**
 * Check Authorization header and ensure provided key is in allowed set.
 * @param {Request} request
 * @param {import('@cloudflare/workers-types').Env} env
 */
function authenticateClient(request, env) {
  const auth = request.headers.get("Authorization") || "";
  const m = auth.match(/^Bearer\s+(.+)$/i);
  if (!m) return false;
  const key = m[1];
  const allowed = new Set((env.CLIENT_API_KEYS || "").split(",").map((k) => k.trim()).filter(Boolean));
  return allowed.has(key);
}

/**
 * Round-robin JWT selection shared across requests.
 */
const jwtState = {
  idx: 0,
};

function getNextJetbrainsJwt(env) {
  const tokens = (env.JETBRAINS_JWTS || "").split(",").map((t) => t.trim()).filter(Boolean);
  if (tokens.length === 0) throw new Error("No JetBrains JWT configured");
  const token = tokens[jwtState.idx];
  jwtState.idx = (jwtState.idx + 1) % tokens.length;
  return token;
}

function loadModels(env) {
  return (env.MODELS || "")
    .split(",")
    .map((id) => id.trim())
    .filter(Boolean)
    .map((id) => ({ id, object: "model", created: Math.floor(Date.now() / 1000), owned_by: "jetbrains-ai" }));
}

function extractTextContent(content) {
  if (typeof content === "string") {
    return content;
  }
  if (Array.isArray(content)) {
    return content
      .filter((item) => item && item.type === "text")
      .map((item) => item.text || "")
      .join(" ");
  }
  return String(content);
}

/* -------------------------------------------------------------------------- */
/*                               /v1/models                                   */
/* -------------------------------------------------------------------------- */

/**
 * Handle GET /v1/models
 */
async function handleListModels(request, env) {
  if (!authenticateClient(request, env)) {
    return unauthorized();
  }
  const data = loadModels(env);
  return jsonResponse({ object: "list", data });
}

/* -------------------------------------------------------------------------- */
/*                          /v1/chat/completions                              */
/* -------------------------------------------------------------------------- */

async function handleChatCompletions(request, env) {
  if (!authenticateClient(request, env)) {
    return unauthorized();
  }

  let body;
  try {
    body = await request.json();
  } catch (err) {
    return jsonResponse({ error: { message: "Invalid JSON body" } }, 400);
  }

  const { model, messages, stream = false } = body;
  if (!model || !messages) {
    return jsonResponse({ error: { message: "model and messages are required" } }, 400);
  }

  const models = loadModels(env).map((m) => m.id);
  if (!models.includes(model)) {
    return jsonResponse({ error: { message: `Model ${model} not found` } }, 404);
  }

  const jetbrainsJwt = getNextJetbrainsJwt(env);

  // Convert OpenAI messages to JetBrains format
  const jbMessages = messages.map((msg) => {
    const text = extractTextContent(msg.content);
    return { type: `${msg.role}_message`, content: text };
  });

  const payload = {
    prompt: "ij.chat.request.new-chat-on-start",
    profile: model,
    chat: { messages: jbMessages },
    parameters: { data: [] },
  };

  const headers = {
    "User-Agent": "ktor-client",
    Accept: "text/event-stream",
    "Content-Type": "application/json",
    "Accept-Charset": "UTF-8",
    "Cache-Control": "no-cache",
    "grazie-agent": "{\"name\":\"aia:cloudflare\",\"version\":\"1.0.0\"}",
    "grazie-authenticate-jwt": jetbrainsJwt,
  };

  const upstreamResp = await fetch("https://api.jetbrains.ai/user/v5/llm/chat/stream/v7", {
    method: "POST",
    headers,
    body: JSON.stringify(payload),
  });

  if (!upstreamResp.ok) {
    const txt = await upstreamResp.text();
    console.error("JetBrains upstream error", upstreamResp.status, txt);
    return jsonResponse({ error: { message: `Upstream error ${upstreamResp.status}` } }, 502);
  }

  if (stream) {
    return createOpenAIStreamResponse(upstreamResp.body, model);
  }

  // If stream = false, aggregate and return once
  const aggregated = await aggregateStreamResponse(upstreamResp.body, model);
  return jsonResponse(aggregated);
}

/* -------------------------------------------------------------------------- */
/*                       Streaming Transformation                             */
/* -------------------------------------------------------------------------- */

function createOpenAIStreamResponse(upstreamBody, model) {
  const encoder = new TextEncoder();
  const decoder = new TextDecoder();
  const sseId = `chatcmpl-${crypto.randomUUID().replace(/-/g, "")}`;
  let firstChunkSent = false;

  const { readable, writable } = new TransformStream();
  const writer = writable.getWriter();

  (async () => {
    const reader = upstreamBody.getReader();
    let buf = "";

    // helper to process complete lines
    const processLine = async (line) => {
      if (!line || line === "data: end") return;
      if (!line.startsWith("data: ")) return;

      try {
        const data = JSON.parse(line.slice(6));
        const type = data.type;
        if (type === "Content") {
          const content = data.content || "";
          if (!content) return;
          let delta;
          if (!firstChunkSent) {
            delta = { role: "assistant", content };
            firstChunkSent = true;
          } else {
            delta = { content };
          }
          const payload = {
            id: sseId,
            object: "chat.completion.chunk",
            created: Math.floor(Date.now() / 1000),
            model,
            choices: [
              {
                delta,
                index: 0,
                finish_reason: null,
              },
            ],
          };
          await writer.write(encoder.encode(`data: ${JSON.stringify(payload)}\n\n`));
        } else if (type === "FinishMetadata") {
          const payload = {
            id: sseId,
            object: "chat.completion.chunk",
            created: Math.floor(Date.now() / 1000),
            model,
            choices: [
              {
                delta: {},
                index: 0,
                finish_reason: "stop",
              },
            ],
          };
          await writer.write(encoder.encode(`data: ${JSON.stringify(payload)}\n\n`));
          // done
        }
      } catch (err) {
        console.warn("Failed to parse upstream line", err, line);
      }
    };

    try {
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        let idx;
        while ((idx = buf.indexOf("\n")) >= 0) {
          const line = buf.slice(0, idx).trim();
          buf = buf.slice(idx + 1);
          await processLine(line);
        }
      }
      // Flush remaining buffer line if any
      if (buf.trim()) await processLine(buf.trim());
    } catch (err) {
      console.error("Error while streaming from upstream", err);
    }

    // EOF marker
    await writer.write(encoder.encode("data: [DONE]\n\n"));
    writer.close();
  })();

  return new Response(readable, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
    },
  });
}

async function aggregateStreamResponse(upstreamBody, model) {
  const decoder = new TextDecoder();
  const reader = upstreamBody.getReader();
  let buf = "";
  let contentParts = [];
  const processLine = (line) => {
    if (!line || line === "data: end") return;
    if (!line.startsWith("data: ")) return;
    try {
      const data = JSON.parse(line.slice(6));
      if (data.type === "Content" && data.content) {
        contentParts.push(data.content);
      }
    } catch {}
  };

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let idx;
    while ((idx = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, idx).trim();
      buf = buf.slice(idx + 1);
      processLine(line);
    }
  }
  if (buf.trim()) processLine(buf.trim());

  const fullContent = contentParts.join("");
  return {
    id: `chatcmpl-${crypto.randomUUID().replace(/-/g, "")}`,
    object: "chat.completion",
    created: Math.floor(Date.now() / 1000),
    model,
    choices: [
      {
        message: {
          role: "assistant",
          content: fullContent,
        },
        index: 0,
        finish_reason: "stop",
      },
    ],
    usage: {
      prompt_tokens: 0,
      completion_tokens: 0,
      total_tokens: 0,
    },
  };
}