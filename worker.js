// worker.js - JetBrains AI OpenAI & Anthropic Compatible API
// Cloudflare Worker implementation matching the Python FastAPI version

// Global state for account rotation
const accountRotationState = {
  currentIndex: 0,
  accounts: [],
  lastConfigLoad: 0,
};

export default {
  async fetch(request, env, ctx) {
    try {
      // Add CORS headers
      const corsHeaders = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type, Authorization, x-api-key',
      };

      if (request.method === 'OPTIONS') {
        return new Response(null, {
          status: 200,
          headers: corsHeaders,
        });
      }

      const url = new URL(request.url);
      const pathname = url.pathname;

      // Route handling
      if (pathname === '/v1/models' && request.method === 'GET') {
        return await handleListModels(request, env, corsHeaders);
      }

      if (pathname === '/v1/chat/completions' && request.method === 'POST') {
        return await handleChatCompletions(request, env, corsHeaders);
      }

      if (pathname === '/v1/messages' && request.method === 'POST') {
        return await handleAnthropicMessages(request, env, corsHeaders);
      }

      return jsonResponse({ error: { message: 'Not Found' } }, 404, corsHeaders);
    } catch (err) {
      console.error('Unhandled worker error:', err);
      return jsonResponse({ error: { message: 'Internal Server Error' } }, 500, {
        'Access-Control-Allow-Origin': '*',
      });
    }
  },
};

/* -------------------------------------------------------------------------- */
/*                              Helper Functions                              */
/* -------------------------------------------------------------------------- */

function jsonResponse(data, status = 200, extraHeaders = {}) {
  return new Response(JSON.stringify(data), {
    status,
    headers: {
      'Content-Type': 'application/json; charset=utf-8',
      ...extraHeaders,
    },
  });
}

function streamResponse(readable, extraHeaders = {}) {
  return new Response(readable, {
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      'Connection': 'keep-alive',
      ...extraHeaders,
    },
  });
}

function generateId() {
  return crypto.randomUUID().replace(/-/g, '');
}

function getCurrentTimestamp() {
  return Math.floor(Date.now() / 1000);
}

/* -------------------------------------------------------------------------- */
/*                            Authentication                                  */
/* -------------------------------------------------------------------------- */

function getValidClientKeys(env) {
  const keys = env.CLIENT_API_KEYS || '';
  return new Set(keys.split(',').map(k => k.trim()).filter(Boolean));
}

function authenticateOpenAI(request, env) {
  const auth = request.headers.get('Authorization') || '';
  const match = auth.match(/^Bearer\s+(.+)$/i);
  if (!match) return false;
  const key = match[1];
  const validKeys = getValidClientKeys(env);
  return validKeys.has(key);
}

function authenticateAnthropic(request, env) {
  const apiKey = request.headers.get('x-api-key');
  if (!apiKey) return false;
  const validKeys = getValidClientKeys(env);
  return validKeys.has(apiKey);
}

function authenticateAnyClient(request, env) {
  // Try x-api-key first (Anthropic style)
  if (authenticateAnthropic(request, env)) return true;
  // Then try Authorization header (OpenAI style)
  return authenticateOpenAI(request, env);
}

/* -------------------------------------------------------------------------- */
/*                         Configuration Loading                             */
/* -------------------------------------------------------------------------- */

function loadModelsConfig(env) {
  try {
    const modelsJson = env.MODELS_JSON || '{}';
    const config = JSON.parse(modelsJson);
    
    let models = [];
    let mappings = {};
    
    if (config.models && Array.isArray(config.models)) {
      models = config.models;
      mappings = config.anthropic_model_mappings || {};
    } else if (Array.isArray(config)) {
      // Legacy format - just array of model names
      models = config;
    } else {
      // Fallback to env variable
      const modelsStr = env.MODELS || '';
      models = modelsStr.split(',').map(m => m.trim()).filter(Boolean);
    }
    
    const modelList = models.map(id => ({
      id,
      object: 'model',
      created: getCurrentTimestamp(),
      owned_by: 'jetbrains-ai',
    }));
    
    return { models: modelList, mappings };
  } catch (err) {
    console.error('Error loading models config:', err);
    // Fallback to basic env variable
    const modelsStr = env.MODELS || '';
    const models = modelsStr.split(',').map(m => m.trim()).filter(Boolean);
    const modelList = models.map(id => ({
      id,
      object: 'model',
      created: getCurrentTimestamp(),
      owned_by: 'jetbrains-ai',
    }));
    return { models: modelList, mappings: {} };
  }
}

function loadJetBrainsAccounts(env) {
  try {
    const accountsJson = env.JETBRAINS_ACCOUNTS || '[]';
    const accounts = JSON.parse(accountsJson);
    
    if (!Array.isArray(accounts)) {
      throw new Error('JETBRAINS_ACCOUNTS should be an array');
    }
    
    return accounts.map(account => ({
      licenseId: account.licenseId,
      authorization: account.authorization,
      jwt: account.jwt,
      last_updated: account.last_updated || 0,
      has_quota: account.has_quota !== false, // Default to true
      last_quota_check: account.last_quota_check || 0,
    }));
  } catch (err) {
    console.error('Error loading JetBrains accounts:', err);
    // Fallback to legacy JWT list
    const jwts = env.JETBRAINS_JWTS || '';
    return jwts.split(',').map(jwt => jwt.trim()).filter(Boolean).map(jwt => ({
      jwt,
      has_quota: true,
      last_quota_check: 0,
    }));
  }
}

async function getNextJetBrainsAccount(env) {
  const now = Date.now();
  
  // Reload accounts periodically or on first load
  if (!accountRotationState.accounts.length || now - accountRotationState.lastConfigLoad > 300000) { // 5 minutes
    accountRotationState.accounts = loadJetBrainsAccounts(env);
    accountRotationState.lastConfigLoad = now;
  }
  
  if (!accountRotationState.accounts.length) {
    throw new Error('No JetBrains accounts configured');
  }
  
  const startIndex = accountRotationState.currentIndex;
  
  for (let i = 0; i < accountRotationState.accounts.length; i++) {
    const account = accountRotationState.accounts[accountRotationState.currentIndex];
    accountRotationState.currentIndex = (accountRotationState.currentIndex + 1) % accountRotationState.accounts.length;
    
    // Check if JWT is valid and account has quota
    if (account.jwt && account.has_quota) {
      return account;
    }
  }
  
  throw new Error('All JetBrains accounts are either invalid or out of quota');
}

/* -------------------------------------------------------------------------- */
/*                           Content Processing                               */
/* -------------------------------------------------------------------------- */

function extractTextContent(content) {
  if (typeof content === 'string') {
    return content;
  }
  if (Array.isArray(content)) {
    return content
      .filter(item => item && item.type === 'text')
      .map(item => item.text || '')
      .join(' ');
  }
  return String(content || '');
}

function convertOpenAIToJetBrains(messages, tools = null) {
  const toolIdToFuncNameMap = {};
  
  // Build tool_call_id to function name mapping from assistant messages
  for (const msg of messages) {
    if (msg.role === 'assistant' && msg.tool_calls) {
      for (const tc of msg.tool_calls) {
        if (tc.id && tc.function?.name) {
          toolIdToFuncNameMap[tc.id] = tc.function.name;
        }
      }
    }
  }
  
  const jetbrainsMessages = [];
  
  for (const msg of messages) {
    const textContent = extractTextContent(msg.content);
    
    if (msg.role === 'user' || msg.role === 'system') {
      jetbrainsMessages.push({
        type: `${msg.role}_message`,
        content: textContent,
      });
    } else if (msg.role === 'assistant') {
      if (msg.tool_calls && msg.tool_calls.length > 0) {
        // JetBrains only supports one function call per message
        const firstToolCall = msg.tool_calls[0];
        toolIdToFuncNameMap[firstToolCall.id] = firstToolCall.function.name;
        jetbrainsMessages.push({
          type: 'assistant_message',
          content: textContent,
          functionCall: {
            functionName: firstToolCall.function.name,
            content: firstToolCall.function.arguments,
          },
        });
      } else {
        jetbrainsMessages.push({
          type: 'assistant_message',
          content: textContent,
        });
      }
    } else if (msg.role === 'tool') {
      const functionName = toolIdToFuncNameMap[msg.tool_call_id];
      if (functionName) {
        jetbrainsMessages.push({
          type: 'function_message',
          content: textContent,
          functionName: functionName,
        });
      }
    }
  }
  
  const data = [];
  if (tools && tools.length > 0) {
    data.push({ type: 'json', fqdn: 'llm.parameters.functions' });
    const jetbrainsTools = tools.map(t => t.function || t);
    data.push({ type: 'json', value: JSON.stringify(jetbrainsTools) });
  }
  
  return { messages: jetbrainsMessages, data };
}

/* -------------------------------------------------------------------------- */
/*                          Anthropic Conversions                            */
/* -------------------------------------------------------------------------- */

function convertAnthropicToOpenAI(anthropicReq) {
  const openaiMessages = [];
  const toolIdToFuncNameMap = {};
  
  // Convert system message
  if (anthropicReq.system) {
    let systemPrompt = '';
    if (typeof anthropicReq.system === 'string') {
      systemPrompt = anthropicReq.system;
    } else if (Array.isArray(anthropicReq.system)) {
      systemPrompt = anthropicReq.system
        .filter(item => item.type === 'text')
        .map(item => item.text)
        .join(' ');
    }
    if (systemPrompt) {
      openaiMessages.push({ role: 'system', content: systemPrompt });
    }
  }
  
  // Convert messages
  for (const msg of anthropicReq.messages) {
    if (msg.role === 'user') {
      const textParts = [];
      
      if (typeof msg.content === 'string') {
        textParts.push(msg.content);
      } else if (Array.isArray(msg.content)) {
        for (const block of msg.content) {
          if (block.type === 'text') {
            textParts.push(block.text);
          } else if (block.type === 'tool_result' && block.tool_use_id) {
            const contentStr = typeof block.content === 'string' 
              ? block.content 
              : JSON.stringify(block.content);
            openaiMessages.push({
              role: 'tool',
              tool_call_id: block.tool_use_id,
              content: contentStr,
            });
          }
        }
      }
      
      if (textParts.length > 0) {
        openaiMessages.push({
          role: 'user',
          content: textParts.join(' '),
        });
      }
    } else if (msg.role === 'assistant') {
      const textParts = [];
      const toolCalls = [];
      
      if (Array.isArray(msg.content)) {
        for (const block of msg.content) {
          if (block.type === 'text') {
            textParts.push(block.text);
          } else if (block.type === 'tool_use' && block.id && block.name) {
            const arguments_ = block.input ? JSON.stringify(block.input) : '{}';
            toolCalls.push({
              id: block.id,
              type: 'function',
              function: {
                name: block.name,
                arguments: arguments_,
              },
            });
            toolIdToFuncNameMap[block.id] = block.name;
          }
        }
      }
      
      const contentText = textParts.length > 0 ? textParts.join(' ') : null;
      openaiMessages.push({
        role: 'assistant',
        content: contentText,
        tool_calls: toolCalls.length > 0 ? toolCalls : undefined,
      });
    }
  }
  
  // Convert tools
  let openaiTools = null;
  if (anthropicReq.tools) {
    openaiTools = anthropicReq.tools.map(t => ({
      type: 'function',
      function: {
        name: t.name,
        description: t.description,
        parameters: t.input_schema,
      },
    }));
  }
  
  return {
    model: anthropicReq.model,
    messages: openaiMessages,
    stream: anthropicReq.stream,
    temperature: anthropicReq.temperature,
    max_tokens: anthropicReq.max_tokens,
    top_p: anthropicReq.top_p,
    tools: openaiTools,
    stop: anthropicReq.stop_sequences,
  };
}

function mapFinishReason(finishReason) {
  switch (finishReason) {
    case 'stop': return 'end_turn';
    case 'length': return 'max_tokens';
    case 'tool_calls': return 'tool_use';
    default: return finishReason;
  }
}

function convertOpenAIToAnthropicResponse(openaiResponse) {
  const message = openaiResponse.choices[0].message;
  const contentBlocks = [];
  
  if (message.content) {
    contentBlocks.push({
      type: 'text',
      text: message.content,
    });
  }
  
  if (message.tool_calls) {
    for (const tc of message.tool_calls) {
      let toolInput;
      try {
        toolInput = JSON.parse(tc.function.arguments);
      } catch {
        toolInput = {
          error: 'invalid JSON in arguments',
          arguments: tc.function.arguments,
        };
      }
      contentBlocks.push({
        type: 'tool_use',
        id: tc.id,
        name: tc.function.name,
        input: toolInput,
      });
    }
  }
  
  return {
    id: openaiResponse.id.replace('chatcmpl-', 'msg_'),
    type: 'message',
    role: 'assistant',
    model: openaiResponse.model,
    content: contentBlocks,
    stop_reason: mapFinishReason(openaiResponse.choices[0].finish_reason),
    stop_sequence: null,
    usage: {
      input_tokens: openaiResponse.usage?.prompt_tokens || 0,
      output_tokens: openaiResponse.usage?.completion_tokens || 0,
    },
  };
}

/* -------------------------------------------------------------------------- */
/*                             Streaming                                     */
/* -------------------------------------------------------------------------- */

function createOpenAIStreamResponse(upstreamBody, model, tools = []) {
  const encoder = new TextEncoder();
  const decoder = new TextDecoder();
  const streamId = `chatcmpl-${generateId()}`;
  let firstChunkSent = false;
  let toolId = 0;

  const { readable, writable } = new TransformStream();
  const writer = writable.getWriter();

  (async () => {
    const reader = upstreamBody.getReader();
    let buffer = '';

    const processLine = async (line) => {
      if (!line || line === 'data: end') return;
      if (!line.startsWith('data: ')) return;

      try {
        const data = JSON.parse(line.slice(6));
        const eventType = data.type;

        if (eventType === 'Content') {
          const content = data.content || '';
          if (!content) return;

          let deltaPayload = {};
          if (!firstChunkSent) {
            deltaPayload = { role: 'assistant', content };
            firstChunkSent = true;
          } else {
            deltaPayload = { content };
          }

          const streamResp = {
            id: streamId,
            object: 'chat.completion.chunk',
            created: getCurrentTimestamp(),
            model,
            choices: [{
              delta: deltaPayload,
              index: 0,
              finish_reason: null,
            }],
          };

          await writer.write(encoder.encode(`data: ${JSON.stringify(streamResp)}\n\n`));
        } else if (eventType === 'FunctionCall') {
          const funcName = data.name;
          const funcArgs = data.content;
          
          if (funcName && tools.length > 0) {
            // Find tool index
            toolId = tools.findIndex(tool => tool.name === funcName);
            if (toolId === -1) toolId = 0;
          }

          const deltaPayload = {
            tool_calls: [{
              index: toolId,
              id: `call_${generateId()}`,
              function: {
                arguments: funcArgs,
                name: funcName,
              },
              type: 'function',
            }],
          };

          const streamResp = {
            id: streamId,
            object: 'chat.completion.chunk',
            created: getCurrentTimestamp(),
            model,
            choices: [{
              delta: deltaPayload,
              index: 0,
              finish_reason: null,
            }],
          };

          await writer.write(encoder.encode(`data: ${JSON.stringify(streamResp)}\n\n`));
        } else if (eventType === 'FinishMetadata') {
          const finalResp = {
            id: streamId,
            object: 'chat.completion.chunk',
            created: getCurrentTimestamp(),
            model,
            choices: [{
              delta: {},
              index: 0,
              finish_reason: 'stop',
            }],
          };

          await writer.write(encoder.encode(`data: ${JSON.stringify(finalResp)}\n\n`));
          await writer.write(encoder.encode('data: [DONE]\n\n'));
          return; // Exit the processing loop
        }
      } catch (err) {
        console.warn('Failed to parse upstream line:', err, line);
      }
    };

    try {
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        let newlineIndex;
        while ((newlineIndex = buffer.indexOf('\n')) >= 0) {
          const line = buffer.slice(0, newlineIndex).trim();
          buffer = buffer.slice(newlineIndex + 1);
          await processLine(line);
        }
      }

      // Process remaining buffer
      if (buffer.trim()) {
        await processLine(buffer.trim());
      }
    } catch (err) {
      console.error('Error while streaming from upstream:', err);
    }

    // Ensure stream ends properly
    await writer.write(encoder.encode('data: [DONE]\n\n'));
    await writer.close();
  })();

  return readable;
}

function createAnthropicStreamResponse(openaiStream, model) {
  const encoder = new TextEncoder();
  const messageId = `msg_${generateId()}`;
  
  const { readable, writable } = new TransformStream();
  const writer = writable.getWriter();
  
  (async () => {
    // Start event
    await writer.write(encoder.encode(`event: message_start\ndata: ${JSON.stringify({
      type: 'message_start',
      message: {
        id: messageId,
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: 0, output_tokens: 0 }
      }
    })}\n\n`));
    
    await writer.write(encoder.encode(`event: ping\ndata: ${JSON.stringify({ type: 'ping' })}\n\n`));
    
    const decoder = new TextDecoder();
    const reader = openaiStream.getReader();
    let buffer = '';
    let contentBlockIndex = 0;
    let textBlockOpen = false;
    const toolBlocks = {};
    
    const processLine = async (line) => {
      if (!line.startsWith('data:') || line.trim() === 'data: [DONE]') {
        return;
      }
      
      const dataStr = line.slice(6).trim();
      try {
        const data = JSON.parse(dataStr);
        if (!data.choices) return;
        
        const delta = data.choices[0].delta || {};
        const finishReason = data.choices[0].finish_reason;
        
        if (delta.content) {
          if (!textBlockOpen) {
            await writer.write(encoder.encode(`event: content_block_start\ndata: ${JSON.stringify({
              type: 'content_block_start',
              index: contentBlockIndex,
              content_block: { type: 'text', text: '' }
            })}\n\n`));
            textBlockOpen = true;
          }
          
          await writer.write(encoder.encode(`event: content_block_delta\ndata: ${JSON.stringify({
            type: 'content_block_delta',
            index: contentBlockIndex,
            delta: { type: 'text_delta', text: delta.content }
          })}\n\n`));
        }
        
        if (delta.tool_calls) {
          if (textBlockOpen) {
            await writer.write(encoder.encode(`event: content_block_stop\ndata: ${JSON.stringify({
              type: 'content_block_stop',
              index: contentBlockIndex
            })}\n\n`));
            textBlockOpen = false;
            contentBlockIndex++;
          }
          
          for (const tc of delta.tool_calls) {
            const idx = tc.index;
            if (!(idx in toolBlocks)) {
              toolBlocks[idx] = {
                id: tc.id,
                name: tc.function?.name,
                args: '',
              };
              await writer.write(encoder.encode(`event: content_block_start\ndata: ${JSON.stringify({
                type: 'content_block_start',
                index: contentBlockIndex + idx,
                content_block: {
                  type: 'tool_use',
                  id: tc.id,
                  name: tc.function?.name,
                  input: {}
                }
              })}\n\n`));
            }
            
            if (tc.function?.arguments) {
              const argsDelta = tc.function.arguments;
              toolBlocks[idx].args += argsDelta;
              await writer.write(encoder.encode(`event: content_block_delta\ndata: ${JSON.stringify({
                type: 'content_block_delta',
                index: contentBlockIndex + idx,
                delta: { type: 'input_json_delta', partial_json: argsDelta }
              })}\n\n`));
            }
          }
        }
        
        if (finishReason) {
          if (textBlockOpen) {
            await writer.write(encoder.encode(`event: content_block_stop\ndata: ${JSON.stringify({
              type: 'content_block_stop',
              index: contentBlockIndex
            })}\n\n`));
          }
          
          for (let i = 0; i < Object.keys(toolBlocks).length; i++) {
            await writer.write(encoder.encode(`event: content_block_stop\ndata: ${JSON.stringify({
              type: 'content_block_stop',
              index: contentBlockIndex + i
            })}\n\n`));
          }
          
          await writer.write(encoder.encode(`event: message_delta\ndata: ${JSON.stringify({
            type: 'message_delta',
            delta: {
              stop_reason: mapFinishReason(finishReason),
              stop_sequence: null
            },
            usage: { input_tokens: 0, output_tokens: 0 }
          })}\n\n`));
          
          await writer.write(encoder.encode(`event: message_stop\ndata: ${JSON.stringify({
            type: 'message_stop'
          })}\n\n`));
          return true; // Signal end
        }
      } catch (err) {
        console.warn('Anthropic adapter JSON decode error:', err, dataStr);
      }
      return false;
    };
    
    try {
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        
        buffer += decoder.decode(value, { stream: true });
        let newlineIndex;
        while ((newlineIndex = buffer.indexOf('\n')) >= 0) {
          const line = buffer.slice(0, newlineIndex).trim();
          buffer = buffer.slice(newlineIndex + 1);
          const shouldEnd = await processLine(line);
          if (shouldEnd) break;
        }
      }
      
      if (buffer.trim()) {
        await processLine(buffer.trim());
      }
    } catch (err) {
      console.error('Error in Anthropic stream adapter:', err);
    }
    
    await writer.close();
  })();
  
  return readable;
}

async function aggregateStreamResponse(upstreamBody, model, tools = []) {
  const decoder = new TextDecoder();
  const reader = upstreamBody.getReader();
  let buffer = '';
  const contentParts = [];
  const toolCallsMap = {};
  let finalFinishReason = 'stop';

  const processLine = (line) => {
    if (!line || line === 'data: end') return;
    if (!line.startsWith('data: ')) return;

    try {
      const data = JSON.parse(line.slice(6));
      if (data.type === 'Content' && data.content) {
        contentParts.push(data.content);
      } else if (data.type === 'FunctionCall') {
        const funcName = data.name;
        const funcArgs = data.content;
        const toolIndex = tools.findIndex(tool => tool.name === funcName);
        const callId = `call_${generateId()}`;
        
        toolCallsMap[toolIndex >= 0 ? toolIndex : 0] = {
          id: callId,
          type: 'function',
          function: {
            name: funcName,
            arguments: funcArgs,
          },
        };
        finalFinishReason = 'tool_calls';
      }
    } catch (err) {
      // Ignore parse errors
    }
  };

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    let newlineIndex;
    while ((newlineIndex = buffer.indexOf('\n')) >= 0) {
      const line = buffer.slice(0, newlineIndex).trim();
      buffer = buffer.slice(newlineIndex + 1);
      processLine(line);
    }
  }

  if (buffer.trim()) {
    processLine(buffer.trim());
  }

  const finalToolCalls = Object.values(toolCallsMap);
  const fullContent = contentParts.join('') || null;

  const message = {
    role: 'assistant',
    content: fullContent,
  };

  if (finalToolCalls.length > 0) {
    message.tool_calls = finalToolCalls;
  }

  return {
    id: `chatcmpl-${generateId()}`,
    object: 'chat.completion',
    created: getCurrentTimestamp(),
    model,
    choices: [{
      message,
      index: 0,
      finish_reason: finalFinishReason,
    }],
    usage: {
      prompt_tokens: 0,
      completion_tokens: 0,
      total_tokens: 0,
    },
  };
}

/* -------------------------------------------------------------------------- */
/*                               Endpoints                                   */
/* -------------------------------------------------------------------------- */

async function handleListModels(request, env, corsHeaders) {
  if (!authenticateAnyClient(request, env)) {
    return jsonResponse({ error: { message: 'Unauthorized' } }, 401, {
      ...corsHeaders,
      'WWW-Authenticate': 'Bearer',
    });
  }

  const { models } = loadModelsConfig(env);
  return jsonResponse({ object: 'list', data: models }, 200, corsHeaders);
}

async function handleChatCompletions(request, env, corsHeaders) {
  if (!authenticateOpenAI(request, env)) {
    return jsonResponse({ error: { message: 'Unauthorized' } }, 401, {
      ...corsHeaders,
      'WWW-Authenticate': 'Bearer',
    });
  }

  let body;
  try {
    body = await request.json();
  } catch (err) {
    return jsonResponse({ error: { message: 'Invalid JSON body' } }, 400, corsHeaders);
  }

  const { model, messages, stream = false, tools } = body;
  if (!model || !messages) {
    return jsonResponse({ error: { message: 'model and messages are required' } }, 400, corsHeaders);
  }

  const { models } = loadModelsConfig(env);
  const modelIds = models.map(m => m.id);
  if (!modelIds.includes(model)) {
    return jsonResponse({ error: { message: `Model ${model} not found` } }, 404, corsHeaders);
  }

  try {
    const account = await getNextJetBrainsAccount(env);
    const { messages: jetbrainsMessages, data } = convertOpenAIToJetBrains(messages, tools);

    const payload = {
      prompt: 'ij.chat.request.new-chat-on-start',
      profile: model,
      chat: { messages: jetbrainsMessages },
      parameters: { data },
    };

    const headers = {
      'User-Agent': 'ktor-client',
      'Accept': 'text/event-stream',
      'Content-Type': 'application/json',
      'Accept-Charset': 'UTF-8',
      'Cache-Control': 'no-cache',
      'grazie-agent': '{"name":"aia:cloudflare","version":"1.0.0"}',
      'grazie-authenticate-jwt': account.jwt,
    };

    const upstreamResp = await fetch('https://api.jetbrains.ai/user/v5/llm/chat/stream/v7', {
      method: 'POST',
      headers,
      body: JSON.stringify(payload),
    });

    if (!upstreamResp.ok) {
      const errorText = await upstreamResp.text();
      console.error('JetBrains upstream error:', upstreamResp.status, errorText);
      
      if (upstreamResp.status === 477) {
        // Quota exceeded
        account.has_quota = false;
        account.last_quota_check = getCurrentTimestamp();
      }
      
      return jsonResponse(
        { error: { message: `Upstream error ${upstreamResp.status}` } },
        502,
        corsHeaders
      );
    }

    if (stream) {
      const openaiStream = createOpenAIStreamResponse(upstreamResp.body, model, tools || []);
      return streamResponse(openaiStream, corsHeaders);
    } else {
      const aggregated = await aggregateStreamResponse(upstreamResp.body, model, tools || []);
      return jsonResponse(aggregated, 200, corsHeaders);
    }
  } catch (err) {
    console.error('Error in chat completions:', err);
    return jsonResponse({ error: { message: err.message } }, 503, corsHeaders);
  }
}

async function handleAnthropicMessages(request, env, corsHeaders) {
  if (!authenticateAnthropic(request, env)) {
    return jsonResponse({ error: { message: 'Unauthorized' } }, 401, corsHeaders);
  }

  let anthropicRequest;
  try {
    anthropicRequest = await request.json();
  } catch (err) {
    return jsonResponse({ error: { message: 'Invalid JSON body' } }, 400, corsHeaders);
  }

  const { model, messages, stream = false } = anthropicRequest;
  if (!model || !messages) {
    return jsonResponse({ error: { message: 'model and messages are required' } }, 400, corsHeaders);
  }

  // Convert to OpenAI format
  const openaiRequest = convertAnthropicToOpenAI(anthropicRequest);
  
  // Apply model mapping
  const { models, mappings } = loadModelsConfig(env);
  if (mappings[openaiRequest.model]) {
    const originalModel = openaiRequest.model;
    openaiRequest.model = mappings[openaiRequest.model];
    console.log(`Model mapping applied: ${originalModel} -> ${openaiRequest.model}`);
  }

  const modelIds = models.map(m => m.id);
  if (!modelIds.includes(openaiRequest.model)) {
    return jsonResponse({ error: { message: `Model ${openaiRequest.model} not found` } }, 404, corsHeaders);
  }

  try {
    const account = await getNextJetBrainsAccount(env);
    const { messages: jetbrainsMessages, data } = convertOpenAIToJetBrains(openaiRequest.messages, openaiRequest.tools);

    const payload = {
      prompt: 'ij.chat.request.new-chat-on-start',
      profile: openaiRequest.model,
      chat: { messages: jetbrainsMessages },
      parameters: { data },
    };

    const headers = {
      'User-Agent': 'ktor-client',
      'Accept': 'text/event-stream',
      'Content-Type': 'application/json',
      'Accept-Charset': 'UTF-8',
      'Cache-Control': 'no-cache',
      'grazie-agent': '{"name":"aia:cloudflare","version":"1.0.0"}',
      'grazie-authenticate-jwt': account.jwt,
    };

    const upstreamResp = await fetch('https://api.jetbrains.ai/user/v5/llm/chat/stream/v7', {
      method: 'POST',
      headers,
      body: JSON.stringify(payload),
    });

    if (!upstreamResp.ok) {
      const errorText = await upstreamResp.text();
      console.error('JetBrains upstream error:', upstreamResp.status, errorText);
      
      if (upstreamResp.status === 477) {
        account.has_quota = false;
        account.last_quota_check = getCurrentTimestamp();
      }
      
      return jsonResponse(
        { error: { message: `Upstream error ${upstreamResp.status}` } },
        502,
        corsHeaders
      );
    }

    if (stream) {
      const openaiStream = createOpenAIStreamResponse(upstreamResp.body, openaiRequest.model, openaiRequest.tools || []);
      const anthropicStream = createAnthropicStreamResponse(openaiStream, openaiRequest.model);
      return streamResponse(anthropicStream, corsHeaders);
    } else {
      const aggregated = await aggregateStreamResponse(upstreamResp.body, openaiRequest.model, openaiRequest.tools || []);
      const anthropicResponse = convertOpenAIToAnthropicResponse(aggregated);
      return jsonResponse(anthropicResponse, 200, corsHeaders);
    }
  } catch (err) {
    console.error('Error in Anthropic messages:', err);
    return jsonResponse({ error: { message: err.message } }, 503, corsHeaders);
  }
}