import json
import time
import uuid
import threading
import asyncio
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

import httpx
import uvicorn
import aiofiles
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

# Configuration
DEFAULT_REQUEST_TIMEOUT = 30.0

# Global variables
VALID_CLIENT_KEYS: set = set()
JETBRAINS_ACCOUNTS: list = []
current_account_index: int = 0
account_rotation_lock = asyncio.Lock()
file_write_lock = asyncio.Lock()
models_data: Dict[str, Any] = {}
anthropic_model_mappings: Dict[str, str] = {}
http_client: Optional[httpx.AsyncClient] = None

# Pydantic Models
class ChatMessage(BaseModel):
    role: str
    content: Optional[Union[str, List[Dict[str, Any]]]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    stream: bool = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    tools: Optional[List[Dict[str, Any]]] = None
    stop: Optional[Union[str, List[str]]] = None


# --- Anthropic-Compatible Models ---


class AnthropicContentBlock(BaseModel):
    type: str
    text: Optional[str] = None
    tool_use_id: Optional[str] = None
    content: Optional[Union[str, List[Dict[str, Any]]]] = None
    id: Optional[str] = None
    name: Optional[str] = None
    input: Optional[Dict[str, Any]] = None


class AnthropicMessage(BaseModel):
    role: str
    content: Union[str, List[AnthropicContentBlock]]


class AnthropicTool(BaseModel):
    name: str
    description: Optional[str] = None
    input_schema: Dict[str, Any]


class AnthropicMessageRequest(BaseModel):
    model: str
    messages: List[AnthropicMessage]
    system: Optional[Union[str, List[Dict[str, Any]]]] = None
    max_tokens: int
    stream: bool = False
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    tools: Optional[List[AnthropicTool]] = None
    stop_sequences: Optional[List[str]] = None


# --- Anthropic-Compatible Response Models ---


class AnthropicUsage(BaseModel):
    input_tokens: int
    output_tokens: int


class AnthropicResponseContent(BaseModel):
    type: str
    id: Optional[str] = None
    name: Optional[str] = None
    input: Optional[Dict[str, Any]] = None
    text: Optional[str] = None


class AnthropicResponseMessage(BaseModel):
    id: str
    type: str = "message"
    role: str = "assistant"
    model: str
    content: List[AnthropicResponseContent]
    stop_reason: Optional[str]
    stop_sequence: Optional[str] = None
    usage: AnthropicUsage


# --- End Anthropic Models ---


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelInfo]


class ChatCompletionChoice(BaseModel):
    message: ChatMessage
    index: int = 0
    finish_reason: str = "stop"


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionChoice]
    usage: Dict[str, int] = Field(
        default_factory=lambda: {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
    )


class StreamChoice(BaseModel):
    delta: Dict[str, Any] = Field(default_factory=dict)
    index: int = 0
    finish_reason: Optional[str] = None


class StreamResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[StreamChoice]


# FastAPI App
app = FastAPI(title="JetBrains AI OpenAI Compatible API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

security = HTTPBearer(auto_error=False)


# Helper functions
def load_models():
    """加载模型配置和映射规则"""
    global anthropic_model_mappings
    try:
        with open("models.json", "r", encoding="utf-8") as f:
            config = json.load(f)

        # 支持新格式（包含 models 和 anthropic_model_mappings）
        if isinstance(config, dict):
            if "models" in config:
                model_ids = config["models"]
                # 加载模型映射配置
                anthropic_model_mappings = config.get("anthropic_model_mappings", {})
                print(f"从 models.json 加载了 {len(anthropic_model_mappings)} 个模型映射规则")
            else:
                # 处理旧格式的字典（如果有其他字段但没有 models）
                model_ids = []
                anthropic_model_mappings = {}
                print("警告: models.json 使用非标准格式，没有找到 models 字段")
        # 支持旧格式（仅包含模型列表）
        elif isinstance(config, list):
            model_ids = config
            anthropic_model_mappings = {}
            print("警告: models.json 使用旧格式，没有找到模型映射配置")
        else:
            print("错误: models.json 格式不正确")
            return {"data": []}

        processed_models = []
        if isinstance(model_ids, list):
            for model_id in model_ids:
                if isinstance(model_id, str):
                    processed_models.append(
                        {
                            "id": model_id,
                            "object": "model",
                            "created": int(time.time()),
                            "owned_by": "jetbrains-ai",
                        }
                    )

        return {"data": processed_models}
    except Exception as e:
        print(f"加载 models.json 时出错: {e}")
        # 设置默认映射规则
        anthropic_model_mappings = {}
        return {"data": []}


async def _save_accounts_to_file():
    """将当前账户状态异步保存到文件"""
    async with file_write_lock:
        try:
            async with aiofiles.open("jetbrainsai.json", "w", encoding="utf-8") as f:
                await f.write(json.dumps(JETBRAINS_ACCOUNTS, indent=2))
        except Exception as e:
            print(f"保存 jetbrainsai.json 文件时出错: {e}")


def load_client_api_keys():
    """加载客户端 API 密钥"""
    global VALID_CLIENT_KEYS
    try:
        with open("client_api_keys.json", "r", encoding="utf-8") as f:
            keys = json.load(f)
            if not isinstance(keys, list):
                print("警告: client_api_keys.json 应包含密钥列表")
                VALID_CLIENT_KEYS = set()
                return
            VALID_CLIENT_KEYS = set(keys)
            if not VALID_CLIENT_KEYS:
                print("警告: client_api_keys.json 为空")
            else:
                print(f"成功加载 {len(VALID_CLIENT_KEYS)} 个客户端 API 密钥")
    except FileNotFoundError:
        print("错误: 未找到 client_api_keys.json")
        VALID_CLIENT_KEYS = set()
    except Exception as e:
        print(f"加载 client_api_keys.json 时出错: {e}")
        VALID_CLIENT_KEYS = set()


def load_jetbrains_accounts():
    """加载 JetBrains AI 认证信息"""
    global JETBRAINS_ACCOUNTS
    try:
        with open("jetbrainsai.json", "r", encoding="utf-8") as f:
            accounts_data = json.load(f)

        if not isinstance(accounts_data, list):
            print("警告: jetbrainsai.json 格式不正确，应为对象列表")
            JETBRAINS_ACCOUNTS = []
            return

        processed_accounts = []
        for account in accounts_data:
            processed_accounts.append(
                {
                    "licenseId": account.get("licenseId"),
                    "authorization": account.get("authorization"),
                    "jwt": account.get("jwt"),
                    "last_updated": account.get("last_updated", 0),
                    "has_quota": account.get("has_quota", True),
                    "last_quota_check": account.get("last_quota_check", 0),
                }
            )

        JETBRAINS_ACCOUNTS = processed_accounts
        if not JETBRAINS_ACCOUNTS:
            print("警告: jetbrainsai.json 中未找到有效的认证信息")
        else:
            print(f"成功加载 {len(JETBRAINS_ACCOUNTS)} 个 JetBrains AI 账户")

    except FileNotFoundError:
        print("错误: 未找到 jetbrainsai.json 文件")
        JETBRAINS_ACCOUNTS = []
    except Exception as e:
        print(f"加载 jetbrainsai.json 时出错: {e}")
        JETBRAINS_ACCOUNTS = []


def get_model_item(model_id: str) -> Optional[Dict]:
    """根据模型ID获取模型配置"""
    for model in models_data.get("data", []):
        if model.get("id") == model_id:
            return model
    return None


async def authenticate_client(
    auth: Optional[HTTPAuthorizationCredentials] = Depends(security),
):
    """客户端认证 (OpenAI-style)"""
    if not VALID_CLIENT_KEYS:
        raise HTTPException(status_code=503, detail="服务不可用: 未配置客户端 API 密钥")

    if not auth or not auth.credentials:
        raise HTTPException(
            status_code=401,
            detail="需要在 Authorization header 中提供 API 密钥",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if auth.credentials not in VALID_CLIENT_KEYS:
        raise HTTPException(status_code=403, detail="无效的客户端 API 密钥")


async def authenticate_any_client(
    auth: Optional[HTTPAuthorizationCredentials] = Depends(security),
    api_key: Optional[str] = Header(None, alias="x-api-key"),
):
    """客户端认证 (支持 OpenAI 和 Anthropic 风格)"""
    if not VALID_CLIENT_KEYS:
        raise HTTPException(status_code=503, detail="服务不可用: 未配置客户端 API 密钥")

    # 优先检查 x-api-key
    if api_key:
        if api_key in VALID_CLIENT_KEYS:
            return
        else:
            raise HTTPException(
                status_code=403, detail="无效的客户端 API 密钥 (x-api-key)"
            )

    # 其次检查 Authorization header
    if auth and auth.credentials:
        if auth.credentials in VALID_CLIENT_KEYS:
            return
        else:
            raise HTTPException(
                status_code=403, detail="无效的客户端 API 密钥 (Bearer token)"
            )

    # 如果两者都未提供
    raise HTTPException(
        status_code=401,
        detail="需要在 Authorization header (Bearer) 或 x-api-key header 中提供 API 密钥",
        headers={"WWW-Authenticate": "Bearer"},
    )


async def authenticate_anthropic_client(
    api_key: Optional[str] = Header(None, alias="x-api-key"),
):
    """客户端认证 (Anthropic-style)"""
    if not VALID_CLIENT_KEYS:
        raise HTTPException(status_code=503, detail="服务不可用: 未配置客户端 API 密钥")

    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="需要在 x-api-key header 中提供 API 密钥",
        )

    if api_key not in VALID_CLIENT_KEYS:
        raise HTTPException(status_code=403, detail="无效的客户端 API 密钥")


async def _check_quota(account: dict):
    """检查指定账户的配额"""
    if not http_client:
        raise HTTPException(status_code=500, detail="HTTP 客户端未初始化")

    # 对于基于许可证的账户，如果 JWT 不存在，则先刷新
    if not account.get("jwt") and account.get("licenseId"):
        await _refresh_jetbrains_jwt(account)

    if not account.get("jwt"):
        account["has_quota"] = False
        return

    try:
        headers = {
            "User-Agent": "ktor-client",
            "Content-Length": "0",
            "Accept-Charset": "UTF-8",
            "grazie-agent": '{"name":"aia:pycharm","version":"251.26094.80.13:251.26094.141"}',
            "grazie-authenticate-jwt": account["jwt"],
        }
        response = await http_client.post(
            "https://api.jetbrains.ai/user/v5/quota/get", headers=headers, timeout=10.0
        )

        if response.status_code == 401 and account.get("licenseId"):
            print(f"JWT for {account['licenseId']} expired, refreshing...")
            await _refresh_jetbrains_jwt(account)
            headers["grazie-authenticate-jwt"] = account["jwt"]
            response = await http_client.post(
                "https://api.jetbrains.ai/user/v5/quota/get", headers=headers, timeout=10.0
            )

        response.raise_for_status()
        quota_data = response.json()
        
        has_quota = quota_data.get("dailyUsed", 0) < quota_data.get("dailyTotal", 1)
        account["has_quota"] = has_quota
        if not has_quota:
            print(f"Account {account.get('licenseId') or 'with static JWT'} has no quota.")

    except Exception as e:
        print(f"Error checking quota for account: {e}")
        # On error, assume it has no quota to be safe
        account["has_quota"] = False
    finally:
        account["last_quota_check"] = time.time()
        await _save_accounts_to_file()


async def _refresh_jetbrains_jwt(account: dict):
    """使用 licenseId 和 authorization 刷新 JWT"""
    if not http_client:
        raise HTTPException(status_code=500, detail="HTTP 客户端未初始化")

    print(f"正在为 licenseId {account['licenseId']} 刷新 JWT...")
    try:
        headers = {
            "User-Agent": "ktor-client",
            "Content-Type": "application/json",
            "Accept-Charset": "UTF-8",
            "authorization": f"Bearer {account['authorization']}",
        }
        payload = {"licenseId": account["licenseId"]}

        response = await http_client.post(
            "https://api.jetbrains.ai/auth/jetbrains-jwt/provide-access/license/v2",
            json=payload,
            headers=headers,
            timeout=DEFAULT_REQUEST_TIMEOUT,
        )
        response.raise_for_status()

        data = response.json()
        if data.get("state") == "PAID" and "token" in data:
            account["jwt"] = data["token"]
            account["last_updated"] = time.time()
            print(f"成功刷新 licenseId {account['licenseId']} 的 JWT")
            await _save_accounts_to_file()
        else:
            print(f"刷新 JWT 失败: 无效的响应状态 {data.get('state')}")
            raise HTTPException(status_code=500, detail=f"刷新 JWT 失败: {data}")

    except httpx.HTTPStatusError as e:
        print(f"刷新 JWT 时 HTTP 错误: {e.response.status_code} {e.response.text}")
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"刷新 JWT 失败: {e.response.text}",
        )
    except Exception as e:
        print(f"刷新 JWT 时发生未知错误: {e}")
        raise HTTPException(status_code=500, detail=f"刷新 JWT 时发生未知错误: {e}")


async def get_next_jetbrains_account() -> dict:
    """轮询获取下一个有配额的 JetBrains 账户"""
    global current_account_index

    if not JETBRAINS_ACCOUNTS:
        raise HTTPException(status_code=503, detail="服务不可用: 未配置 JetBrains 账户")

    async with account_rotation_lock:
        start_index = current_account_index
        for _ in range(len(JETBRAINS_ACCOUNTS)):
            account = JETBRAINS_ACCOUNTS[current_account_index]
            current_account_index = (current_account_index + 1) % len(
                JETBRAINS_ACCOUNTS
            )

            # 如果状态是 stale，检查配额
            is_quota_stale = (
                time.time() - account.get("last_quota_check", 0) > 3600
            )  # 1 hour cache
            if account.get("has_quota") and is_quota_stale:
                await _check_quota(account)

            if account.get("has_quota"):
                # 如果是基于许可证的账户，检查 JWT 是否需要刷新
                if account.get("licenseId"):
                    is_jwt_stale = (
                        time.time() - account.get("last_updated", 0) > 12 * 3600
                    )
                    if not account.get("jwt") or is_jwt_stale:
                        await _refresh_jetbrains_jwt(account)
                        # 刷新 JWT 后可能需要重新检查配额
                        if not account.get("has_quota"):
                            await _check_quota(account)
                            if not account.get("has_quota"):
                                continue

                if account.get("jwt"):
                    return account

        # 循环完成，没有找到可用的账户
        raise HTTPException(status_code=429, detail="所有 JetBrains 账户均已超出配额或无效")


# FastAPI 生命周期事件
@app.on_event("startup")
async def startup():
    global models_data, http_client
    models_data = load_models()
    load_client_api_keys()
    load_jetbrains_accounts()
    http_client = httpx.AsyncClient(timeout=None)
    print("JetBrains AI OpenAI Compatible API 服务器已启动")


@app.on_event("shutdown")
async def shutdown():
    global http_client
    if http_client:
        await http_client.aclose()


# API 端点
@app.get("/v1/models", response_model=ModelList)
async def list_models(_: None = Depends(authenticate_any_client)):
    """列出可用模型"""
    model_list = [
        ModelInfo(
            id=model.get("id", ""),
            created=model.get("created", int(time.time())),
            owned_by=model.get("owned_by", "jetbrains-ai"),
        )
        for model in models_data.get("data", [])
    ]
    return ModelList(data=model_list)


async def openai_stream_adapter(
    api_stream_generator: AsyncGenerator[str, None],
    model_name: str,
    tools: Optional[List[Dict[str, Any]]],
) -> AsyncGenerator[str, None]:
    """将 JetBrains API 的流转换为 OpenAI 格式的 SSE"""
    stream_id = f"chatcmpl-{uuid.uuid4().hex}"
    first_chunk_sent = False
    tool_id = 0

    try:
        async for line in api_stream_generator:
            if not line or line == "data: end":
                continue

            if line.startswith("data: "):
                try:
                    data = json.loads(line[6:])
                    event_type = data.get("type")

                    if event_type == "Content":
                        content = data.get("content", "")
                        if not content:
                            continue

                        delta_payload = {}
                        if not first_chunk_sent:
                            delta_payload = {"role": "assistant", "content": content}
                            first_chunk_sent = True
                        else:
                            delta_payload = {"content": content}

                        stream_resp = StreamResponse(
                            id=stream_id,
                            model=model_name,
                            choices=[StreamChoice(delta=delta_payload)],
                        )
                        yield f"data: {stream_resp.json()}\n\n"

                    elif event_type == "FunctionCall":
                        func_name = data.get("name", None)
                        func_argu = data.get("content", None)
                        if func_name and tools:
                            for tool_id, tool in enumerate(tools):
                                if tool["name"] == func_name:
                                    break

                        delta_payload = {
                            "tool_calls": [
                                {
                                    "index": tool_id,
                                    "id": f"call_{uuid.uuid4().hex}",
                                    "function": {
                                        "arguments": func_argu,
                                        "name": func_name,
                                    },
                                    "type": "function" if func_name else None,
                                }
                            ]
                        }
                        stream_resp = StreamResponse(
                            id=stream_id,
                            model=model_name,
                            choices=[StreamChoice(delta=delta_payload)],
                        )
                        yield f"data: {stream_resp.json()}\n\n"

                    elif event_type == "FinishMetadata":
                        final_resp = StreamResponse(
                            id=stream_id,
                            model=model_name,
                            choices=[StreamChoice(delta={}, finish_reason="stop")],
                        )
                        yield f"data: {final_resp.json()}\n\n"
                        break
                except json.JSONDecodeError:
                    print(f"警告: 无法解析的 JSON 行: {line}")
                    continue

        yield "data: [DONE]\n\n"

    except Exception as e:
        print(f"流式适配器错误: {e}")
        error_resp = StreamResponse(
            id=stream_id,
            model=model_name,
            choices=[
                StreamChoice(
                    delta={"role": "assistant", "content": f"内部错误: {str(e)}"},
                    index=0,
                    finish_reason="stop",
                )
            ],
        )
        yield f"data: {error_resp.json()}\n\n"
        yield "data: [DONE]\n\n"


async def aggregate_stream_for_non_stream_response(
    openai_sse_stream: AsyncGenerator[str, None], model_name: str
) -> ChatCompletionResponse:
    """聚合流式响应为完整响应"""
    content_parts = []
    tool_calls_map = {}
    final_finish_reason = "stop"

    async for sse_line in openai_sse_stream:
        if sse_line.startswith("data: ") and sse_line.strip() != "data: [DONE]":
            try:
                data = json.loads(sse_line[6:].strip())
                if not data.get("choices"):
                    continue

                choice = data["choices"][0]
                delta = choice.get("delta", {})

                if choice.get("finish_reason"):
                    final_finish_reason = choice.get("finish_reason")

                if delta.get("content"):
                    content_parts.append(delta["content"])

                if "tool_calls" in delta:
                    for tc_chunk in delta["tool_calls"]:
                        idx = tc_chunk["index"]
                        if idx not in tool_calls_map:
                            tool_calls_map[idx] = {
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            }

                        if tc_chunk.get("id"):
                            tool_calls_map[idx]["id"] = tc_chunk["id"]

                        func_chunk = tc_chunk.get("function", {})
                        if func_chunk.get("name"):
                            tool_calls_map[idx]["function"]["name"] = func_chunk["name"]
                        if func_chunk.get("arguments"):
                            tool_calls_map[idx]["function"]["arguments"] += func_chunk[
                                "arguments"
                            ]
            except json.JSONDecodeError:
                print(f"警告: 聚合时无法解析的 JSON 行: {sse_line}")

    final_tool_calls = []
    for k, v in sorted(tool_calls_map.items()):
        if "id" not in v:
            v["id"] = f"call_{uuid.uuid4().hex}"
        final_tool_calls.append(v)

    full_content = "".join(content_parts) or None

    if final_tool_calls:
        message = ChatMessage(
            role="assistant", content=full_content, tool_calls=final_tool_calls
        )
        final_finish_reason = "tool_calls"
    else:
        message = ChatMessage(role="assistant", content=full_content)

    return ChatCompletionResponse(
        model=model_name,
        choices=[
            ChatCompletionChoice(
                message=message,
                finish_reason=final_finish_reason,
            )
        ],
    )


def extract_text_content(content: Optional[Union[str, List[Dict[str, Any]]]]) -> str:
    """从消息内容中提取文本内容"""
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        # 处理多模态消息格式，提取所有文本内容
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))
        return " ".join(text_parts)
    return ""


@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest, _: None = Depends(authenticate_client)
):
    """创建聊天完成"""
    model_config = get_model_item(request.model)
    if not model_config:
        raise HTTPException(status_code=404, detail=f"模型 {request.model} 未找到")

    account = await get_next_jetbrains_account()
    auth_token = account["jwt"]

    # 从历史消息中创建 tool_call_id 到 function_name 的映射
    tool_id_to_func_name_map = {}
    for m in request.messages:
        if m.role == "assistant" and m.tool_calls:
            for tc in m.tool_calls:
                if tc.get("id") and tc.get("function", {}).get("name"):
                    tool_id_to_func_name_map[tc["id"]] = tc["function"]["name"]

    # 将 OpenAI 格式的消息转换为 JetBrains 格式
    jetbrains_messages = []
    for msg in request.messages:
        # 提取文本内容，处理多模态消息格式
        text_content = extract_text_content(msg.content)

        if msg.role in ["user", "system"]:
            jetbrains_messages.append(
                {"type": f"{msg.role}_message", "content": text_content}
            )

        elif msg.role == "assistant":
            if msg.tool_calls:
                # 只处理第一个工具调用，以匹配 JetBrains API 的限制
                first_tool_call = msg.tool_calls[0]
                tool_id_to_func_name_map[first_tool_call["id"]] = first_tool_call[
                    "function"
                ]["name"]
                jetbrains_messages.append(
                    {
                        "type": "assistant_message",
                        "content": text_content,
                        "functionCall": {
                            "functionName": first_tool_call["function"]["name"],
                            "content": first_tool_call["function"]["arguments"],
                        },
                    }
                )
            else:
                jetbrains_messages.append(
                    {"type": "assistant_message", "content": text_content}
                )

        elif msg.role == "tool":
            function_name = tool_id_to_func_name_map.get(msg.tool_call_id)
            if function_name:
                jetbrains_messages.append(
                    {
                        "type": "function_message",
                        "content": text_content,
                        "functionName": function_name,
                    }
                )
            else:
                print(
                    f"警告: 无法为 tool_call_id {msg.tool_call_id} 找到对应的函数调用"
                )
        else:
            jetbrains_messages.append({"type": "user_message", "content": text_content})

    data = []
    tools = None
    if request.tools:
        data.append({"type": "json", "fqdn": "llm.parameters.functions"})
        tools = [t["function"] for t in request.tools]
        data.append({"type": "json", "value": json.dumps(tools)})

    # 创建 API 请求的 payload
    payload = {
        "prompt": "ij.chat.request.new-chat-on-start",
        "profile": request.model,
        "chat": {"messages": jetbrains_messages},
        "parameters": {"data": data},
    }

    headers = {
        "User-Agent": "ktor-client",
        "Accept": "text/event-stream",
        "Content-Type": "application/json",
        "Accept-Charset": "UTF-8",
        "Cache-Control": "no-cache",
        "grazie-agent": '{"name":"aia:pycharm","version":"251.26094.80.13:251.26094.141"}',
        "grazie-authenticate-jwt": auth_token,
    }

    async def api_stream_generator():
        """一个包装 httpx 请求的异步生成器"""
        try:
            async with http_client.stream(
                "POST",
                "https://api.jetbrains.ai/user/v5/llm/chat/stream/v7",
                json=payload,
                headers=headers,
            ) as response:
                if response.status_code == 477:
                    print(f"Account {account.get('licenseId') or 'with static JWT'} has no quota (received 477).")
                    account["has_quota"] = False
                    account["last_quota_check"] = time.time()
                    await _save_accounts_to_file()
                response.raise_for_status()
                async for line in response.aiter_lines():
                    yield line
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 477:
                print(f"Account {account.get('licenseId') or 'with static JWT'} has no quota (received 477).")
                account["has_quota"] = False
                account["last_quota_check"] = time.time()
                await _save_accounts_to_file()
            raise e

    # 创建 OpenAI 格式的流
    openai_sse_stream = openai_stream_adapter(
        api_stream_generator(), request.model, tools or []
    )

    # 返回流式或非流式响应
    if request.stream:
        return StreamingResponse(openai_sse_stream, media_type="text/event-stream")
    else:
        return await aggregate_stream_for_non_stream_response(
            openai_sse_stream, request.model
        )


def convert_anthropic_to_openai(
    anthropic_req: AnthropicMessageRequest,
) -> ChatCompletionRequest:
    openai_messages = []
    tool_id_to_func_name_map = {}

    if anthropic_req.system:
        system_prompt = ""
        if isinstance(anthropic_req.system, str):
            system_prompt = anthropic_req.system
        elif isinstance(anthropic_req.system, list):
            system_prompt = " ".join(
                [
                    item.get("text", "")
                    for item in anthropic_req.system
                    if isinstance(item, dict) and item.get("type") == "text"
                ]
            )
        if system_prompt:
            openai_messages.append(ChatMessage(role="system", content=system_prompt))

    for msg in anthropic_req.messages:
        if msg.role == "user":
            text_parts = []
            if isinstance(msg.content, str):
                text_parts.append(msg.content)
            else:
                for block in msg.content:
                    if block.type == "text":
                        text_parts.append(block.text)
                    elif block.type == "tool_result" and block.tool_use_id:
                        content_str = (
                            block.content
                            if isinstance(block.content, str)
                            else json.dumps(block.content)
                        )
                        openai_messages.append(
                            ChatMessage(
                                role="tool",
                                tool_call_id=block.tool_use_id,
                                content=content_str,
                            )
                        )

            if text_parts:
                openai_messages.append(
                    ChatMessage(role="user", content=" ".join(text_parts))
                )

        elif msg.role == "assistant":
            text_parts = []
            tool_calls = []
            if isinstance(msg.content, list):
                for block in msg.content:
                    if block.type == "text":
                        text_parts.append(block.text)
                    elif block.type == "tool_use" and block.id and block.name:
                        arguments = (
                            json.dumps(block.input) if block.input is not None else "{}"
                        )
                        tool_calls.append(
                            {
                                "id": block.id,
                                "type": "function",
                                "function": {
                                    "name": block.name,
                                    "arguments": arguments,
                                },
                            }
                        )
                        tool_id_to_func_name_map[block.id] = block.name

            content_text = " ".join(text_parts) if text_parts else None
            openai_messages.append(
                ChatMessage(
                    role="assistant",
                    content=content_text,
                    tool_calls=tool_calls if tool_calls else None,
                )
            )

    openai_tools = None
    if anthropic_req.tools:
        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.input_schema,
                },
            }
            for t in anthropic_req.tools
        ]

    return ChatCompletionRequest(
        model=anthropic_req.model,
        messages=openai_messages,
        stream=anthropic_req.stream,
        temperature=anthropic_req.temperature,
        max_tokens=anthropic_req.max_tokens,
        top_p=anthropic_req.top_p,
        tools=openai_tools,
        stop=anthropic_req.stop_sequences,
    )


def map_finish_reason(finish_reason: Optional[str]) -> Optional[str]:
    if finish_reason == "stop":
        return "end_turn"
    if finish_reason == "length":
        return "max_tokens"
    if finish_reason == "tool_calls":
        return "tool_use"
    return finish_reason


async def openai_to_anthropic_stream_adapter(
    openai_stream: AsyncGenerator[str, None], model_name: str
) -> AsyncGenerator[str, None]:
    message_id = f"msg_{uuid.uuid4().hex.replace('-', '')}"
    yield f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': {'id': message_id, 'type': 'message', 'role': 'assistant', 'model': model_name, 'content': [], 'stop_reason': None, 'stop_sequence': None, 'usage': {'input_tokens': 0, 'output_tokens': 0}}})}\n\n"
    yield f"event: ping\ndata: {json.dumps({'type': 'ping'})}\n\n"

    content_block_index = 0
    text_block_open = False
    tool_blocks = {}  # index -> {id, name, args}

    async for sse_line in openai_stream:
        if not sse_line.startswith("data:") or sse_line.strip() == "data: [DONE]":
            continue

        data_str = sse_line[6:].strip()
        try:
            data = json.loads(data_str)
            if not data.get("choices"):
                continue

            delta = data["choices"][0].get("delta", {})
            finish_reason = data["choices"][0].get("finish_reason")

            if delta.get("content"):
                if not text_block_open:
                    yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': content_block_index, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
                    text_block_open = True

                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': content_block_index, 'delta': {'type': 'text_delta', 'text': delta['content']}})}\n\n"

            if delta.get("tool_calls"):
                if text_block_open:
                    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': content_block_index})}\n\n"
                    text_block_open = False
                    content_block_index += 1

                for tc in delta["tool_calls"]:
                    idx = tc["index"]
                    if idx not in tool_blocks:
                        tool_blocks[idx] = {
                            "id": tc.get("id"),
                            "name": tc.get("function", {}).get("name"),
                            "args": "",
                        }
                        yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': content_block_index + idx, 'content_block': {'type': 'tool_use', 'id': tc.get('id'), 'name': tc.get('function', {}).get('name'), 'input': {}}})}\n\n"

                    if tc.get("function", {}).get("arguments"):
                        args_delta = tc["function"]["arguments"]
                        tool_blocks[idx]["args"] += args_delta
                        yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': content_block_index + idx, 'delta': {'type': 'input_json_delta', 'partial_json': args_delta}})}\n\n"

            if finish_reason:
                if text_block_open:
                    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': content_block_index})}\n\n"

                for i in range(len(tool_blocks)):
                    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': content_block_index + i})}\n\n"

                message_delta_data = {
                    "type": "message_delta",
                    "delta": {
                        "stop_reason": map_finish_reason(finish_reason),
                        "stop_sequence": None,
                    },
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                }
                yield f"event: message_delta\ndata: {json.dumps(message_delta_data)}\n\n"
                break
        except json.JSONDecodeError:
            print(f"Anthropic adapter JSON decode error: {data_str}")
            continue

    yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"


def convert_openai_to_anthropic_response(
    resp: ChatCompletionResponse,
) -> AnthropicResponseMessage:
    message = resp.choices[0].message
    content_blocks = []

    if message.content:
        content_blocks.append(
            AnthropicResponseContent(type="text", text=message.content)
        )

    if message.tool_calls:
        for tc in message.tool_calls:
            try:
                tool_input = json.loads(tc["function"]["arguments"])
            except json.JSONDecodeError:
                tool_input = {
                    "error": "invalid JSON in arguments",
                    "arguments": tc["function"]["arguments"],
                }
            content_blocks.append(
                AnthropicResponseContent(
                    type="tool_use",
                    id=tc["id"],
                    name=tc["function"]["name"],
                    input=tool_input,
                )
            )

    return AnthropicResponseMessage(
        id=resp.id.replace("chatcmpl-", "msg_"),
        model=resp.model,
        content=content_blocks,
        stop_reason=map_finish_reason(resp.choices[0].finish_reason),
        usage=AnthropicUsage(
            input_tokens=resp.usage.get("prompt_tokens", 0),
            output_tokens=resp.usage.get("completion_tokens", 0),
        ),
    )


@app.post("/v1/messages", response_model=None)
async def messages_completions(
    request: AnthropicMessageRequest, _: None = Depends(authenticate_anthropic_client)
):
    """创建符合 Anthropic 规范的聊天完成"""
    openai_request = convert_anthropic_to_openai(request)

    # Apply model mapping specifically for /v1/messages endpoint using config from models.json
    if openai_request.model in anthropic_model_mappings:
        original_model = openai_request.model
        openai_request.model = anthropic_model_mappings[openai_request.model]
        print(f"Model mapping applied: {original_model} -> {openai_request.model}")

    model_config = get_model_item(openai_request.model)
    if not model_config:
        raise HTTPException(
            status_code=404, detail=f"模型 {openai_request.model} 未找到"
        )

    account = await get_next_jetbrains_account()
    auth_token = account["jwt"]

    tool_id_to_func_name_map = {}
    for m in openai_request.messages:
        if m.role == "assistant" and m.tool_calls:
            for tc in m.tool_calls:
                if tc.get("id") and tc.get("function", {}).get("name"):
                    tool_id_to_func_name_map[tc["id"]] = tc["function"]["name"]

    jetbrains_messages = []
    for msg in openai_request.messages:
        text_content = extract_text_content(msg.content)
        if msg.role in ["user", "system"]:
            jetbrains_messages.append(
                {"type": f"{msg.role}_message", "content": text_content}
            )
        elif msg.role == "assistant":
            if msg.tool_calls:
                first_tool_call = msg.tool_calls[0]
                tool_id_to_func_name_map[first_tool_call["id"]] = first_tool_call[
                    "function"
                ]["name"]
                jetbrains_messages.append(
                    {
                        "type": "assistant_message",
                        "content": text_content,
                        "functionCall": {
                            "functionName": first_tool_call["function"]["name"],
                            "content": first_tool_call["function"]["arguments"],
                        },
                    }
                )
            else:
                jetbrains_messages.append(
                    {"type": "assistant_message", "content": text_content}
                )
        elif msg.role == "tool":
            function_name = tool_id_to_func_name_map.get(msg.tool_call_id)
            if function_name:
                jetbrains_messages.append(
                    {
                        "type": "function_message",
                        "content": text_content,
                        "functionName": function_name,
                    }
                )

    data = []
    tools = None
    if openai_request.tools:
        data.append({"type": "json", "fqdn": "llm.parameters.functions"})
        tools = [t["function"] for t in openai_request.tools]
        data.append({"type": "json", "value": json.dumps(tools)})

    payload = {
        "prompt": "ij.chat.request.new-chat-on-start",
        "profile": openai_request.model,
        "chat": {"messages": jetbrains_messages},
        "parameters": {"data": data},
    }
    headers = {
        "User-Agent": "ktor-client",
        "Accept": "text/event-stream",
        "Content-Type": "application/json",
        "Accept-Charset": "UTF-8",
        "Cache-Control": "no-cache",
        "grazie-agent": '{"name":"aia:pycharm","version":"251.26094.80.13:251.26094.141"}',
        "grazie-authenticate-jwt": auth_token,
    }

    async def api_stream_generator():
        try:
            async with http_client.stream(
                "POST",
                "https://api.jetbrains.ai/user/v5/llm/chat/stream/v7",
                json=payload,
                headers=headers,
            ) as response:
                if response.status_code == 477:
                    print(f"Account {account.get('licenseId') or 'with static JWT'} has no quota (received 477).")
                    account["has_quota"] = False
                    account["last_quota_check"] = time.time()
                    await _save_accounts_to_file()
                response.raise_for_status()
                async for line in response.aiter_lines():
                    yield line
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 477:
                print(f"Account {account.get('licenseId') or 'with static JWT'} has no quota (received 477).")
                account["has_quota"] = False
                account["last_quota_check"] = time.time()
                await _save_accounts_to_file()
            raise e

    openai_sse_stream = openai_stream_adapter(
        api_stream_generator(), openai_request.model, tools or []
    )

    if openai_request.stream:
        anthropic_stream = openai_to_anthropic_stream_adapter(
            openai_sse_stream, openai_request.model
        )
        return StreamingResponse(anthropic_stream, media_type="text/event-stream")
    else:
        openai_response = await aggregate_stream_for_non_stream_response(
            openai_sse_stream, openai_request.model
        )
        return convert_openai_to_anthropic_response(openai_response)


# 主程序入口
if __name__ == "__main__":
    import os

    # 创建示例配置文件（如果不存在）
    if not os.path.exists("client_api_keys.json"):
        with open("client_api_keys.json", "w", encoding="utf-8") as f:
            json.dump(["sk-your-custom-key-here"], f, indent=2)
        print("已创建示例 client_api_keys.json 文件")

    if not os.path.exists("jetbrainsai.json"):
        with open("jetbrainsai.json", "w", encoding="utf-8") as f:
            json.dump([{"jwt": "your-jwt-here"}], f, indent=2)
        print("已创建示例 jetbrainsai.json 文件")

    if not os.path.exists("models.json"):
        with open("models.json", "w", encoding="utf-8") as f:
            example_config = {
                "models": ["anthropic-claude-3.5-sonnet"],
                "anthropic_model_mappings": {
                    "claude-3.5-sonnet": "anthropic-claude-3.5-sonnet",
                    "sonnet": "anthropic-claude-3.5-sonnet"
                }
            }
            json.dump(example_config, f, indent=2)
        print("已创建示例 models.json 文件")

    print("正在启动 JetBrains AI OpenAI Compatible API 服务器...")
    print("端点:")
    print("  GET  /v1/models")
    print("  POST /v1/chat/completions")
    print("  POST /v1/messages")
    print("\n在 Authorization header 中使用客户端 API 密钥 (Bearer sk-xxx)")

    uvicorn.run(app, host="0.0.0.0", port=8000)
