"""FastAPI app: endpoints, auth middleware, SSE streaming."""

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .config import Config, load_config
from .openrouter import OpenRouterClient, OpenRouterError
from .router import (
    AllModelsFailedError,
    passthrough_completion,
    resolve_mode,
    route_completion,
)
from .store import Store

logger = logging.getLogger(__name__)

STREAM_CHUNK_CHARS = 400
UNAUTHENTICATED_PATHS = {"/healthz"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = load_config()
    if not config.openrouter.api_key:
        logger.warning("OPENROUTER_API_KEY missing — calls to OpenRouter will fail")
    if not config.api_keys:
        logger.warning("No local api_keys in config — endpoints are open without auth")
    app.state.config = config
    app.state.client = OpenRouterClient(
        base_url=config.openrouter.base_url,
        api_key=config.openrouter.api_key,
        timeout=config.request_timeout_seconds,
        pricing_refresh_hours=config.pricing_refresh_hours,
    )
    app.state.store = Store(config.db_path)
    await app.state.store.init()
    try:
        await app.state.client.prices()
    except Exception:
        logger.warning("Failed to prefetch prices at startup", exc_info=True)
    yield
    await app.state.client.close()
    await app.state.store.close()


app = FastAPI(title="Ratatosk Router", lifespan=lifespan)


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    config: Config = request.app.state.config
    if request.url.path not in UNAUTHENTICATED_PATHS and config.api_keys:
        header = request.headers.get("authorization", "")
        token = header.removeprefix("Bearer ").strip()
        if token not in config.api_keys:
            return JSONResponse(
                status_code=401,
                content={"error": {"message": "Invalid API key", "type": "invalid_request_error"}},
            )
    return await call_next(request)


def _sse_stream(response: dict):
    """Stream a completed response as fake OpenAI chunks (SSE)."""
    completion_id = response.get("id", f"chatcmpl-{uuid.uuid4().hex}")
    created = response.get("created", int(time.time()))
    model = response.get("model", "")
    choice = response["choices"][0]
    content = choice["message"].get("content") or ""

    def chunk(delta: dict, finish_reason: str | None = None, **extra) -> str:
        payload = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
            **extra,
        }
        return f"data: {json.dumps(payload)}\n\n"

    yield chunk({"role": "assistant", "content": ""})
    for i in range(0, len(content), STREAM_CHUNK_CHARS):
        yield chunk({"content": content[i : i + STREAM_CHUNK_CHARS]})
    tool_calls = choice["message"].get("tool_calls") or []
    if tool_calls:
        yield chunk(
            {
                "tool_calls": [
                    {
                        "index": index,
                        "id": call.get("id"),
                        "type": call.get("type", "function"),
                        "function": call.get("function", {}),
                    }
                    for index, call in enumerate(tool_calls)
                ]
            }
        )
    extra = {}
    if response.get("usage"):
        extra["usage"] = response["usage"]
    if response.get("frugal"):
        extra["frugal"] = response["frugal"]
    yield chunk({}, finish_reason=choice.get("finish_reason", "stop"), **extra)
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    config: Config = request.app.state.config
    client: OpenRouterClient = request.app.state.client
    store: Store = request.app.state.store

    try:
        body = await request.json()
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON body")
    if not isinstance(body, dict) or not body.get("messages"):
        raise HTTPException(status_code=400, detail="Field 'messages' is required")

    request_id = uuid.uuid4().hex
    model_requested = body.get("model")
    stream = bool(body.get("stream"))
    mode = resolve_mode(model_requested, config)

    try:
        if mode.kind == "passthrough":
            result = await passthrough_completion(body, mode, config, client)
        else:
            result = await route_completion(body, mode, config, client)
    except (OpenRouterError, AllModelsFailedError) as exc:
        await store.log_request(
            request_id=request_id,
            mode=mode.name if mode.kind == "route" else "passthrough",
            model_requested=model_requested,
            error=str(exc),
        )
        status = getattr(exc, "status_code", None) or 502
        raise HTTPException(status_code=status, detail=str(exc))

    await store.log_request(
        request_id=request_id,
        mode=result.mode,
        model_requested=model_requested,
        classifier_tier=result.classifier_tier,
        difficulty=result.difficulty,
        final_model=result.final_model,
        final_score=result.final_score,
        attempts=[
            {"model": a.model, "score": a.score, "cost_usd": a.cost_usd, "error": a.error}
            for a in result.attempts
        ],
        escalations=result.escalations,
        total_cost_usd=result.total_cost_usd,
        baseline_cost_usd=result.baseline_cost_usd,
    )

    if stream:
        return StreamingResponse(_sse_stream(result.response), media_type="text/event-stream")
    return JSONResponse(result.response)


@app.get("/v1/models")
async def list_models(request: Request):
    config: Config = request.app.state.config
    created = int(time.time())
    ids = ["auto", "auto:strict", "auto:cheap"] + [step.id for step in config.ladder]
    return {
        "object": "list",
        "data": [
            {"id": model_id, "object": "model", "created": created, "owned_by": "ratatosk"}
            for model_id in ids
        ],
    }


@app.get("/stats")
async def stats(request: Request):
    return await request.app.state.store.stats()


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}
