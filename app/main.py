"""FastAPI app: endpoints, auth middleware, SSE streaming."""

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask

from .audit import audit_model, run_audit, should_audit
from .config import Config, load_config
from .embeddings import EmbeddingClient, NeighborRouter, pack
from .features import FEATURE_VERSION
from .feedback import EXPLICIT_SOURCE, implicit_negative
from .learning import TierLearner
from .predictor import PredictorRuntime
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
SINCE_UNITS = {"h": 3600, "d": 86400, "w": 604800}


def _parse_since(since: str | None) -> str | None:
    """`7d` / `24h` / `2w` ⇒ ISO timestamp; anything else is passed through."""
    if not since:
        return None
    unit = SINCE_UNITS.get(since[-1])
    if unit is None:
        return since
    try:
        amount = float(since[:-1])
    except ValueError:
        return since
    return (datetime.now(timezone.utc) - timedelta(seconds=amount * unit)).isoformat()


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
    app.state.learner = TierLearner(config, app.state.store)
    try:
        await app.state.learner.refresh()
    except Exception:
        logger.warning("Failed to load learned tiers at startup", exc_info=True)
    app.state.embeddings = EmbeddingClient(config) if config.embeddings.enabled else None
    app.state.neighbors = NeighborRouter(config, app.state.store, app.state.embeddings)
    if config.embeddings.enabled:
        try:
            await app.state.neighbors.refresh()
        except Exception:
            logger.warning("Failed to load neighbour index at startup", exc_info=True)
    app.state.predictor = PredictorRuntime(config)
    app.state.predictor.load()
    try:
        await app.state.client.prices()
    except Exception:
        logger.warning("Failed to prefetch prices at startup", exc_info=True)
    yield
    await app.state.client.close()
    if app.state.embeddings is not None:
        await app.state.embeddings.close()
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
    learner: TierLearner = request.app.state.learner
    neighbors: NeighborRouter = request.app.state.neighbors
    predictor: PredictorRuntime = request.app.state.predictor
    predictor.maybe_reload()

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
    await _record_implicit_feedback(store, config, body["messages"])

    try:
        if mode.kind == "passthrough":
            result = await passthrough_completion(body, mode, config, client)
        else:
            result = await route_completion(
                body, mode, config, client, learner, neighbors,
                predictor if config.predictor.enabled else None,
            )
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
        task_type=result.task_type,
        tier_source=result.tier_source,
        final_model=result.final_model,
        final_score=result.final_score,
        attempts=result.attempt_rows(),
        escalations=result.escalations,
        total_cost_usd=result.total_cost_usd,
        baseline_cost_usd=result.baseline_cost_usd,
        threshold=result.threshold,
        prompt_chars=result.prompt_chars,
        prompt_text=result.prompt_text,
        explored=result.explored,
        explore_prob=result.explore_prob,
        answer_hash=result.answer_hash,
        cache_hit=result.cache_hit,
    )
    await _log_prompt_vector(store, config, request_id, result)
    await _log_prompt_features(store, config, request_id, result)

    # Runs after the response is handed to the client, so auditing costs the
    # request no latency at all.
    background = _audit_task(request, config, store, request_id, body, result)

    if stream:
        return StreamingResponse(
            _sse_stream(result.response),
            media_type="text/event-stream",
            background=background,
        )
    return JSONResponse(result.response, background=background)


async def _record_implicit_feedback(store: Store, config: Config, messages: list) -> None:
    """Label the previous answer when this turn reads as a complaint about it."""
    try:
        signal = implicit_negative(messages, config)
        if signal is None:
            return
        fingerprint, source = signal
        request_id = await store.request_by_answer_hash(fingerprint)
        if request_id is None:
            return  # the answer came from somewhere other than this proxy
        await store.record_feedback(request_id, ok=False, source=source)
        logger.info("Implicit negative feedback for %s (%s)", request_id, source)
    except Exception:
        logger.warning("Failed to record implicit feedback", exc_info=True)


async def _log_prompt_features(store: Store, config: Config, request_id: str, result) -> None:
    """The local model's training data: hashed features, never the prompt text."""
    if not config.predictor.log_features or result.features is None:
        return
    try:
        await store.log_prompt_features(
            request_id=request_id,
            feature_version=FEATURE_VERSION,
            buckets=config.predictor.buckets,
            prompt_chars=result.prompt_chars,
            features=result.features,
        )
    except Exception:
        logger.warning("Failed to store prompt features", exc_info=True)


async def _log_prompt_vector(store: Store, config: Config, request_id: str, result) -> None:
    if not config.embeddings.enabled or result.embedding is None:
        return
    try:
        await store.log_prompt_vector(
            request_id=request_id,
            model=config.embeddings.model,
            vector=pack(result.embedding),
            ladder_index=result.final_index,
            score=result.final_score,
            answer_text=result.final_answer if config.embeddings.cache_enabled else None,
        )
    except Exception:
        logger.warning("Failed to store prompt vector", exc_info=True)


def _audit_task(request: Request, config: Config, store: Store, request_id: str, body: dict, result):
    """A shadow audit for a sampled accepted answer, or None."""
    if result.cache_hit or not should_audit(config, result.final_score):
        return None
    return BackgroundTask(
        run_audit,
        client=request.app.state.client,
        config=config,
        store=store,
        request_id=request_id,
        messages=body["messages"],
        answer=result.final_answer or "",
        model=result.final_model,
        task_type=result.task_type,
        difficulty=result.difficulty,
        cheap_score=result.final_score,
        threshold=result.threshold,
    )


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
async def stats(request: Request, since: str | None = None):
    """`since` accepts an ISO timestamp or a window like `7d` / `24h`."""
    learner: TierLearner = request.app.state.learner
    store: Store = request.app.state.store
    window = _parse_since(since)
    payload = await store.stats(window)
    payload["skill_matrix"] = learner.skill_matrix()
    payload["judge_calibration"] = await store.audit_stats(window)
    payload["feedback"] = await store.feedback_stats(window)
    return payload


@app.post("/v1/feedback")
async def feedback(request: Request):
    """Ground truth from the client: `{"request_id": ..., "ok": false}`.

    Negative feedback overrides the judge's score when the skill matrix is
    rebuilt, so a model the judge liked but the user didn't stops looking good.
    """
    config: Config = request.app.state.config
    store: Store = request.app.state.store
    if not config.feedback.enabled:
        raise HTTPException(status_code=404, detail="Feedback is disabled")
    try:
        body = await request.json()
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON body")
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Body must be an object")
    request_id = body.get("request_id")
    ok = body.get("ok")
    if not isinstance(request_id, str) or not isinstance(ok, bool):
        raise HTTPException(
            status_code=400, detail="Fields 'request_id' (string) and 'ok' (bool) are required"
        )
    note = body.get("note")
    recorded = await store.record_feedback(
        request_id, ok=ok, source=EXPLICIT_SOURCE, note=note if isinstance(note, str) else None
    )
    if not recorded:
        raise HTTPException(status_code=404, detail=f"Unknown request_id {request_id}")
    return {"recorded": True, "request_id": request_id, "ok": ok}


@app.get("/learn")
async def learn(request: Request):
    """The learned knowledge: pass rates per cell and the start rungs they imply."""
    config: Config = request.app.state.config
    learner: TierLearner = request.app.state.learner
    return {
        "enabled": config.learning.enabled,
        "policy": config.learning.policy,
        "min_samples": config.learning.min_samples,
        "lookback_days": config.learning.lookback_days,
        "refreshed_at": learner.refreshed_at,
        "cells": learner.cell_count,
        "exploration": {
            "enabled": config.exploration.enabled,
            "epsilon": config.exploration.epsilon,
        },
        "verifiers": {
            "enabled": config.verifiers.enabled,
            "enforce": config.verifiers.enforce,
        },
        "audit": {
            "enabled": config.audit.enabled,
            "sample_pct": config.audit.sample_pct,
            "model": audit_model(config),
        },
        "neighbors": {
            "enabled": config.embeddings.enabled,
            "model": config.embeddings.model,
            "vectors": request.app.state.neighbors.vector_count,
            "refreshed_at": request.app.state.neighbors.refreshed_at,
            "cache_enabled": config.embeddings.cache_enabled,
        },
        "judge_calibration": await request.app.state.store.audit_stats(),
        "feedback": await request.app.state.store.feedback_stats(),
        "tiers": learner.learned_tiers(),
        "skill_matrix": learner.skill_matrix(),
    }


@app.get("/predictor")
async def predictor_status(request: Request):
    """The local routing model: whether it is routing, and how it was evaluated."""
    config: Config = request.app.state.config
    runtime: PredictorRuntime = request.app.state.predictor
    model = runtime.model
    return {
        "enabled": config.predictor.enabled,
        "active": runtime.active,
        "status": runtime.status,
        "path": config.predictor.path,
        "policy": config.predictor.policy,
        "log_features": config.predictor.log_features,
        "model": None if model is None else {
            "trained_at": model.trained_at,
            "examples": model.examples,
            "ladder": model.ladder,
            "thresholds": model.thresholds,
            "deployable": model.deployable,
            "deploy_reason": model.deploy_reason,
            "metrics": model.metrics,
        },
    }


@app.post("/predictor/reload")
async def predictor_reload(request: Request):
    """Load a freshly trained model now instead of waiting for the file check."""
    runtime: PredictorRuntime = request.app.state.predictor
    runtime.load()
    return {"active": runtime.active, "status": runtime.status}


@app.post("/learn/refresh")
async def learn_refresh(request: Request):
    """Recompute the skill matrix (and neighbour index) without waiting for the TTL."""
    config: Config = request.app.state.config
    learner: TierLearner = request.app.state.learner
    neighbors: NeighborRouter = request.app.state.neighbors
    try:
        cells = await learner.refresh()
        vectors = await neighbors.refresh() if config.embeddings.enabled else 0
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Refresh failed: {exc}")
    return {"cells": cells, "vectors": vectors, "refreshed_at": learner.refreshed_at}


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}
