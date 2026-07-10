"""Routing loop: classify → generate → judge → escalate."""

import logging
from dataclasses import dataclass, field

from .classifier import classify
from .config import Config
from .judge import content_to_text, judge_answer
from .openrouter import OpenRouterError
from .pricing import attempt_cost

logger = logging.getLogger(__name__)

GENERATION_KEYS_TO_DROP = {"model", "stream", "stream_options"}


@dataclass
class Mode:
    """Interpretation of request field `model`."""

    kind: str  # "route" | "passthrough"
    name: str
    threshold: int = 0
    max_escalations: int = 0
    passthrough_model: str | None = None


def resolve_mode(model: str | None, config: Config) -> Mode:
    judge = config.judge
    if model == "auto:strict":
        return Mode("route", "auto:strict", judge.strict_threshold, judge.max_escalations)
    if model == "auto:cheap":
        return Mode("route", "auto:cheap", judge.cheap_threshold, judge.cheap_max_escalations)
    if model and "/" in model:
        # Exact OpenRouter model ID ⇒ passthrough without routing.
        return Mode("passthrough", model, passthrough_model=model)
    # "auto", omitted, or unknown ⇒ routing active.
    return Mode("route", "auto", judge.threshold, judge.max_escalations)


@dataclass
class Attempt:
    model: str
    response: dict | None
    score: int | None
    reason: str
    cost_usd: float
    error: str | None = None


@dataclass
class RouteResult:
    response: dict  # OpenAI format + frugal field
    mode: str
    classifier_tier: int | None
    difficulty: str | None
    final_model: str
    final_score: int | None
    attempts: list[Attempt] = field(default_factory=list)
    escalations: int = 0
    total_cost_usd: float = 0.0
    baseline_cost_usd: float = 0.0


class AllModelsFailedError(Exception):
    pass


def _generation_payload(body: dict, model: str) -> dict:
    payload = {k: v for k, v in body.items() if k not in GENERATION_KEYS_TO_DROP}
    payload["model"] = model
    return payload


async def _cost_of(client, model: str, response: dict) -> float:
    price_in, price_out = await client.price_for(model)
    return attempt_cost(response.get("usage"), price_in, price_out)


async def _baseline_cost(client, config: Config, response: dict) -> float:
    """Estimated cost if same tokens went directly to most expensive ladder model."""
    return await _cost_of(client, config.ladder[-1].id, response)


async def route_completion(body: dict, mode: Mode, config: Config, client) -> RouteResult:
    """Execute steps 1-4 of the routing flow and return the best/accepted answer."""
    messages = body.get("messages") or []
    ladder = [step.id for step in config.ladder]

    difficulty, start_index, classifier_cost = await classify(client, config, messages)
    total_cost = classifier_cost

    attempts: list[Attempt] = []
    accepted: Attempt | None = None
    best: Attempt | None = None
    escalations = 0
    index = start_index

    while index < len(ladder):
        model = ladder[index]
        try:
            response = await client.chat_completion(_generation_payload(body, model))
        except OpenRouterError as exc:
            # Model down after retries ⇒ skip to next ladder model
            # (does not count toward max_escalations — no response to judge).
            logger.warning("Generation with %s failed: %s", model, exc)
            attempts.append(Attempt(model, None, None, "", 0.0, error=str(exc)))
            index += 1
            continue

        cost = await _cost_of(client, model, response)
        total_cost += cost
        message = response["choices"][0]["message"]

        if message.get("tool_calls"):
            # Tool calls can't be quality-scored as text — pass through
            # directly without judging (agentic clients judge via the result).
            accepted = Attempt(model, response, None, "tool_calls", cost)
            attempts.append(accepted)
            break

        answer = content_to_text(message.get("content"))
        score, reason, judge_cost = await judge_answer(client, config, messages, answer)
        total_cost += judge_cost
        attempt = Attempt(model, response, score, reason, cost)
        attempts.append(attempt)

        if score is None:
            # Judge down/unparseable ⇒ pass through (fail open).
            accepted = attempt
            break
        if score >= mode.threshold:
            accepted = attempt
            break
        if best is None or score > (best.score or 0):
            best = attempt
        if escalations >= mode.max_escalations:
            break
        escalations += 1
        index += 1

    final = accepted or best
    if final is None or final.response is None:
        raise AllModelsFailedError(
            "; ".join(a.error or "unknown error" for a in attempts) or "ladder is empty"
        )

    baseline = await _baseline_cost(client, config, final.response)
    frugal = {
        "attempts": [
            {"model": a.model, "score": a.score, "cost_usd": round(a.cost_usd, 8)}
            | ({"error": a.error} if a.error else {})
            for a in attempts
        ],
        "final_model": final.model,
        "classifier_tier": start_index,
        "difficulty": difficulty,
        "total_cost_usd": round(total_cost, 8),
        "baseline_cost_usd": round(baseline, 8),
        "saved_pct": round((baseline - total_cost) / baseline * 100, 1)
        if baseline > 0
        else 0.0,
    }
    response = dict(final.response)
    response["frugal"] = frugal

    return RouteResult(
        response=response,
        mode=mode.name,
        classifier_tier=start_index,
        difficulty=difficulty,
        final_model=final.model,
        final_score=final.score,
        attempts=attempts,
        escalations=escalations,
        total_cost_usd=total_cost,
        baseline_cost_usd=baseline,
    )


async def passthrough_completion(body: dict, mode: Mode, config: Config, client) -> RouteResult:
    """Exact model ID: forward directly, but log cost."""
    payload = _generation_payload(body, mode.passthrough_model)
    response = await client.chat_completion(payload)
    cost = await _cost_of(client, mode.passthrough_model, response)
    return RouteResult(
        response=response,
        mode="passthrough",
        classifier_tier=None,
        difficulty=None,
        final_model=mode.passthrough_model,
        final_score=None,
        attempts=[Attempt(mode.passthrough_model, response, None, "", cost)],
        total_cost_usd=cost,
        baseline_cost_usd=cost,
    )
