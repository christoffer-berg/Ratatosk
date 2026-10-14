"""Routing loop: classify → pick start rung → generate → verify → judge → escalate."""

import logging
import time
import uuid
from dataclasses import dataclass, field

from .classifier import classify, prompt_chars, task_text
from .config import Config
from .embeddings import NEIGHBORS_SOURCE, Neighborhood
from .features import dumps as dump_features
from .features import extract as extract_features
from .feedback import fingerprint_text, fingerprint_tool_calls
from .judge import content_to_text, judge_answer
from .learning import CONFIG_SOURCE, apply_exploration, config_start_index
from .openrouter import OpenRouterError
from .predictor import PREDICTOR_SOURCE
from .pricing import attempt_cost
from .verifiers import Verdict, verify

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
    ladder_index: int | None = None
    error: str | None = None
    verdict: Verdict | None = None
    accepted: bool = False

    @property
    def blocked(self) -> bool:
        """A deterministic verifier rejected this answer outright."""
        return self.verdict is not None and not self.verdict.passed


@dataclass
class RouteResult:
    response: dict  # OpenAI format + frugal field
    mode: str
    classifier_tier: int | None
    difficulty: str | None
    final_model: str
    final_score: int | None
    task_type: str | None = None
    tier_source: str | None = None
    threshold: int | None = None
    prompt_chars: int | None = None
    prompt_text: str | None = None
    explored: bool = False
    explore_prob: float | None = None
    answer_hash: str | None = None
    cache_hit: bool = False
    embedding: list[float] | None = None
    final_index: int | None = None
    final_answer: str | None = None
    features: str | None = None  # serialised hashed features, for training
    predicted_pass: list[float] | None = None
    attempts: list[Attempt] = field(default_factory=list)
    escalations: int = 0
    total_cost_usd: float = 0.0
    baseline_cost_usd: float = 0.0

    def attempt_rows(self) -> list[dict]:
        """Per-attempt records for the log (and thus for the learner)."""
        return [
            {
                "model": a.model,
                "ladder_index": a.ladder_index,
                "score": a.score,
                "cost_usd": a.cost_usd,
                "error": a.error,
                "accepted": a.accepted,
                "verifier": a.verdict.name if a.verdict else None,
                "verifier_passed": a.verdict.passed if a.verdict else None,
                "prompt_tokens": _usage(a.response, "prompt_tokens"),
                "completion_tokens": _usage(a.response, "completion_tokens"),
            }
            for a in self.attempts
        ]


def _usage(response: dict | None, key: str) -> int | None:
    usage = (response or {}).get("usage") or {}
    value = usage.get(key)
    return value if isinstance(value, int) else None


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


async def _baseline_rung(learner, config: Config, classification, threshold: int) -> int:
    """Where the skill matrix, then the config tiers, would start this request."""
    if learner is None:
        return config_start_index(config, classification.difficulty)
    await learner.maybe_refresh()
    return learner.start_index(classification.task_type, classification.difficulty, threshold)[0]


async def _start_rung(
    neighborhood: Neighborhood, learner, config: Config, classification, threshold: int,
    predictor=None, features: dict | None = None, chars: int = 0, client=None,
) -> tuple[int, str, list[float] | None]:
    """The trained model, then neighbours, then the skill matrix, then config.

    Each source defers to the next when it has nothing trustworthy to say: the
    model when it is missing or failed its deployment check, neighbours when the
    neighbourhood is sparse, the matrix when a cell has too few samples.
    """
    if predictor is not None and features is not None:
        decision = await predictor.decide(features, threshold, chars, client)
        if decision is not None:
            rung, probs = decision
            # Trust region: never more than one rung below the non-ML router —
            # as far as exploration reaches, so the move is backed by evidence
            # and a wrong model wastes at most one cheap attempt. Must match
            # training.model_start, which is what the deploy check evaluated.
            baseline = await _baseline_rung(learner, config, classification, threshold)
            return max(rung, baseline - 1), PREDICTOR_SOURCE, probs
    rung = neighborhood.start_index(threshold, config.learning.min_pass_rate)
    if rung is not None:
        return min(rung, len(config.ladder) - 1), NEIGHBORS_SOURCE, None
    if learner is None:
        return config_start_index(config, classification.difficulty), CONFIG_SOURCE, None
    await learner.maybe_refresh()
    rung, source = learner.start_index(
        classification.task_type, classification.difficulty, threshold
    )
    return rung, source, None


def _cached_result(
    neighbor, mode: Mode, classification, messages: list[dict], config: Config, cost: float
) -> RouteResult:
    """Serve a near-identical prompt's answer verbatim: 100% saving, no model call."""
    response = {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": neighbor.model or "cache",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": neighbor.answer},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "frugal": {
            "attempts": [],
            "final_model": neighbor.model,
            "classifier_tier": None,
            "difficulty": classification.difficulty,
            "task_type": classification.task_type,
            "tier_source": "cache",
            "cache_hit": True,
            "cache_similarity": round(neighbor.similarity, 4),
            "cache_source_request": neighbor.request_id,
            "total_cost_usd": round(cost, 8),
            "baseline_cost_usd": 0.0,
            "saved_pct": 100.0,
        },
    }
    return RouteResult(
        response=response,
        mode=mode.name,
        classifier_tier=None,
        difficulty=classification.difficulty,
        task_type=classification.task_type,
        tier_source="cache",
        threshold=mode.threshold,
        prompt_chars=prompt_chars(messages),
        prompt_text=task_text(messages) if config.log_prompts else None,
        cache_hit=True,
        final_model=neighbor.model or "cache",
        final_score=neighbor.score,
        final_answer=neighbor.answer,
        answer_hash=fingerprint_text(neighbor.answer),
        total_cost_usd=cost,
        baseline_cost_usd=0.0,
    )


async def route_completion(
    body: dict, mode: Mode, config: Config, client, learner=None, neighbors=None,
    predictor=None,
) -> RouteResult:
    """Execute the routing flow and return the best/accepted answer."""
    messages = body.get("messages") or []
    ladder = [step.id for step in config.ladder]

    classification = await classify(client, config, messages)
    total_cost = classification.cost_usd

    neighborhood = Neighborhood()
    if neighbors is not None:
        neighborhood = await neighbors.lookup(task_text(messages))
        hit = neighborhood.cache_hit(config.embeddings.cache_similarity)
        if hit is not None and config.embeddings.cache_enabled:
            result = _cached_result(hit, mode, classification, messages, config, total_cost)
            result.embedding = neighborhood.embedding
            return result

    features = None
    if config.predictor.log_features or predictor is not None:
        features = extract_features(
            messages,
            classification.task_type,
            classification.difficulty,
            has_tools=bool(body.get("tools")),
            buckets=config.predictor.buckets,
        )
    start_index, tier_source, predicted_pass = await _start_rung(
        neighborhood, learner, config, classification, mode.threshold,
        predictor=predictor, features=features, chars=prompt_chars(messages), client=client,
    )
    # The chance of exploring is recorded with the request: evaluating the
    # model offline weighs explored requests by 1/that, as it was at the time.
    explore_prob = (
        config.exploration.epsilon if config.exploration.enabled and start_index > 0 else 0.0
    )
    explored_index = apply_exploration(start_index, config)
    explored = explored_index != start_index
    start_index = explored_index

    attempts: list[Attempt] = []
    accepted: Attempt | None = None
    best: Attempt | None = None
    best_score = -1
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
            attempts.append(Attempt(model, None, None, "", 0.0, index, error=str(exc)))
            index += 1
            continue

        cost = await _cost_of(client, model, response)
        total_cost += cost
        message = response["choices"][0]["message"]

        if message.get("tool_calls"):
            # Tool calls can't be quality-scored as text — pass through
            # directly without judging (agentic clients judge via the result).
            accepted = Attempt(model, response, None, "tool_calls", cost, index, accepted=True)
            attempts.append(accepted)
            break

        answer = content_to_text(message.get("content"))
        verdict = verify(classification.task_type, answer, config)
        # The judge still runs on an answer a verifier rejected: the pair of
        # scores is what measures how much the cheap judge can be trusted.
        score, reason, judge_cost = await judge_answer(client, config, messages, answer)
        total_cost += judge_cost
        attempt = Attempt(model, response, score, reason, cost, index, verdict=verdict)
        attempts.append(attempt)

        blocked = attempt.blocked and config.verifiers.enforce
        if blocked:
            logger.info(
                "Verifier %s rejected %s despite judge score %s: %s",
                verdict.name, model, score, verdict.detail,
            )
        elif score is None:
            # Judge down/unparseable ⇒ pass through (fail open).
            accepted = attempt
            break
        elif score >= mode.threshold:
            accepted = attempt
            break

        effective = 1 if blocked else (score or 0)
        if effective > best_score:
            best, best_score = attempt, effective
        if escalations >= mode.max_escalations:
            break
        escalations += 1
        index += 1

    final = accepted or best
    if final is None or final.response is None:
        raise AllModelsFailedError(
            "; ".join(a.error or "unknown error" for a in attempts) or "ladder is empty"
        )
    final.accepted = True

    final_message = final.response["choices"][0]["message"]
    final_answer = content_to_text(final_message.get("content"))
    if final_message.get("tool_calls"):
        answer_hash = fingerprint_tool_calls(final_message["tool_calls"])
    else:
        answer_hash = fingerprint_text(final_answer)

    baseline = await _baseline_cost(client, config, final.response)
    frugal = {
        "attempts": [
            {"model": a.model, "score": a.score, "cost_usd": round(a.cost_usd, 8)}
            | ({"error": a.error} if a.error else {})
            | ({"verifier": a.verdict.name, "verifier_passed": a.verdict.passed}
               if a.verdict else {})
            for a in attempts
        ],
        "final_model": final.model,
        "classifier_tier": start_index,
        "difficulty": classification.difficulty,
        "task_type": classification.task_type,
        "tier_source": tier_source,
        "explored": explored,
        **({"predicted_pass": [round(p, 3) for p in predicted_pass]} if predicted_pass else {}),
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
        difficulty=classification.difficulty,
        task_type=classification.task_type,
        tier_source=tier_source,
        threshold=mode.threshold,
        prompt_chars=prompt_chars(messages),
        prompt_text=task_text(messages) if config.log_prompts else None,
        explored=explored,
        explore_prob=explore_prob,
        answer_hash=answer_hash,
        embedding=neighborhood.embedding,
        features=dump_features(features) if features and config.predictor.log_features else None,
        predicted_pass=predicted_pass,
        final_index=final.ladder_index,
        final_answer=final_answer,
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
    messages = body.get("messages") or []
    message = response["choices"][0]["message"]
    return RouteResult(
        response=response,
        mode="passthrough",
        classifier_tier=None,
        difficulty=None,
        final_model=mode.passthrough_model,
        final_score=None,
        prompt_chars=prompt_chars(messages),
        prompt_text=task_text(messages) if config.log_prompts else None,
        answer_hash=(
            fingerprint_tool_calls(message["tool_calls"])
            if message.get("tool_calls")
            else fingerprint_text(content_to_text(message.get("content")))
        ),
        final_answer=content_to_text(message.get("content")),
        attempts=[Attempt(mode.passthrough_model, response, None, "", cost, accepted=True)],
        total_cost_usd=cost,
        baseline_cost_usd=cost,
    )
