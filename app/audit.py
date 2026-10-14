"""Shadow audits: re-judge a sample of accepted answers with a strong model.

The routing loop only ever observes the failures the *cheap* judge catches. If
that judge over-scores fluent-but-wrong answers — the known failure mode of cheap
judges — the learned pass rates are optimistically wrong and the router quietly
learns to under-route, with nothing in the loop to notice.

The fix is to re-score a small random sample with a flagship judge, after the
client already has its response, and log the disagreement. That yields a measured
false-accept rate per task type, which is the number that says whether the cheap
judge can be trusted at all.
"""

import logging
import random

from .config import Config
from .judge import judge_answer

logger = logging.getLogger(__name__)

_rng = random.Random()


def audit_model(config: Config) -> str:
    """The strong judge: configured, else the most expensive ladder rung."""
    return config.audit.model or config.ladder[-1].id


def should_audit(config: Config, score: int | None, rng: random.Random | None = None) -> bool:
    """Audit only *accepted* answers — a rejected one was already escalated."""
    if not config.audit.enabled or score is None:
        return False
    return (rng or _rng).random() < config.audit.sample_pct


async def run_audit(
    client,
    config: Config,
    store,
    request_id: str,
    messages: list[dict],
    answer: str,
    model: str,
    task_type: str | None,
    difficulty: str | None,
    cheap_score: int,
    threshold: int,
) -> None:
    """Re-judge with the strong model and log the disagreement. Never raises."""
    strong = audit_model(config)
    try:
        score, reason, cost = await judge_answer(client, config, messages, answer, model=strong)
        if score is None:
            logger.warning("Shadow audit for %s produced no score", request_id)
            return
        await store.log_audit(
            request_id=request_id,
            judge_model=strong,
            model=model,
            task_type=task_type,
            difficulty=difficulty,
            cheap_score=cheap_score,
            strong_score=score,
            threshold=threshold,
            cost_usd=cost,
            reason=reason,
        )
        if score < threshold <= cheap_score:
            logger.info(
                "Shadow audit: cheap judge accepted %s at %d, strong judge scored %d (%s)",
                model, cheap_score, score, reason[:120],
            )
    except Exception:
        logger.warning("Shadow audit failed for %s", request_id, exc_info=True)
