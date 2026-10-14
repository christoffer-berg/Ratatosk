"""Quality judge: scores candidate answers 1-10 with a cheap model."""

import json
import logging
import re

from .config import Config
from .pricing import attempt_cost

logger = logging.getLogger(__name__)

JUDGE_PROMPT = """\
You are a strict quality reviewer. Judge whether the ANSWER solves the TASK.
Criteria: correctness, completeness, follows instructions, no hallucinations.
Answer ONLY with JSON: {{"score": <1-10>, "reason": "<one sentence>"}}

TASK: {task}

ANSWER: {answer}"""


def extract_first_json(text: str) -> dict | None:
    """Robust JSON parsing: regex-extract first {...}, fallback on garbage."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    candidate = match.group(0)
    # Try full match first, then progressively shorter prefixes ending with }.
    while candidate:
        try:
            parsed = json.loads(candidate)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            end = candidate.rfind("}", 0, len(candidate) - 1)
            if end == -1:
                return None
            candidate = candidate[: end + 1]
    return None


def content_to_text(content) -> str:
    """Extract text from message content (string or multimodal list)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


def last_user_task(messages: list[dict]) -> str:
    """Judge only needs the last user turn."""
    for message in reversed(messages):
        if message.get("role") == "user":
            return content_to_text(message.get("content"))
    return ""


async def judge_answer(
    client, config: Config, messages: list[dict], answer: str, model: str | None = None
) -> tuple[int | None, str, float]:
    """Returns (score, reason, cost_usd). score=None ⇒ judge down/unparseable
    — caller should pass through the answer (fail open) and log warning.

    `model` overrides the configured judge, which is how shadow audits re-score
    an accepted answer with a stronger model."""
    model = model or config.judge.model
    prompt = JUDGE_PROMPT.format(task=last_user_task(messages), answer=answer)
    try:
        response = await client.chat_completion(
            {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": 200,
            }
        )
    except Exception:
        logger.warning("Judge call failed — passing through response", exc_info=True)
        return None, "judge unavailable", 0.0

    price_in, price_out = await client.price_for(model)
    cost = attempt_cost(response.get("usage"), price_in, price_out)

    text = content_to_text(response["choices"][0]["message"].get("content"))
    parsed = extract_first_json(text)
    if parsed is None or not isinstance(parsed.get("score"), (int, float)):
        logger.warning("Unparseable judge response: %r — passing through", text[:200])
        return None, "unparseable judge response", cost

    score = max(1, min(10, int(parsed["score"])))
    return score, str(parsed.get("reason", "")), cost
