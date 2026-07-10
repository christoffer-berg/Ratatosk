"""Difficulty classification: selects starting model in the ladder."""

import logging

from .config import Config
from .judge import content_to_text, extract_first_json
from .pricing import attempt_cost

logger = logging.getLogger(__name__)

DIFFICULTIES = ("trivial", "easy", "medium", "hard", "expert")
DEFAULT_DIFFICULTY = "medium"

CLASSIFIER_PROMPT = """\
Rate how difficult the following task is for a language model.
Answer ONLY with JSON: {{"difficulty": "trivial|easy|medium|hard|expert"}}

trivial: factual question, formatting, simple translation
easy: summarization, simple code, standard questions
medium: multi-step reasoning, moderately complex code, analysis
hard: complex code/math, long context, subtle requirements
expert: research-level, very long/complex task

TASK:
{task}"""

HEAD_CHARS = 2000
TAIL_CHARS = 500


def task_text(messages: list[dict]) -> str:
    """Entire messages history as text, truncated to ~2000 chars + last 500."""
    parts = []
    for message in messages:
        text = content_to_text(message.get("content"))
        if text:
            parts.append(f"{message.get('role', 'user')}: {text}")
    full = "\n".join(parts)
    if len(full) <= HEAD_CHARS + TAIL_CHARS:
        return full
    return f"{full[:HEAD_CHARS]}\n[...]\n{full[-TAIL_CHARS:]}"


async def classify(
    client, config: Config, messages: list[dict]
) -> tuple[str, int, float]:
    """Returns (difficulty, start_index, cost_usd). Error/timeout ⇒ medium."""
    prompt = CLASSIFIER_PROMPT.format(task=task_text(messages))
    difficulty = DEFAULT_DIFFICULTY
    cost = 0.0
    try:
        response = await client.chat_completion(
            {
                "model": config.classifier.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": 50,
            }
        )
        price_in, price_out = await client.price_for(config.classifier.model)
        cost = attempt_cost(response.get("usage"), price_in, price_out)
        text = content_to_text(response["choices"][0]["message"].get("content"))
        parsed = extract_first_json(text)
        if parsed and parsed.get("difficulty") in DIFFICULTIES:
            difficulty = parsed["difficulty"]
        else:
            logger.warning(
                "Unparseable classifier response: %r — defaulting to medium", text[:200]
            )
    except Exception:
        logger.warning(
            "Classifier call failed — defaulting to medium", exc_info=True
        )

    start_index = config.classifier.tiers.get(difficulty, 1)
    start_index = max(0, min(start_index, len(config.ladder) - 1))
    return difficulty, start_index, cost
