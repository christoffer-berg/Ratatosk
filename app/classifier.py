"""Prompt classification: difficulty + task type, used to pick a start rung."""

import logging
from dataclasses import dataclass

from .config import Config
from .judge import content_to_text, extract_first_json
from .pricing import attempt_cost

logger = logging.getLogger(__name__)

DIFFICULTIES = ("trivial", "easy", "medium", "hard", "expert")
DEFAULT_DIFFICULTY = "medium"

# Model competence is not one-dimensional — a cheap model can be great at
# summarization and hopeless at math — so difficulty alone is a poor routing key.
TASK_TYPES = (
    "code",
    "math",
    "extraction",
    "summarization",
    "factual_qa",
    "creative",
    "translation",
    "other",
)
DEFAULT_TASK_TYPE = "other"

CLASSIFIER_PROMPT = """\
Rate how difficult the following task is for a language model, and what kind of task it is.
Answer ONLY with JSON: {{"difficulty": "trivial|easy|medium|hard|expert", "task_type": "code|math|extraction|summarization|factual_qa|creative|translation|other"}}

difficulty:
trivial: factual question, formatting, simple translation
easy: summarization, simple code, standard questions
medium: multi-step reasoning, moderately complex code, analysis
hard: complex code/math, long context, subtle requirements
expert: research-level, very long/complex task

task_type:
code: writing, explaining, reviewing or debugging code
math: arithmetic, symbolic math, quantitative reasoning
extraction: pulling structured data or specific facts out of supplied text
summarization: condensing, rewriting or reformatting supplied text
factual_qa: questions answered from general world knowledge
creative: fiction, poetry, brainstorming, marketing copy
translation: converting text between natural languages
other: anything that fits none of the above

TASK:
{task}"""

HEAD_CHARS = 2000
TAIL_CHARS = 500


@dataclass
class Classification:
    difficulty: str
    task_type: str
    cost_usd: float


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


def prompt_chars(messages: list[dict]) -> int:
    """Untruncated length of the conversation, logged as a cheap input feature."""
    return sum(len(content_to_text(message.get("content"))) for message in messages)


async def classify(client, config: Config, messages: list[dict]) -> Classification:
    """Error/timeout ⇒ (medium, other) so routing degrades instead of failing."""
    prompt = CLASSIFIER_PROMPT.format(task=task_text(messages))
    difficulty = DEFAULT_DIFFICULTY
    task_type = DEFAULT_TASK_TYPE
    cost = 0.0
    try:
        response = await client.chat_completion(
            {
                "model": config.classifier.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": 80,
            }
        )
        price_in, price_out = await client.price_for(config.classifier.model)
        cost = attempt_cost(response.get("usage"), price_in, price_out)
        text = content_to_text(response["choices"][0]["message"].get("content"))
        parsed = extract_first_json(text) or {}
        if parsed.get("difficulty") in DIFFICULTIES:
            difficulty = parsed["difficulty"]
        else:
            logger.warning(
                "Unparseable classifier difficulty: %r — defaulting to medium", text[:200]
            )
        if parsed.get("task_type") in TASK_TYPES:
            task_type = parsed["task_type"]
    except Exception:
        logger.warning("Classifier call failed — defaulting to medium", exc_info=True)

    return Classification(difficulty=difficulty, task_type=task_type, cost_usd=cost)
