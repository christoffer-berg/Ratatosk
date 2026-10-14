"""Prompt features for the local routing model.

Two kinds, laid out in one sparse vector:

* dense — a handful of structured signals (length, code fences, tool use,
  the classifier's labels) at fixed indices [0, DENSE_SIZE);
* hashed — word unigrams and bigrams of the conversation, hashed into
  `buckets` slots after the dense block.

The hashing trick keeps the model a fixed size whatever the vocabulary, and it
means requests can log *features* instead of prompt text: the training data
never has to contain what anyone actually wrote. Hashed n-grams are not a
privacy guarantee — a determined attacker with a vocabulary can probe them —
but they are not the prompt either.

`FEATURE_VERSION` must be bumped whenever extraction changes, so stored
vectors from an older scheme are never mixed with new ones.
"""

import hashlib
import json
import math
import re

from .classifier import DIFFICULTIES, TASK_TYPES, task_text
from .judge import content_to_text

FEATURE_VERSION = 1

TOKEN = re.compile(r"[a-z0-9_]+")
CODE_FENCE = re.compile(r"```")
MAX_TOKENS = 600  # cap per request so one huge prompt can't dominate a row

DENSE_NAMES = (
    ["bias", "log_chars", "log_messages", "log_last_user_chars", "code_fences",
     "has_tools", "digit_ratio", "has_question", "has_system"]
    + [f"task_type={t}" for t in TASK_TYPES]
    + [f"difficulty={d}" for d in DIFFICULTIES]
)
DENSE_SIZE = len(DENSE_NAMES)
_DENSE_INDEX = {name: i for i, name in enumerate(DENSE_NAMES)}


def _bucket(token: str, buckets: int) -> int:
    """Stable across processes — Python's own hash() is salted per run."""
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return DENSE_SIZE + int.from_bytes(digest, "little") % buckets


def _last_user_text(messages: list[dict]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return content_to_text(message.get("content"))
    return ""


def extract(
    messages: list[dict],
    task_type: str | None,
    difficulty: str | None,
    has_tools: bool = False,
    buckets: int = 8192,
) -> dict[int, float]:
    """Sparse feature vector {index: value} for one request."""
    full = "\n".join(content_to_text(m.get("content")) for m in messages)
    last_user = _last_user_text(messages)
    chars = len(full)

    features: dict[int, float] = {
        _DENSE_INDEX["bias"]: 1.0,
        _DENSE_INDEX["log_chars"]: math.log1p(chars) / 10.0,
        _DENSE_INDEX["log_messages"]: math.log1p(len(messages)) / 3.0,
        _DENSE_INDEX["log_last_user_chars"]: math.log1p(len(last_user)) / 10.0,
        _DENSE_INDEX["code_fences"]: min(len(CODE_FENCE.findall(full)) / 2, 5) / 5.0,
        _DENSE_INDEX["has_tools"]: 1.0 if has_tools else 0.0,
        _DENSE_INDEX["digit_ratio"]: (sum(c.isdigit() for c in full) / chars) if chars else 0.0,
        _DENSE_INDEX["has_question"]: 1.0 if "?" in last_user else 0.0,
        _DENSE_INDEX["has_system"]: 1.0 if any(m.get("role") == "system" for m in messages) else 0.0,
    }
    if task_type in TASK_TYPES:
        features[_DENSE_INDEX[f"task_type={task_type}"]] = 1.0
    if difficulty in DIFFICULTIES:
        features[_DENSE_INDEX[f"difficulty={difficulty}"]] = 1.0

    # Text features come from the same truncated view the classifier sees, so
    # a 100k-token paste costs the same to featurise as a one-liner.
    tokens = TOKEN.findall(task_text(messages).lower())[:MAX_TOKENS]
    grams = set(tokens)
    grams.update(f"{a} {b}" for a, b in zip(tokens, tokens[1:]))
    if grams:
        # Binary presence, scaled so the hashed block has unit norm: a long
        # prompt and a short one contribute comparably.
        weight = 1.0 / math.sqrt(len(grams))
        for gram in grams:
            index = _bucket(gram, buckets)
            features[index] = features.get(index, 0.0) + weight
    return {i: v for i, v in features.items() if v}


def dumps(features: dict[int, float]) -> str:
    """Compact JSON for the prompt_features table."""
    indices = sorted(features)
    return json.dumps({"i": indices, "v": [round(features[i], 5) for i in indices]},
                      separators=(",", ":"))


def loads(blob: str) -> dict[int, float]:
    data = json.loads(blob)
    return dict(zip(data["i"], data["v"]))
