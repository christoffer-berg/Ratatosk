"""Ground-truth labels from clients, which beat the judge as a training signal.

Two sources, both cheap:

* explicit — `POST /v1/feedback` with `{"request_id": ..., "ok": false}`.
* implicit — the next turn in a conversation. When a user answers with "no,
  that's wrong" or an agent feeds back a tool error, the previous answer was bad,
  whatever the judge thought of it.

The proxy is stateless, so a follow-up arrives as a fresh request carrying the
whole history. The answer being reacted to is matched by hashing it and looking
the hash up in the log — which also means no answer text has to be stored to
make the link.
"""

import hashlib
import json
import logging
import re

from .config import Config
from .judge import content_to_text

logger = logging.getLogger(__name__)

EXPLICIT_SOURCE = "explicit"
REGRET_SOURCE = "user_regret"
TOOL_ERROR_SOURCE = "tool_error"

# Deliberately narrow. A false negative label is worse than a missed one: it
# teaches the router that a good model is bad.
REGRET = re.compile(
    r"""(
        that'?s\s+(wrong|not\s+right|incorrect|not\s+what)
      | that\s+(is|isn'?t|didn'?t|doesn'?t)\s+(wrong|right|correct|work)
      | (still|it'?s)\s+(wrong|broken|failing|not\s+working)
      | do(esn'?t|es\s+not)\s+work
      | did\s?n'?t\s+work
      | try\s+again
      | ^no[,.!]?\s
      | ^nope\b
      | ^wrong\b
      | not\s+what\s+i\s+(asked|wanted|meant)
      | you\s+(got\s+it\s+wrong|misunderstood)
    )""",
    re.I | re.X,
)

TOOL_ERROR = re.compile(
    r"""(
        ^\s*\{?\s*"?error"?\s*[:=]
      | \btraceback\s*\(most\s+recent
      | \b[a-z]{2,}(error|exception)\s*[:(]   # PermissionError: / ValueError(
      | \bcommand\s+not\s+found\b
      | \bno\s+such\s+file\b
      | \bpermission\s+denied\b
      | \bexit\s+(code|status)\s+[1-9]
      | \bfailed\s+with\b
    )""",
    re.I | re.X,
)


def answer_fingerprint(message: dict) -> str | None:
    """Stable hash of an assistant turn — text, or its tool calls if it made any.

    Lets a later turn be matched to the logged request that produced it without
    keeping the answer text around.
    """
    if not message or message.get("role") != "assistant":
        return None
    calls = message.get("tool_calls")
    if calls:
        payload = json.dumps(
            [
                {
                    "name": (call.get("function") or {}).get("name"),
                    "arguments": (call.get("function") or {}).get("arguments"),
                }
                for call in calls
            ],
            sort_keys=True,
        )
    else:
        payload = content_to_text(message.get("content"))
    payload = payload.strip()
    if not payload:
        return None
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def fingerprint_text(answer: str) -> str | None:
    """Fingerprint for an answer this proxy just produced."""
    answer = (answer or "").strip()
    return hashlib.sha256(answer.encode("utf-8")).hexdigest() if answer else None


def fingerprint_tool_calls(tool_calls: list) -> str | None:
    return answer_fingerprint({"role": "assistant", "tool_calls": tool_calls})


def _preceding_assistant(messages: list[dict], before: int) -> dict | None:
    for message in reversed(messages[:before]):
        if message.get("role") == "assistant":
            return message
        if message.get("role") == "user":
            return None  # a user turn in between ⇒ not a direct reaction
    return None


def implicit_negative(messages: list[dict], config: Config) -> tuple[str, str] | None:
    """(fingerprint, source) of an answer the latest turn is complaining about.

    Returns None unless the reaction is unambiguous — a short "that's wrong"
    style reply, or a tool result that clearly carries an error.
    """
    if not config.feedback.enabled or not config.feedback.implicit or len(messages) < 2:
        return None
    last = messages[-1]
    role = last.get("role")
    text = content_to_text(last.get("content"))
    if not text:
        return None

    if role == "user":
        # A long message is a new instruction that happens to mention a problem,
        # not a verdict on the previous answer.
        if len(text) > config.feedback.max_regret_chars or not REGRET.search(text.strip()):
            return None
        source = REGRET_SOURCE
    elif role == "tool":
        if not TOOL_ERROR.search(text):
            return None
        source = TOOL_ERROR_SOURCE
    else:
        return None

    previous = _preceding_assistant(messages, len(messages) - 1)
    fingerprint = answer_fingerprint(previous) if previous else None
    return (fingerprint, source) if fingerprint else None
