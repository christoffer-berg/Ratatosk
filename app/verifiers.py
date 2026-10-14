"""Deterministic verifiers: zero-cost, unfoolable quality signals.

An LLM judge can be talked into accepting a fluent wrong answer; a Python parser
cannot. Wherever a task has a mechanical correctness check, run it — it costs
nothing, it catches false accepts the cheap judge misses, and (because the
result is logged next to the judge's score) it measures how much the judge can
be trusted per task type.

Deliberately conservative: a verifier reports a failure only when it is certain.
Anything ambiguous returns None, which leaves the judgement to the LLM judge.
"""

import ast
import json
import logging
import math
import re

from dataclasses import dataclass

logger = logging.getLogger(__name__)

CODE_BLOCK = re.compile(r"```([A-Za-z0-9_+#.-]*)[ \t]*\n(.*?)```", re.DOTALL)

PYTHON_TAGS = {"python", "py", "python3"}
JSON_TAGS = {"json"}

# Fragments, transcripts and diffs are not expected to parse as a module.
NOT_A_MODULE = re.compile(
    r"""^\s*(
        \.\.\.            # elided body
      | \#\s*\.\.\.       # elided body, commented
      | >>>|\.\.\.\        # REPL transcript
      | [+-]{3}\s         # diff header
      | @@                # diff hunk
      | <[A-Za-z_ ]+>     # <your code here> placeholder
    )""",
    re.M | re.VERBOSE,
)

# "2 + 2 = 5" / "12 * 3 = 36". Only flags claims it can fully evaluate.
ARITHMETIC = re.compile(
    r"(?<![\w.])(\d[\d\s+\-*/×÷().,]{2,60}?)\s*=\s*(-?\d(?:[\d,]*\d)?(?:\.\d+)?)(?!\w)"
)

_MATH_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.USub, ast.UAdd,
)


@dataclass
class Verdict:
    """A deterministic check that actually ran. `passed=False` is hard evidence."""

    name: str
    passed: bool
    detail: str = ""


def _strip_thousands(number: str) -> str:
    return number.replace(",", "") if re.fullmatch(r"-?\d{1,3}(,\d{3})+(\.\d+)?", number) else number


def _safe_arithmetic(expression: str) -> float | None:
    """Evaluate a pure-arithmetic expression, or None if it isn't one.

    No names, no calls, and no `**` — an exponent tower would hang the process.
    """
    expression = expression.replace("×", "*").replace("÷", "/").strip().rstrip(".")
    expression = re.sub(r"(?<=\d),(?=\d{3}\b)", "", expression)
    if not expression or not re.fullmatch(r"[\d\s+\-*/().]+", expression):
        return None
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if not isinstance(node, _MATH_NODES):
            return None
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
            return None
    try:
        return float(eval(compile(tree, "<arithmetic>", "eval"), {"__builtins__": {}}, {}))
    except (ArithmeticError, ValueError):
        return None


def check_arithmetic(answer: str) -> Verdict | None:
    """Evaluate every `<expression> = <number>` claim the answer makes."""
    checked = 0
    for expression, claimed in ARITHMETIC.findall(answer):
        if not any(op in expression for op in "+-*/×÷"):
            continue  # "x = 5" is a definition, not a claim
        actual = _safe_arithmetic(expression)
        if actual is None:
            continue
        try:
            expected = float(_strip_thousands(claimed))
        except ValueError:
            continue
        checked += 1
        if not math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-9):
            return Verdict(
                "arithmetic",
                False,
                f"{expression.strip()} = {actual:g}, answer claims {expected:g}",
            )
    return Verdict("arithmetic", True, f"{checked} claim(s) check out") if checked else None


def check_code_blocks(answer: str) -> Verdict | None:
    """Syntax-check fenced Python and JSON blocks. Untagged blocks are skipped."""
    checked = 0
    for tag, body in CODE_BLOCK.findall(answer):
        tag = tag.lower()
        if not body.strip() or NOT_A_MODULE.search(body):
            continue
        if tag in PYTHON_TAGS:
            checked += 1
            try:
                ast.parse(body)
            except SyntaxError as exc:
                return Verdict("python_syntax", False, f"line {exc.lineno}: {exc.msg}")
            except ValueError as exc:  # e.g. null bytes
                return Verdict("python_syntax", False, str(exc))
        elif tag in JSON_TAGS:
            checked += 1
            try:
                json.loads(body)
            except json.JSONDecodeError as exc:
                return Verdict("json_syntax", False, f"line {exc.lineno}: {exc.msg}")
    return Verdict("code_syntax", True, f"{checked} block(s) parse") if checked else None


# Which verifier applies to which classifier label. Code blocks are checked for
# any task type — a summarisation answer that includes broken Python is still
# broken — while arithmetic is only trusted where numbers are the point.
ARITHMETIC_TASK_TYPES = {"math", "extraction", "other"}


def verify(task_type: str, answer: str, config=None) -> Verdict | None:
    """Best available deterministic verdict for an answer, or None if none applies."""
    if config is not None and not config.verifiers.enabled:
        return None
    if not answer:
        return None
    try:
        verdict = check_code_blocks(answer)
        if verdict is not None and not verdict.passed:
            return verdict
        if task_type in ARITHMETIC_TASK_TYPES:
            arithmetic = check_arithmetic(answer)
            if arithmetic is not None and not arithmetic.passed:
                return arithmetic
            verdict = verdict or arithmetic
        return verdict
    except Exception:
        # A verifier must never take down a request it was only advising on.
        logger.warning("Verifier crashed — ignoring", exc_info=True)
        return None
