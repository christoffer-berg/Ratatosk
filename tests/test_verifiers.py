"""Tests for the deterministic verifiers."""

from app.config import ClassifierConfig, Config, JudgeConfig, LadderStep
from app.verifiers import check_arithmetic, check_code_blocks, verify


def make_config(**overrides) -> Config:
    config = Config(
        ladder=[LadderStep(id="a/b")],
        classifier=ClassifierConfig(model="m"),
        judge=JudgeConfig(model="j"),
    )
    for key, value in overrides.items():
        setattr(config.verifiers, key, value)
    return config


def block(tag: str, code: str) -> str:
    return f"Here you go:\n```{tag}\n{code}\n```\n"


def test_broken_python_is_a_hard_failure():
    verdict = check_code_blocks(block("python", "def f(:\n    pass"))
    assert verdict.name == "python_syntax"
    assert verdict.passed is False


def test_valid_python_passes():
    verdict = check_code_blocks(block("python", "def f():\n    return 1"))
    assert verdict.passed is True


def test_untagged_blocks_are_not_judged():
    assert check_code_blocks(block("", "def f(:")) is None


def test_pseudocode_and_transcripts_are_skipped():
    # A fragment with an elided body is not meant to parse as a module.
    assert check_code_blocks(block("python", "def f():\n    ...")) is None
    assert check_code_blocks(block("python", ">>> f()\n1")) is None
    assert check_code_blocks(block("python", "class A:\n    <your code here>")) is None


def test_broken_json_is_a_hard_failure():
    verdict = check_code_blocks(block("json", '{"a": 1,}'))
    assert verdict.name == "json_syntax"
    assert verdict.passed is False


def test_no_code_blocks_yields_no_verdict():
    assert check_code_blocks("just prose, no code") is None


def test_wrong_arithmetic_is_caught():
    verdict = check_arithmetic("Adding them, 2 + 2 = 5.")
    assert verdict.passed is False
    assert "claims 5" in verdict.detail


def test_correct_arithmetic_passes():
    assert check_arithmetic("Adding them, 2 + 2 = 4.").passed is True


def test_every_claim_is_checked_not_just_the_first():
    verdict = check_arithmetic("First 12 * 3 = 36, then 10 / 4 = 3.")
    assert verdict.passed is False
    assert "10 / 4" in verdict.detail


def test_variable_definitions_are_not_arithmetic_claims():
    assert check_arithmetic("Let x = 5 and y = 7.") is None


def test_thousands_separators_are_understood():
    assert check_arithmetic("Revenue is 1,200 * 3 = 3,600 total.").passed is True


def test_unevaluable_expressions_are_left_alone():
    assert check_arithmetic("Speed = 12 m/s and cost = $4") is None


def test_exponent_towers_cannot_hang_the_process():
    # `**` is rejected outright rather than evaluated.
    assert check_arithmetic("9 ** 9 ** 9 = 1") is None


def test_arithmetic_is_not_trusted_for_creative_tasks():
    story = "In the story 2 + 2 = 5, which is the point."
    assert verify("creative", story) is None
    assert verify("math", story).passed is False


def test_code_is_checked_regardless_of_task_type():
    # A summarisation answer containing broken Python is still broken.
    assert verify("summarization", block("python", "def f(:")).passed is False


def test_disabled_verifiers_return_nothing():
    assert verify("code", block("python", "def f(:"), make_config(enabled=False)) is None


def test_a_failing_verdict_wins_over_a_passing_one():
    answer = block("python", "x = 1") + "and 2 + 2 = 5"
    assert verify("math", answer).passed is False


def test_empty_answer_is_not_verdicted():
    assert verify("code", "") is None
