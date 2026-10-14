"""Tests for explicit and implicit ground-truth signals."""

from app.config import ClassifierConfig, Config, JudgeConfig, LadderStep
from app.feedback import (
    REGRET_SOURCE,
    TOOL_ERROR_SOURCE,
    answer_fingerprint,
    fingerprint_text,
    implicit_negative,
)


def make_config(**overrides) -> Config:
    config = Config(
        ladder=[LadderStep(id="a/b")],
        classifier=ClassifierConfig(model="m"),
        judge=JudgeConfig(model="j"),
    )
    for key, value in overrides.items():
        setattr(config.feedback, key, value)
    return config


ANSWER = {"role": "assistant", "content": "The capital is Lyon."}
TOOL_CALL = {
    "role": "assistant",
    "content": None,
    "tool_calls": [
        {"id": "1", "type": "function",
         "function": {"name": "read_file", "arguments": '{"path": "x"}'}}
    ],
}


def convo(*turns):
    return [{"role": "user", "content": "What is the capital of France?"}, *turns]


def test_fingerprint_links_a_reply_back_to_the_logged_answer():
    # The proxy hashes the answer it produced; the follow-up hashes the same
    # text off the wire. No answer text has to be stored to make the link.
    assert answer_fingerprint(ANSWER) == fingerprint_text("The capital is Lyon.")


def test_fingerprint_ignores_surrounding_whitespace():
    assert answer_fingerprint({"role": "assistant", "content": " hi \n"}) == fingerprint_text("hi")


def test_fingerprint_covers_tool_calls():
    assert answer_fingerprint(TOOL_CALL) is not None
    assert answer_fingerprint(TOOL_CALL) != answer_fingerprint(ANSWER)


def test_non_assistant_turns_have_no_fingerprint():
    assert answer_fingerprint({"role": "user", "content": "hi"}) is None
    assert answer_fingerprint({"role": "assistant", "content": "  "}) is None


def test_short_regret_marks_the_previous_answer_bad():
    messages = convo(ANSWER, {"role": "user", "content": "No, that's wrong."})
    fingerprint, source = implicit_negative(messages, make_config())
    assert source == REGRET_SOURCE
    assert fingerprint == answer_fingerprint(ANSWER)


def test_various_phrasings_are_recognised():
    for text in ["try again", "nope", "that doesn't work", "still broken",
                 "not what I asked for", "wrong"]:
        messages = convo(ANSWER, {"role": "user", "content": text})
        assert implicit_negative(messages, make_config()) is not None, text


def test_ordinary_follow_ups_are_not_negative():
    for text in ["Thanks! Now translate it.", "What about Germany?", "ok"]:
        messages = convo(ANSWER, {"role": "user", "content": text})
        assert implicit_negative(messages, make_config()) is None, text


def test_a_long_message_is_a_new_instruction_not_a_verdict():
    # "doesn't work" buried in a long request should not label the last answer.
    text = "The deploy doesn't work " + "and here is a lot more context " * 20
    messages = convo(ANSWER, {"role": "user", "content": text})
    assert implicit_negative(messages, make_config()) is None


def test_reaction_must_directly_follow_the_answer():
    messages = convo(
        ANSWER,
        {"role": "user", "content": "ok"},
        {"role": "user", "content": "no, that's wrong"},
    )
    assert implicit_negative(messages, make_config()) is None


def test_tool_error_marks_the_tool_call_bad():
    messages = convo(TOOL_CALL, {"role": "tool", "content": "Traceback (most recent call last):"})
    fingerprint, source = implicit_negative(messages, make_config())
    assert source == TOOL_ERROR_SOURCE
    assert fingerprint == answer_fingerprint(TOOL_CALL)


def test_successful_tool_results_are_not_negative():
    messages = convo(TOOL_CALL, {"role": "tool", "content": "print('hello')"})
    assert implicit_negative(messages, make_config()) is None


def test_error_shaped_tool_payloads_are_recognised():
    for text in ['{"error": "no such file"}', "command not found: fo",
                 "exit code 1", "PermissionError: denied"]:
        messages = convo(TOOL_CALL, {"role": "tool", "content": text})
        assert implicit_negative(messages, make_config()) is not None, text


def test_implicit_can_be_switched_off():
    messages = convo(ANSWER, {"role": "user", "content": "no, that's wrong"})
    assert implicit_negative(messages, make_config(implicit=False)) is None
    assert implicit_negative(messages, make_config(enabled=False)) is None


def test_a_conversation_with_no_prior_answer_is_ignored():
    assert implicit_negative([{"role": "user", "content": "no, wrong"}], make_config()) is None
