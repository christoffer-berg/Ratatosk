"""Tests for routing loop with mocked OpenRouter client."""

import json

import pytest

from app.config import ClassifierConfig, Config, JudgeConfig, LadderStep
from app.judge import extract_first_json
from app.openrouter import OpenRouterError
from app.router import resolve_mode, route_completion

LADDER = [
    "cheap/model-a",
    "mid/model-b",
    "big/model-c",
    "flagship/model-d",
]

JUDGE_MODEL = "cheap/judge"
CLASSIFIER_MODEL = "cheap/classifier"


def make_config(**judge_overrides) -> Config:
    return Config(
        ladder=[LadderStep(id=model_id) for model_id in LADDER],
        classifier=ClassifierConfig(model=CLASSIFIER_MODEL),
        judge=JudgeConfig(model=JUDGE_MODEL, **judge_overrides),
    )


def completion(content: str, model: str, tool_calls: list | None = None) -> dict:
    message: dict = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
        message["content"] = None
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if tool_calls else "stop",
            }
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
    }


TOOL_CALL = {
    "id": "call_abc123",
    "type": "function",
    "function": {"name": "read_file", "arguments": '{"path": "main.py"}'},
}


class FakeClient:
    """Mocked OpenRouter client: scripted responses for classifier, judge, and generation."""

    def __init__(
        self,
        difficulty: str = "easy",
        scores: list[int] | None = None,
        failing_models: set[str] | None = None,
        judge_raises: bool = False,
        judge_garbage: bool = False,
        classifier_raises: bool = False,
        tool_call_models: set[str] | None = None,
    ):
        self.difficulty = difficulty
        self.scores = list(scores or [])
        self.failing_models = failing_models or set()
        self.judge_raises = judge_raises
        self.judge_garbage = judge_garbage
        self.classifier_raises = classifier_raises
        self.tool_call_models = tool_call_models or set()
        self.generation_calls: list[str] = []
        self.judge_calls = 0

    async def chat_completion(self, payload: dict) -> dict:
        model = payload["model"]
        if model == CLASSIFIER_MODEL and "Rate how difficult" in payload["messages"][0]["content"]:
            if self.classifier_raises:
                raise OpenRouterError("classifier down", status_code=503)
            return completion(json.dumps({"difficulty": self.difficulty}), model)
        if model == JUDGE_MODEL and "strict quality reviewer" in payload["messages"][0]["content"]:
            self.judge_calls += 1
            if self.judge_raises:
                raise OpenRouterError("judge down", status_code=503)
            if self.judge_garbage:
                return completion("not json at all", model)
            score = self.scores.pop(0)
            return completion(json.dumps({"score": score, "reason": "test"}), model)
        # Generating model
        self.generation_calls.append(model)
        if model in self.failing_models:
            raise OpenRouterError("model down", status_code=502)
        if model in self.tool_call_models:
            return completion("", model, tool_calls=[dict(TOOL_CALL)])
        return completion(f"response from {model}", model)

    async def price_for(self, model: str) -> tuple[float, float]:
        # Flagship is 10x more expensive so baseline > total in tests.
        if model == LADDER[-1]:
            return (1e-5, 2e-5)
        return (1e-6, 2e-6)


BODY = {"model": "auto", "messages": [{"role": "user", "content": "What is 2+2?"}]}


def mode_for(config: Config, model: str = "auto"):
    return resolve_mode(model, config)


async def test_accepts_first_answer_when_score_meets_threshold():
    config = make_config()
    client = FakeClient(difficulty="easy", scores=[8])
    result = await route_completion(dict(BODY), mode_for(config), config, client)

    assert client.generation_calls == [LADDER[0]]
    assert result.final_model == LADDER[0]
    assert result.escalations == 0
    frugal = result.response["frugal"]
    assert frugal["final_model"] == LADDER[0]
    assert frugal["classifier_tier"] == 0
    assert len(frugal["attempts"]) == 1
    assert frugal["attempts"][0]["score"] == 8


async def test_escalates_until_threshold_met():
    config = make_config()
    client = FakeClient(difficulty="easy", scores=[4, 6, 9])
    result = await route_completion(dict(BODY), mode_for(config), config, client)

    assert client.generation_calls == LADDER[:3]
    assert result.final_model == LADDER[2]
    assert result.escalations == 2
    assert [a["score"] for a in result.response["frugal"]["attempts"]] == [4, 6, 9]


async def test_returns_best_attempt_when_ladder_exhausted():
    config = make_config()
    client = FakeClient(difficulty="easy", scores=[3, 6, 4, 5])
    result = await route_completion(dict(BODY), mode_for(config), config, client)

    # max_escalations=3 ⇒ 4 attempts, none reach 7 ⇒ best (score 6) is returned.
    assert client.generation_calls == LADDER
    assert result.final_model == LADDER[1]
    assert result.final_score == 6


async def test_cheap_mode_allows_only_one_escalation():
    config = make_config()
    client = FakeClient(difficulty="easy", scores=[1, 2])
    result = await route_completion(dict(BODY), mode_for(config, "auto:cheap"), config, client)

    assert client.generation_calls == LADDER[:2]
    assert result.final_score == 2  # best of the two


async def test_strict_mode_uses_higher_threshold():
    config = make_config()
    client = FakeClient(difficulty="easy", scores=[7, 8])
    result = await route_completion(dict(BODY), mode_for(config, "auto:strict"), config, client)

    # 7 is not enough when threshold is 8.
    assert client.generation_calls == LADDER[:2]
    assert result.final_score == 8


async def test_classifier_selects_start_index():
    config = make_config()
    client = FakeClient(difficulty="hard", scores=[9])
    result = await route_completion(dict(BODY), mode_for(config), config, client)

    assert client.generation_calls == [LADDER[2]]
    assert result.response["frugal"]["classifier_tier"] == 2


async def test_classifier_failure_defaults_to_medium():
    config = make_config()
    client = FakeClient(classifier_raises=True, scores=[9])
    result = await route_completion(dict(BODY), mode_for(config), config, client)

    assert client.generation_calls == [LADDER[1]]  # medium ⇒ index 1


async def test_judge_failure_fails_open():
    config = make_config()
    client = FakeClient(difficulty="easy", judge_raises=True)
    result = await route_completion(dict(BODY), mode_for(config), config, client)

    assert client.generation_calls == [LADDER[0]]
    assert result.final_model == LADDER[0]
    assert result.final_score is None


async def test_unparseable_judge_fails_open():
    config = make_config()
    client = FakeClient(difficulty="easy", judge_garbage=True)
    result = await route_completion(dict(BODY), mode_for(config), config, client)

    assert client.generation_calls == [LADDER[0]]
    assert result.final_model == LADDER[0]


async def test_failed_model_skips_to_next_in_ladder():
    config = make_config()
    client = FakeClient(difficulty="easy", scores=[8], failing_models={LADDER[0]})
    result = await route_completion(dict(BODY), mode_for(config), config, client)

    assert client.generation_calls == LADDER[:2]
    assert result.final_model == LADDER[1]
    errors = [a for a in result.response["frugal"]["attempts"] if a.get("error")]
    assert len(errors) == 1


async def test_total_cost_includes_classifier_and_judge():
    config = make_config()
    client = FakeClient(difficulty="easy", scores=[8])
    result = await route_completion(dict(BODY), mode_for(config), config, client)

    # 3 anrop (classifier + generering + judge) à 100*1e-6 + 50*2e-6 = 2e-4
    assert result.total_cost_usd == pytest.approx(6e-4)
    # baseline: samma tokens till flaggskeppspris = 100*1e-5 + 50*2e-5 = 2e-3
    assert result.baseline_cost_usd == pytest.approx(2e-3)
    assert result.response["frugal"]["saved_pct"] == pytest.approx(70.0)


def test_resolve_mode():
    config = make_config()
    assert resolve_mode("auto", config).threshold == 7
    assert resolve_mode(None, config).kind == "route"
    assert resolve_mode("gpt-4o-unknown", config).kind == "route"
    strict = resolve_mode("auto:strict", config)
    assert strict.threshold == 8
    cheap = resolve_mode("auto:cheap", config)
    assert (cheap.threshold, cheap.max_escalations) == (6, 1)
    passthrough = resolve_mode("anthropic/claude-sonnet-5", config)
    assert passthrough.kind == "passthrough"
    assert passthrough.passthrough_model == "anthropic/claude-sonnet-5"


async def test_tool_call_response_skips_judge():
    config = make_config()
    client = FakeClient(difficulty="easy", tool_call_models={LADDER[0]})
    result = await route_completion(dict(BODY), mode_for(config), config, client)

    assert client.generation_calls == [LADDER[0]]
    assert client.judge_calls == 0  # judge is never called for tool calls
    assert result.final_model == LADDER[0]
    message = result.response["choices"][0]["message"]
    assert message["tool_calls"][0]["id"] == "call_abc123"


def test_sse_stream_emits_tool_calls():
    from app.main import _sse_stream

    response = completion("", "cheap/model-a", tool_calls=[dict(TOOL_CALL)])
    events = list(_sse_stream(response))

    tool_chunks = [
        e
        for e in events
        if e.startswith("data: {")
        and json.loads(e.removeprefix("data: "))["choices"][0]["delta"].get("tool_calls")
    ]
    assert len(tool_chunks) == 1
    payload = json.loads(tool_chunks[0].removeprefix("data: "))
    call = payload["choices"][0]["delta"]["tool_calls"][0]
    assert call["index"] == 0
    assert call["id"] == "call_abc123"
    assert call["function"]["name"] == "read_file"
    # finish_reason from upstream should be preserved
    assert any('"finish_reason": "tool_calls"' in e for e in events)
    assert events[-1] == "data: [DONE]\n\n"


def test_extract_first_json_handles_garbage():
    assert extract_first_json('blah {"score": 7, "reason": "ok"} blah') == {
        "score": 7,
        "reason": "ok",
    }
    assert extract_first_json("no json here") is None
    assert extract_first_json('```json\n{"difficulty": "easy"}\n```') == {
        "difficulty": "easy"
    }
