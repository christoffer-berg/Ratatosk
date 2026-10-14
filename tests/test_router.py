"""Tests for routing loop with mocked OpenRouter client."""

import json

import pytest

from app.config import ClassifierConfig, Config, JudgeConfig, LadderStep
from app.embeddings import Neighbor, Neighborhood
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
    config = Config(
        ladder=[LadderStep(id=model_id) for model_id in LADDER],
        classifier=ClassifierConfig(model=CLASSIFIER_MODEL),
        judge=JudgeConfig(model=JUDGE_MODEL, **judge_overrides),
    )
    # Exploration is stochastic — opt in per test rather than per run.
    config.exploration.enabled = False
    return config


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
        task_type: str = "factual_qa",
        answers: dict | None = None,
        scores: list[int] | None = None,
        failing_models: set[str] | None = None,
        judge_raises: bool = False,
        judge_garbage: bool = False,
        classifier_raises: bool = False,
        tool_call_models: set[str] | None = None,
    ):
        self.difficulty = difficulty
        self.task_type = task_type
        self.answers = answers or {}
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
            return completion(
                json.dumps({"difficulty": self.difficulty, "task_type": self.task_type}), model
            )
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
        return completion(self.answers.get(model, f"response from {model}"), model)

    async def price_for(self, model: str) -> tuple[float, float]:
        # Flagship is 10x more expensive so baseline > total in tests.
        if model == LADDER[-1]:
            return (1e-5, 2e-5)
        return (1e-6, 2e-6)


BODY = {"model": "auto", "messages": [{"role": "user", "content": "What is 2+2?"}]}


class FakeLearner:
    """Stands in for TierLearner: returns a fixed start rung."""

    def __init__(self, index: int, source: str = "learned"):
        self.index = index
        self.source = source
        self.refreshed = 0
        self.asked: tuple | None = None

    async def maybe_refresh(self) -> None:
        self.refreshed += 1

    def start_index(self, task_type: str, difficulty: str, threshold: int) -> tuple[int, str]:
        self.asked = (task_type, difficulty, threshold)
        return self.index, self.source


class FakeNeighbors:
    """Stands in for NeighborRouter: returns a fixed neighbourhood."""

    def __init__(self, neighborhood: Neighborhood | None = None):
        self.neighborhood = neighborhood or Neighborhood()
        self.looked_up: str | None = None

    async def lookup(self, text: str) -> Neighborhood:
        self.looked_up = text
        return self.neighborhood


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


async def test_frugal_reports_task_type_and_config_tier_source():
    config = make_config()
    client = FakeClient(difficulty="easy", task_type="code", scores=[8])
    result = await route_completion(dict(BODY), mode_for(config), config, client)

    frugal = result.response["frugal"]
    assert frugal["task_type"] == "code"
    assert frugal["tier_source"] == "config"
    assert result.task_type == "code"


async def test_classifier_without_task_type_defaults_to_other():
    config = make_config()
    client = FakeClient(difficulty="easy", task_type="not-a-real-type", scores=[8])
    result = await route_completion(dict(BODY), mode_for(config), config, client)

    assert result.task_type == "other"


async def test_learner_start_rung_overrides_config_tier():
    config = make_config()
    # config maps hard ⇒ index 2; the learner has evidence for index 0.
    client = FakeClient(difficulty="hard", task_type="code", scores=[8])
    learner = FakeLearner(0)
    result = await route_completion(dict(BODY), mode_for(config), config, client, learner)

    assert client.generation_calls == [LADDER[0]]
    assert learner.asked == ("code", "hard", 7)
    assert learner.refreshed == 1
    assert result.response["frugal"]["classifier_tier"] == 0
    assert result.response["frugal"]["tier_source"] == "learned"


async def test_learner_is_asked_with_the_mode_threshold():
    config = make_config()
    client = FakeClient(difficulty="easy", scores=[9])
    learner = FakeLearner(0)
    await route_completion(dict(BODY), mode_for(config, "auto:strict"), config, client, learner)

    assert learner.asked[2] == 8


async def test_attempt_rows_carry_ladder_index():
    config = make_config()
    client = FakeClient(difficulty="easy", scores=[4, 9])
    result = await route_completion(dict(BODY), mode_for(config), config, client)

    assert [row["ladder_index"] for row in result.attempt_rows()] == [0, 1]
    assert [row["model"] for row in result.attempt_rows()] == LADDER[:2]


async def test_prompt_text_is_only_logged_when_enabled():
    config = make_config()
    client = FakeClient(difficulty="easy", scores=[8])
    result = await route_completion(dict(BODY), mode_for(config), config, client)
    assert result.prompt_text is None
    assert result.prompt_chars == len("What is 2+2?")

    config.log_prompts = True
    client = FakeClient(difficulty="easy", scores=[8])
    result = await route_completion(dict(BODY), mode_for(config), config, client)
    assert "What is 2+2?" in result.prompt_text


BROKEN_PY = "Here:\n```python\ndef f(:\n```"
GOOD_PY = "Here:\n```python\ndef f():\n    return 1\n```"


async def test_verifier_forces_escalation_past_a_judge_that_was_fooled():
    config = make_config()
    # The judge loves the broken code; the parser knows better.
    client = FakeClient(
        difficulty="easy", task_type="code",
        answers={LADDER[0]: BROKEN_PY, LADDER[1]: GOOD_PY}, scores=[9, 9],
    )
    result = await route_completion(dict(BODY), mode_for(config), config, client)

    assert client.generation_calls == LADDER[:2]
    assert result.final_model == LADDER[1]
    attempts = result.response["frugal"]["attempts"]
    assert attempts[0]["verifier_passed"] is False
    assert attempts[1]["verifier_passed"] is True


async def test_verifier_verdict_is_recorded_without_enforcement():
    config = make_config()
    config.verifiers.enforce = False
    client = FakeClient(
        difficulty="easy", task_type="code", answers={LADDER[0]: BROKEN_PY}, scores=[9]
    )
    result = await route_completion(dict(BODY), mode_for(config), config, client)

    # Still logged as a failure — it just no longer overrides the judge.
    assert client.generation_calls == [LADDER[0]]
    assert result.attempt_rows()[0]["verifier_passed"] is False


async def test_a_blocked_answer_is_still_returned_when_the_ladder_runs_out():
    config = make_config()
    client = FakeClient(
        difficulty="expert", task_type="code",
        answers={LADDER[3]: BROKEN_PY}, scores=[9],
    )
    result = await route_completion(dict(BODY), mode_for(config), config, client)

    # Nothing better exists, so the client gets the best available answer.
    assert result.final_model == LADDER[3]


async def test_verifier_does_not_fire_when_the_judge_already_rejected():
    config = make_config()
    client = FakeClient(
        difficulty="easy", task_type="code",
        answers={LADDER[0]: GOOD_PY, LADDER[1]: GOOD_PY}, scores=[3, 9],
    )
    result = await route_completion(dict(BODY), mode_for(config), config, client)
    assert result.final_model == LADDER[1]


async def test_exploration_starts_one_rung_lower():
    config = make_config()
    config.exploration.enabled = True
    config.exploration.epsilon = 1.0  # always explore
    client = FakeClient(difficulty="hard", scores=[9])
    result = await route_completion(dict(BODY), mode_for(config), config, client)

    assert client.generation_calls == [LADDER[1]]  # hard ⇒ 2, explored down to 1
    assert result.explored is True
    assert result.response["frugal"]["explored"] is True


async def test_exploration_never_goes_below_the_cheapest_rung():
    config = make_config()
    config.exploration.enabled = True
    config.exploration.epsilon = 1.0
    client = FakeClient(difficulty="easy", scores=[9])
    result = await route_completion(dict(BODY), mode_for(config), config, client)

    assert client.generation_calls == [LADDER[0]]
    assert result.explored is False


async def test_neighbours_outrank_the_learned_tier():
    config = make_config()
    neighborhood = Neighborhood(
        embedding=[0.1, 0.2],
        neighbors=[Neighbor(f"r{i}", 0.9, 0, 9, LADDER[0], None) for i in range(3)],
    )
    client = FakeClient(difficulty="hard", scores=[9])
    learner = FakeLearner(2)
    result = await route_completion(
        dict(BODY), mode_for(config), config, client, learner, FakeNeighbors(neighborhood)
    )

    assert client.generation_calls == [LADDER[0]]
    assert result.tier_source == "neighbors"
    assert result.embedding == [0.1, 0.2]


async def test_sparse_neighbourhood_falls_back_to_the_learner():
    config = make_config()
    client = FakeClient(difficulty="hard", scores=[9])
    learner = FakeLearner(1)
    result = await route_completion(
        dict(BODY), mode_for(config), config, client, learner, FakeNeighbors()
    )

    assert client.generation_calls == [LADDER[1]]
    assert result.tier_source == "learned"


async def test_semantic_cache_returns_the_stored_answer_without_a_model_call():
    config = make_config()
    config.embeddings.cache_enabled = True
    neighborhood = Neighborhood(
        embedding=[0.5],
        neighbors=[Neighbor("r1", 0.99, 0, 9, LADDER[0], "the cached answer")],
    )
    client = FakeClient(difficulty="easy")
    result = await route_completion(
        dict(BODY), mode_for(config), config, client, None, FakeNeighbors(neighborhood)
    )

    assert client.generation_calls == []  # nothing was generated
    assert result.cache_hit is True
    assert result.response["choices"][0]["message"]["content"] == "the cached answer"
    assert result.response["frugal"]["saved_pct"] == 100.0
    assert result.response["frugal"]["cache_source_request"] == "r1"


async def test_cache_is_not_used_when_disabled_or_too_dissimilar():
    config = make_config()  # cache_enabled defaults to False
    neighborhood = Neighborhood(
        neighbors=[Neighbor("r1", 0.99, 0, 9, LADDER[0], "the cached answer")]
    )
    client = FakeClient(difficulty="easy", scores=[9])
    result = await route_completion(
        dict(BODY), mode_for(config), config, client, None, FakeNeighbors(neighborhood)
    )
    assert result.cache_hit is False
    assert client.generation_calls == [LADDER[0]]


async def test_answer_hash_matches_the_feedback_fingerprint():
    from app.feedback import answer_fingerprint

    config = make_config()
    client = FakeClient(difficulty="easy", scores=[9])
    result = await route_completion(dict(BODY), mode_for(config), config, client)

    # This is what lets a later "no, that's wrong" find the request that
    # produced the answer being complained about.
    echoed = {"role": "assistant", "content": f"response from {LADDER[0]}"}
    assert result.answer_hash == answer_fingerprint(echoed)


async def test_accepted_attempt_is_flagged_in_the_log():
    config = make_config()
    client = FakeClient(difficulty="easy", scores=[4, 9])
    result = await route_completion(dict(BODY), mode_for(config), config, client)

    assert [row["accepted"] for row in result.attempt_rows()] == [False, True]


class FakePredictor:
    """Stands in for PredictorRuntime."""

    def __init__(self, decision=None):
        self.decision = decision
        self.asked = None

    async def decide(self, features, threshold, chars, client):
        self.asked = (features, threshold, chars)
        return self.decision


async def test_trained_model_outranks_every_other_source():
    config = make_config()
    client = FakeClient(difficulty="hard", scores=[9])
    neighborhood = Neighborhood(
        neighbors=[Neighbor(f"r{i}", 0.9, 0, 9, LADDER[0], None) for i in range(3)]
    )
    predictor = FakePredictor((3, [0.1, 0.2, 0.5, 1.0]))
    result = await route_completion(
        dict(BODY), mode_for(config), config, client, FakeLearner(1),
        FakeNeighbors(neighborhood), predictor,
    )

    # The neighbours said rung 0 and the matrix rung 1; the model's rung 3 wins.
    assert client.generation_calls == [LADDER[3]]
    assert result.tier_source == "predictor"
    assert result.response["frugal"]["predicted_pass"] == [0.1, 0.2, 0.5, 1.0]
    features, threshold, _ = predictor.asked
    assert threshold == 7 and features  # asked with this request's features


async def test_the_model_may_start_at_most_one_rung_below_the_matrix():
    # Trust region: one rung below is as far as exploration reaches, so it is
    # as far as the model's evidence goes.
    config = make_config()
    client = FakeClient(difficulty="hard", scores=[9])
    result = await route_completion(
        dict(BODY), mode_for(config), config, client, FakeLearner(3), None,
        FakePredictor((0, [0.99, 0.99, 0.99, 1.0])),
    )
    assert client.generation_calls == [LADDER[2]]
    assert result.tier_source == "predictor"


async def test_exploration_probability_is_recorded_with_the_request():
    config = make_config()
    config.exploration.enabled = True
    config.exploration.epsilon = 0.0  # never actually explore, but it's on
    client = FakeClient(difficulty="hard", scores=[9])
    result = await route_completion(dict(BODY), mode_for(config), config, client)
    assert result.explore_prob == 0.0

    config.exploration.epsilon = 0.2
    client = FakeClient(difficulty="hard", scores=[9, 9])
    result = await route_completion(dict(BODY), mode_for(config), config, client)
    assert result.explore_prob == 0.2

    # On the cheapest rung there is nowhere lower to explore.
    client = FakeClient(difficulty="easy", scores=[9])
    result = await route_completion(dict(BODY), mode_for(config), config, client)
    assert result.explore_prob == 0.0


async def test_an_inactive_model_defers_to_the_next_source():
    config = make_config()
    client = FakeClient(difficulty="hard", scores=[9])
    result = await route_completion(
        dict(BODY), mode_for(config), config, client, FakeLearner(1), None, FakePredictor(None),
    )
    assert result.tier_source == "learned"
    assert "predicted_pass" not in result.response["frugal"]


async def test_features_are_captured_for_training_without_the_prompt():
    config = make_config()
    client = FakeClient(difficulty="easy", scores=[9])
    result = await route_completion(dict(BODY), mode_for(config), config, client)

    assert result.features is not None
    assert "2+2" not in result.features


async def test_feature_logging_can_be_switched_off():
    config = make_config()
    config.predictor.log_features = False
    client = FakeClient(difficulty="easy", scores=[9])
    result = await route_completion(dict(BODY), mode_for(config), config, client)
    assert result.features is None


async def test_attempts_record_token_counts():
    config = make_config()
    client = FakeClient(difficulty="easy", scores=[4, 9])
    result = await route_completion(dict(BODY), mode_for(config), config, client)
    rows = result.attempt_rows()
    assert [(r["prompt_tokens"], r["completion_tokens"]) for r in rows] == [(100, 50), (100, 50)]


def test_extract_first_json_handles_garbage():
    assert extract_first_json('blah {"score": 7, "reason": "ok"} blah') == {
        "score": 7,
        "reason": "ok",
    }
    assert extract_first_json("no json here") is None
    assert extract_first_json('```json\n{"difficulty": "easy"}\n```') == {
        "difficulty": "easy"
    }
