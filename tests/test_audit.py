"""Tests for shadow audits — measuring the cheap judge's false-accept rate."""

import json
import random

from app.audit import audit_model, run_audit, should_audit
from app.config import ClassifierConfig, Config, JudgeConfig, LadderStep

LADDER = ["cheap/a", "mid/b", "flagship/c"]


def make_config(**overrides) -> Config:
    config = Config(
        ladder=[LadderStep(id=m) for m in LADDER],
        classifier=ClassifierConfig(model="cheap/classifier"),
        judge=JudgeConfig(model="cheap/judge"),
    )
    for key, value in overrides.items():
        setattr(config.audit, key, value)
    return config


class FakeClient:
    def __init__(self, score=3, raises=False):
        self.score = score
        self.raises = raises
        self.judged_with: list[str] = []

    async def chat_completion(self, payload):
        self.judged_with.append(payload["model"])
        if self.raises:
            raise RuntimeError("judge down")
        content = json.dumps({"score": self.score, "reason": "actually wrong"})
        return {
            "choices": [{"message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }

    async def price_for(self, model):
        return (1e-5, 2e-5)


class FakeStore:
    def __init__(self):
        self.audits: list[dict] = []

    async def log_audit(self, **kwargs):
        self.audits.append(kwargs)


MESSAGES = [{"role": "user", "content": "What is 2+2?"}]


def test_audit_model_defaults_to_the_top_rung():
    assert audit_model(make_config()) == "flagship/c"
    assert audit_model(make_config(model="strong/judge")) == "strong/judge"


def test_sampling_respects_the_configured_rate():
    config = make_config(enabled=True, sample_pct=0.02)
    rng = random.Random(0)
    sampled = sum(should_audit(config, 8, rng) for _ in range(10000))
    assert 100 < sampled < 350  # ~2%


def test_unscored_and_disabled_are_never_audited():
    assert should_audit(make_config(enabled=True, sample_pct=1.0), None) is False
    assert should_audit(make_config(enabled=False, sample_pct=1.0), 8) is False


async def test_audit_logs_a_disagreement():
    client, store = FakeClient(score=3), FakeStore()
    await run_audit(
        client, make_config(enabled=True), store, "req1", MESSAGES, "an answer",
        model="cheap/a", task_type="math", difficulty="easy", cheap_score=9, threshold=7,
    )

    assert client.judged_with == ["flagship/c"]  # re-judged by the strong model
    (logged,) = store.audits
    assert (logged["cheap_score"], logged["strong_score"]) == (9, 3)
    assert logged["model"] == "cheap/a"
    assert logged["task_type"] == "math"
    assert logged["cost_usd"] > 0


async def test_audit_logs_agreement_too():
    client, store = FakeClient(score=9), FakeStore()
    await run_audit(
        client, make_config(enabled=True), store, "req1", MESSAGES, "an answer",
        model="cheap/a", task_type="math", difficulty="easy", cheap_score=9, threshold=7,
    )
    assert store.audits[0]["strong_score"] == 9


async def test_a_failing_audit_never_raises():
    # It runs after the client already has its response; it must not blow up.
    client, store = FakeClient(raises=True), FakeStore()
    await run_audit(
        client, make_config(enabled=True), store, "req1", MESSAGES, "an answer",
        model="cheap/a", task_type="math", difficulty="easy", cheap_score=9, threshold=7,
    )
    assert store.audits == []
