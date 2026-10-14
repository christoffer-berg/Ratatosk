"""Tests for the local routing model and its runtime."""

import os
import random
import time

import pytest

from app.config import ClassifierConfig, Config, JudgeConfig, LadderStep
from app.features import FEATURE_VERSION
from app.predictor import (
    Head, PredictorRuntime, RouterModel, choose_rung, expected_costs, sigmoid,
)

LADDER = ["cheap/a", "mid/b", "big/c"]


def make_config(tmp_path, **overrides) -> Config:
    config = Config(
        ladder=[LadderStep(id=m) for m in LADDER],
        classifier=ClassifierConfig(model="cheap/classifier"),
        judge=JudgeConfig(model="cheap/judge"),
    )
    config.predictor.path = str(tmp_path / "model.json")
    config.predictor.buckets = 16
    for key, value in overrides.items():
        setattr(config.predictor, key, value)
    return config


def model_with(biases: list[float], **kwargs) -> RouterModel:
    """A model whose heads ignore the prompt and return fixed probabilities."""
    heads = []
    for b in biases:
        head = Head(32)
        head.weights[0] = b
        heads.append(head)
    defaults = dict(
        ladder=list(LADDER), thresholds=[7], buckets=16, feature_version=FEATURE_VERSION,
        heads={7: heads}, mean_cost={"cheap/a": 1e-4, "mid/b": 4e-4, "big/c": 2e-3},
        completion_tokens={"cheap/a": 200, "mid/b": 200, "big/c": 200},
        deployable=True, deploy_reason="test",
    )
    defaults.update(kwargs)
    return RouterModel(**defaults)


class Prices:
    def __init__(self, table=None):
        self.table = table if table is not None else {
            "cheap/a": (1e-6, 2e-6), "mid/b": (4e-6, 8e-6), "big/c": (2e-5, 4e-5),
            "cheap/judge": (1e-7, 2e-7),
        }

    async def price_for(self, model):
        return self.table.get(model, (0.0, 0.0))


BIAS = {0: 1.0}


def test_sigmoid_is_numerically_safe():
    assert sigmoid(-1000) == 0.0 and sigmoid(1000) == 1.0
    assert sigmoid(0) == 0.5


def test_a_head_learns_a_separable_signal():
    head, rng = Head(4), random.Random(0)
    for _ in range(400):
        signal = rng.random() < 0.5
        head.update({0: 1.0, 1: 1.0 if signal else 0.0}, 1.0 if signal else 0.0, 1.0, 0.5, 0.0)
    assert head.score({0: 1.0, 1: 1.0}) > 0.9
    assert head.score({0: 1.0}) < 0.1


def test_out_of_range_features_are_ignored():
    head = Head(4)
    assert head.score({999: 1.0}) == 0.5
    head.update({999: 1.0}, 1.0, 1.0, 0.5, 0.0)  # must not raise


def test_predictions_are_monotone_up_the_ladder():
    # A pricier rung is never predicted less likely to pass.
    model = model_with([2.0, -1.0, 0.5])
    p = model.predict(BIAS, 7)
    assert p[0] == p[1] >= sigmoid(-1.0)
    assert p == sorted(p)


def test_unknown_threshold_yields_no_prediction():
    assert model_with([0.0, 0.0, 0.0]).predict(BIAS, 9) is None


def test_save_and_load_round_trip(tmp_path):
    model = model_with([1.5, -0.5, 2.0], examples=42, metrics={"x": 1})
    path = str(tmp_path / "m.json")
    model.save(path)
    loaded = RouterModel.load(path)
    assert loaded.ladder == model.ladder and loaded.examples == 42
    assert loaded.metrics == {"x": 1} and loaded.deployable is True
    assert loaded.predict(BIAS, 7) == pytest.approx(model.predict(BIAS, 7), abs=1e-6)
    assert not os.path.exists(path + ".tmp")


def test_expected_cost_recursion():
    assert expected_costs([0.9, 0.95, 1.0], [1, 10, 100]) == pytest.approx([2.5, 15.0, 100])


def test_expected_cost_policy_starts_where_it_is_cheapest_in_expectation():
    assert choose_rung([0.9, 0.95, 1.0], [1, 10, 100], "expected_cost", 0.8) == 0
    # A cheap rung that nearly always fails is not worth the detour.
    assert choose_rung([0.01, 0.95, 1.0], [5, 6, 100], "expected_cost", 0.8) == 1


def test_pass_rate_policy_takes_the_first_likely_rung():
    assert choose_rung([0.5, 0.85, 1.0], [1, 10, 100], "pass_rate", 0.8) == 1
    assert choose_rung([0.1, 0.2, 0.3], [1, 10, 100], "pass_rate", 0.8) == 2


def runtime_with(tmp_path, model=None, **overrides):
    config = make_config(tmp_path, **overrides)
    if model is not None:
        model.save(config.predictor.path)
    runtime = PredictorRuntime(config)
    runtime.load()
    return runtime


def test_a_missing_model_is_inactive(tmp_path):
    runtime = runtime_with(tmp_path)
    assert not runtime.active
    assert "app.train" in runtime.status


def test_a_deployable_matching_model_is_active(tmp_path):
    runtime = runtime_with(tmp_path, model_with([0.0, 0.0, 0.0]))
    assert runtime.active and runtime.status == "active"


@pytest.mark.parametrize("change, reason", [
    ({"ladder": ["other/x", "mid/b", "big/c"]}, "different ladder"),
    ({"feature_version": FEATURE_VERSION - 1}, "older feature scheme"),
    ({"buckets": 99}, "bucket count"),
    ({"deployable": False, "deploy_reason": "too few"}, "not deployable"),
])
def test_an_unusable_model_never_routes(tmp_path, change, reason):
    runtime = runtime_with(tmp_path, model_with([0.0, 0.0, 0.0], **change))
    assert not runtime.active
    assert reason in runtime.status


def test_allow_undeployable_overrides_the_gate(tmp_path):
    model = model_with([0.0, 0.0, 0.0], deployable=False, deploy_reason="x")
    assert runtime_with(tmp_path, model, allow_undeployable=True).active


def test_disabled_predictor_is_inactive(tmp_path):
    assert not runtime_with(tmp_path, model_with([0.0, 0.0, 0.0]), enabled=False).active


def test_a_corrupt_file_is_reported_not_raised(tmp_path):
    config = make_config(tmp_path)
    with open(config.predictor.path, "w") as f:
        f.write("{not json")
    runtime = PredictorRuntime(config)
    assert runtime.load() is False and "failed to load" in runtime.status


def test_retrained_file_is_picked_up_without_restart(tmp_path):
    runtime = runtime_with(tmp_path, model_with([0.0, 0.0, 0.0], examples=1))
    model_with([0.0, 0.0, 0.0], examples=2).save(runtime._config.predictor.path)
    os.utime(runtime._config.predictor.path, (time.time() + 5, time.time() + 5))
    runtime._checked_at = 0.0  # skip the throttle
    runtime.maybe_reload()
    assert runtime.model.examples == 2


async def test_rung_costs_use_live_prices_and_include_judging(tmp_path):
    runtime = runtime_with(tmp_path, model_with([0.0, 0.0, 0.0]))
    costs = await runtime.rung_costs(400, Prices())
    assert costs == sorted(costs) and costs[0] > 0
    no_judge = await runtime.rung_costs(400, Prices({**Prices().table, "cheap/judge": (0, 0)}))
    assert all(c > n for c, n in zip(costs, no_judge))


async def test_rung_costs_fall_back_to_history_when_prices_are_missing(tmp_path):
    runtime = runtime_with(tmp_path, model_with([0.0, 0.0, 0.0]))
    costs = await runtime.rung_costs(400, Prices({}))
    assert costs == pytest.approx([1e-4, 4e-4, 2e-3])


async def test_decide_routes_cheap_when_the_cheap_rung_will_pass(tmp_path):
    runtime = runtime_with(tmp_path, model_with([4.0, 4.0, 4.0]))
    rung, probs = await runtime.decide(BIAS, 7, 400, Prices())
    assert rung == 0 and probs[0] > 0.95


async def test_decide_skips_a_cheap_rung_that_will_fail(tmp_path):
    runtime = runtime_with(tmp_path, model_with([-6.0, 4.0, 4.0]))
    rung, _ = await runtime.decide(BIAS, 7, 400, Prices())
    assert rung == 1


async def test_decide_defers_when_inactive_or_threshold_unknown(tmp_path):
    assert await runtime_with(tmp_path).decide(BIAS, 7, 400, Prices()) is None
    active = runtime_with(tmp_path, model_with([0.0, 0.0, 0.0]))
    assert await active.decide(BIAS, 9, 400, Prices()) is None


def test_unpriced_rungs_are_filled_from_the_rung_below():
    from app.predictor import fill_ladder_costs
    assert fill_ladder_costs([1.0, 4.0, 16.0, None]) == [1.0, 4.0, 16.0, 16.0]
    assert fill_ladder_costs([None, 4.0, 16.0]) == [4.0, 4.0, 16.0]
    assert fill_ladder_costs([1.0, 0.0, 16.0]) == [1.0, 1.0, 16.0]  # $0 means unknown
    assert fill_ladder_costs([None, 0.0]) is None


async def test_a_flagship_with_no_price_and_no_history_is_not_free(tmp_path):
    # Regression: an unpriced top rung used to cost $0, and the expected-cost
    # recursion then routed everything straight to the flagship.
    model = model_with([4.0, 4.0, 4.0], mean_cost={"cheap/a": 1e-4, "mid/b": 4e-4})
    runtime = runtime_with(tmp_path, model)
    prices = Prices({"cheap/a": (1e-6, 2e-6), "mid/b": (4e-6, 8e-6)})
    costs = await runtime.rung_costs(400, prices)
    assert costs[2] >= costs[1] > 0
    rung, _ = await runtime.decide(BIAS, 7, 400, prices)
    assert rung == 0
