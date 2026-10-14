"""Tests for the learned start-rung selection."""

import pytest

from app.config import ClassifierConfig, Config, JudgeConfig, LadderStep, LearningConfig
from app.learning import CONFIG_SOURCE, LEARNED_SOURCE, TierLearner

LADDER = ["cheap/a", "mid/b", "big/c", "flagship/d"]

# Cell costs used throughout: adjacent rungs are close enough that pass rates,
# not raw price, decide the winner.
COST = {"cheap/a": 1e-4, "mid/b": 1.2e-4, "big/c": 2e-4, "flagship/d": 5e-4}


def make_config(**learning_overrides) -> Config:
    return Config(
        ladder=[LadderStep(id=model_id) for model_id in LADDER],
        classifier=ClassifierConfig(model="cheap/classifier"),
        judge=JudgeConfig(model="cheap/judge"),
        learning=LearningConfig(**learning_overrides),
    )


def rows(cell: tuple[str, str], scores_by_model: dict[str, dict[int, int]], costs=None):
    """Flatten {model: {score: count}} into the shape store.attempt_scores returns."""
    costs = costs or COST
    task_type, difficulty = cell
    return [
        {
            "task_type": task_type,
            "difficulty": difficulty,
            "model": model,
            "score": score,
            "samples": count,
            "cost_sum": costs[model] * count,
        }
        for model, scores in scores_by_model.items()
        for score, count in scores.items()
    ]


class FakeStore:
    def __init__(self, rows=None, raises=False):
        self.rows = rows or []
        self.raises = raises
        self.calls = 0

    async def attempt_scores(self, since: str) -> list[dict]:
        self.calls += 1
        if self.raises:
            raise RuntimeError("db down")
        return list(self.rows)


async def learner_for(rows_, **learning_overrides) -> TierLearner:
    config = make_config(**learning_overrides)
    learner = TierLearner(config, FakeStore(rows_))
    await learner.refresh()
    return learner


CODE_HARD = ("code", "hard")  # config tier map says index 2


async def test_sparse_cell_falls_back_to_config():
    learner = await learner_for(rows(CODE_HARD, {"cheap/a": {9: 5}}))
    assert learner.start_index("code", "hard", 7) == (2, CONFIG_SOURCE)


async def test_unseen_cell_falls_back_to_config():
    learner = await learner_for(rows(CODE_HARD, {"cheap/a": {9: 40}}))
    assert learner.start_index("math", "easy", 7) == (0, CONFIG_SOURCE)


async def test_learns_to_start_below_the_config_tier():
    # The cheap model actually passes hard code tasks 95% of the time, so the
    # config's "hard ⇒ start at index 2" is needlessly expensive.
    learner = await learner_for(
        rows(CODE_HARD, {"cheap/a": {9: 38, 2: 2}, "mid/b": {9: 36, 2: 4}})
    )
    assert learner.start_index("code", "hard", 7) == (0, LEARNED_SOURCE)


async def test_expected_cost_skips_a_rung_that_usually_fails():
    # cheap/a passes only 10% ⇒ starting there almost always costs an extra hop.
    learner = await learner_for(
        rows(CODE_HARD, {"cheap/a": {9: 4, 2: 36}, "mid/b": {9: 38, 2: 2}})
    )
    assert learner.start_index("code", "hard", 7) == (1, LEARNED_SOURCE)


async def test_threshold_changes_the_learned_rung():
    # cheap/a scores exactly 7 — good enough for "auto", not for "auto:strict".
    learner = await learner_for(
        rows(CODE_HARD, {"cheap/a": {7: 38, 2: 2}, "mid/b": {9: 38, 2: 2}})
    )
    assert learner.start_index("code", "hard", 7)[0] == 0
    assert learner.start_index("code", "hard", 8)[0] == 1


async def test_pass_rate_policy_picks_cheapest_clearing_the_bar():
    learner = await learner_for(
        rows(CODE_HARD, {"cheap/a": {9: 19, 2: 21}, "mid/b": {9: 38, 2: 2}}),
        policy="pass_rate",
    )
    assert learner.start_index("code", "hard", 7) == (1, LEARNED_SOURCE)


async def test_pass_rate_policy_falls_back_when_nothing_clears_the_bar():
    learner = await learner_for(
        rows(CODE_HARD, {"cheap/a": {9: 19, 2: 21}, "mid/b": {9: 20, 2: 20}}),
        policy="pass_rate",
    )
    assert learner.start_index("code", "hard", 7) == (2, CONFIG_SOURCE)


async def test_missing_prices_fall_back_to_config():
    # Unknown models price at $0, which would make every rung look equally cheap.
    zero = dict.fromkeys(LADDER, 0.0)
    learner = await learner_for(
        rows(CODE_HARD, {"cheap/a": {9: 38, 2: 2}, "mid/b": {9: 38, 2: 2}}, costs=zero)
    )
    assert learner.start_index("code", "hard", 7) == (2, CONFIG_SOURCE)


async def test_tail_is_priced_from_other_cells():
    # This cell has only ever used cheap/a; mid/b's cost comes from another cell.
    data = rows(CODE_HARD, {"cheap/a": {9: 4, 2: 36}}) + rows(
        ("math", "easy"), {"mid/b": {9: 30}}
    )
    learner = await learner_for(data)
    # cheap/a is still the only candidate with enough samples, but it is now
    # priced against a real escalation instead of being unevaluable.
    assert learner.start_index("code", "hard", 7) == (0, LEARNED_SOURCE)


async def test_disabled_learning_always_uses_config():
    learner = await learner_for(
        rows(CODE_HARD, {"cheap/a": {9: 40}}), enabled=False
    )
    assert learner.start_index("code", "hard", 7) == (2, CONFIG_SOURCE)


async def test_refresh_failure_leaves_config_tiers_in_place():
    config = make_config()
    learner = TierLearner(config, FakeStore(raises=True))
    await learner.maybe_refresh()  # must not raise
    assert learner.start_index("code", "hard", 7) == (2, CONFIG_SOURCE)


async def test_maybe_refresh_respects_ttl():
    config = make_config(refresh_minutes=60)
    store = FakeStore(rows(CODE_HARD, {"cheap/a": {9: 40}}))
    learner = TierLearner(config, store)
    await learner.maybe_refresh()
    await learner.maybe_refresh()
    assert store.calls == 1


async def test_skill_matrix_reports_pass_rates():
    learner = await learner_for(rows(CODE_HARD, {"cheap/a": {9: 30, 2: 10}}))
    matrix = learner.skill_matrix(threshold=7)
    assert matrix == [
        {
            "task_type": "code",
            "difficulty": "hard",
            "model": "cheap/a",
            "samples": 40,
            "pass_rate": 0.75,
            "avg_cost_usd": pytest.approx(1e-4),
        }
    ]


async def test_learned_tiers_reports_source_per_cell():
    data = rows(CODE_HARD, {"cheap/a": {9: 38, 2: 2}}) + rows(
        ("math", "easy"), {"cheap/a": {9: 3}}
    )
    learner = await learner_for(data)
    tiers = {(t["task_type"], t["difficulty"]): t for t in learner.learned_tiers(threshold=7)}
    assert tiers[("code", "hard")]["source"] == LEARNED_SOURCE
    assert tiers[("code", "hard")]["config_start_index"] == 2
    assert tiers[("math", "easy")]["source"] == CONFIG_SOURCE  # only 3 samples
