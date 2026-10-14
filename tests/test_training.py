"""Tests for turning the log into a routing model, and judging it."""

import random

import pytest

from app.config import ClassifierConfig, Config, JudgeConfig, LadderStep
from app.features import FEATURE_VERSION, dumps, extract
from app.store import Store
from app.predictor import Head, RouterModel
from app.training import (
    Example, NotEnoughData, _deploy_decision, _t95, auc, build_examples, replay, simulate,
    train,
)

LADDER = ["cheap/a", "mid/b", "big/c", "flagship/d"]


def example(scores: dict[int, int]) -> Example:
    return Example("r", "2026-01-01", "easy", min(scores), {0: 1.0}, 10, dict(scores),
                   {r: 1e-4 * (r + 1) for r in scores})


# --- partial labels -----------------------------------------------------------

def test_observed_labels_are_kept_as_is():
    assert example({1: 3, 2: 9}).labels(7, 4, impute=False) == {
        1: (False, False), 2: (True, False),
    }


def test_a_pass_implies_passes_above_it():
    labels = example({1: 9}).labels(7, 4, impute=True)
    assert labels[2] == (True, True) and labels[3] == (True, True)


def test_a_failure_implies_failures_below_it():
    labels = example({2: 3, 3: 9}).labels(7, 4, impute=True)
    assert labels[0] == (False, True) and labels[1] == (False, True)


def test_nothing_is_inferred_below_a_pass():
    # Whether a cheaper model would have passed is exactly what we don't know.
    labels = example({2: 9}).labels(7, 4, impute=True)
    assert 0 not in labels and 1 not in labels


def test_contradictory_evidence_is_not_imputed():
    # A pass below and a failure above (judge noise): rung 2 stays unknown.
    labels = example({1: 9, 3: 3}).labels(7, 4, impute=True)
    assert 2 not in labels


def test_labels_depend_on_the_threshold():
    assert example({0: 7}).labels(7, 4, False)[0] == (True, False)
    assert example({0: 7}).labels(8, 4, False)[0] == (False, False)


def test_build_examples_groups_attempts_and_drops_out_of_range_rungs():
    feats = dumps({0: 1.0})
    rows = [
        {"request_id": "b", "created_at": "2", "difficulty": "easy", "start_index": 0,
         "features": feats, "prompt_chars": 5, "ladder_index": 0, "score": 3, "cost_usd": 1.0},
        {"request_id": "b", "created_at": "2", "difficulty": "easy", "start_index": 0,
         "features": feats, "prompt_chars": 5, "ladder_index": 9, "score": 9, "cost_usd": 1.0},
        {"request_id": "a", "created_at": "1", "difficulty": "hard", "start_index": 2,
         "features": feats, "prompt_chars": 5, "ladder_index": 2, "score": 9, "cost_usd": 2.0},
    ]
    examples = build_examples(rows, 4)
    assert [e.request_id for e in examples] == ["a", "b"]  # oldest first
    assert examples[1].scores == {0: 3}


# --- metrics ------------------------------------------------------------------

def test_auc_on_known_cases():
    assert auc([0.1, 0.4, 0.35, 0.8], [False, False, True, True]) == pytest.approx(0.75)
    assert auc([0.1, 0.9], [False, True]) == 1.0
    assert auc([0.5, 0.5], [False, True]) == 0.5  # ties count half
    assert auc([0.1, 0.2], [True, True]) is None  # undefined with one class


# --- the replay walk ----------------------------------------------------------

COSTS = [1.0, 4.0, 16.0, 64.0]


def test_walk_charges_each_attempt_until_a_pass():
    cost, attempts, success = simulate(example({0: 3, 1: 9}), 0, 7, COSTS, 4, 3, True)
    assert (cost, attempts, success) == (5.0, 2.0, 1.0)


def test_walk_gives_up_on_unknown_outcomes_without_a_fill():
    assert simulate(example({2: 9}), 0, 7, COSTS, 4, 3, True) is None


def test_walk_fills_unknown_outcomes_in_expectation():
    # Rungs 0 and 1 unknown at p=0.5 each; rung 2 known pass.
    cost, attempts, success = simulate(example({2: 9}), 0, 7, COSTS, 4, 3, True,
                                       fill=lambda r: 0.5)
    assert cost == pytest.approx(1 + 0.5 * 4 + 0.25 * 16)
    assert attempts == pytest.approx(1.75)
    assert success == pytest.approx(1.0)


def test_walk_respects_max_escalations():
    cost, attempts, success = simulate(example({0: 3, 1: 3, 2: 3, 3: 9}), 0, 7, COSTS, 4, 1, True)
    assert (attempts, success) == (2.0, 0.0)


# --- the doubly robust replay ---------------------------------------------------

def fixed_model(p: float) -> RouterModel:
    """A model that believes P(pass) = p at every rung, whatever the prompt."""
    import math
    heads = []
    for _ in LADDER:
        head = Head(32)
        head.weights[0] = math.log(p / (1 - p))
        heads.append(head)
    return RouterModel(
        ladder=list(LADDER), thresholds=[7], buckets=16, feature_version=FEATURE_VERSION,
        heads={7: heads}, mean_cost={"cheap/a": 0.25, "mid/b": 1.0, "big/c": 4.0, "flagship/d": 16.0},
    )


def logged_world(n: int, epsilon: float, true_pass: float, seed: int = 0) -> list[Example]:
    """The router starts every request on big/c (rung 2); a coin flip with
    probability `epsilon` starts it one rung lower, on mid/b, which truly passes
    with probability `true_pass`. big/c always passes."""
    rng = random.Random(seed)
    examples = []
    for i in range(n):
        explored = rng.random() < epsilon
        if explored:
            if rng.random() < true_pass:
                scores, costs = {1: 9}, {1: 1.0}
            else:
                scores, costs = {1: 3, 2: 9}, {1: 1.0, 2: 4.0}
        else:
            scores, costs = {2: 9}, {2: 4.0}
        examples.append(Example(
            f"r{i}", f"2026-01-{i:06d}", "hard", 1 if explored else 2, {0: 1.0}, 10,
            scores, costs, task_type="other", explored=explored, explore_prob=epsilon,
            threshold=7,
        ))
    return examples


async def test_replay_recovers_the_truth_from_a_confidently_wrong_model(tmp_path):
    # The model believes mid/b passes 90% of the time; it really passes 20%.
    # Starting there truly costs 1 + 0.8 * 4 = 4.2, against 4.0 for the router.
    # Trusting the model's beliefs would claim a 65% saving — the doubly robust
    # correction from explored requests must bring it back to about -5%.
    config = make_config(tmp_path)
    config.exploration.epsilon = 0.5
    summary = await replay(fixed_model(0.9), logged_world(8000, 0.5, 0.2), config, 7)

    model = summary["policies"]["model"]
    assert model["mean_cost_usd"] == pytest.approx(4.2, abs=0.25)
    assert summary["policies"]["baseline"]["mean_cost_usd"] == pytest.approx(4.0)
    assert summary["saving_vs_baseline"]["estimate"] == pytest.approx(-0.05, abs=0.07)
    assert summary["saving_vs_baseline"]["lower_bound_95"] < 0


async def test_replay_credits_a_model_that_really_is_cheaper(tmp_path):
    # Now mid/b truly passes 90%: starting there costs 1 + 0.1 * 4 = 1.4.
    config = make_config(tmp_path)
    config.exploration.epsilon = 0.5
    summary = await replay(fixed_model(0.9), logged_world(8000, 0.5, 0.9), config, 7)
    assert summary["policies"]["model"]["mean_cost_usd"] == pytest.approx(1.4, abs=0.15)
    assert summary["saving_vs_baseline"]["lower_bound_95"] > 0.5


async def test_replay_weighs_by_the_probability_recorded_at_routing_time(tmp_path):
    # The config says 5% today, but these requests were explored at 50%. Using
    # today's value would multiply every correction tenfold.
    config = make_config(tmp_path)
    config.exploration.epsilon = 0.05
    summary = await replay(fixed_model(0.9), logged_world(8000, 0.5, 0.2), config, 7)
    assert summary["policies"]["model"]["mean_cost_usd"] == pytest.approx(4.2, abs=0.25)


async def test_replay_counts_the_evidence_behind_downward_moves(tmp_path):
    config = make_config(tmp_path)
    examples = logged_world(1000, 0.1, 0.5, seed=4)
    summary = await replay(fixed_model(0.9), examples, config, 7)
    assert summary["saving_vs_baseline"]["evidence"] == sum(e.explored for e in examples)


async def test_replay_skips_requests_routed_under_another_threshold(tmp_path):
    config = make_config(tmp_path)
    examples = logged_world(100, 0.5, 0.5)
    for e in examples[:40]:
        e.threshold = 8  # auto:strict traffic stopped at a different bar
    summary = await replay(fixed_model(0.9), examples, config, 7)
    assert summary["covered"] == 60


def prompt_aware_model() -> RouterModel:
    """Feature 1 marks a hard prompt: it fails on rungs 0 and 1, passes on 2."""
    heads = []
    for rung in range(len(LADDER)):
        head = Head(32)
        head.weights[0] = 4.0  # easy prompts pass everywhere
        head.weights[1] = -8.0 if rung < 2 else 0.0
        heads.append(head)
    return RouterModel(
        ladder=list(LADDER), thresholds=[7], buckets=16, feature_version=FEATURE_VERSION,
        heads={7: heads}, mean_cost={"cheap/a": 0.25, "mid/b": 1.0, "big/c": 4.0, "flagship/d": 16.0},
    )


def bottom_rung_world(n: int, seed: int = 0) -> list[Example]:
    """The router starts everything on the cheapest rung, so nothing can be
    explored below it. Hard prompts climb 0 -> 1 -> 2, wasting two attempts."""
    rng = random.Random(seed)
    examples = []
    for i in range(n):
        hard = rng.random() < 0.5
        if hard:
            scores, costs, feats = {0: 3, 1: 3, 2: 9}, {0: 0.25, 1: 1.0, 2: 4.0}, {0: 1.0, 1: 1.0}
        else:
            scores, costs, feats = {0: 9}, {0: 0.25}, {0: 1.0}
        examples.append(Example(
            f"r{i}", f"2026-01-{i:06d}", "easy", 0, feats, 10, scores, costs,
            task_type="other", explored=False, explore_prob=0.0, threshold=7,
        ))
    return examples


async def test_a_model_that_only_moves_up_needs_no_exploration_evidence(tmp_path):
    # Sending hard prompts straight to rung 2 skips two wasted attempts. Those
    # outcomes are all in the log, so the saving is exact — no coin flips needed.
    config = make_config(tmp_path)
    config.learning.enabled = False
    summary_ = await replay(prompt_aware_model(), bottom_rung_world(400), config, 7)

    assert summary_["saving_vs_baseline"]["evidence"] == 0
    worst = summary_["worst_case_saving"]
    assert worst["lower_bound_95"] > 0
    ok, reason = _deploy_decision(summary_, [])
    assert ok and "even if every downward move failed" in reason


def test_t_quantiles_widen_with_thin_evidence():
    assert _t95(1) > _t95(5) > _t95(20) > _t95(1000) == pytest.approx(1.645)


# --- the deployment gate ------------------------------------------------------

def summary(estimate=0.2, lower=0.1, evidence=50, compared=100, covered=100,
            model_success=1.0, base_success=1.0, unsupported=0.0):
    return {
        "covered": covered,
        "policies": {
            "model": {"success_rate": model_success, "unsupported": unsupported},
            "baseline": {"success_rate": base_success, "unsupported": 0.0},
        },
        "saving_vs_baseline": {"compared": compared, "evidence": evidence,
                               "estimate": estimate, "lower_bound_95": lower},
    }


def test_gate_deploys_a_saving_proven_at_95_percent():
    ok, reason = _deploy_decision(summary(), [])
    assert ok and "20.0% cheaper" in reason and "at least 10.0%" in reason


def test_gate_blocks_a_saving_indistinguishable_from_zero():
    ok, reason = _deploy_decision(summary(estimate=0.2, lower=-0.01), [])
    assert not ok and "not distinguishable from zero" in reason


def test_gate_blocks_a_model_that_buys_savings_with_failures():
    ok, reason = _deploy_decision(summary(model_success=0.90, base_success=0.99), [])
    assert not ok and "success rate" in reason


def test_gate_blocks_thin_evidence_however_good_it_looks():
    ok, reason = _deploy_decision(summary(estimate=0.5, lower=0.4, evidence=6), [])
    assert not ok and "6 explored requests" in reason


def test_gate_blocks_choices_exploration_never_covered():
    ok, reason = _deploy_decision(summary(unsupported=0.2), [])
    assert not ok and "below anything exploration tried" in reason


def test_gate_deploys_on_the_worst_case_bound_alone():
    s = summary(estimate=0.3, lower=-0.5, evidence=0)
    s["worst_case_saving"] = {"estimate": 0.2, "lower_bound_95": 0.1, "success_rate": 1.0}
    ok, reason = _deploy_decision(s, [])
    assert ok and "at least 10.0%" in reason and "even if every downward move failed" in reason


def test_the_worst_case_route_still_requires_quality():
    s = summary(estimate=0.3, lower=-0.5, evidence=0, base_success=0.99)
    s["worst_case_saving"] = {"estimate": 0.2, "lower_bound_95": 0.1, "success_rate": 0.90}
    assert not _deploy_decision(s, [])[0]


def test_gate_blocks_when_too_little_could_be_replayed():
    assert not _deploy_decision(summary(covered=5), [])[0]
    assert not _deploy_decision(summary(compared=5), [])[0]


# --- end to end ---------------------------------------------------------------

def make_config(tmp_path, **predictor) -> Config:
    config = Config(
        ladder=[LadderStep(id=m) for m in LADDER],
        classifier=ClassifierConfig(model="c", tiers={"easy": 0, "hard": 2}),
        judge=JudgeConfig(model="j", threshold=7, strict_threshold=8, cheap_threshold=6),
    )
    config.db_path = str(tmp_path / "t.sqlite3")
    config.predictor.min_examples = 50
    config.predictor.epochs = 4
    for k, v in predictor.items():
        setattr(config.predictor, k, v)
    return config


async def populate(store, config, n, seed=3, epsilon=0.3):
    """Prompts mentioning 'translate' pass on any rung; 'prove' needs rung 2.

    Most traffic is classified 'hard' (static tier 2) and explored one rung
    down with probability `epsilon`, recorded the way the router records it.
    A slice classified 'easy' starts on the cheapest rung. Only the words tell
    the two kinds of prompt apart.
    """
    rng = random.Random(seed)
    for k in range(n):
        easy_prompt = rng.random() < 0.5
        text = ("translate this to french " if easy_prompt else "prove this theorem ") + str(k % 7)
        true_rung = 0 if easy_prompt else 2
        difficulty = "easy" if rng.random() < 0.15 else "hard"
        router = 0 if difficulty == "easy" else 2
        explored = router > 0 and rng.random() < epsilon
        start = router - explored
        attempts = []
        for r in range(start, 4):
            passed = r >= true_rung
            attempts.append({"model": LADDER[r], "ladder_index": r, "score": 9 if passed else 3,
                             "cost_usd": 1e-4 * 4 ** r, "prompt_tokens": 20,
                             "completion_tokens": 100, "accepted": passed})
            if passed:
                break
        rid = f"r{k}"
        await store.log_request(
            request_id=rid, mode="auto", model_requested="auto", classifier_tier=start,
            difficulty=difficulty, task_type="other", tier_source="config",
            final_model=attempts[-1]["model"], final_score=attempts[-1]["score"],
            attempts=attempts, threshold=7, prompt_chars=len(text), explored=explored,
            explore_prob=epsilon if router > 0 else 0.0,
        )
        messages = [{"role": "user", "content": text}]
        await store.log_prompt_features(
            rid, FEATURE_VERSION, config.predictor.buckets, len(text),
            dumps(extract(messages, "other", difficulty, buckets=config.predictor.buckets)),
        )


@pytest.fixture
async def store(tmp_path):
    s = Store(str(tmp_path / "t.sqlite3"))
    await s.init()
    yield s
    await s.close()


async def test_refuses_to_train_on_too_little_data(tmp_path, store):
    config = make_config(tmp_path)
    await populate(store, config, 10)
    with pytest.raises(NotEnoughData):
        await train(config, store)


async def test_force_trains_anyway(tmp_path, store):
    config = make_config(tmp_path)
    await populate(store, config, 10)
    report = await train(config, store, force=True)
    assert report.examples == 10 and not report.deployable


async def test_learns_which_prompts_the_cheap_model_can_handle(tmp_path, store):
    config = make_config(tmp_path)
    # populate() logs with the static tiers, so make them the baseline router
    # too; see the next test for what happens when the two disagree.
    config.learning.enabled = False
    # Enough traffic that the held-out slice holds 20+ explored requests to
    # ground the saving — with fewer, the gate (rightly) waits.
    await populate(store, config, 2000)
    report = await train(config, store)

    model = report.model
    easy = extract([{"role": "user", "content": "translate this to french 1"}], "other", "hard",
                   buckets=config.predictor.buckets)
    hard = extract([{"role": "user", "content": "prove this theorem 1"}], "other", "hard",
                   buckets=config.predictor.buckets)
    # Same classifier labels; the words alone separate them.
    assert model.predict(easy, 7)[1] > 0.8
    assert model.predict(hard, 7)[1] < 0.2
    assert report.deployable, report.deploy_reason
    saving = report.replay["saving_vs_baseline"]
    assert saving["evidence"] >= 20 and saving["lower_bound_95"] > 0


async def test_a_baseline_that_moved_below_the_logged_traffic_is_caught(tmp_path, store):
    # The traffic was routed by the static tiers (rung 2), but the skill matrix
    # rebuilt from it now starts lower, so the model — one rung below *that* —
    # would start where no request ever went. There is no evidence about those
    # choices, and the gate must say so rather than guess.
    config = make_config(tmp_path)
    await populate(store, config, 1000)
    report = await train(config, store)
    assert report.replay["policies"]["model"]["unsupported"] > 0.05
    assert not report.deployable
    assert "below anything exploration tried" in report.deploy_reason


async def test_trains_a_head_set_per_mode_threshold(tmp_path, store):
    config = make_config(tmp_path)
    await populate(store, config, 100)
    report = await train(config, store)
    assert sorted(report.model.heads) == [6, 7, 8]
    assert all(len(h) == len(LADDER) for h in report.model.heads.values())


async def test_holdout_is_the_newest_traffic(tmp_path, store):
    config = make_config(tmp_path, holdout=0.25)
    await populate(store, config, 100)
    report = await train(config, store, refit=False)
    assert (report.train_examples, report.test_examples) == (75, 25)
    assert report.model.examples == 75


async def test_refit_trains_the_saved_model_on_everything(tmp_path, store):
    config = make_config(tmp_path, holdout=0.25)
    await populate(store, config, 100)
    report = await train(config, store, refit=True)
    assert report.model.examples == 100
    assert report.model.metrics["refit_on_all_data"] is True


async def test_features_from_another_scheme_are_ignored(tmp_path, store):
    config = make_config(tmp_path)
    await populate(store, config, 60)
    config.predictor.buckets = 4096  # stored rows were written with 8192
    with pytest.raises(NotEnoughData):
        await train(config, store)


async def test_token_statistics_are_carried_into_the_model(tmp_path, store):
    config = make_config(tmp_path)
    await populate(store, config, 100)
    model = (await train(config, store)).model
    assert model.completion_tokens["cheap/a"] == pytest.approx(100)
    assert model.mean_cost["mid/b"] == pytest.approx(4e-4)
    assert model.chars_per_token > 0


async def test_a_never_reached_top_rung_does_not_block_the_replay(tmp_path, store):
    # Regression: populate() never escalates to the flagship, so it has no cost
    # history. The replay used to give up on every request, and the model could
    # never deploy however good it was.
    config = make_config(tmp_path)
    await populate(store, config, 200)
    report = await train(config, store)
    assert report.replay["coverage"] == 1.0
    assert report.model.mean_cost["flagship/d"] == report.model.mean_cost["big/c"]
