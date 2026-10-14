"""Turn the request log into a routing model, and decide if it's good enough.

The hard part is not the regression, it's the labels. A request only tells us
about the rungs it actually tried: one that started at Sonnet and passed says
nothing about whether Haiku would have. Two consequences shape this module.

* Labels are *partial*. The ladder is declared cheapest-to-strongest, so a pass
  at rung i implies a pass above it and a failure at rung i implies a failure
  below it. With `impute` on, those inferred labels are added at reduced weight.
  What can never be inferred is a pass *below* the cheapest rung tried — that
  data only comes from exploration (`exploration.epsilon`).

* The model is never trusted far from evidence. It may start a request at most
  one rung below where the skill matrix (or config) would — exactly as far as
  exploration reaches — so every downward move it makes is one the log has
  outcomes for, and a wrong model costs at most one wasted cheap attempt.

* Accuracy is the wrong yardstick. A model that is 90% right but wrong on the
  expensive cases loses money. So the held-out set is *replayed*: every policy
  is walked up the ladder and charged what it would have cost.

The replay has the same partial-label problem, which is why the model is
held to one rung below the router: there, exploration's coin flips provide an
unbiased sample of real outcomes, and a doubly robust estimator uses them to
correct the model's own predictions (see `replay`). Filling unknowns with the
model's beliefs alone was tried first and, against ground truth, overstated a
stale model's saving by up to 16 points. The model deploys only if the 95%
lower confidence bound on its saving over the baseline router is positive.
"""

import logging
import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from .config import Config
from .features import DENSE_SIZE, FEATURE_VERSION, loads
from .learning import TierLearner, config_start_index
from .predictor import Head, RouterModel, choose_rung, fill_ladder_costs

logger = logging.getLogger(__name__)

SEED = 1729
MIN_TEST_REQUESTS = 30
SUCCESS_TOLERANCE = 0.01  # a cheaper policy may lose at most 1 point of success rate
MAX_UNSUPPORTED = 0.05  # share of choices allowed outside what exploration covered
MIN_EVIDENCE = 20  # explored requests that must ground the model's downward moves

# One-sided 95% Student-t quantiles by degrees of freedom (interpolated). The
# saving estimate's noise comes from the few explored requests, each weighted
# 1/epsilon, so its degrees of freedom are set by how many there are — with
# thin evidence the bound widens instead of pretending the mean is normal.
_T95 = [(1, 6.314), (2, 2.920), (3, 2.353), (4, 2.132), (5, 2.015), (7, 1.895),
        (10, 1.812), (15, 1.753), (20, 1.725), (30, 1.697), (60, 1.671), (120, 1.658)]


def _t95(df: int) -> float:
    if df <= 1:
        return _T95[0][1]
    for (d0, t0), (d1, t1) in zip(_T95, _T95[1:]):
        if df <= d1:
            return t0 + (t1 - t0) * (df - d0) / (d1 - d0)
    return 1.645


@dataclass
class Example:
    request_id: str
    created_at: str
    difficulty: str | None
    start_index: int | None
    features: dict[int, float]
    prompt_chars: int
    scores: dict[int, int] = field(default_factory=dict)  # rung -> effective score
    costs: dict[int, float] = field(default_factory=dict)  # rung -> observed cost
    task_type: str | None = None
    explored: bool = False  # started one rung below the router's choice by coin flip
    explore_prob: float | None = None  # the chance of that, when it was routed
    threshold: int | None = None  # the judge threshold this request was routed under

    @property
    def router_start(self) -> int | None:
        """Where the router of the day chose to start, before exploration moved it."""
        if self.start_index is None:
            return None
        return self.start_index + (1 if self.explored else 0)

    def labels(self, threshold: int, rungs: int, impute: bool) -> dict[int, tuple[bool, bool]]:
        """{rung: (passed, imputed)} — observed outcomes, plus monotone inferences."""
        observed = {i: s >= threshold for i, s in self.scores.items()}
        labels = {i: (passed, False) for i, passed in observed.items()}
        if not impute:
            return labels
        for rung in range(rungs):
            if rung in observed:
                continue
            passes_below = any(p for i, p in observed.items() if i < rung)
            fails_above = any(not p for i, p in observed.items() if i > rung)
            if passes_below and not fails_above:
                labels[rung] = (True, True)
            elif fails_above and not passes_below:
                labels[rung] = (False, True)
        return labels


@dataclass
class TrainingReport:
    examples: int
    train_examples: int
    test_examples: int
    thresholds: list[int]
    head_metrics: list[dict]
    calibration: list[dict]
    replay: dict
    deployable: bool
    deploy_reason: str
    model: RouterModel | None = None
    refit: bool = False


class NotEnoughData(Exception):
    pass


def build_examples(rows: list[dict], ladder_size: int) -> list[Example]:
    """Group attempt rows (oldest first) into one Example per request."""
    examples: dict[str, Example] = {}
    for row in rows:
        index = row["ladder_index"]
        if index is None or not 0 <= index < ladder_size:
            continue  # the ladder has changed since this was logged
        example = examples.get(row["request_id"])
        if example is None:
            example = Example(
                request_id=row["request_id"],
                created_at=row["created_at"],
                difficulty=row["difficulty"],
                start_index=row["start_index"],
                features=loads(row["features"]),
                prompt_chars=row["prompt_chars"] or 0,
                task_type=row.get("task_type"),
                explored=bool(row.get("explored")),
                explore_prob=row.get("explore_prob"),
                threshold=row.get("threshold"),
            )
            examples[row["request_id"]] = example
        example.scores[index] = row["score"]
        example.costs[index] = row["cost_usd"] or 0.0
    return sorted(examples.values(), key=lambda e: (e.created_at, e.request_id))


def _token_stats(rows: list[dict], ladder: list[str]) -> tuple[float, dict, dict]:
    """chars-per-token, mean completion tokens and mean attempt cost per model."""
    chars = tokens = 0
    seen: set[str] = set()
    completion: dict[str, list[int]] = {}
    cost: dict[str, list[float]] = {}
    for row in rows:
        model = row["model"]
        if row["request_id"] not in seen and row["prompt_tokens"] and row["prompt_chars"]:
            seen.add(row["request_id"])
            chars += row["prompt_chars"]
            tokens += row["prompt_tokens"]
        if row["completion_tokens"]:
            completion.setdefault(model, []).append(row["completion_tokens"])
        if row["cost_usd"]:
            cost.setdefault(model, []).append(row["cost_usd"])
    mean = lambda values: sum(values) / len(values)  # noqa: E731
    # Rungs the window never reached still need a price, or no walk that might
    # escalate onto them could be costed — and the model could never deploy.
    observed = [mean(cost[m]) if cost.get(m) else None for m in ladder]
    filled = fill_ladder_costs(observed) or []
    return (
        chars / tokens if tokens else 4.0,
        {m: mean(v) for m, v in completion.items() if m in ladder},
        dict(zip(ladder, filled)),
    )


def _logit(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def fit(examples: list[Example], config: Config, thresholds: list[int],
        ladder: list[str]) -> dict[int, list[Head]]:
    """Train one head per (threshold, rung)."""
    predictor = config.predictor
    size = DENSE_SIZE + predictor.buckets
    rng = random.Random(SEED)
    heads: dict[int, list[Head]] = {}
    for threshold in thresholds:
        heads[threshold] = []
        for rung in range(len(ladder)):
            data = []
            for example in examples:
                label = example.labels(threshold, len(ladder), predictor.impute).get(rung)
                if label is None:
                    continue
                passed, imputed = label
                weight = predictor.imputed_weight if imputed else 1.0
                data.append((example.features, 1.0 if passed else 0.0, weight))
            head = Head(size)
            # Start from the (smoothed) base rate, so a head with little or
            # one-sided data predicts something sensible instead of 0.5.
            positives = sum(w for _, y, w in data if y)
            total = sum(w for _, _, w in data)
            head.weights[0] = _logit((positives + 1) / (total + 2))
            for _ in range(predictor.epochs):
                rng.shuffle(data)
                for features, label, weight in data:
                    head.update(features, label, weight, predictor.learning_rate, predictor.l2)
            heads[threshold].append(head)
    return heads


class _ExampleScores:
    """Feeds training examples to TierLearner as if they were the attempts table."""

    def __init__(self, examples: list[Example], ladder: list[str]):
        counts: dict[tuple, list] = {}
        for e in examples:
            for rung, score in e.scores.items():
                key = (e.task_type, e.difficulty, ladder[rung], score)
                entry = counts.setdefault(key, [0, 0.0])
                entry[0] += 1
                entry[1] += e.costs.get(rung, 0.0)
        self.rows = [
            {"task_type": t, "difficulty": d, "model": m, "score": s,
             "samples": n, "cost_sum": c}
            for (t, d, m, s), (n, c) in counts.items() if t and d
        ]

    async def attempt_scores(self, since: str) -> list[dict]:
        return self.rows


async def baseline_router(examples: list[Example], config: Config) -> TierLearner:
    """The router the model is clamped against: skill matrix, then config tiers.

    Rebuilt from the training split with the same TierLearner the server uses,
    so the trust region in the replay is the one the runtime will apply.
    """
    ladder = [step.id for step in config.ladder]
    learner = TierLearner(config, _ExampleScores(examples, ladder))
    await learner.refresh()
    return learner


def model_start(probs: list[float], costs: list[float], baseline: int, config: Config) -> int:
    """The model's start rung, kept within one rung below the baseline router.

    One rung is exactly how far exploration reaches, so every downward move the
    model makes is one there is logged evidence about — and a wrong model costs
    at most one wasted cheap attempt, the same bound exploration already accepts.
    Moving *up* is unrestricted: it costs money, never quality, and its outcomes
    are known from the attempts that were actually made.
    """
    predictor = config.predictor
    rung = choose_rung(probs, costs, predictor.policy, predictor.min_pass_prob)
    return max(rung, baseline - 1)


def auc(scores: list[float], labels: list[bool]) -> float | None:
    """Area under the ROC curve via the rank-sum formulation (ties averaged)."""
    pos = sum(labels)
    neg = len(labels) - pos
    if not pos or not neg:
        return None
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    rank_sum = sum(r for r, y in zip(ranks, labels) if y)
    return (rank_sum - pos * (pos + 1) / 2) / (pos * neg)


def _log_loss(p: float, y: bool) -> float:
    p = min(max(p, 1e-7), 1 - 1e-7)
    return -math.log(p) if y else -math.log(1 - p)


def evaluate_heads(model: RouterModel, train: list[Example], test: list[Example],
                   threshold: int) -> tuple[list[dict], list[dict]]:
    """Per-rung metrics on *observed* held-out labels, plus a calibration table.

    Every head is compared with a constant predictor of its training base rate:
    a model whose log loss doesn't beat that has learned nothing about prompts.
    """
    rungs = len(model.ladder)
    metrics, pairs = [], []
    pooled_model = pooled_base = 0.0
    pooled_n = 0
    for rung in range(rungs):
        train_labels = [e.scores[rung] >= threshold for e in train if rung in e.scores]
        base = (sum(train_labels) + 1) / (len(train_labels) + 2)
        preds, ys = [], []
        for example in test:
            if rung not in example.scores:
                continue
            p = model.predict(example.features, threshold)[rung]
            preds.append(p)
            ys.append(example.scores[rung] >= threshold)
        pairs.extend(zip(preds, ys))
        n = len(ys)
        pooled_model += sum(_log_loss(p, y) for p, y in zip(preds, ys))
        pooled_base += sum(_log_loss(base, y) for y in ys)
        pooled_n += n
        metrics.append({
            "rung": rung,
            "model": model.ladder[rung],
            "test_n": n,
            "test_pass_rate": round(sum(ys) / n, 3) if n else None,
            "auc": None if (a := auc(preds, ys)) is None else round(a, 3),
            "log_loss": round(sum(_log_loss(p, y) for p, y in zip(preds, ys)) / n, 4) if n else None,
            "baseline_log_loss": round(sum(_log_loss(base, y) for y in ys) / n, 4) if n else None,
            "brier": round(sum((p - y) ** 2 for p, y in zip(preds, ys)) / n, 4) if n else None,
            "mean_predicted": round(sum(preds) / n, 4) if n else None,
        })
    calibration = []
    for b in range(10):
        lo, hi = b / 10, (b + 1) / 10
        bucket = [(p, y) for p, y in pairs if lo <= p < hi or (b == 9 and p == 1.0)]
        if bucket:
            calibration.append({
                "predicted": f"{lo:.1f}-{hi:.1f}",
                "n": len(bucket),
                "mean_predicted": round(sum(p for p, _ in bucket) / len(bucket), 3),
                "observed": round(sum(y for _, y in bucket) / len(bucket), 3),
            })
    pooled = {
        "n": pooled_n,
        "log_loss": round(pooled_model / pooled_n, 4) if pooled_n else None,
        "baseline_log_loss": round(pooled_base / pooled_n, 4) if pooled_n else None,
    }
    metrics.append({"rung": "all", "pooled": pooled})
    return metrics, calibration


def _attempt_costs(example: Example, model: RouterModel) -> list[float] | None:
    """What one attempt at each rung costs *for this request*.

    Observed where the rung was tried; elsewhere, an observed cost rescaled by
    the ratio of the two models' average attempt cost — which carries this
    request's length over to the rung it never reached.
    """
    anchor = next(((i, c) for i, c in example.costs.items() if c), None)
    costs = []
    for rung, model_id in enumerate(model.ladder):
        if rung in example.costs and example.costs[rung]:
            costs.append(example.costs[rung])
            continue
        mean_here = model.mean_cost.get(model_id)
        if anchor is None or not mean_here:
            return None
        anchor_rung, anchor_cost = anchor
        mean_anchor = model.mean_cost.get(model.ladder[anchor_rung])
        if not mean_anchor:
            return None
        costs.append(anchor_cost * mean_here / mean_anchor)
    return costs


def simulate(example: Example, start: int, threshold: int, costs: list[float],
             rungs: int, max_escalations: int, impute: bool,
             fill=None) -> tuple[float, float, float] | None:
    """Expected (cost, attempts, success) of walking the ladder from `start`.

    Known outcomes (observed, or imputed by monotonicity) are certain. An
    unknown rung takes its pass probability from `fill(rung)`; with no `fill`
    the walk gives up and returns None, which is the observed-only replay.
    """
    labels = example.labels(threshold, rungs, impute)
    cost = attempts = success = 0.0
    reach = 1.0  # probability the walk gets this far
    for step, rung in enumerate(range(start, rungs)):
        if step > max_escalations:
            break
        label = labels.get(rung)
        if label is not None:
            q = 1.0 if label[0] else 0.0
        elif fill is None:
            return None
        else:
            q = fill(rung)
        cost += reach * costs[rung]
        attempts += reach
        success += reach * q
        reach *= 1.0 - q
        if reach <= 0.0:
            break
    return cost, attempts, success


def _propensity(example: Example, configured: float) -> float:
    """P(this request was explored), as it stood when it was routed.

    Requests logged before the probability was recorded fall back to today's
    configured epsilon — correct only if it hasn't been changed since.
    """
    return example.explore_prob if example.explore_prob is not None else configured


def model_walk(probs: list[float], costs: list[float], start: int,
               max_escalations: int) -> tuple[float, float, float]:
    """Expected (cost, attempts, success) from `start` using only the model.

    The doubly robust estimator's "direct method" term. It must not peek at
    this request's real outcomes — that independence is what lets the
    correction term below remove its bias.
    """
    cost = attempts = success = 0.0
    reach = 1.0
    for step, rung in enumerate(range(start, len(probs))):
        if step > max_escalations:
            break
        cost += reach * costs[rung]
        attempts += reach
        success += reach * probs[rung]
        reach *= 1.0 - probs[rung]
    return cost, attempts, success


async def replay(model: RouterModel, test: list[Example], config: Config, threshold: int,
                 baseline: TierLearner | None = None) -> dict:
    """Estimate what each policy would have cost on the held-out requests.

    For a start rung at or above where the router of the day started (`s0`),
    the outcome is known: those rungs were tried, or are implied by a pass
    below them. One rung lower is where the estimate has to be earned. There,
    exploration moved a random ε of requests by coin flip, so they are an
    unbiased sample of what *would* happen — and the doubly robust estimator

        V = model_walk(b) + [explored] / ε * (observed - model_walk(b))

    uses them to correct the model's own prediction. Its expectation is the
    true value whatever the model believes; a stale or overconfident model only
    makes it noisier, which the confidence bound then has to overcome. Lower
    than that, nothing was ever observed: such choices are counted as
    unsupported. (The runtime trust region keeps the model out of there.)
    """
    rungs = len(model.ladder)
    predictor = config.predictor
    epsilon = config.exploration.epsilon if config.exploration.enabled else 0.0
    policies = ["model", "baseline", "static_tiers", "as_routed", "always_cheapest",
                "always_strongest"]
    totals = {p: [0.0, 0.0, 0.0] for p in policies}  # cost, attempts, success
    unsupported = {p: 0 for p in policies}
    savings: list[float] = []  # per request: baseline cost - model cost
    worst_savings: list[float] = []  # same, charging every downward move as a failure
    worst_success = paired_base_success = 0.0
    paired_base_cost = 0.0
    corrections = 0  # explored requests that ground the model's downward moves
    oracle_cost, oracle_n, covered = 0.0, 0, 0
    # The policy decides on what the runtime could know (average attempt costs
    # per model) and is charged what this request actually cost.
    decision_costs = [model.mean_cost.get(m, 0.0) for m in model.ladder]

    for example in test:
        if example.threshold is not None and example.threshold != threshold:
            continue  # routed under another mode: its walk stopped at a different bar
        s0 = example.router_start
        costs = _attempt_costs(example, model)
        if costs is None or s0 is None:
            continue
        observed = simulate(example, example.start_index, threshold, costs, rungs,
                            config.judge.max_escalations, impute=False)
        if observed is None:
            continue
        covered += 1
        probs = model.predict(example.features, threshold)
        decide_on = decision_costs if all(decision_costs) else costs
        static = config_start_index(config, example.difficulty or "medium")
        base = static
        if baseline is not None:
            base = baseline.start_index(
                example.task_type or "", example.difficulty or "medium", threshold
            )[0]
        starts = {
            "model": model_start(probs, decide_on, base, config),
            "baseline": base,
            "static_tiers": static,
            "as_routed": example.start_index,
            "always_cheapest": 0,
            "always_strongest": rungs - 1,
        }

        values, supported = {}, {}
        for name, b in starts.items():
            supported[name] = True
            if b >= s0:
                # Known from the log (observed, or implied by a pass below).
                value = simulate(example, b, threshold, costs, rungs,
                                 config.judge.max_escalations, predictor.impute,
                                 fill=lambda rung: probs[rung])
            elif b == s0 - 1 and (eps := _propensity(example, epsilon)) > 0:
                # Both terms are priced with this request's own per-rung costs, so
                # the correction measures pass/fail surprises, not the spread of
                # completion lengths — which, multiplied by 1/ε, would drown it.
                dm = model_walk(probs, costs, b, config.judge.max_escalations)
                if example.explored:  # this request *was* started at b, by coin flip
                    value = tuple(d + (o - d) / eps for d, o in zip(dm, observed))
                    if name == "model":
                        corrections += 1
                else:
                    value = dm
            else:
                supported[name] = False
                unsupported[name] += 1
                value = model_walk(probs, costs, b, config.judge.max_escalations)
            values[name] = value
            for i in range(3):
                totals[name][i] += value[i]
        # Compare only where both sides are estimable: an unsupported baseline
        # choice would be valued with the model's own beliefs.
        if supported["model"] and supported["baseline"]:
            savings.append(values["baseline"][0] - values["model"][0])
            paired_base_cost += values["baseline"][0]
            paired_base_success += values["baseline"][2]
            worst = values["model"]
            if starts["model"] == s0 - 1:
                # Worst case for a move one rung down: the cheap attempt fails
                # and the request continues from where the router would have
                # started — known exactly, with one escalation already spent.
                rest = simulate(example, s0, threshold, costs, rungs,
                                config.judge.max_escalations - 1, predictor.impute,
                                fill=lambda rung: probs[rung])
                worst = (costs[s0 - 1] + rest[0], 1 + rest[1], rest[2])
            worst_savings.append(values["baseline"][0] - worst[0])
            worst_success += worst[2]

        labels = example.labels(threshold, rungs, predictor.impute)
        cheapest = next((r for r in range(rungs) if labels.get(r, (False,))[0]), None)
        if cheapest is not None and all(r in labels for r in range(cheapest)):
            oracle_cost += costs[cheapest]
            oracle_n += 1

    def mean(total: float, digits: int) -> float | None:
        return round(total / covered, digits) if covered else None

    base_cost = paired_base_cost / len(savings) if savings else 0.0

    def bound(values: list[float], df: int) -> tuple[float, float]:
        if len(values) < 2:
            return 0.0, 0.0
        avg = sum(values) / len(values)
        var = sum((v - avg) ** 2 for v in values) / (len(values) - 1)
        return avg, avg - _t95(df) * math.sqrt(var / len(values))

    avg, lower = bound(savings, corrections - 1)
    # The worst-case bound has no 1/epsilon weights — every term is a logged
    # outcome or a known cost — so its noise is ordinary and n is its evidence.
    worst_avg, worst_lower = bound(worst_savings, len(worst_savings) - 1)
    pairs = len(savings)
    return {
        "threshold": threshold,
        "requests": len(test),
        "covered": covered,
        "coverage": round(covered / len(test), 3) if test else 0.0,
        "method": "doubly robust within one rung of the logged start; exact above it",
        "epsilon": epsilon,
        "policies": {
            name: {
                "mean_cost_usd": mean(t[0], 8),
                "mean_attempts": mean(t[1], 3),
                "success_rate": mean(t[2], 3),
                "unsupported": round(unsupported[name] / covered, 3) if covered else None,
            }
            for name, t in totals.items()
        },
        "saving_vs_baseline": {
            "compared": pairs,
            "evidence": corrections,
            "estimate": round(avg / base_cost, 4) if base_cost else None,
            "lower_bound_95": round(lower / base_cost, 4) if base_cost else None,
            "baseline_success": round(paired_base_success / pairs, 3) if pairs else None,
        },
        # Assumption-free: every move below the router's start is charged as a
        # failure, so this needs no exploration evidence at all.
        "worst_case_saving": {
            "estimate": round(worst_avg / base_cost, 4) if base_cost else None,
            "lower_bound_95": round(worst_lower / base_cost, 4) if base_cost else None,
            "success_rate": round(worst_success / pairs, 3) if pairs else None,
        },
        "oracle_mean_cost_usd": round(oracle_cost / oracle_n, 8) if oracle_n else None,
    }


def _deploy_decision(replay_summary: dict, head_metrics: list[dict]) -> tuple[bool, str]:
    """Deploy only on a saving over the baseline router that is proven at 95%.

    Two ways to prove it. The worst-case bound charges every downward move as a
    wasted attempt, so it holds without any exploration data — it is what lets a
    model that mostly sends hard prompts *up* deploy straight away. Failing that,
    the doubly robust estimate can prove a saving from downward moves, but only
    with enough explored requests behind it.
    """
    covered = replay_summary["covered"]
    if covered < MIN_TEST_REQUESTS:
        return False, f"only {covered} held-out requests could be replayed (need {MIN_TEST_REQUESTS})"
    model = replay_summary["policies"]["model"]
    base = replay_summary["policies"]["baseline"]
    if model["unsupported"] > MAX_UNSUPPORTED:
        return False, (
            f"{model['unsupported']:.0%} of its choices start below anything exploration "
            "tried, so their outcome can't be estimated — let the router collect more "
            "evidence (exploration.epsilon) before retraining"
        )
    saving = replay_summary["saving_vs_baseline"]
    if saving["compared"] < MIN_TEST_REQUESTS:
        return False, (
            f"only {saving['compared']} held-out requests let the model and the baseline "
            "router be compared on logged evidence"
        )
    worst = replay_summary.get("worst_case_saving") or {}
    base_success = saving.get("baseline_success", base["success_rate"])
    if (
        worst.get("lower_bound_95") is not None
        and worst["lower_bound_95"] > 0
        and worst["success_rate"] >= base_success - SUCCESS_TOLERANCE
    ):
        return True, (
            f"at least {worst['lower_bound_95']:.1%} cheaper than the baseline router "
            "(skill matrix, then config) with 95% confidence, even if every downward move "
            f"failed — {saving['estimate']:.1%} expected — at success rate "
            f"{worst['success_rate']:.3f} vs {base_success:.3f}"
        )
    if model["success_rate"] < base["success_rate"] - SUCCESS_TOLERANCE:
        return False, (
            f"success rate {model['success_rate']:.3f} is below the baseline router's "
            f"{base['success_rate']:.3f}"
        )
    if saving["evidence"] < MIN_EVIDENCE:
        worst_note = ""
        if worst.get("lower_bound_95") is not None:
            worst_note = (
                f"Its worst-case saving ({worst['lower_bound_95']:+.1%} at 95%) isn't proof "
                "either, and "
            )
        return False, (
            f"no proven saving over the baseline router. {worst_note}the expected "
            f"{saving['estimate']:+.1%} rests on {saving['evidence']} explored requests "
            f"where it would start lower; {MIN_EVIDENCE} are needed — more traffic (or a "
            "higher exploration.epsilon) will get there"
        )
    if saving["lower_bound_95"] is None or saving["lower_bound_95"] <= 0:
        return False, (
            f"estimated saving {saving['estimate']:+.1%} vs the baseline router is not "
            f"distinguishable from zero (95% lower bound {saving['lower_bound_95']:+.1%}) — "
            "more traffic will narrow it"
        )
    return True, (
        f"{saving['estimate']:.1%} cheaper than the baseline router (skill matrix, then "
        f"config) — at least {saving['lower_bound_95']:.1%} with 95% confidence — at "
        f"success rate {model['success_rate']:.3f} vs {base['success_rate']:.3f}"
    )


def should_replace(existing: RouterModel | None, candidate: RouterModel) -> tuple[bool, str]:
    """Whether a freshly trained candidate may overwrite the model on disk.

    A retrain that fails its deployment check must not un-deploy a model that
    passed one. Otherwise routing falls back to the static tiers, exploration
    stops reaching the cheap rungs the candidate needed evidence about, and the
    next retrain has *less* to go on — the router flaps instead of converging.
    """
    if candidate.deployable:
        return True, "candidate passed its deployment check"
    if existing is None:
        return True, "no model on disk yet"
    if existing.ladder != candidate.ladder or existing.feature_version != candidate.feature_version:
        return True, "the model on disk was trained for a different ladder or feature scheme"
    if not existing.deployable:
        return True, "the model on disk was not deployable either"
    return False, "kept the deployed model — the candidate failed its deployment check"


async def train(config: Config, store, refit: bool = True, force: bool = False) -> TrainingReport:
    """Train, evaluate on the newest slice of the log, and (by default) refit on all of it."""
    predictor = config.predictor
    ladder = [step.id for step in config.ladder]
    thresholds = sorted({config.judge.threshold, config.judge.strict_threshold,
                         config.judge.cheap_threshold})
    since = (datetime.now(timezone.utc) - timedelta(days=predictor.lookback_days)).isoformat()
    rows = await store.training_rows(since, FEATURE_VERSION, predictor.buckets)
    examples = build_examples(rows, len(ladder))
    if len(examples) < predictor.min_examples and not force:
        raise NotEnoughData(
            f"{len(examples)} usable requests in the last {predictor.lookback_days:g} days; "
            f"need {predictor.min_examples} (predictor.min_examples). Keep routing with "
            f"`predictor.log_features: true` and try again, or pass --force."
        )
    if not examples:
        raise NotEnoughData("no requests with stored features — nothing to train on")

    # Split by time, not at random: the model is judged on traffic newer than
    # anything it learned from, which is the situation it will actually face.
    cut = max(1, int(len(examples) * (1 - predictor.holdout)))
    train_set, test_set = examples[:cut], examples[cut:]
    chars_per_token, completion_tokens, mean_cost = _token_stats(rows, ladder)

    def build(data: list[Example]) -> RouterModel:
        return RouterModel(
            ladder=ladder,
            thresholds=thresholds,
            buckets=predictor.buckets,
            feature_version=FEATURE_VERSION,
            heads=fit(data, config, thresholds, ladder),
            chars_per_token=chars_per_token,
            completion_tokens=completion_tokens,
            mean_cost=mean_cost,
            trained_at=datetime.now(timezone.utc).isoformat(),
            examples=len(data),
        )

    model = build(train_set)
    main = config.judge.threshold
    head_metrics, calibration = evaluate_heads(model, train_set, test_set, main)
    baseline = await baseline_router(train_set, config)
    replay_summary = await replay(model, test_set, config, main, baseline)
    deployable, reason = _deploy_decision(replay_summary, head_metrics)

    final = build(examples) if refit else model
    final.metrics = {
        "evaluated_on": "time-ordered holdout",
        "train_examples": len(train_set),
        "test_examples": len(test_set),
        "heads": head_metrics,
        "calibration": calibration,
        "replay": replay_summary,
        "refit_on_all_data": refit,
    }
    final.deployable, final.deploy_reason = deployable, reason
    return TrainingReport(
        examples=len(examples),
        train_examples=len(train_set),
        test_examples=len(test_set),
        thresholds=thresholds,
        head_metrics=head_metrics,
        calibration=calibration,
        replay=replay_summary,
        deployable=deployable,
        deploy_reason=reason,
        model=final,
        refit=refit,
    )
