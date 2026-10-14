"""A small local model that predicts which ladder rung will pass, per prompt.

The skill matrix (learning.py) answers "how often does Haiku pass hard code
tasks?". This answers "will Haiku pass *this* prompt?" — a logistic regression
per (judge threshold, rung) over the sparse features in features.py, trained
on the request log. It needs no GPU, no extra dependencies and no API call:
inference is a sparse dot product, microseconds per request.

Pass probabilities feed the same expected-cost recursion the matrix uses,

    E[i] = c_i + (1 - p_i) * E[i+1],    E[last] = c_last

but with probabilities for this prompt and costs from live prices, so the
choice tracks both what the prompt looks like and what models cost today.
"""

import base64
import json
import logging
import math
import os
import time
from array import array
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

PREDICTOR_SOURCE = "predictor"
MODEL_FORMAT = 1


def sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _encode(weights: array) -> str:
    return base64.b64encode(array("f", weights).tobytes()).decode("ascii")


def _decode(blob: str) -> array:
    values = array("f")
    values.frombytes(base64.b64decode(blob))
    return array("d", values)


class Head:
    """One logistic regression: P(pass | features) for a single rung.

    AdaGrad keeps a per-weight learning rate, which suits hashed sparse
    features — rare n-grams get large steps, common ones settle quickly —
    and needs almost no tuning.
    """

    def __init__(self, size: int, weights: array | None = None):
        self.weights = weights if weights is not None else array("d", bytes(8 * size))
        self._g2: array | None = None

    def score(self, features: dict[int, float]) -> float:
        w = self.weights
        return sigmoid(sum(w[i] * v for i, v in features.items() if i < len(w)))

    def update(self, features: dict[int, float], label: float, weight: float,
               lr: float, l2: float) -> float:
        """One AdaGrad step; returns the example's log loss before the step."""
        if self._g2 is None:
            self._g2 = array("d", bytes(8 * len(self.weights)))
        p = self.score(features)
        grad = (p - label) * weight
        w, g2 = self.weights, self._g2
        for i, v in features.items():
            if i >= len(w):
                continue
            g = grad * v + l2 * w[i]
            g2[i] += g * g
            w[i] -= lr * g / (math.sqrt(g2[i]) + 1e-8)
        p = min(max(p, 1e-7), 1 - 1e-7)
        return -(label * math.log(p) + (1 - label) * math.log(1 - p)) * weight


@dataclass
class RouterModel:
    """Trained heads plus everything needed to know whether they still apply."""

    ladder: list[str]
    thresholds: list[int]
    buckets: int
    feature_version: int
    heads: dict[int, list[Head]]  # threshold -> one head per rung
    # Token statistics for pricing an attempt that hasn't happened yet.
    chars_per_token: float = 4.0
    completion_tokens: dict[str, float] = field(default_factory=dict)
    mean_cost: dict[str, float] = field(default_factory=dict)
    trained_at: str | None = None
    examples: int = 0
    metrics: dict = field(default_factory=dict)
    deployable: bool = False
    deploy_reason: str = ""

    def predict(self, features: dict[int, float], threshold: int) -> list[float] | None:
        """P(pass) per rung, made monotone in rung — or None for an unknown threshold.

        The ladder is declared cheapest-to-strongest, so a pricier rung should
        never be predicted *less* likely to pass. Independent heads trained on
        sparse data can disagree with that; a running max restores it.
        """
        heads = self.heads.get(threshold)
        if heads is None:
            return None  # a mode this model was never trained for
        probs, running = [], 0.0
        for head in heads:
            running = max(running, head.score(features))
            probs.append(running)
        return probs

    def save(self, path: str) -> None:
        payload = {
            "format": MODEL_FORMAT,
            "ladder": self.ladder,
            "thresholds": self.thresholds,
            "buckets": self.buckets,
            "feature_version": self.feature_version,
            "chars_per_token": self.chars_per_token,
            "completion_tokens": self.completion_tokens,
            "mean_cost": self.mean_cost,
            "trained_at": self.trained_at,
            "examples": self.examples,
            "metrics": self.metrics,
            "deployable": self.deployable,
            "deploy_reason": self.deploy_reason,
            "heads": {
                str(t): [_encode(h.weights) for h in heads] for t, heads in self.heads.items()
            },
        }
        tmp = f"{path}.tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f)
        os.replace(tmp, path)  # atomic: a running server never reads half a file

    @classmethod
    def load(cls, path: str) -> "RouterModel":
        with open(path) as f:
            payload = json.load(f)
        if payload.get("format") != MODEL_FORMAT:
            raise ValueError(f"Unsupported model format {payload.get('format')}")
        return cls(
            ladder=payload["ladder"],
            thresholds=payload["thresholds"],
            buckets=payload["buckets"],
            feature_version=payload["feature_version"],
            heads={
                int(t): [Head(0, _decode(blob)) for blob in blobs]
                for t, blobs in payload["heads"].items()
            },
            chars_per_token=payload.get("chars_per_token", 4.0),
            completion_tokens=payload.get("completion_tokens", {}),
            mean_cost=payload.get("mean_cost", {}),
            trained_at=payload.get("trained_at"),
            examples=payload.get("examples", 0),
            metrics=payload.get("metrics", {}),
            deployable=payload.get("deployable", False),
            deploy_reason=payload.get("deploy_reason", ""),
        )


def fill_ladder_costs(costs: list[float | None]) -> list[float] | None:
    """Price every rung, given prices for only some of them.

    A rung nothing has ever reached has no cost history, and a model missing
    from the price cache has no price; left at $0 it would look free, and the
    expected-cost recursion would happily start on the flagship. The ladder is
    declared cheapest-to-strongest, so the least an unpriced rung can cost is
    the priciest rung below it — a lower bound, like learning.py's. Leading
    gaps take the first known price above them, an upper bound that errs
    against the cheap rung. None when nothing is priced at all.
    """
    known = [c if c else None for c in costs]
    if not any(known):
        return None
    filled: list[float | None] = []
    floor: float | None = None
    for cost in known:
        if cost is None:
            filled.append(floor)
        else:
            floor = cost if floor is None else max(floor, cost)
            filled.append(cost)
    first = next(c for c in filled if c is not None)
    return [first if c is None else c for c in filled]


def expected_costs(probs: list[float], costs: list[float]) -> list[float]:
    """E[i] = c_i + (1 - p_i) * E[i+1] — the cost of starting at each rung."""
    expected = [0.0] * len(costs)
    expected[-1] = costs[-1]
    for i in range(len(costs) - 2, -1, -1):
        expected[i] = costs[i] + (1 - probs[i]) * expected[i + 1]
    return expected


def choose_rung(probs: list[float], costs: list[float], policy: str,
                min_pass_prob: float) -> int:
    """Start rung under the configured policy.

    "expected_cost" minimises dollars and accepts extra round trips to do it;
    "pass_rate" takes the cheapest rung likely enough to pass first time.
    """
    if policy == "pass_rate":
        for i, p in enumerate(probs):
            if p >= min_pass_prob:
                return i
        return len(probs) - 1
    expected = expected_costs(probs, costs)
    return min(range(len(expected)), key=lambda i: expected[i])


class PredictorRuntime:
    """Loads the trained model, keeps it fresh, and answers routing queries.

    Refuses to route — returning None so the matrix or config decides — whenever
    the model cannot be trusted: missing, trained for a different ladder or
    feature scheme, or never shown to beat the static tiers.
    """

    RELOAD_CHECK_SECONDS = 30

    def __init__(self, config):
        self._config = config
        self.model: RouterModel | None = None
        self.status = "not loaded"
        self._mtime: float | None = None
        self._checked_at = 0.0

    def load(self) -> bool:
        path = self._config.predictor.path
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            self.model, self.status = None, f"no model at {path} — run `python -m app.train`"
            return False
        try:
            model = RouterModel.load(path)
        except Exception as exc:
            self.model, self.status = None, f"failed to load {path}: {exc}"
            logger.warning("Failed to load routing model", exc_info=True)
            return False
        self._mtime = mtime
        self.model = model
        self.status = self._usability(model) or "active"
        logger.info("Routing model loaded (%s): %s", path, self.status)
        return True

    def maybe_reload(self) -> None:
        """Pick up a retrained model without a restart (cheap stat, throttled)."""
        now = time.monotonic()
        if now - self._checked_at < self.RELOAD_CHECK_SECONDS:
            return
        self._checked_at = now
        try:
            mtime = os.path.getmtime(self._config.predictor.path)
        except OSError:
            return
        if mtime != self._mtime:
            self.load()

    def _usability(self, model: RouterModel) -> str | None:
        """Why this model must not route, or None if it may."""
        from .features import FEATURE_VERSION

        ladder = [step.id for step in self._config.ladder]
        if model.ladder != ladder:
            return "inactive: trained for a different ladder — retrain"
        if model.feature_version != FEATURE_VERSION:
            return "inactive: trained with an older feature scheme — retrain"
        if model.buckets != self._config.predictor.buckets:
            return "inactive: bucket count differs from config — retrain"
        if not model.deployable and not self._config.predictor.allow_undeployable:
            return f"inactive: not deployable ({model.deploy_reason})"
        return None

    @property
    def active(self) -> bool:
        return (
            self._config.predictor.enabled
            and self.model is not None
            and self._usability(self.model) is None
        )

    async def rung_costs(self, prompt_chars: int, client) -> list[float] | None:
        """Estimated dollar cost of one attempt at each rung, at today's prices."""
        model = self.model
        prompt_tokens = prompt_chars / max(model.chars_per_token, 1.0)
        judge_in, judge_out = await client.price_for(self._config.judge.model)
        costs = []
        for model_id in model.ladder:
            completion = model.completion_tokens.get(model_id, 300.0)
            price_in, price_out = await client.price_for(model_id)
            if price_in or price_out:
                costs.append(prompt_tokens * price_in + completion * price_out)
            else:
                # Price cache down or model unlisted: fall back to what attempts
                # at this rung actually cost historically.
                costs.append(model.mean_cost.get(model_id))
        filled = fill_ladder_costs(costs)
        if filled is None:
            return None
        # Every attempt is also judged; that overhead is what makes a failed
        # cheap attempt more than free.
        judging = (prompt_tokens + 150) * judge_in + 60 * judge_out
        return [
            cost + judging + model.completion_tokens.get(m, 300.0) * judge_in
            for cost, m in zip(filled, model.ladder)
        ]

    async def decide(self, features: dict[int, float], threshold: int,
                     prompt_chars: int, client) -> tuple[int, list[float]] | None:
        """(start rung, P(pass) per rung), or None to defer to the next source."""
        if not self.active:
            return None
        probs = self.model.predict(features, threshold)
        if probs is None:
            return None
        costs = await self.rung_costs(prompt_chars, client)
        if costs is None:
            return None
        predictor = self._config.predictor
        rung = choose_rung(probs, costs, predictor.policy, predictor.min_pass_prob)
        return rung, probs
