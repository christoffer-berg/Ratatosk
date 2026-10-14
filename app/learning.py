"""Learned routing: turns logged attempts into a start rung per (task_type, difficulty).

The judge already labels every attempt, so the request log is training data:
P(score >= threshold | model, task_type, difficulty). This module aggregates that
into a skill matrix and uses it to pick where on the ladder to start, falling back
to the static `classifier.tiers` map wherever a cell has too little data.
"""

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from .config import Config

logger = logging.getLogger(__name__)

CONFIG_SOURCE = "config"
LEARNED_SOURCE = "learned"
NEIGHBORS_SOURCE = "neighbors"

_rng = random.Random()


def apply_exploration(index: int, config: Config, rng: random.Random | None = None) -> int:
    """Epsilon-greedy: occasionally start one rung below the chosen one.

    Only ever explores *downward*. The judge escalates a bad cheap answer back
    up, so an unlucky exploration costs one extra cheap attempt rather than a
    wrong answer — which makes the escalation ladder a natural bound on the cost
    of every bad arm pull.
    """
    exploration = config.exploration
    if not exploration.enabled or index <= 0:
        return index
    if (rng or _rng).random() >= exploration.epsilon:
        return index
    return index - 1


@dataclass
class ModelStats:
    """Judged attempts for one model within one (task_type, difficulty) cell."""

    samples: int = 0
    cost_sum: float = 0.0
    scores: dict[int, int] = field(default_factory=dict)

    @property
    def avg_cost(self) -> float:
        return self.cost_sum / self.samples if self.samples else 0.0

    def pass_rate(self, threshold: int) -> float:
        if not self.samples:
            return 0.0
        passed = sum(n for score, n in self.scores.items() if score >= threshold)
        return passed / self.samples


def config_start_index(config: Config, difficulty: str) -> int:
    """Static fallback: the hand-written tier map from config.yaml."""
    index = config.classifier.tiers.get(difficulty, 1)
    return max(0, min(index, len(config.ladder) - 1))


class TierLearner:
    """Caches the skill matrix and resolves start rungs from it.

    Fails open like the rest of the pipeline: any problem reading the log leaves
    the cache as it is and routing falls back to the config tiers.
    """

    def __init__(self, config: Config, store):
        self._config = config
        self._store = store
        self._cells: dict[tuple[str, str], dict[str, ModelStats]] = {}
        self._model_costs: dict[str, float] = {}
        self._refreshed_at: float | None = None
        self._refreshed_wall: str | None = None
        self._lock = asyncio.Lock()

    @property
    def refreshed_at(self) -> str | None:
        return self._refreshed_wall

    @property
    def cell_count(self) -> int:
        return len(self._cells)

    async def refresh(self) -> int:
        """Rebuild the matrix from the last `lookback_days` of attempts."""
        learning = self._config.learning
        since = (
            datetime.now(timezone.utc) - timedelta(days=learning.lookback_days)
        ).isoformat()
        cells: dict[tuple[str, str], dict[str, ModelStats]] = {}
        totals: dict[str, ModelStats] = {}
        for row in await self._store.attempt_scores(since):
            models = cells.setdefault((row["task_type"], row["difficulty"]), {})
            stats = models.setdefault(row["model"], ModelStats())
            stats.samples += row["samples"]
            stats.cost_sum += row["cost_sum"]
            stats.scores[row["score"]] = stats.scores.get(row["score"], 0) + row["samples"]
            overall = totals.setdefault(row["model"], ModelStats())
            overall.samples += row["samples"]
            overall.cost_sum += row["cost_sum"]
        self._cells = cells
        # Cell-independent average cost per attempt, used to price rungs this
        # particular cell has never escalated to.
        self._model_costs = {model: s.avg_cost for model, s in totals.items() if s.samples}
        self._refreshed_at = time.monotonic()
        self._refreshed_wall = datetime.now(timezone.utc).isoformat()
        logger.info("Learned tiers refreshed: %d cells with data", len(cells))
        return len(cells)

    async def maybe_refresh(self) -> None:
        """Refresh if the cache is older than `refresh_minutes`."""
        if not self._config.learning.enabled:
            return
        ttl = self._config.learning.refresh_minutes * 60
        if self._refreshed_at is not None and time.monotonic() - self._refreshed_at <= ttl:
            return
        async with self._lock:
            if self._refreshed_at is not None and time.monotonic() - self._refreshed_at <= ttl:
                return
            try:
                await self.refresh()
            except Exception:
                # Back off for a full TTL rather than retrying on every request.
                self._refreshed_at = time.monotonic()
                logger.warning(
                    "Failed to refresh learned tiers — using config tiers", exc_info=True
                )

    def start_index(self, task_type: str, difficulty: str, threshold: int) -> tuple[int, str]:
        """(ladder index, source) — "learned" when the cell had enough data to decide."""
        fallback = config_start_index(self._config, difficulty)
        if not self._config.learning.enabled:
            return fallback, CONFIG_SOURCE
        stats = self._cells.get((task_type, difficulty))
        if not stats:
            return fallback, CONFIG_SOURCE
        ladder = [step.id for step in self._config.ladder]
        if self._config.learning.policy == "pass_rate":
            index = self._by_pass_rate(ladder, stats, threshold)
        else:
            index = self._by_expected_cost(ladder, stats, threshold)
        if index is None:
            return fallback, CONFIG_SOURCE
        return index, LEARNED_SOURCE

    def _by_pass_rate(self, ladder, stats, threshold) -> int | None:
        """Cheapest rung whose measured pass rate clears `min_pass_rate`."""
        learning = self._config.learning
        for index, model in enumerate(ladder):
            model_stats = stats.get(model)
            if model_stats is None or model_stats.samples < learning.min_samples:
                continue
            if model_stats.pass_rate(threshold) >= learning.min_pass_rate:
                return index
        return None

    def _by_expected_cost(self, ladder, stats, threshold) -> int | None:
        """Rung minimising cost + (1 - pass rate) * cost of escalating onward.

        Only rungs with at least `min_samples` judged attempts can start a chain.
        A rung with less evidence is assumed to pass when it appears in another
        rung's tail — the ladder's own premise is that pricier means more capable,
        so an unmeasured rung terminates a chain rather than inflating it.

        This minimises money, not latency: with the usual ~10x jumps between
        rungs, starting low and letting the judge escalate is cheaper in dollars
        even at a mediocre pass rate. Use the "pass_rate" policy instead to bound
        the number of round trips.
        """
        min_samples = self._config.learning.min_samples
        last = len(ladder) - 1
        costs: list[float | None] = []
        rates: list[float] = []
        known: list[bool] = []
        for model in ladder:
            model_stats = stats.get(model)
            has_data = model_stats is not None and model_stats.samples > 0
            enough = has_data and model_stats.samples >= min_samples
            # Fall back to this model's average across all cells, so a rung this
            # cell has never reached can still be priced into a tail.
            cost = model_stats.avg_cost if has_data else self._model_costs.get(model)
            costs.append(cost)
            rates.append(model_stats.pass_rate(threshold) if enough else 1.0)
            known.append(enough)

        # A rung nothing has ever used has no price at all. The ladder is
        # declared cost-ordered, so the cheapest it can be is the priciest rung
        # below it — a lower bound on the escalation penalty, which keeps the
        # chain computable instead of abandoning the cell to the config tiers.
        floor: float | None = None
        for index, cost in enumerate(costs):
            if cost is None:
                costs[index] = floor
            elif floor is None or cost > floor:
                floor = cost

        expected: list[float | None] = [None] * len(ladder)
        for index in range(last, -1, -1):
            cost = costs[index]
            if cost is None:
                continue
            if index == last or rates[index] >= 1.0:
                expected[index] = cost
                continue
            tail = expected[index + 1]
            if tail is None:
                continue
            expected[index] = cost + (1 - rates[index]) * tail

        candidates = [i for i in range(len(ladder)) if known[i] and expected[i] is not None]
        if not candidates or all(expected[i] == 0 for i in candidates):
            # Every candidate costs $0 ⇒ prices were missing when these were
            # logged, so cost can't discriminate between rungs.
            return None
        return min(candidates, key=lambda i: expected[i])

    def skill_matrix(self, threshold: int | None = None) -> list[dict]:
        """P(pass | model, task_type, difficulty) — the learned knowledge, flattened."""
        threshold = threshold if threshold is not None else self._config.judge.threshold
        rows = []
        for (task_type, difficulty), models in sorted(self._cells.items()):
            for model, stats in sorted(models.items()):
                rows.append(
                    {
                        "task_type": task_type,
                        "difficulty": difficulty,
                        "model": model,
                        "samples": stats.samples,
                        "pass_rate": round(stats.pass_rate(threshold), 3),
                        "avg_cost_usd": round(stats.avg_cost, 8),
                    }
                )
        return rows

    def learned_tiers(self, threshold: int | None = None) -> list[dict]:
        """The resolved start rung per cell, with the source that decided it."""
        threshold = threshold if threshold is not None else self._config.judge.threshold
        ladder = [step.id for step in self._config.ladder]
        rows = []
        for task_type, difficulty in sorted(self._cells):
            index, source = self.start_index(task_type, difficulty, threshold)
            rows.append(
                {
                    "task_type": task_type,
                    "difficulty": difficulty,
                    "start_index": index,
                    "start_model": ladder[index],
                    "source": source,
                    "config_start_index": config_start_index(self._config, difficulty),
                }
            )
        return rows
