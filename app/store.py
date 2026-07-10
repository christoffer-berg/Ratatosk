"""SQLite logging: one row per request, plus queries for /stats."""

import asyncio
import json
from datetime import datetime, timezone

import aiosqlite

SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    mode TEXT NOT NULL,
    model_requested TEXT,
    classifier_tier INTEGER,
    difficulty TEXT,
    final_model TEXT,
    final_score INTEGER,
    num_attempts INTEGER NOT NULL DEFAULT 0,
    escalations INTEGER NOT NULL DEFAULT 0,
    total_cost_usd REAL NOT NULL DEFAULT 0,
    baseline_cost_usd REAL NOT NULL DEFAULT 0,
    attempts_json TEXT,
    error TEXT
);
"""


class Store:
    def __init__(self, path: str):
        self._path = path
        self._db: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()

    async def init(self) -> None:
        self._db = await aiosqlite.connect(self._path)
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(SCHEMA)
        await self._db.commit()

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    async def log_request(
        self,
        request_id: str,
        mode: str,
        model_requested: str | None,
        classifier_tier: int | None = None,
        difficulty: str | None = None,
        final_model: str | None = None,
        final_score: int | None = None,
        attempts: list[dict] | None = None,
        escalations: int = 0,
        total_cost_usd: float = 0.0,
        baseline_cost_usd: float = 0.0,
        error: str | None = None,
    ) -> None:
        async with self._lock:
            await self._db.execute(
                """INSERT INTO requests
                   (id, created_at, mode, model_requested, classifier_tier, difficulty,
                    final_model, final_score, num_attempts, escalations,
                    total_cost_usd, baseline_cost_usd, attempts_json, error)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    request_id,
                    datetime.now(timezone.utc).isoformat(),
                    mode,
                    model_requested,
                    classifier_tier,
                    difficulty,
                    final_model,
                    final_score,
                    len(attempts or []),
                    escalations,
                    total_cost_usd,
                    baseline_cost_usd,
                    json.dumps(attempts or []),
                    error,
                ),
            )
            await self._db.commit()

    async def stats(self) -> dict:
        async with self._lock:
            cursor = await self._db.execute(
                """SELECT
                     COUNT(*) AS requests,
                     COALESCE(SUM(total_cost_usd), 0) AS total_cost_usd,
                     COALESCE(SUM(baseline_cost_usd), 0) AS baseline_cost_usd,
                     COALESCE(SUM(baseline_cost_usd - total_cost_usd), 0) AS saved_usd,
                     COALESCE(AVG(CASE WHEN escalations > 0 THEN 1.0 ELSE 0.0 END), 0)
                       AS escalation_rate,
                     COUNT(error) AS errors
                   FROM requests"""
            )
            totals = dict(await cursor.fetchone())

            cursor = await self._db.execute(
                """SELECT final_model,
                          COUNT(*) AS requests,
                          COALESCE(SUM(total_cost_usd), 0) AS total_cost_usd
                   FROM requests
                   WHERE final_model IS NOT NULL
                   GROUP BY final_model
                   ORDER BY requests DESC"""
            )
            per_model = [dict(row) for row in await cursor.fetchall()]

            cursor = await self._db.execute(
                """SELECT final_score AS score, COUNT(*) AS requests
                   FROM requests
                   WHERE final_score IS NOT NULL
                   GROUP BY final_score
                   ORDER BY final_score"""
            )
            score_distribution = [dict(row) for row in await cursor.fetchall()]

        baseline = totals["baseline_cost_usd"]
        return {
            "requests": totals["requests"],
            "errors": totals["errors"],
            "total_cost_usd": round(totals["total_cost_usd"], 6),
            "baseline_cost_usd": round(baseline, 6),
            "saved_usd": round(totals["saved_usd"], 6),
            "saved_pct": round(totals["saved_usd"] / baseline * 100, 1)
            if baseline > 0
            else 0.0,
            "escalation_rate": round(totals["escalation_rate"], 3),
            "per_model": per_model,
            "score_distribution": score_distribution,
        }
