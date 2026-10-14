"""SQLite logging: one row per request, one per attempt, plus queries for /stats."""

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
    error TEXT,
    task_type TEXT,
    tier_source TEXT,
    prompt_chars INTEGER,
    prompt_text TEXT,
    explored INTEGER NOT NULL DEFAULT 0,
    answer_hash TEXT,
    cache_hit INTEGER NOT NULL DEFAULT 0,
    explore_prob REAL
);

-- Normalised copy of the per-attempt data: "model X scored 5 on a hard code
-- task" is the signal the learner aggregates, and it needs to be queryable.
CREATE TABLE IF NOT EXISTS attempts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    seq INTEGER NOT NULL,
    model TEXT NOT NULL,
    ladder_index INTEGER,
    task_type TEXT,
    difficulty TEXT,
    threshold INTEGER,
    score INTEGER,
    cost_usd REAL NOT NULL DEFAULT 0,
    error TEXT,
    accepted INTEGER NOT NULL DEFAULT 0,
    verifier TEXT,
    verifier_passed INTEGER,
    prompt_tokens INTEGER,
    completion_tokens INTEGER
);

-- Shadow audits: the same answer re-scored by a strong judge, so the cheap
-- judge's false-accept rate is measured rather than assumed.
CREATE TABLE IF NOT EXISTS audits (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    judge_model TEXT NOT NULL,
    model TEXT,
    task_type TEXT,
    difficulty TEXT,
    cheap_score INTEGER,
    strong_score INTEGER,
    threshold INTEGER,
    cost_usd REAL NOT NULL DEFAULT 0,
    reason TEXT
);

-- Ground truth from clients: explicit thumbs, or inferred from the next turn.
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    ok INTEGER NOT NULL,
    source TEXT NOT NULL,
    note TEXT
);

-- Hashed prompt features (features.py) — the local model's training input.
-- Stored instead of prompt text, so training needs no record of the wording.
CREATE TABLE IF NOT EXISTS prompt_features (
    request_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    feature_version INTEGER NOT NULL,
    buckets INTEGER NOT NULL,
    prompt_chars INTEGER,
    features TEXT NOT NULL
);

-- Prompt embeddings for nearest-neighbour routing and the semantic cache.
CREATE TABLE IF NOT EXISTS prompt_vectors (
    request_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    model TEXT NOT NULL,
    vector BLOB NOT NULL,
    ladder_index INTEGER,
    score INTEGER,
    answer_text TEXT
);

"""

# Indexes are applied *after* the migration: some of them cover columns that an
# older database has yet to gain.
INDEXES = """
CREATE INDEX IF NOT EXISTS idx_requests_created ON requests (created_at);
CREATE INDEX IF NOT EXISTS idx_requests_answer_hash ON requests (answer_hash);
CREATE INDEX IF NOT EXISTS idx_attempts_cell
    ON attempts (task_type, difficulty, model, created_at);
CREATE INDEX IF NOT EXISTS idx_feedback_request ON feedback (request_id);
CREATE INDEX IF NOT EXISTS idx_audits_created ON audits (created_at);
CREATE INDEX IF NOT EXISTS idx_vectors_created ON prompt_vectors (created_at);
CREATE INDEX IF NOT EXISTS idx_features_created ON prompt_features (created_at);
CREATE INDEX IF NOT EXISTS idx_attempts_request ON attempts (request_id);
"""

# Columns added after the first release; ALTERed in when an older db is opened.
ADDED_COLUMNS = {
    "requests": {
        "task_type": "TEXT",
        "tier_source": "TEXT",
        "prompt_chars": "INTEGER",
        "prompt_text": "TEXT",
        "explored": "INTEGER NOT NULL DEFAULT 0",
        "answer_hash": "TEXT",
        "cache_hit": "INTEGER NOT NULL DEFAULT 0",
        # The chance this request had of being explored, as configured when it
        # was routed. Off-policy evaluation weighs it by 1/that, so it must be
        # the value in effect then — not whatever the config says today.
        "explore_prob": "REAL",
    },
    "attempts": {
        "accepted": "INTEGER NOT NULL DEFAULT 0",
        "verifier": "TEXT",
        "verifier_passed": "INTEGER",
        "prompt_tokens": "INTEGER",
        "completion_tokens": "INTEGER",
    },
}

# The label the learning code trusts, shared by the skill matrix and the
# training set: harder evidence beats the judge. A verifier rejection, or a
# client saying the accepted answer was wrong, is a failure whatever it scored.
EFFECTIVE_SCORE = """
    CASE
      WHEN a.verifier_passed = 0 THEN 1
      WHEN a.accepted = 1 AND f.ok = 0 THEN 1
      ELSE a.score
    END"""

FEEDBACK_JOIN = """
    LEFT JOIN (SELECT request_id, MIN(ok) AS ok FROM feedback GROUP BY request_id) f
      ON f.request_id = a.request_id"""


def _verifier_summary(rows: list[dict]) -> dict:
    """{verifier: {passed, failed}} — failures are caught false accepts."""
    summary: dict[str, dict[str, int]] = {}
    for row in rows:
        entry = summary.setdefault(row["verifier"], {"passed": 0, "failed": 0})
        entry["passed" if row["verifier_passed"] else "failed"] += row["count"]
    return summary


class Store:
    def __init__(self, path: str):
        self._path = path
        self._db: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()

    async def init(self) -> None:
        self._db = await aiosqlite.connect(self._path)
        try:
            self._db.row_factory = aiosqlite.Row
            await self._db.executescript(SCHEMA)
            await self._migrate()
            await self._db.executescript(INDEXES)
            await self._db.commit()
        except Exception:
            # aiosqlite runs the connection on a non-daemon thread; leaving it
            # open after a failed init would wedge interpreter shutdown.
            await self.close()
            raise

    async def _migrate(self) -> None:
        """Add columns missing from a database created by an earlier version."""
        for table, columns in ADDED_COLUMNS.items():
            cursor = await self._db.execute(f"PRAGMA table_info({table})")
            existing = {row["name"] for row in await cursor.fetchall()}
            for name, kind in columns.items():
                if name not in existing:
                    await self._db.execute(
                        f"ALTER TABLE {table} ADD COLUMN {name} {kind}"
                    )

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
        task_type: str | None = None,
        tier_source: str | None = None,
        final_model: str | None = None,
        final_score: int | None = None,
        attempts: list[dict] | None = None,
        escalations: int = 0,
        total_cost_usd: float = 0.0,
        baseline_cost_usd: float = 0.0,
        threshold: int | None = None,
        prompt_chars: int | None = None,
        prompt_text: str | None = None,
        explored: bool = False,
        answer_hash: str | None = None,
        cache_hit: bool = False,
        explore_prob: float | None = None,
        error: str | None = None,
    ) -> None:
        created_at = datetime.now(timezone.utc).isoformat()
        attempts = attempts or []
        async with self._lock:
            await self._db.execute(
                """INSERT INTO requests
                   (id, created_at, mode, model_requested, classifier_tier, difficulty,
                    final_model, final_score, num_attempts, escalations,
                    total_cost_usd, baseline_cost_usd, attempts_json, error,
                    task_type, tier_source, prompt_chars, prompt_text,
                    explored, answer_hash, cache_hit, explore_prob)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    request_id,
                    created_at,
                    mode,
                    model_requested,
                    classifier_tier,
                    difficulty,
                    final_model,
                    final_score,
                    len(attempts),
                    escalations,
                    total_cost_usd,
                    baseline_cost_usd,
                    json.dumps(attempts),
                    error,
                    task_type,
                    tier_source,
                    prompt_chars,
                    prompt_text,
                    int(explored),
                    answer_hash,
                    int(cache_hit),
                    explore_prob,
                ),
            )
            await self._db.executemany(
                """INSERT INTO attempts
                   (request_id, created_at, seq, model, ladder_index, task_type,
                    difficulty, threshold, score, cost_usd, error,
                    accepted, verifier, verifier_passed, prompt_tokens, completion_tokens)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        request_id,
                        created_at,
                        seq,
                        attempt.get("model"),
                        attempt.get("ladder_index"),
                        task_type,
                        difficulty,
                        threshold,
                        attempt.get("score"),
                        attempt.get("cost_usd") or 0.0,
                        attempt.get("error"),
                        int(bool(attempt.get("accepted"))),
                        attempt.get("verifier"),
                        None
                        if attempt.get("verifier_passed") is None
                        else int(attempt["verifier_passed"]),
                        attempt.get("prompt_tokens"),
                        attempt.get("completion_tokens"),
                    )
                    for seq, attempt in enumerate(attempts)
                    if attempt.get("model")
                ],
            )
            await self._db.commit()

    async def attempt_scores(self, since: str) -> list[dict]:
        """Score histogram + cost per (task_type, difficulty, model) since a timestamp.

        Returned per score rather than pre-aggregated so the learner can compute
        pass rates against any threshold (auto / auto:cheap / auto:strict differ).

        The judge's score is overridden by harder evidence where it exists: an
        answer a deterministic verifier rejected, or one the client told us was
        wrong, counts as a failure however well it scored. Without this the
        matrix would only ever learn from the mistakes the cheap judge caught.
        """
        async with self._lock:
            cursor = await self._db.execute(
                f"""SELECT a.task_type AS task_type,
                          a.difficulty AS difficulty,
                          a.model AS model,
                          {EFFECTIVE_SCORE} AS score,
                          COUNT(*) AS samples,
                          COALESCE(SUM(a.cost_usd), 0) AS cost_sum
                   FROM attempts a
                   {FEEDBACK_JOIN}
                   WHERE a.created_at >= ?
                     AND a.score IS NOT NULL
                     AND a.task_type IS NOT NULL
                     AND a.difficulty IS NOT NULL
                   GROUP BY a.task_type, a.difficulty, a.model, score""",
                (since,),
            )
            return [dict(row) for row in await cursor.fetchall()]

    async def log_audit(
        self,
        request_id: str,
        judge_model: str,
        model: str | None,
        task_type: str | None,
        difficulty: str | None,
        cheap_score: int | None,
        strong_score: int | None,
        threshold: int | None,
        cost_usd: float = 0.0,
        reason: str | None = None,
    ) -> None:
        async with self._lock:
            await self._db.execute(
                """INSERT INTO audits
                   (request_id, created_at, judge_model, model, task_type, difficulty,
                    cheap_score, strong_score, threshold, cost_usd, reason)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    request_id,
                    datetime.now(timezone.utc).isoformat(),
                    judge_model,
                    model,
                    task_type,
                    difficulty,
                    cheap_score,
                    strong_score,
                    threshold,
                    cost_usd,
                    reason,
                ),
            )
            await self._db.commit()

    async def record_feedback(
        self, request_id: str, ok: bool, source: str, note: str | None = None
    ) -> bool:
        """False if no such request — feedback for an unknown id is dropped."""
        async with self._lock:
            cursor = await self._db.execute(
                "SELECT 1 FROM requests WHERE id = ?", (request_id,)
            )
            if await cursor.fetchone() is None:
                return False
            await self._db.execute(
                """INSERT INTO feedback (request_id, created_at, ok, source, note)
                   VALUES (?, ?, ?, ?, ?)""",
                (request_id, datetime.now(timezone.utc).isoformat(), int(ok), source, note),
            )
            await self._db.commit()
            return True

    async def request_by_answer_hash(self, answer_hash: str) -> str | None:
        """Most recent request that produced this exact answer."""
        async with self._lock:
            cursor = await self._db.execute(
                """SELECT id FROM requests
                   WHERE answer_hash = ?
                   ORDER BY created_at DESC LIMIT 1""",
                (answer_hash,),
            )
            row = await cursor.fetchone()
            return row["id"] if row else None

    async def log_prompt_vector(
        self,
        request_id: str,
        model: str,
        vector: bytes,
        ladder_index: int | None,
        score: int | None,
        answer_text: str | None,
    ) -> None:
        async with self._lock:
            await self._db.execute(
                """INSERT OR REPLACE INTO prompt_vectors
                   (request_id, created_at, model, vector, ladder_index, score, answer_text)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    request_id,
                    datetime.now(timezone.utc).isoformat(),
                    model,
                    vector,
                    ladder_index,
                    score,
                    answer_text,
                ),
            )
            await self._db.commit()

    async def log_prompt_features(
        self,
        request_id: str,
        feature_version: int,
        buckets: int,
        prompt_chars: int | None,
        features: str,
    ) -> None:
        async with self._lock:
            await self._db.execute(
                """INSERT OR REPLACE INTO prompt_features
                   (request_id, created_at, feature_version, buckets, prompt_chars, features)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    request_id,
                    datetime.now(timezone.utc).isoformat(),
                    feature_version,
                    buckets,
                    prompt_chars,
                    features,
                ),
            )
            await self._db.commit()

    async def training_rows(self, since: str, feature_version: int, buckets: int) -> list[dict]:
        """One row per judged attempt on a request with stored features, oldest first.

        Only features from the current extraction scheme are returned — vectors
        from an older FEATURE_VERSION or bucket count are silently incompatible.
        """
        async with self._lock:
            cursor = await self._db.execute(
                f"""SELECT r.id AS request_id,
                          r.created_at AS created_at,
                          r.difficulty AS difficulty,
                          r.task_type AS task_type,
                          r.classifier_tier AS start_index,
                          r.tier_source AS tier_source,
                          r.explored AS explored,
                          r.explore_prob AS explore_prob,
                          p.features AS features,
                          p.prompt_chars AS prompt_chars,
                          a.ladder_index AS ladder_index,
                          a.model AS model,
                          a.threshold AS threshold,
                          a.score AS raw_score,
                          {EFFECTIVE_SCORE} AS score,
                          a.cost_usd AS cost_usd,
                          a.prompt_tokens AS prompt_tokens,
                          a.completion_tokens AS completion_tokens
                   FROM prompt_features p
                   JOIN requests r ON r.id = p.request_id
                   JOIN attempts a ON a.request_id = p.request_id
                   {FEEDBACK_JOIN}
                   WHERE p.created_at >= ?
                     AND p.feature_version = ?
                     AND p.buckets = ?
                     AND a.score IS NOT NULL
                     AND a.ladder_index IS NOT NULL
                     AND r.cache_hit = 0
                   ORDER BY r.created_at, r.id, a.seq""",
                (since, feature_version, buckets),
            )
            return [dict(row) for row in await cursor.fetchall()]

    async def prompt_vectors(self, since: str, limit: int) -> list[dict]:
        """Most recent vectors, newest first, for the in-memory scan."""
        async with self._lock:
            cursor = await self._db.execute(
                """SELECT request_id, model, vector, ladder_index, score, answer_text
                   FROM prompt_vectors
                   WHERE created_at >= ?
                   ORDER BY created_at DESC
                   LIMIT ?""",
                (since, limit),
            )
            return [dict(row) for row in await cursor.fetchall()]

    async def audit_stats(self, since: str | None = None) -> dict:
        """How often the cheap judge accepted what the strong judge rejects."""
        where = "WHERE created_at >= ?" if since else ""
        params = (since,) if since else ()
        async with self._lock:
            cursor = await self._db.execute(
                f"""SELECT COUNT(*) AS audits,
                          COALESCE(SUM(CASE WHEN cheap_score >= threshold
                                             AND strong_score < threshold
                                        THEN 1 ELSE 0 END), 0) AS false_accepts,
                          COALESCE(SUM(CASE WHEN cheap_score < threshold
                                             AND strong_score >= threshold
                                        THEN 1 ELSE 0 END), 0) AS false_rejects,
                          COALESCE(AVG(cheap_score - strong_score), 0) AS mean_score_gap,
                          COALESCE(SUM(cost_usd), 0) AS cost_usd
                   FROM audits {where}""",
                params,
            )
            totals = dict(await cursor.fetchone())

            cursor = await self._db.execute(
                f"""SELECT task_type,
                          COUNT(*) AS audits,
                          COALESCE(SUM(CASE WHEN cheap_score >= threshold
                                             AND strong_score < threshold
                                        THEN 1 ELSE 0 END), 0) AS false_accepts,
                          COALESCE(AVG(cheap_score - strong_score), 0) AS mean_score_gap
                   FROM audits
                   WHERE task_type IS NOT NULL
                   {"AND created_at >= ?" if since else ""}
                   GROUP BY task_type
                   ORDER BY audits DESC""",
                params,
            )
            per_task_type = [dict(row) for row in await cursor.fetchall()]

        audits = totals["audits"]
        return {
            "audits": audits,
            "false_accepts": totals["false_accepts"],
            "false_accept_rate": round(totals["false_accepts"] / audits, 3) if audits else 0.0,
            "false_rejects": totals["false_rejects"],
            "mean_score_gap": round(totals["mean_score_gap"], 2),
            "cost_usd": round(totals["cost_usd"], 6),
            "per_task_type": per_task_type,
        }

    async def feedback_stats(self, since: str | None = None) -> dict:
        where = "WHERE created_at >= ?" if since else ""
        params = (since,) if since else ()
        async with self._lock:
            cursor = await self._db.execute(
                f"""SELECT source, ok, COUNT(*) AS count
                   FROM feedback {where}
                   GROUP BY source, ok""",
                params,
            )
            rows = [dict(row) for row in await cursor.fetchall()]
        by_source: dict[str, dict[str, int]] = {}
        for row in rows:
            entry = by_source.setdefault(row["source"], {"ok": 0, "not_ok": 0})
            entry["ok" if row["ok"] else "not_ok"] += row["count"]
        return {
            "total": sum(r["count"] for r in rows),
            "negative": sum(r["count"] for r in rows if not r["ok"]),
            "by_source": by_source,
        }

    async def stats(self, since: str | None = None) -> dict:
        where = "WHERE created_at >= ?" if since else ""
        params = (since,) if since else ()
        async with self._lock:
            cursor = await self._db.execute(
                f"""SELECT
                     COUNT(*) AS requests,
                     COALESCE(SUM(total_cost_usd), 0) AS total_cost_usd,
                     COALESCE(SUM(baseline_cost_usd), 0) AS baseline_cost_usd,
                     COALESCE(SUM(baseline_cost_usd - total_cost_usd), 0) AS saved_usd,
                     COALESCE(AVG(CASE WHEN escalations > 0 THEN 1.0 ELSE 0.0 END), 0)
                       AS escalation_rate,
                     COUNT(error) AS errors
                   FROM requests {where}""",
                params,
            )
            totals = dict(await cursor.fetchone())

            cursor = await self._db.execute(
                f"""SELECT final_model,
                          COUNT(*) AS requests,
                          COALESCE(SUM(total_cost_usd), 0) AS total_cost_usd
                   FROM requests
                   WHERE final_model IS NOT NULL
                   {"AND created_at >= ?" if since else ""}
                   GROUP BY final_model
                   ORDER BY requests DESC""",
                params,
            )
            per_model = [dict(row) for row in await cursor.fetchall()]

            cursor = await self._db.execute(
                f"""SELECT final_score AS score, COUNT(*) AS requests
                   FROM requests
                   WHERE final_score IS NOT NULL
                   {"AND created_at >= ?" if since else ""}
                   GROUP BY final_score
                   ORDER BY final_score""",
                params,
            )
            score_distribution = [dict(row) for row in await cursor.fetchall()]

            cursor = await self._db.execute(
                f"""SELECT task_type, COUNT(*) AS requests,
                          COALESCE(SUM(total_cost_usd), 0) AS total_cost_usd,
                          COALESCE(AVG(escalations), 0) AS avg_escalations
                   FROM requests
                   WHERE task_type IS NOT NULL
                   {"AND created_at >= ?" if since else ""}
                   GROUP BY task_type
                   ORDER BY requests DESC""",
                params,
            )
            per_task_type = [dict(row) for row in await cursor.fetchall()]

            cursor = await self._db.execute(
                f"""SELECT tier_source, COUNT(*) AS requests
                   FROM requests
                   WHERE tier_source IS NOT NULL
                   {"AND created_at >= ?" if since else ""}
                   GROUP BY tier_source""",
                params,
            )
            tier_sources = {row["tier_source"]: row["requests"] for row in await cursor.fetchall()}

            cursor = await self._db.execute(
                f"""SELECT COALESCE(SUM(explored), 0) AS explored,
                          COALESCE(SUM(cache_hit), 0) AS cache_hits
                   FROM requests {where}""",
                params,
            )
            extras = dict(await cursor.fetchone())

            cursor = await self._db.execute(
                f"""SELECT verifier, verifier_passed, COUNT(*) AS count
                   FROM attempts
                   WHERE verifier IS NOT NULL
                   {"AND created_at >= ?" if since else ""}
                   GROUP BY verifier, verifier_passed""",
                params,
            )
            verifier_rows = [dict(row) for row in await cursor.fetchall()]

        baseline = totals["baseline_cost_usd"]
        return {
            "since": since,
            "requests": totals["requests"],
            "errors": totals["errors"],
            "total_cost_usd": round(totals["total_cost_usd"], 6),
            "baseline_cost_usd": round(baseline, 6),
            "saved_usd": round(totals["saved_usd"], 6),
            "saved_pct": round(totals["saved_usd"] / baseline * 100, 1)
            if baseline > 0
            else 0.0,
            "escalation_rate": round(totals["escalation_rate"], 3),
            "explored": extras["explored"],
            "cache_hits": extras["cache_hits"],
            "verifiers": _verifier_summary(verifier_rows),
            "per_model": per_model,
            "per_task_type": per_task_type,
            "tier_sources": tier_sources,
            "score_distribution": score_distribution,
        }
