"""Tests for SQLite logging, the attempts table, and schema migration."""

import aiosqlite
import pytest

from app.store import Store


@pytest.fixture
async def store(tmp_path):
    store = Store(str(tmp_path / "test.sqlite3"))
    await store.init()
    yield store
    await store.close()


ATTEMPTS = [
    {"model": "cheap/a", "ladder_index": 0, "score": 4, "cost_usd": 1e-4, "error": None},
    {"model": "mid/b", "ladder_index": 1, "score": 9, "cost_usd": 3e-4, "error": None},
]


async def log(store, request_id="r1", **overrides):
    kwargs = dict(
        request_id=request_id,
        mode="auto",
        model_requested="auto",
        classifier_tier=0,
        difficulty="hard",
        task_type="code",
        tier_source="learned",
        final_model="mid/b",
        final_score=9,
        attempts=ATTEMPTS,
        escalations=1,
        total_cost_usd=4e-4,
        baseline_cost_usd=2e-3,
        threshold=7,
        prompt_chars=12,
    )
    kwargs.update(overrides)
    await store.log_request(**kwargs)


async def test_attempts_are_written_as_queryable_rows(store):
    await log(store)

    rows = await store.attempt_scores("1970-01-01T00:00:00+00:00")
    by_model = {row["model"]: row for row in rows}
    assert by_model["cheap/a"]["score"] == 4
    assert by_model["cheap/a"]["task_type"] == "code"
    assert by_model["cheap/a"]["difficulty"] == "hard"
    assert by_model["mid/b"]["samples"] == 1
    assert by_model["mid/b"]["cost_sum"] == pytest.approx(3e-4)


async def test_attempt_scores_aggregates_repeats(store):
    for i in range(3):
        await log(store, request_id=f"r{i}")

    rows = await store.attempt_scores("1970-01-01T00:00:00+00:00")
    cheap = next(row for row in rows if row["model"] == "cheap/a")
    assert cheap["samples"] == 3
    assert cheap["cost_sum"] == pytest.approx(3e-4)


async def test_attempt_scores_excludes_unjudged_and_windows_by_time(store):
    await log(
        store,
        attempts=[
            {"model": "cheap/a", "ladder_index": 0, "score": None, "cost_usd": 1e-4},
            {"model": "mid/b", "ladder_index": 1, "score": 8, "cost_usd": 3e-4},
        ],
    )

    assert {row["model"] for row in await store.attempt_scores("1970-01-01T00:00:00+00:00")} == {
        "mid/b"
    }
    assert await store.attempt_scores("2999-01-01T00:00:00+00:00") == []


async def test_error_rows_log_without_attempts(store):
    await store.log_request(
        request_id="err", mode="auto", model_requested="auto", error="everything is down"
    )

    stats = await store.stats()
    assert stats["requests"] == 1
    assert stats["errors"] == 1
    assert await store.attempt_scores("1970-01-01T00:00:00+00:00") == []


async def test_stats_reports_task_type_and_tier_source(store):
    await log(store, request_id="r1")
    await log(store, request_id="r2", task_type="math", tier_source="config")

    stats = await store.stats()
    assert stats["tier_sources"] == {"learned": 1, "config": 1}
    assert {row["task_type"] for row in stats["per_task_type"]} == {"code", "math"}
    assert stats["escalation_rate"] == 1.0


async def test_stats_since_filters_by_window(store):
    await log(store)

    assert (await store.stats("1970-01-01T00:00:00+00:00"))["requests"] == 1
    windowed = await store.stats("2999-01-01T00:00:00+00:00")
    assert windowed["requests"] == 0
    assert windowed["per_model"] == []
    assert windowed["saved_pct"] == 0.0


async def test_prompt_text_round_trips_when_supplied(store):
    await log(store, prompt_text="user: What is 2+2?")

    async with aiosqlite.connect(store._path) as db:
        cursor = await db.execute("SELECT prompt_text, prompt_chars FROM requests")
        assert await cursor.fetchone() == ("user: What is 2+2?", 12)


async def test_migrates_a_database_from_the_previous_schema(tmp_path):
    path = str(tmp_path / "old.sqlite3")
    async with aiosqlite.connect(path) as db:
        await db.execute(
            """CREATE TABLE requests (
                   id TEXT PRIMARY KEY, created_at TEXT NOT NULL, mode TEXT NOT NULL,
                   model_requested TEXT, classifier_tier INTEGER, difficulty TEXT,
                   final_model TEXT, final_score INTEGER,
                   num_attempts INTEGER NOT NULL DEFAULT 0,
                   escalations INTEGER NOT NULL DEFAULT 0,
                   total_cost_usd REAL NOT NULL DEFAULT 0,
                   baseline_cost_usd REAL NOT NULL DEFAULT 0,
                   attempts_json TEXT, error TEXT)"""
        )
        await db.execute(
            "INSERT INTO requests (id, created_at, mode) VALUES ('old', '2025-01-01', 'auto')"
        )
        await db.commit()

    store = Store(path)
    await store.init()
    try:
        await log(store, request_id="new")
        stats = await store.stats()
        assert stats["requests"] == 2  # the pre-migration row survives
        assert stats["tier_sources"] == {"learned": 1}
        assert len(await store.attempt_scores("1970-01-01T00:00:00+00:00")) == 2
    finally:
        await store.close()


async def test_negative_feedback_overrides_a_passing_judge_score(store):
    await log(store, attempts=[
        {"model": "cheap/a", "ladder_index": 0, "score": 9, "cost_usd": 1e-4, "accepted": True},
    ])

    (before,) = await store.attempt_scores("1970-01-01T00:00:00+00:00")
    assert before["score"] == 9

    assert await store.record_feedback("r1", ok=False, source="explicit") is True
    (after,) = await store.attempt_scores("1970-01-01T00:00:00+00:00")
    # The judge liked it; the user did not. The user wins.
    assert after["score"] == 1


async def test_positive_feedback_leaves_the_score_alone(store):
    await log(store, attempts=[
        {"model": "cheap/a", "ladder_index": 0, "score": 8, "cost_usd": 1e-4, "accepted": True},
    ])
    await store.record_feedback("r1", ok=True, source="explicit")

    (row,) = await store.attempt_scores("1970-01-01T00:00:00+00:00")
    assert row["score"] == 8


async def test_feedback_only_discredits_the_accepted_attempt(store):
    await log(store, attempts=[
        {"model": "cheap/a", "ladder_index": 0, "score": 4, "cost_usd": 1e-4, "accepted": False},
        {"model": "mid/b", "ladder_index": 1, "score": 9, "cost_usd": 3e-4, "accepted": True},
    ])
    await store.record_feedback("r1", ok=False, source="user_regret")

    scores = {r["model"]: r["score"] for r in await store.attempt_scores("1970-01-01T00:00:00+00:00")}
    assert scores == {"cheap/a": 4, "mid/b": 1}


async def test_a_failed_verifier_overrides_the_judge(store):
    await log(store, attempts=[
        {"model": "cheap/a", "ladder_index": 0, "score": 9, "cost_usd": 1e-4,
         "verifier": "python_syntax", "verifier_passed": False},
    ])

    (row,) = await store.attempt_scores("1970-01-01T00:00:00+00:00")
    assert row["score"] == 1


async def test_a_passing_verifier_does_not_change_the_score(store):
    await log(store, attempts=[
        {"model": "cheap/a", "ladder_index": 0, "score": 8, "cost_usd": 1e-4,
         "verifier": "code_syntax", "verifier_passed": True},
    ])

    (row,) = await store.attempt_scores("1970-01-01T00:00:00+00:00")
    assert row["score"] == 8


async def test_feedback_for_an_unknown_request_is_dropped(store):
    assert await store.record_feedback("nope", ok=False, source="explicit") is False


async def test_audit_stats_measure_the_false_accept_rate(store):
    await log(store)
    for cheap, strong in [(9, 3), (9, 8), (5, 9), (8, 8)]:
        await store.log_audit(
            request_id="r1", judge_model="flagship/j", model="cheap/a",
            task_type="math", difficulty="easy",
            cheap_score=cheap, strong_score=strong, threshold=7, cost_usd=1e-4,
        )

    stats = await store.audit_stats()
    assert stats["audits"] == 4
    assert stats["false_accepts"] == 1  # (9, 3): accepted at 9, strong judge says 3
    assert stats["false_accept_rate"] == 0.25
    assert stats["false_rejects"] == 1  # (5, 9)
    assert stats["per_task_type"][0]["task_type"] == "math"


async def test_feedback_stats_group_by_source(store):
    await log(store, request_id="r1")
    await log(store, request_id="r2")
    await store.record_feedback("r1", ok=False, source="user_regret")
    await store.record_feedback("r2", ok=True, source="explicit")

    stats = await store.feedback_stats()
    assert stats["total"] == 2
    assert stats["negative"] == 1
    assert stats["by_source"]["user_regret"] == {"ok": 0, "not_ok": 1}


async def test_stats_report_exploration_and_verifiers(store):
    await log(store, request_id="r1", explored=True, attempts=[
        {"model": "cheap/a", "ladder_index": 0, "score": 9, "cost_usd": 1e-4,
         "verifier": "python_syntax", "verifier_passed": False},
    ])
    await log(store, request_id="r2", cache_hit=True)

    stats = await store.stats()
    assert stats["explored"] == 1
    assert stats["cache_hits"] == 1
    assert stats["verifiers"] == {"python_syntax": {"passed": 0, "failed": 1}}


async def test_prompt_vectors_round_trip_newest_first(store):
    await log(store, request_id="r1")
    await store.log_prompt_vector("r1", "emb", b"\x00\x01", 0, 9, "an answer")

    (row,) = await store.prompt_vectors("1970-01-01T00:00:00+00:00", 10)
    assert row["vector"] == b"\x00\x01"
    assert (row["ladder_index"], row["score"], row["answer_text"]) == (0, 9, "an answer")
    assert await store.prompt_vectors("2999-01-01T00:00:00+00:00", 10) == []


async def test_prompt_vectors_are_replaced_not_duplicated(store):
    await log(store, request_id="r1")
    await store.log_prompt_vector("r1", "emb", b"\x00", 0, 9, None)
    await store.log_prompt_vector("r1", "emb", b"\x01", 1, 8, None)

    rows = await store.prompt_vectors("1970-01-01T00:00:00+00:00", 10)
    assert len(rows) == 1 and rows[0]["vector"] == b"\x01"


async def test_training_rows_join_features_with_effective_scores(store):
    await log(store, request_id="r1", attempts=[
        {"model": "cheap/a", "ladder_index": 0, "score": 9, "cost_usd": 1e-4, "accepted": True,
         "prompt_tokens": 20, "completion_tokens": 100},
    ])
    await store.log_prompt_features("r1", 1, 8192, 12, '{"i":[0],"v":[1.0]}')
    await store.record_feedback("r1", ok=False, source="explicit")

    (row,) = await store.training_rows("1970-01-01T00:00:00+00:00", 1, 8192)
    assert row["raw_score"] == 9
    assert row["score"] == 1  # the client's verdict overrides the judge
    assert (row["prompt_tokens"], row["completion_tokens"]) == (20, 100)
    assert row["features"] == '{"i":[0],"v":[1.0]}'


async def test_training_rows_skip_other_feature_schemes_and_requests_without_features(store):
    await log(store, request_id="old")
    await log(store, request_id="new")
    await log(store, request_id="bare")
    await store.log_prompt_features("old", 0, 8192, 12, '{"i":[],"v":[]}')
    await store.log_prompt_features("new", 1, 8192, 12, '{"i":[],"v":[]}')

    rows = await store.training_rows("1970-01-01T00:00:00+00:00", 1, 8192)
    assert {r["request_id"] for r in rows} == {"new"}
    assert await store.training_rows("1970-01-01T00:00:00+00:00", 1, 4096) == []


async def test_cache_hits_are_not_training_data(store):
    await log(store, request_id="hit", cache_hit=True)
    await store.log_prompt_features("hit", 1, 8192, 12, '{"i":[],"v":[]}')
    assert await store.training_rows("1970-01-01T00:00:00+00:00", 1, 8192) == []
