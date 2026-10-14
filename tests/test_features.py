"""Tests for the hashed prompt features the local model trains on."""

import subprocess
import sys

import pytest

from app.features import (
    DENSE_NAMES, DENSE_SIZE, MAX_TOKENS, _bucket, dumps, extract, loads,
)

USER = [{"role": "user", "content": "Write a Python function that parses CSV?"}]


def dense(features):
    return {DENSE_NAMES[i]: v for i, v in features.items() if i < DENSE_SIZE}


def test_hashing_is_stable_across_processes():
    # Python's own hash() is salted per process; a model trained in one process
    # and served in another would silently see different features.
    code = "from app.features import _bucket; print(_bucket('parse csv', 8192))"
    other = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                           check=True).stdout.strip()
    assert int(other) == _bucket("parse csv", 8192)


def test_hashed_indices_sit_after_the_dense_block():
    for token in ["a", "python", "parse csv", "x" * 50]:
        assert DENSE_SIZE <= _bucket(token, 100) < DENSE_SIZE + 100


def test_dense_features_reflect_the_request():
    features = extract(USER, "code", "easy", has_tools=True)
    d = dense(features)
    assert d["bias"] == 1.0
    assert d["task_type=code"] == 1.0 and d["difficulty=easy"] == 1.0
    assert d["has_tools"] == 1.0
    assert d["has_question"] == 1.0
    assert "task_type=math" not in d  # zeros are dropped


def test_unknown_labels_add_no_one_hot():
    d = dense(extract(USER, "not-a-type", None))
    assert not any(k.startswith(("task_type=", "difficulty=")) for k in d)


def test_hashed_block_has_unit_norm():
    # Enough buckets that no two n-grams collide, so the norm is exactly 1.
    features = extract(USER, "code", "easy", buckets=1 << 24)
    hashed = [v for i, v in features.items() if i >= DENSE_SIZE]
    assert sum(v * v for v in hashed) == pytest.approx(1.0)


def test_similar_prompts_share_features_and_different_ones_dont():
    a = extract([{"role": "user", "content": "translate this sentence to french"}], None, None)
    b = extract([{"role": "user", "content": "translate this sentence to german"}], None, None)
    c = extract([{"role": "user", "content": "prove the lemma by induction"}], None, None)
    hashed = lambda f: {i for i in f if i >= DENSE_SIZE}  # noqa: E731
    assert len(hashed(a) & hashed(b)) > len(hashed(a) & hashed(c))


def test_a_huge_prompt_is_capped():
    huge = [{"role": "user", "content": " ".join(f"w{i}" for i in range(20000))}]
    hashed = [i for i in extract(huge, None, None) if i >= DENSE_SIZE]
    assert len(hashed) <= 2 * MAX_TOKENS


def test_round_trip_through_storage():
    features = extract(USER, "code", "easy")
    restored = loads(dumps(features))
    assert restored.keys() == features.keys()
    assert all(abs(restored[i] - features[i]) < 1e-4 for i in features)


def test_stored_form_contains_no_prompt_text():
    assert "csv" not in dumps(extract(USER, "code", "easy")).lower()
