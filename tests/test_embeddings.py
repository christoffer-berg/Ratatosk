"""Tests for nearest-neighbour routing and the semantic cache."""

from app.config import ClassifierConfig, Config, JudgeConfig, LadderStep
from app.embeddings import Neighbor, Neighborhood, NeighborRouter, normalize, pack, unpack

LADDER = ["cheap/a", "mid/b", "big/c"]


def make_config(**overrides) -> Config:
    config = Config(
        ladder=[LadderStep(id=m) for m in LADDER],
        classifier=ClassifierConfig(model="m"),
        judge=JudgeConfig(model="j"),
    )
    config.embeddings.enabled = True
    config.embeddings.min_neighbors = 2
    config.embeddings.min_similarity = 0.5
    for key, value in overrides.items():
        setattr(config.embeddings, key, value)
    return config


class FakeStore:
    def __init__(self, rows=None):
        self.rows = rows or []

    async def prompt_vectors(self, since, limit):
        return self.rows[:limit]


class FakeEmbeddings:
    def __init__(self, vector=None, fails=False):
        self.vector = vector or [1.0, 0.0]
        self.fails = fails
        self.calls = 0

    async def embed(self, text):
        self.calls += 1
        return None if self.fails else self.vector


def row(request_id, vector, ladder_index=0, score=9, answer=None):
    return {
        "request_id": request_id,
        "model": "emb",
        "vector": pack(vector),
        "ladder_index": ladder_index,
        "score": score,
        "answer_text": answer,
    }


def test_pack_unpack_roundtrip():
    assert [round(v, 4) for v in unpack(pack([0.1, -0.2, 0.3]))] == [0.1, -0.2, 0.3]


def test_normalize_handles_a_zero_vector():
    assert normalize([0.0, 0.0]) == [0.0, 0.0]


async def test_similar_prompts_are_found_and_dissimilar_ones_are_not():
    store = FakeStore([row("near", [1.0, 0.05]), row("far", [-1.0, 0.0])])
    router = NeighborRouter(make_config(min_neighbors=1), store, FakeEmbeddings([1.0, 0.0]))
    await router.refresh()

    result = await router.lookup("a prompt")
    assert [n.request_id for n in result.neighbors] == ["near"]


async def test_neighbours_pick_the_cheapest_rung_that_worked():
    rows = [row(f"r{i}", [1.0, 0.0], ladder_index=0, score=9) for i in range(3)]
    router = NeighborRouter(make_config(), FakeStore(rows), FakeEmbeddings([1.0, 0.0]))
    await router.refresh()

    result = await router.lookup("a prompt")
    assert result.start_index(threshold=7, min_pass_rate=0.8) == 0


async def test_neighbours_skip_a_rung_that_failed_for_them():
    rows = [row("a1", [1.0, 0.0], 0, 3), row("a2", [1.0, 0.0], 0, 4),
            row("b1", [1.0, 0.0], 1, 9), row("b2", [1.0, 0.0], 1, 8)]
    router = NeighborRouter(make_config(), FakeStore(rows), FakeEmbeddings([1.0, 0.0]))
    await router.refresh()

    result = await router.lookup("a prompt")
    assert result.start_index(threshold=7, min_pass_rate=0.8) == 1


def test_a_single_neighbour_is_not_a_pass_rate():
    neighborhood = Neighborhood(neighbors=[Neighbor("r1", 0.99, 0, 9, "cheap/a", None)])
    assert neighborhood.start_index(7, 0.8) is None


async def test_a_sparse_neighbourhood_defers_to_the_skill_matrix():
    router = NeighborRouter(
        make_config(min_neighbors=5), FakeStore([row("only", [1.0, 0.0])]),
        FakeEmbeddings([1.0, 0.0]),
    )
    await router.refresh()

    result = await router.lookup("a prompt")
    assert result.neighbors == []
    assert result.embedding is not None  # still logged, so the index can grow
    assert result.start_index(7, 0.8) is None


async def test_a_failed_embedding_call_yields_nothing():
    router = NeighborRouter(make_config(), FakeStore([row("r", [1.0, 0.0])]), FakeEmbeddings(fails=True))
    await router.refresh()

    result = await router.lookup("a prompt")
    assert result.embedding is None and result.neighbors == []


async def test_disabled_embeddings_never_call_the_model():
    embeddings = FakeEmbeddings()
    router = NeighborRouter(make_config(enabled=False), FakeStore(), embeddings)
    result = await router.lookup("a prompt")
    assert embeddings.calls == 0 and result.neighbors == []


async def test_vectors_of_a_different_dimension_are_ignored():
    # Changing embedding model mid-log must not crash the scan.
    store = FakeStore([row("old", [1.0, 0.0, 0.0]), row("new", [1.0, 0.0])])
    router = NeighborRouter(make_config(min_neighbors=1), store, FakeEmbeddings([1.0, 0.0]))
    await router.refresh()

    result = await router.lookup("a prompt")
    assert [n.request_id for n in result.neighbors] == ["new"]


def test_cache_hit_requires_high_similarity_and_a_stored_answer():
    hit = Neighbor("r1", 0.99, 0, 9, "cheap/a", "cached answer")
    no_answer = Neighbor("r2", 0.99, 0, 9, "cheap/a", None)
    assert Neighborhood(neighbors=[hit]).cache_hit(0.97) is hit
    assert Neighborhood(neighbors=[hit]).cache_hit(0.999) is None
    assert Neighborhood(neighbors=[no_answer]).cache_hit(0.97) is None
