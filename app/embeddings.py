"""Nearest-neighbour routing over logged prompts, plus a semantic answer cache.

The skill matrix in learning.py buckets prompts into (task_type, difficulty)
cells. Embeddings are the finer-grained version of the same idea: instead of
"hard code tasks pass on Haiku 80% of the time", it can say "prompts *like this
one* passed on Haiku 9 times out of 10, start there". As a bonus, a prompt that
is nearly identical to one already answered can reuse the answer outright, at
100% saving.

Scanning is brute-force cosine in pure Python over the most recent
`max_vectors` prompts, which is fine for thousands of vectors at small
dimensionality and is why this is off by default. A vector index (or numpy) is
the upgrade path if the log outgrows it.
"""

import asyncio
import logging
import math
import time
from array import array
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from operator import mul

import httpx

from .config import Config

logger = logging.getLogger(__name__)

NEIGHBORS_SOURCE = "neighbors"


def pack(vector: list[float]) -> bytes:
    return array("f", vector).tobytes()


def unpack(blob: bytes) -> list[float]:
    values = array("f")
    values.frombytes(blob)
    return list(values)


def normalize(vector: list[float]) -> list[float]:
    """Unit-length, so cosine similarity is just a dot product."""
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm else vector


class EmbeddingClient:
    """Minimal OpenAI-compatible `POST /embeddings` client.

    Defaults to the OpenRouter base URL and key, which serve `POST /embeddings`
    with provider-prefixed ids like `openai/text-embedding-3-small`. Point
    `embeddings.base_url` elsewhere — OpenAI, or a local server such as
    llama.cpp or text-embeddings-inference — to use another provider.
    """

    def __init__(self, config: Config):
        embeddings = config.embeddings
        base_url = embeddings.base_url or config.openrouter.base_url
        api_key = embeddings.api_key or config.openrouter.api_key
        self._model = embeddings.model
        self._dimensions = embeddings.dimensions
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
            timeout=config.request_timeout_seconds,
        )

    async def embed(self, text: str) -> list[float] | None:
        """None on any failure — routing falls back to the skill matrix."""
        payload: dict = {"model": self._model, "input": text}
        if self._dimensions:
            payload["dimensions"] = self._dimensions
        try:
            response = await self._client.post("/embeddings", json=payload)
            response.raise_for_status()
            return response.json()["data"][0]["embedding"]
        except Exception:
            logger.warning("Embedding call failed — skipping neighbour routing", exc_info=True)
            return None

    async def close(self) -> None:
        await self._client.aclose()


@dataclass
class Neighbor:
    request_id: str
    similarity: float
    ladder_index: int | None
    score: int | None
    model: str | None
    answer: str | None


@dataclass
class Neighborhood:
    """The result of one embedding lookup."""

    embedding: list[float] | None = None
    neighbors: list[Neighbor] = field(default_factory=list)

    def cache_hit(self, min_similarity: float) -> Neighbor | None:
        """A near-identical prompt whose answer can be reused verbatim."""
        for neighbor in self.neighbors:
            if neighbor.similarity >= min_similarity and neighbor.answer:
                return neighbor
        return None

    def start_index(self, threshold: int, min_pass_rate: float) -> int | None:
        """Cheapest rung the neighbours actually succeeded on."""
        by_rung: dict[int, list[int]] = {}
        for neighbor in self.neighbors:
            if neighbor.ladder_index is None or neighbor.score is None:
                continue
            by_rung.setdefault(neighbor.ladder_index, []).append(neighbor.score)
        for rung in sorted(by_rung):
            scores = by_rung[rung]
            if len(scores) < 2:
                continue  # one anecdote is not a pass rate
            if sum(s >= threshold for s in scores) / len(scores) >= min_pass_rate:
                return rung
        return None


class NeighborRouter:
    """Caches prompt vectors in memory and scans them per request."""

    def __init__(self, config: Config, store, client: EmbeddingClient | None = None):
        self._config = config
        self._store = store
        self._client = client
        self._vectors: list[tuple[str, list[float], int | None, int | None, str | None, str | None]] = []
        self._refreshed_at: float | None = None
        self._refreshed_wall: str | None = None
        self._lock = asyncio.Lock()

    @property
    def refreshed_at(self) -> str | None:
        return self._refreshed_wall

    @property
    def vector_count(self) -> int:
        return len(self._vectors)

    async def refresh(self) -> int:
        embeddings = self._config.embeddings
        since = (
            datetime.now(timezone.utc) - timedelta(days=self._config.learning.lookback_days)
        ).isoformat()
        rows = await self._store.prompt_vectors(since, embeddings.max_vectors)
        vectors = []
        for row in rows:
            vector = normalize(unpack(row["vector"]))
            vectors.append(
                (
                    row["request_id"],
                    vector,
                    row["ladder_index"],
                    row["score"],
                    row["model"],
                    row["answer_text"],
                )
            )
        self._vectors = vectors
        self._refreshed_at = time.monotonic()
        self._refreshed_wall = datetime.now(timezone.utc).isoformat()
        logger.info("Neighbour index refreshed: %d vectors", len(vectors))
        return len(vectors)

    async def maybe_refresh(self) -> None:
        if not self._config.embeddings.enabled:
            return
        ttl = self._config.embeddings.refresh_minutes * 60
        if self._refreshed_at is not None and time.monotonic() - self._refreshed_at <= ttl:
            return
        async with self._lock:
            if self._refreshed_at is not None and time.monotonic() - self._refreshed_at <= ttl:
                return
            try:
                await self.refresh()
            except Exception:
                self._refreshed_at = time.monotonic()
                logger.warning("Failed to refresh neighbour index", exc_info=True)

    def search(self, embedding: list[float]) -> Neighborhood:
        """Brute-force cosine scan of the cached vectors."""
        embeddings = self._config.embeddings
        query = normalize(embedding)
        scored = []
        for request_id, vector, ladder_index, score, model, answer in self._vectors:
            if len(vector) != len(query):
                continue  # embedding model or dimensionality changed
            similarity = sum(map(mul, query, vector))
            if similarity >= embeddings.min_similarity:
                scored.append(
                    Neighbor(request_id, similarity, ladder_index, score, model, answer)
                )
        scored.sort(key=lambda n: n.similarity, reverse=True)
        return Neighborhood(embedding=embedding, neighbors=scored[: embeddings.neighbors])

    async def lookup(self, text: str) -> Neighborhood:
        """Embed a prompt and find its neighbours. Empty on any failure."""
        if not self._config.embeddings.enabled or self._client is None or not text:
            return Neighborhood()
        embedding = await self._client.embed(text)
        if embedding is None:
            return Neighborhood()
        await self.maybe_refresh()
        neighborhood = self.search(embedding)
        if len(neighborhood.neighbors) < self._config.embeddings.min_neighbors:
            # Too sparse to be evidence — keep the embedding so the request is
            # still logged, but let the skill matrix decide the rung.
            return Neighborhood(embedding=embedding, neighbors=[])
        return neighborhood
