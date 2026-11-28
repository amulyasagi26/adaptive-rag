"""Cost-aware routing across retrieval indexes with different granularities."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .chunking import Chunk


class VectorRetriever(Protocol):
    chunks: list[Chunk]

    def search_vector(self, query_vector: np.ndarray, top_k: int = 5) -> list[dict]: ...


class QueryEmbedder(Protocol):
    def embed_query(self, query: str) -> np.ndarray: ...


@dataclass(frozen=True)
class RouteCandidate:
    chunk_size: int
    top_score: float
    retrieved_tokens: int
    objective: float


@dataclass(frozen=True)
class RouteDecision:
    chunk_size: int
    results: list[dict]
    candidates: tuple[RouteCandidate, ...]


class AdaptiveRouter:
    """Select a chunk-size index using retrieval confidence and context cost.

    The objective is ``top_score - cost_weight * retrieved_tokens``.  A higher
    ``cost_weight`` favors smaller retrieved contexts; setting it to zero gives
    confidence-only routing.  This is a transparent baseline router rather
    than a trained policy, so its behavior is easy to reproduce and inspect.
    """

    def __init__(
        self,
        retrievers: dict[int, VectorRetriever],
        embedder: QueryEmbedder,
        top_k: int = 5,
        cost_weight: float = 0.0005,
    ) -> None:
        if not retrievers:
            raise ValueError("At least one retriever is required")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if cost_weight < 0:
            raise ValueError("cost_weight cannot be negative")
        self.retrievers = dict(retrievers)
        self.embedder = embedder
        self.top_k = top_k
        self.cost_weight = cost_weight

    def route(self, query: str) -> RouteDecision:
        """Return the selected index, retrieved results, and audit candidates."""
        query_vector = np.asarray(self.embedder.embed_query(query), dtype="float32")
        candidates: list[RouteCandidate] = []
        results_by_size: dict[int, list[dict]] = {}

        for chunk_size, retriever in sorted(self.retrievers.items()):
            results = retriever.search_vector(query_vector, top_k=self.top_k)
            results_by_size[chunk_size] = results
            top_score = float(results[0]["score"]) if results else float("-inf")
            retrieved_tokens = sum(item["chunk"].token_count for item in results)
            candidates.append(
                RouteCandidate(
                    chunk_size=chunk_size,
                    top_score=top_score,
                    retrieved_tokens=retrieved_tokens,
                    objective=top_score - self.cost_weight * retrieved_tokens,
                )
            )

        selected = max(candidates, key=lambda candidate: (candidate.objective, -candidate.chunk_size))
        return RouteDecision(selected.chunk_size, results_by_size[selected.chunk_size], tuple(candidates))

    def search(self, query: str) -> list[dict]:
        """Convenience method returning results from the selected index."""
        return self.route(query).results
