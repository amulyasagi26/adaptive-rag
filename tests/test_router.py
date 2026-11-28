import numpy as np

from src.chunking import Chunk
from src.router import AdaptiveRouter


def make_result(score, token_count):
    chunk = Chunk("d:0", "d", "Page", "text", token_count, (0,), ("q",))
    return [{"score": score, "chunk": chunk}]


class FakeRetriever:
    def __init__(self, score, token_count):
        self.results = make_result(score, token_count)
        self.chunks = [self.results[0]["chunk"]]

    def search_vector(self, query_vector, top_k=5):
        assert query_vector.dtype == np.float32
        return self.results


class FakeEmbedder:
    def embed_query(self, query):
        return np.array([1.0, 0.0])


def test_router_prefers_lower_cost_when_confidence_is_close():
    router = AdaptiveRouter(
        {256: FakeRetriever(0.80, 10), 512: FakeRetriever(0.90, 1000)},
        FakeEmbedder(),
        cost_weight=0.0005,
    )

    decision = router.route("question")

    assert decision.chunk_size == 256
    assert len(decision.candidates) == 2
    assert decision.results[0]["score"] == 0.80


def test_router_can_run_without_cost_penalty():
    router = AdaptiveRouter(
        {256: FakeRetriever(0.80, 10), 512: FakeRetriever(0.90, 1000)},
        FakeEmbedder(),
        cost_weight=0.0,
    )

    assert router.route("question").chunk_size == 512
