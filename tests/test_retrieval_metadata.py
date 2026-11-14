import numpy as np
from src.chunking import Chunk
from src.evaluation import supporting_fact_recall


def test_supporting_fact_recall_uses_chunk_metadata():
    chunks = [Chunk("d:0", "d", "Page", "text", 2, (0, 2), ("q",))]
    assert supporting_fact_recall(chunks, [("Page", 0), ("Page", 2)]) == 1.0
    assert np.isclose(supporting_fact_recall(chunks, [("Page", 1)]), 0.0)
