import numpy as np

from src.data import Document
from src.semantic_chunking import SemanticChunkingConfig, semantic_chunk_document


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(range(len(text.split())))

    def decode(self, token_ids, skip_special_tokens=True):
        return " ".join(f"token_{token_id}" for token_id in token_ids)


class FakeEmbedder:
    tokenizer = FakeTokenizer()

    def __init__(self, vectors):
        self.vectors = np.asarray(vectors, dtype="float32")

    def embed_texts(self, texts, batch_size=16):
        assert len(texts) == len(self.vectors)
        return self.vectors


def test_semantic_chunking_splits_at_similarity_boundary_and_keeps_metadata():
    document = Document("d1", "Page", ["one two", "three four", "five"], ("q1",))
    embedder = FakeEmbedder([[1, 0], [1, 0], [0, 1]])

    chunks = semantic_chunk_document(
        document, embedder, SemanticChunkingConfig(similarity_threshold=0.7, max_tokens=10)
    )

    assert [chunk.source_sentence_indices for chunk in chunks] == [(0, 1), (2,)]
    assert [chunk.token_count for chunk in chunks] == [4, 1]
    assert all(chunk.source_question_ids == ("q1",) for chunk in chunks)


def test_semantic_chunking_respects_hard_token_limit():
    document = Document("d1", "Page", ["one two three four five six"], ("q1",))
    chunks = semantic_chunk_document(
        document,
        FakeEmbedder([[1, 0]]),
        SemanticChunkingConfig(max_tokens=4),
    )

    assert [chunk.token_count for chunk in chunks] == [4, 2]
    assert all(chunk.source_sentence_indices == (0,) for chunk in chunks)
