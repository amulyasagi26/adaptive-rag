import numpy as np

from src.data import Document
from src.semantic_chunking import semantic_chunk_documents


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(range(len(text.split())))

    def decode(self, token_ids, skip_special_tokens=True):
        return " ".join(f"token_{i}" for i in token_ids)


class CountingEmbedder:
    tokenizer = Tokenizer()

    def __init__(self):
        self.calls = 0
        self.batch_sizes = []

    def embed_texts(self, texts, batch_size=16):
        self.calls += 1
        self.batch_sizes.append(len(texts))
        return np.ones((len(texts), 2), dtype="float32")


def test_semantic_chunk_documents_batches_all_sentences_once():
    documents = [
        Document("d1", "One", ["one sentence"], ("q1",)),
        Document("d2", "Two", ["another sentence", "more text"], ("q2",)),
    ]
    embedder = CountingEmbedder()

    chunks = semantic_chunk_documents(documents, embedder)

    assert len(chunks) == 2
    assert embedder.calls == 1
    assert embedder.batch_sizes == [3]
