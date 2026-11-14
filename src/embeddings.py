"""BGE-M3 embedding generation."""

import numpy as np


class BGEEmbedder:
    """Thin wrapper around SentenceTransformers; model loading is lazy until construction."""
    def __init__(self, model_name: str = "BAAI/bge-m3", device: str | None = None):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError("Install sentence-transformers to use BGE-M3") from exc
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.tokenizer = self.model.tokenizer

    def embed_texts(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        vectors = self.model.encode(texts, batch_size=batch_size, convert_to_numpy=True,
                                    normalize_embeddings=True, show_progress_bar=True)
        return np.asarray(vectors, dtype="float32")

    def embed_chunks(self, chunks: list[object], batch_size: int = 16) -> np.ndarray:
        return self.embed_texts([chunk.text for chunk in chunks], batch_size)

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_texts([query], batch_size=1)[0]
