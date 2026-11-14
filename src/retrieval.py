"""FAISS indexing, persistence, and top-k retrieval."""

import json
from dataclasses import asdict
from pathlib import Path
import numpy as np
from .chunking import Chunk


class FaissRetriever:
    def __init__(self, index: object, chunks: list[Chunk]):
        self.index, self.chunks = index, chunks

    @classmethod
    def build(cls, chunks: list[Chunk], embeddings: np.ndarray) -> "FaissRetriever":
        try:
            import faiss
        except ImportError as exc:
            raise ImportError("Install faiss-cpu to build the retrieval index") from exc
        if len(chunks) != len(embeddings):
            raise ValueError("There must be one embedding per chunk")
        index = faiss.IndexFlatIP(int(embeddings.shape[1]))
        index.add(np.asarray(embeddings, dtype="float32"))
        return cls(index, chunks)

    def search(self, query: str, embedder: object, top_k: int = 5) -> list[dict]:
        query_vector = np.asarray([embedder.embed_query(query)], dtype="float32")
        scores, positions = self.index.search(query_vector, min(top_k, len(self.chunks)))
        return [{"score": float(score), "chunk": self.chunks[int(position)]}
                for score, position in zip(scores[0], positions[0]) if position >= 0]

    def save(self, directory: str | Path) -> None:
        try:
            import faiss
        except ImportError as exc:
            raise ImportError("Install faiss-cpu to save the retrieval index") from exc
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / "index.faiss"))
        (directory / "metadata.json").write_text(json.dumps([asdict(chunk) for chunk in self.chunks], indent=2), encoding="utf-8")
