"""Embedding-based semantic document chunking.

The chunker uses cosine similarity between adjacent sentences as a boundary
signal, while retaining a hard token limit so retrieved context remains
bounded.  Model loading is deliberately left to the caller; any embedder with
an ``embed_texts`` method and a tokenizer can be used.
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .chunking import Chunk, Tokenizer
from .data import Document


class SentenceEmbedder(Protocol):
    tokenizer: Tokenizer

    def embed_texts(self, texts: list[str], batch_size: int = 16) -> np.ndarray: ...


@dataclass(frozen=True)
class SemanticChunkingConfig:
    similarity_threshold: float = 0.7
    max_tokens: int = 512
    batch_size: int = 16

    def __post_init__(self) -> None:
        if not 0 <= self.similarity_threshold <= 1:
            raise ValueError("similarity_threshold must be between 0 and 1")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))


def _split_long_sentence(
    document: Document,
    sentence_index: int,
    token_ids: list[int],
    tokenizer: Tokenizer,
    chunk_size: int,
    chunk_number: int,
) -> tuple[list[Chunk], int]:
    chunks: list[Chunk] = []
    for start in range(0, len(token_ids), chunk_size):
        ids = token_ids[start : start + chunk_size]
        chunks.append(
            Chunk(
                chunk_id=f"{document.document_id}:{chunk_number}",
                document_id=document.document_id,
                title=document.title,
                text=tokenizer.decode(ids),
                token_count=len(ids),
                source_sentence_indices=(sentence_index,),
                source_question_ids=document.source_question_ids,
            )
        )
        chunk_number += 1
    return chunks, chunk_number


def semantic_chunk_document(
    document: Document,
    embedder: SentenceEmbedder,
    config: SemanticChunkingConfig | None = None,
    sentence_embeddings: np.ndarray | None = None,
) -> list[Chunk]:
    """Chunk one document at semantic boundaries under a token budget."""
    config = config or SemanticChunkingConfig()
    if not document.sentences:
        return []

    tokenizer = embedder.tokenizer
    token_ids = [tokenizer.encode(sentence, add_special_tokens=False) for sentence in document.sentences]
    embeddings = np.asarray(
        sentence_embeddings
        if sentence_embeddings is not None
        else embedder.embed_texts(document.sentences, batch_size=config.batch_size),
        dtype="float32",
    )
    if len(embeddings) != len(document.sentences):
        raise ValueError("The embedder must return one vector per sentence")

    boundaries = {
        index + 1
        for index in range(len(document.sentences) - 1)
        if _cosine_similarity(embeddings[index], embeddings[index + 1]) < config.similarity_threshold
    }

    chunks: list[Chunk] = []
    current_sentences: list[str] = []
    current_indices: list[int] = []
    current_tokens = 0
    chunk_number = 0

    def flush() -> None:
        nonlocal current_sentences, current_indices, current_tokens, chunk_number
        if not current_sentences:
            return
        chunks.append(
            Chunk(
                chunk_id=f"{document.document_id}:{chunk_number}",
                document_id=document.document_id,
                title=document.title,
                text=" ".join(current_sentences),
                token_count=current_tokens,
                source_sentence_indices=tuple(current_indices),
                source_question_ids=document.source_question_ids,
            )
        )
        chunk_number += 1
        current_sentences, current_indices, current_tokens = [], [], 0

    for index, (sentence, sentence_ids) in enumerate(zip(document.sentences, token_ids)):
        if len(sentence_ids) > config.max_tokens:
            flush()
            long_chunks, chunk_number = _split_long_sentence(
                document, index, sentence_ids, tokenizer, config.max_tokens, chunk_number
            )
            chunks.extend(long_chunks)
            continue

        semantic_boundary = index in boundaries and bool(current_sentences)
        exceeds_budget = current_tokens + len(sentence_ids) > config.max_tokens
        if semantic_boundary or exceeds_budget:
            flush()
        current_sentences.append(sentence)
        current_indices.append(index)
        current_tokens += len(sentence_ids)

    flush()
    return chunks


def semantic_chunk_documents(
    documents: list[Document],
    embedder: SentenceEmbedder,
    config: SemanticChunkingConfig | None = None,
) -> list[Chunk]:
    """Chunk documents independently while preserving document order."""
    config = config or SemanticChunkingConfig()
    sentence_texts = [sentence for document in documents for sentence in document.sentences]
    if not sentence_texts:
        return []
    all_embeddings = np.asarray(
        embedder.embed_texts(sentence_texts, batch_size=config.batch_size), dtype="float32"
    )
    if len(all_embeddings) != len(sentence_texts):
        raise ValueError("The embedder must return one vector per sentence")

    chunks: list[Chunk] = []
    offset = 0
    for document in documents:
        count = len(document.sentences)
        chunks.extend(
            semantic_chunk_document(
                document,
                embedder,
                config,
                sentence_embeddings=all_embeddings[offset : offset + count],
            )
        )
        offset += count
    return chunks
