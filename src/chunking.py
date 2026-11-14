"""Token-based document chunking with source sentence provenance."""

from dataclasses import dataclass
from typing import Protocol
from .data import Document


class Tokenizer(Protocol):
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]: ...
    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str: ...


class WhitespaceTokenizer:
    """Deterministic tokenizer used for tests and offline smoke tests."""
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return list(range(len(text.split())))

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        return " ".join(f"token_{i}" for i in token_ids)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    title: str
    text: str
    token_count: int
    source_sentence_indices: tuple[int, ...]
    source_question_ids: tuple[str, ...]


def chunk_document(document: Document, tokenizer: Tokenizer, chunk_size: int = 512) -> list[Chunk]:
    """Split a document into non-overlapping chunks and preserve sentence spans."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    sentence_tokens = [tokenizer.encode(sentence, add_special_tokens=False) for sentence in document.sentences]
    chunks, current_ids, current_sentences = [], [], []
    chunk_number = 0

    def flush() -> None:
        nonlocal chunk_number, current_ids, current_sentences
        if not current_ids:
            return
        chunks.append(Chunk(f"{document.document_id}:{chunk_number}", document.document_id, document.title,
                            tokenizer.decode(current_ids), len(current_ids), tuple(current_sentences), document.source_question_ids))
        chunk_number += 1
        current_ids, current_sentences = [], []

    for sentence_index, tokens in enumerate(sentence_tokens):
        if current_ids and len(current_ids) + len(tokens) > chunk_size:
            flush()
        if len(tokens) <= chunk_size:
            current_ids.extend(tokens)
            current_sentences.append(sentence_index)
        else:
            for start in range(0, len(tokens), chunk_size):
                current_ids, current_sentences = tokens[start:start + chunk_size], [sentence_index]
                flush()
    flush()
    return chunks


def chunk_documents(documents: list[Document], tokenizer: Tokenizer, chunk_size: int = 512) -> list[Chunk]:
    return [chunk for document in documents for chunk in chunk_document(document, tokenizer, chunk_size)]
