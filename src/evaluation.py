"""Small retrieval-evaluation helpers."""

from .chunking import Chunk


def supporting_fact_recall(retrieved_chunks: list[Chunk], supporting_facts: list[tuple[str, int]]) -> float:
    """Return the fraction of labeled title/sentence pairs covered by retrieved chunks."""
    target = set(supporting_facts)
    if not target:
        return 0.0
    found = {(chunk.title, index) for chunk in retrieved_chunks for index in chunk.source_sentence_indices}
    return len(target & found) / len(target)


def retrieved_token_count(retrieved_chunks: list[Chunk]) -> int:
    """Count the tokenizer tokens carried by a retrieved context."""
    return sum(chunk.token_count for chunk in retrieved_chunks)
