"""Small retrieval and answer-quality evaluation helpers."""

import re
import string

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


def normalize_answer(answer: str) -> str:
    """Normalize an answer for simple exact-match and token-F1 scoring."""
    answer = answer.lower()
    answer = answer.translate(str.maketrans("", "", string.punctuation))
    answer = re.sub(r"\b(a|an|the)\b", " ", answer)
    return " ".join(answer.split())


def answer_exact_match(prediction: str, reference: str) -> float:
    """Return 1 when normalized prediction and reference are identical."""
    return float(normalize_answer(prediction) == normalize_answer(reference))


def answer_token_f1(prediction: str, reference: str) -> float:
    """Return token-level F1 after standard answer normalization."""
    predicted_tokens = normalize_answer(prediction).split()
    reference_tokens = normalize_answer(reference).split()
    if not predicted_tokens or not reference_tokens:
        return float(predicted_tokens == reference_tokens)

    overlap = sum((min(predicted_tokens.count(token), reference_tokens.count(token)) for token in set(predicted_tokens)))
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)
