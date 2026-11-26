"""HotpotQA loading and development-corpus preparation."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class HotpotExample:
    question_id: str
    question: str
    answer: str
    contexts: list[dict[str, Any]]
    supporting_facts: list[tuple[str, int]]


@dataclass(frozen=True)
class Document:
    document_id: str
    title: str
    sentences: list[str]
    source_question_ids: tuple[str, ...]


def parse_hotpot_record(record: dict[str, Any]) -> HotpotExample:
    """Parse one record while keeping answers/labels out of document text."""
    raw_context = record["context"]

    if isinstance(raw_context, dict):
        context_pairs = zip(raw_context["title"], raw_context["sentences"])
    else:
        context_pairs = raw_context

    contexts = [{"title": title, "sentences": list(sentences)} for title, sentences in context_pairs]
    labels = record.get("supporting_facts", {})
    if isinstance(labels, dict) and "title" in labels and "sent_id" in labels:
        supporting = [(str(title), int(index)) for title, index in zip(labels["title"], labels["sent_id"])]
    else:
        supporting = [(str(title), int(index)) for title, indices in labels.items() for index in indices]
    return HotpotExample(
        question_id=str(record.get("_id", record.get("id", ""))),
        question=str(record["question"]), answer=str(record["answer"]),
        contexts=contexts, supporting_facts=supporting,
    )


def load_hotpotqa(sample_size: int = 50, seed: int = 14, split: str = "train") -> tuple[list[HotpotExample], list[Document]]:
    """Load a deterministic sample and return examples plus a pooled context corpus."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("Install datasets to load HotpotQA: python -m pip install datasets") from exc
    dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split)
    sample_size = min(sample_size, len(dataset))
    dataset = dataset.shuffle(seed=seed).select(range(sample_size))
    examples = [parse_hotpot_record(row) for row in dataset]
    pooled: dict[str, dict[str, Any]] = {}
    for example in examples:
        for context in example.contexts:
            key = f"{context['title']}::{context['sentences']!r}"
            item = pooled.setdefault(key, {"title": context["title"], "sentences": context["sentences"], "questions": []})
            item["questions"].append(example.question_id)
    documents = [Document(str(i), item["title"], item["sentences"], tuple(sorted(set(item["questions"])))) for i, item in enumerate(pooled.values())]
    return examples, documents
