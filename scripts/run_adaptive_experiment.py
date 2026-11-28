#!/usr/bin/env python
"""Run fixed, semantic, adaptive, and optional generation evaluations.

This script is intentionally model-heavy. Use it in Colab or a machine with
enough memory for BGE-M3 and optionally Qwen.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

# Allow direct execution as ``python scripts/run_adaptive_experiment.py``.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.chunking import chunk_documents
from src.data import load_hotpotqa
from src.embeddings import BGEEmbedder
from src.evaluation import (
    answer_exact_match,
    answer_token_f1,
    retrieved_token_count,
    supporting_fact_recall,
)
from src.generation import QwenGenerator
from src.retrieval import FaissRetriever
from src.router import AdaptiveRouter
from src.semantic_chunking import SemanticChunkingConfig, semantic_chunk_documents


def build_retriever(chunks, embedder, output_dir: Path, name: str) -> FaissRetriever:
    print(f"Embedding {len(chunks)} chunks for {name}...")
    embeddings = embedder.embed_chunks(chunks)
    retriever = FaissRetriever.build(chunks, embeddings)
    retriever.save(output_dir / name)
    return retriever


def retrieve_metrics(example, results, latency_ms, method, chunk_size=None):
    chunks = [item["chunk"] for item in results]
    return {
        "question_id": example.question_id,
        "question": example.question,
        "method": method,
        "chunk_size": chunk_size,
        "supporting_fact_recall": supporting_fact_recall(chunks, example.supporting_facts),
        "retrieved_tokens": retrieved_token_count(chunks),
        "retrieval_latency_ms": latency_ms,
        "retrieved_titles": [chunk.title for chunk in chunks],
        "similarity_scores": [item["score"] for item in results],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=14)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--cost-weight", type=float, default=0.0005)
    parser.add_argument("--output-dir", type=Path, default=Path("results/adaptive_experiment"))
    parser.add_argument("--generation", action="store_true", help="Also run Qwen answer generation")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    examples, documents = load_hotpotqa(sample_size=args.sample_size, seed=args.seed)
    print(f"Loaded {len(examples)} questions and {len(documents)} pooled documents")

    embedder = BGEEmbedder(device="cuda" if __import__("torch").cuda.is_available() else "cpu")
    fixed_retrievers = {}
    for chunk_size in (256, 512, 1024):
        chunks = chunk_documents(documents, embedder.tokenizer, chunk_size=chunk_size)
        fixed_retrievers[chunk_size] = build_retriever(
            chunks, embedder, args.output_dir / "indexes", f"fixed_{chunk_size}"
        )

    semantic_chunks = semantic_chunk_documents(
        documents,
        embedder,
        SemanticChunkingConfig(similarity_threshold=0.7, max_tokens=512),
    )
    semantic_retriever = build_retriever(
        semantic_chunks, embedder, args.output_dir / "indexes", "semantic"
    )
    router = AdaptiveRouter(
        fixed_retrievers, embedder, top_k=args.top_k, cost_weight=args.cost_weight
    )

    rows = []
    generation_rows = []
    generator = QwenGenerator(device="auto") if args.generation else None
    for index, example in enumerate(examples, start=1):
        print(f"Evaluating {index}/{len(examples)}", end="\r")

        for chunk_size, retriever in fixed_retrievers.items():
            start = time.perf_counter()
            fixed_results = retriever.search(example.question, embedder, args.top_k)
            rows.append(
                retrieve_metrics(
                    example,
                    fixed_results,
                    (time.perf_counter() - start) * 1000,
                    "fixed",
                    chunk_size,
                )
            )

        start = time.perf_counter()
        decision = router.route(example.question)
        rows.append(
            retrieve_metrics(
                example,
                decision.results,
                (time.perf_counter() - start) * 1000,
                "adaptive",
                decision.chunk_size,
            )
        )

        start = time.perf_counter()
        semantic_results = semantic_retriever.search(example.question, embedder, args.top_k)
        rows.append(
            retrieve_metrics(
                example,
                semantic_results,
                (time.perf_counter() - start) * 1000,
                "semantic",
                None,
            )
        )

        if generator is not None:
            start = time.perf_counter()
            answer = generator.generate_answer(example.question, decision.results)
            generation_rows.append(
                {
                    "question_id": example.question_id,
                    "question": example.question,
                    "generated_answer": answer,
                    "ground_truth": example.answer,
                    "answer_exact_match": answer_exact_match(answer, example.answer),
                    "answer_token_f1": answer_token_f1(answer, example.answer),
                    "generation_latency_ms": (time.perf_counter() - start) * 1000,
                }
            )
    print()

    detail_path = args.output_dir / "retrieval_results.json"
    detail_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    details_df = pd.DataFrame(rows)
    details_df.to_csv(args.output_dir / "retrieval_results.csv", index=False)
    summary = (
        details_df.groupby(["method", "chunk_size"], dropna=False)
        .agg(
            questions_evaluated=("question_id", "count"),
            mean_supporting_fact_recall=("supporting_fact_recall", "mean"),
            mean_retrieved_tokens=("retrieved_tokens", "mean"),
            mean_retrieval_latency_ms=("retrieval_latency_ms", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(args.output_dir / "retrieval_summary.csv", index=False)
    print(summary.to_string(index=False))

    if generation_rows:
        generation_df = pd.DataFrame(generation_rows)
        generation_df.to_csv(args.output_dir / "generation_results.csv", index=False)
        print(generation_df[["answer_exact_match", "answer_token_f1", "generation_latency_ms"]].mean())


if __name__ == "__main__":
    main()
