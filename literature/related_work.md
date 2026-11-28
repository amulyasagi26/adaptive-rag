# Related work notes

The project compares fixed-size retrieval with semantic chunking and a transparent query-time cost-aware router. The literature section should position these components separately: chunk construction, query-adaptive granularity selection, and grounded answer generation.

The initial comparison set includes:

- [Mix-of-Granularity](https://arxiv.org/abs/2406.00456) (Zhong et al.; later published at COLING 2025), which dynamically selects knowledge-source granularity with a query router.
- [SmartChunk Retrieval](https://arxiv.org/abs/2602.22225), a query-aware chunk-planning and compression approach for efficient document RAG.
- [Adaptive Chunking](https://aclanthology.org/2026.lrec-1.903/), which selects chunking methods using document-level intrinsic metrics.
- Semantic chunking and related chunking-strategy evaluations, which provide the non-routing baseline for comparing natural-boundary segmentation against fixed token windows.

The current implementation is a transparent heuristic router, not a reproduction of the learned MoG or SmartChunk planners. The final report should preserve that distinction and record each method's dataset, routing signal, cost objective, and evaluation metrics.
