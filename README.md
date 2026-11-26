# Team 14 — Cost-Aware Adaptive RAG

This repository contains the reproducible retrieval baselines for Team 14's CSC 803 research project. The current implementation uses HotpotQA distractor contexts, fixed chunk sizes of 256, 512, and 1024 tokens, BGE-M3 embeddings, FAISS inner-product indexes, and top-5 retrieval.

The research investigates whether adaptive chunk-size selection can improve the trade-off between retrieval quality, token usage, and response latency compared to fixed-size retrieval approaches.

This is a development baseline, not an official HotpotQA benchmark implementation. The initial sample is small and the corpus is a pooled, deduplicated collection of contexts attached to the selected questions. Answers and supporting-fact annotations are retained only as evaluation metadata and are never included in indexed text.

## Repository layout

```text
.
├── notebooks/
│   ├── 01_baseline_512.ipynb
│   └── 02_fixed_size_indexes.ipynb
├── src/
│   ├── chunking.py       # Token-based chunking and source-sentence metadata
│   ├── data.py           # HotpotQA loading and corpus preparation
│   ├── embeddings.py     # BGE-M3 wrapper
│   ├── evaluation.py     # Lightweight retrieval metrics
│   ├── generation.py     # Reserved for the later generation phase
│   ├── retrieval.py      # FAISS index construction and search
│   └── router.py         # Reserved for the later adaptive-router phase
├── tests/
├── literature/related_work.md
├── indexes/              # Local generated indexes; ignored by Git
└── results/
    ├── fixed_size_retrieval_results.csv
    ├── fixed_size_retrieval_results.json
    └── fixed_size_retrieval_summary.csv
```

## Colab-first setup

Google Colab is the primary environment for model and dataset work. Local IDEs are used for editing and testing the reusable Python modules. The notebooks mount Google Drive and write generated indexes, metadata, and retrieval outputs there so they survive runtime resets.

1. Push or clone this repository to GitHub.
2. Open `notebooks/01_baseline_512.ipynb` or `notebooks/02_fixed_size_indexes.ipynb` in Colab.
3. Set `GITHUB_REPO_URL` in the setup cell to your repository URL, or upload/open the notebook from an already-cloned checkout.
4. Run the cells in order and approve the Google Drive mount.

The notebooks select CUDA when `torch.cuda.is_available()` is true and otherwise use CPU. They also place Hugging Face caches under the Drive project directory. This avoids downloading models or datasets during local development.

## Local setup

Python 3.10+ is recommended. Create an environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate          # macOS/Linux
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For Windows PowerShell, activate the environment using:

```powershell
.\.venv\Scripts\Activate.ps1
```

`faiss-cpu`, `sentence-transformers`, and the Hugging Face packages may download model or package assets. No download is performed by the repository tests. The first baseline run downloads the HotpotQA dataset and `BAAI/bge-m3` model into the configured Colab/Drive cache, so run it only when network access and sufficient Drive space are available. A GPU is helpful but not required; BGE-M3 inference and indexing can run on CPU, albeit slowly.

## Run the baseline

The original baseline notebook is the primary Colab demonstration. It loads a deterministic 50-question HotpotQA distractor sample, builds a pooled document corpus, chunks documents at 512 tokenizer tokens, generates normalized BGE-M3 embeddings, builds and saves a FAISS index plus JSON metadata, and retrieves the top five chunks for a question.

The index is saved as `hotpotqa_512/index.faiss` with `metadata.json` under the Drive artifact directory, and retrieval output is saved as JSON.

### Milestone 1 output format

The baseline demonstrates evidence retrieval, not answer generation. Its output should have this shape; the text and scores below are illustrative only:

```text
Input question
Which university did the author of a particular book attend?

Retrieved evidence
Chunk 1 — Wikipedia article | Rank 1
Relevant paragraph about the author's education...

Chunk 2 — Wikipedia article | Rank 2
Relevant paragraph about the book and author...
```

The notebook prints the actual question, rank, similarity score, title, chunk ID, token count, and retrieved text after the Colab pipeline runs.

For an offline smoke test, use `WhitespaceTokenizer` and synthetic embeddings from the tests. That checks parsing, chunk boundaries, and metadata without downloading anything.

## Fixed-size retrieval experiments

The fixed-size retrieval experiments extend the original 512-token baseline by constructing and evaluating three separate FAISS indexes using chunk sizes of 256, 512, and 1024 tokens.

The purpose of these experiments is to establish fixed-size retrieval baselines for comparison with the proposed adaptive chunk-routing approach.

### Experimental configuration

| Component | Configuration |
|---|---|
| Dataset | HotpotQA distractor |
| Dataset split | Train |
| Sample size | 50 questions |
| Random seed | 14 |
| Pooled documents | 492 |
| Chunk sizes | 256, 512, 1024 tokens |
| Embedding model | BAAI/bge-m3 |
| Embedding dimensions | 1024 |
| Vector index | FAISS IndexFlatIP |
| Retrieval method | Top-5 similarity search |
| Execution environment | Google Colab |
| GPU | NVIDIA T4 |
| Artifact storage | Google Drive |

All three configurations use the same dataset sample, document corpus, embedding model, and retrieval settings to maintain consistency across the experiments.

### Chunk-size analysis

| Metric | 256 tokens | 512 tokens | 1024 tokens |
|---|---:|---:|---:|
| Total chunks | 536 | 492 | 492 |
| Minimum chunk length | 9 | 13 | 13 |
| Maximum chunk length | 255 | 487 | 487 |
| Average chunk length | 125.30 | 136.50 | 136.50 |
| Total chunk tokens | 67,159 | 67,159 | 67,159 |

The 256-token configuration produced more chunks because some documents exceeded its token limit.

The 512-token and 1024-token configurations produced identical chunk contents because the documents in the pilot corpus did not exceed the 512-token limit.

### Evaluation metrics

- Supporting-fact recall@5: The proportion of labeled supporting facts covered by the five retrieved chunks, based on source-sentence metadata.
- Retrieved token count: The total number of tokenizer tokens contained in the retrieved chunks.
- Retrieval latency: The elapsed time required to embed the query and perform the FAISS similarity search.

### Experimental results

| Metric | 256 tokens | 512 tokens | 1024 tokens |
|---|---:|---:|---:|
| Questions evaluated | 50 | 50 | 50 |
| Mean supporting-fact recall@5 | 0.925 | 0.925 | 0.925 |
| Mean retrieved tokens | 595.20 | 626.46 | 626.46 |
| Mean retrieval latency (ms) | 70.73 | 56.41 | 55.80 |

### Key findings

- All three fixed-size configurations achieved a mean supporting-fact recall@5 of 92.5%.
- Question-level comparisons showed identical supporting-fact recall across all 50 evaluated questions.
- The 256-token configuration reduced mean retrieved token usage by approximately 4.99% compared to the 512-token and 1024-token configurations without reducing measured supporting-fact recall.
- The 256-token configuration recorded higher mean retrieval latency despite retrieving fewer tokens.
- The 512-token and 1024-token configurations produced identical chunk contents and retrieved token counts in the pilot experiment.
- The current results establish fixed-size retrieval baselines but do not demonstrate an advantage from adaptive routing.

### Experimental artifacts

The implementation and experimental outputs are available in the following files:

- [Fixed-size retrieval notebook](notebooks/02_fixed_size_indexes.ipynb)
- [Detailed retrieval results (CSV)](results/fixed_size_retrieval_results.csv)
- [Detailed retrieval results (JSON)](results/fixed_size_retrieval_results.json)
- [Retrieval summary (CSV)](results/fixed_size_retrieval_summary.csv)

The detailed results contain 150 retrieval evaluations, representing 50 questions evaluated across three chunk-size configurations.

Generated FAISS indexes and associated metadata are stored in Google Drive and are not committed to GitHub.

## HotpotQA compatibility

Two compatibility fixes were implemented in `src/data.py`:

- Updated the dataset identifier to `hotpotqa/hotpot_qa`.
- Added support for dictionary-based context formatting while preserving support for the original list-based format.

The fixed-size retrieval notebook retains the temporary compatibility code cell used during the original Colab experiment to preserve the experimental workflow.

## Testing

Install `pytest` if it is not already available, then run:

```bash
python -m pytest -q
```

The tests cover the local data schema, chunk size and boundary behavior, source metadata, and a dependency-free retrieval metadata path.

An additional test verifies dictionary-based HotpotQA context parsing.

Current local test results: 4 passed.

Model downloads, BGE-M3 inference, HotpotQA network loading, and FAISS persistence are not run automatically.

## Reproducibility and limitations

- The sample seed, model name, chunk sizes, top-k, and corpus policy are recorded in the notebooks.
- The corpus is pooled from distractor contexts for the sampled questions; this must be reported alongside any result.
- The experiments use a small 50-question development sample and are not an official HotpotQA benchmark.
- The 512-token and 1024-token configurations produced identical chunk contents, limiting the comparison between these two configurations.
- Supporting-fact recall measures evidence coverage and does not directly measure generated answer accuracy.
- Retrieved token count is used as a proxy for context cost rather than actual monetary cost.
- Retrieval latency measurements are preliminary and may be influenced by the execution environment and fixed evaluation order.
- The original notebook outputs were generated using the temporary compatibility fixes. The permanent source changes have passed local unit tests, but the full notebook has not been rerun against those changes.
- Adaptive routing, semantic chunking, Qwen generation, and full end-to-end evaluation remain future work.

## Next steps

- Evaluate whether a larger or revised document corpus provides more meaningful differences between chunk-size configurations.
- Implement the adaptive chunk-size router.
- Integrate Qwen answer generation.
- Evaluate answer quality, token usage, and end-to-end response latency.
- Compare adaptive routing against the fixed-size retrieval baselines.