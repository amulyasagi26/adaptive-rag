# Team 14 — Cost-Aware Adaptive RAG

This repository contains the first, reproducible retrieval baseline for Team 14's CSC 803 research project. The current implementation uses HotpotQA distractor contexts, 512-token chunks, BGE-M3 embeddings, a FAISS inner-product index, and top-5 retrieval.

This is a development baseline, not an official HotpotQA benchmark implementation. The initial sample is small and the corpus is a pooled, deduplicated collection of contexts attached to the selected questions. Answers and supporting-fact annotations are retained only as evaluation metadata and are never included in indexed text.

## Repository layout

```text
.
├── notebooks/01_baseline_512.ipynb
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
└── results/              # Local experiment outputs; ignored by Git
```

## Colab-first setup

Google Colab is the primary environment for model and dataset work. IntelliJ/Codex are used for editing and testing the reusable Python modules. The notebook mounts Google Drive and writes generated indexes, metadata, and retrieval outputs there so they survive runtime resets.

1. Push or clone this repository to GitHub.
2. Open `notebooks/01_baseline_512.ipynb` in Colab.
3. Set `GITHUB_REPO_URL` in the setup cell to your repository URL, or upload/open the notebook from an already-cloned checkout.
4. Run the cells in order and approve the Google Drive mount.

The notebook selects CUDA when `torch.cuda.is_available()` is true and otherwise uses CPU. It also places Hugging Face caches under the Drive project directory. This avoids downloading models or datasets to the Mac during local development.

## Local setup

Python 3.10+ is recommended. Create an environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate          # macOS/Linux
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`faiss-cpu`, `sentence-transformers`, and the Hugging Face packages may download model or package assets. No download is performed by the repository tests. The first baseline run downloads the HotpotQA dataset and `BAAI/bge-m3` model into the configured Colab/Drive cache, so run it only when network access and sufficient Drive space are available. A GPU is helpful but not required; BGE-M3 inference and indexing can run on CPU, albeit slowly.

## Run the baseline

The notebook is the primary Colab demonstration. It will load a deterministic 50-question HotpotQA distractor sample, build a pooled document corpus, chunk documents at 512 tokenizer tokens, generate normalized BGE-M3 embeddings, build/save a FAISS index plus JSON metadata, and retrieve the top five chunks for a question. The index is saved as `hotpotqa_512/index.faiss` with `metadata.json` under the Drive artifact directory, and retrieval output is saved as JSON.

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

## Testing

Install `pytest` if it is not already available, then run `python -m pytest -q`. The tests cover the local data schema, chunk size/boundary behavior, source metadata, and a dependency-free retrieval metadata path. Model downloads, BGE-M3 inference, HotpotQA network loading, and FAISS persistence are not run automatically.

## Reproducibility and limitations

- The sample seed, model name, chunk size, top-k, and corpus policy are recorded in the notebook.
- The corpus is pooled from distractor contexts for the sampled questions; this must be reported alongside any result.
- Supporting-fact recall is implemented as a utility, but it is not meaningful until a real HotpotQA sample has been indexed and retrieved.
- The adaptive router, 256/1024 indexes, semantic chunking, Qwen generation, and full experiment runner are intentionally not implemented in this phase.
