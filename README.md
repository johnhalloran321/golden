# Golden

[![arXiv](https://img.shields.io/badge/arXiv-2505.23634-b31b1b.svg)](https://arxiv.org/abs/2505.23634)
[![arXiv](https://img.shields.io/badge/arXiv-2605.11217-b31b1b.svg)](https://arxiv.org/abs/2605.11217)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](pyproject.toml)

Custom LangChain retrieval APIs for rapid ML experimentation: a drop-in, GPU-aware replacement for LangChain's Chroma vector store (`Golden_Retriever`), with a HuggingFace-`transformers`-backed embedding function. An optional FAISS-backed alternative (`Golden_Faiss_Retriever`) is also included for when you want an in-process index with no separate DB backend.

Golden is the retrieval backbone behind **RAG-Pref**, the training-free preference alignment method from [*"Leveraging RAG for Training-Free Alignment of LLMs"*](https://arxiv.org/abs/2605.11217), a companion to [*"MCP Safety Training: Learning to Refuse Falsely Benign MCP Exploits using Improved Preference Alignment"*](https://arxiv.org/abs/2505.23634) — see [mcp_safety_training](https://github.com/johnhalloran321/mcp_safety_training) for the full training/eval pipeline that uses it.

## Why Golden?

`Golden_Retriever` (and, optionally, `Golden_Faiss_Retriever`) are subclasses of LangChain's `Chroma`/`FAISS`, so they're used exactly like the originals, but with a few gaps in the underlying libraries closed:

- **OOM-safe embedding batch size.** `batch_size="auto"` starts from `max_batch_size` and backs off automatically on CUDA OOM (via a `find_executable_batch_size` decorator adapted from HuggingFace Accelerate), instead of crashing partway through embedding a large corpus.
- **Extensive hardware optimizations.** Embedding a corpus leverages `torch` throughout, with GPU-aware batching tuned for throughput rather than naive per-document embedding calls.
- **ChromaDB's undocumented upsert limit.** Chroma silently truncates or errors past ~5,461 vectors per `upsert` call on some backends; `add_texts`/`from_texts` batch above that threshold automatically.
- **GPU-aware by default.** Auto-detects CUDA → MPS (Apple Silicon) → CPU, rather than hardcoding a device.
- **A small embedding model zoo** (`sentence-transformers/all-MiniLM-L6-v2`, `togethercomputer/m2-bert-80M-8k-retrieval`, `thenlper/gte-large`, `Alibaba-NLP/gte-Qwen2-1.5B-instruct`) with the correct pooling/normalization wired up per-model, so switching embedding models doesn't require re-deriving the right pooling strategy.
- **Persist-and-reload without re-specifying embedding config.** `persist_directory` saves the embedding model/tokenizer settings alongside the index; `Golden_Retriever.load(path)` picks them back up automatically.

## Install

Install the package locally after cloning:

```bash
git clone https://github.com/johnhalloran321/golden.git
cd golden
make install
# OR
pip install .
```

### Development

Install in editable mode, with test/lint dependencies:

```bash
make develop
# OR
pip install -e .[all]
```

Run the test suite:

```bash
make test
# OR
pytest tests/
```

## Quickstart

```python
from golden.golden_retriever import Golden_Retriever

texts = ["A file system MCP tool call.", "A benign request to read a config file."]
db = Golden_Retriever.from_texts(texts, similarity_fn="l2", persist_directory="my_db")

for doc, score in db.similarity_search_with_score("read a yaml config", k=2):
    print(score, doc.page_content)

# Reload later without re-specifying the embedding model
db = Golden_Retriever.load("my_db")
```

Use it wherever and however you'd use plain ChromaDB via LangChain. The optional `Golden_Faiss_Retriever` (see below) implements the same `from_texts`/`from_documents`/`similarity_search_with_score` interface if you'd rather not run a separate DB backend.

## API

### `Golden_Retriever` (`golden.golden_retriever`)

A `langchain_community.vectorstores.Chroma` subclass.

```
similarity_fn = "l2"        # l2 = euclidean, cosine = cosine, ip = inner product (unnormalized cosine)
language = "python"         # chunking language: ['cpp', 'go', 'java', 'kotlin', 'js', 'ts', 'php', 'proto',
                             #  'python', 'rst', 'ruby', 'rust', 'scala', 'swift', 'markdown', 'latex',
                             #  'html', 'sol', 'csharp', 'cobol']
```

### `Golden_Faiss_Retriever` (`golden.golden_faiss_retriever`) — optional

The same API as `Golden_Retriever`, backed by FAISS instead of Chroma, for when you want an in-process index with no separate DB backend. This is not part of the standard install — `faiss` is not pulled in by `pip install .`, `make install`, or `make develop`. Install it explicitly:

```bash
pip install golden_retriever[faiss]
# or, from a local clone:
pip install -e .[faiss]
```

### `Embedding` (`golden.golden_embeddings`)

A HuggingFace-`transformers`-backed LangChain `Embeddings` implementation.

**Constructor:**

```
model_id = "togethercomputer/m2-bert-80M-8k-retrieval"  # transformers model id
tokenizer_id = "bert-base-uncased"                      # transformers tokenizer id
max_seq_length = 8192                                   # max context window
max_batch_size = 64                                      # max batch size to consider
batch_size = "auto"                                      # starting from max_batch_size, auto-detect the
                                                          #  largest batch size that fits in memory
quantization_config = None                                # None, or a BitsAndBytesConfig instance
```

**`embed_documents`:**

```
language = "python"     # chunking language (see list above)
chunk_size = 8192
chunk_overlap = 20
batch_size = 32          # set to "auto" to auto-detect the batch size which fits in memory
num_workers = 3          # CPU workers for data loading
device = "cuda"          # "cuda", "mps", or "cpu"; auto-detected if not specified
```

## Citation

If you use Golden, please cite the paper it was built for:

```bibtex
@article{halloran2026ragpref,
  title   = {Leveraging RAG for Training-Free Alignment of LLMs},
  author  = {Halloran, John},
  journal = {arXiv preprint arXiv:2605.11217},
  year    = {2026}
}

@article{halloran2025mcpsafety,
  title   = {MCP Safety Training: Learning to Refuse Falsely Benign MCP Exploits using Improved Preference Alignment},
  author  = {Halloran, John},
  journal = {arXiv preprint arXiv:2505.23634},
  year    = {2025}
}
```

## License

Apache License 2.0 — see [LICENSE](LICENSE). `golden/utils.py` includes code adapted from [HuggingFace Accelerate](https://github.com/huggingface/accelerate), also Apache 2.0.
