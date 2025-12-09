# Semantic-Lexical-Retrieval-for-BBC-News Articles

This project implements a **hybrid information retrieval (IR) system** for BBC News articles using both **lexical** and **semantic** search techniques. It supports:

* **TF-IDF retrieval**
* **BM25 retrieval**
* **Semantic embedding retrieval** using SentenceTransformers
* **Hybrid retrieval** using Reciprocal Rank Fusion (RRF)
* **Interactive Jupyter Notebook interface**
* **End-to-end pipeline**: raw BBC dataset → corpus → chunks → embeddings → IR demo
* **IR evaluation** with Precision@k, Recall@k, and nDCG@k

All code is fully modularized, with each file handling a self-contained part of the pipeline.

---

# 1. Project Structure

```
Semantic-Lexical-Retrieval-for-News/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── utils.py
│   ├── build_corpus.py
│   ├── build_chunks.py
│   ├── build_lexical.py
│   ├── build_embeddings.py
│   ├── retrieve_lexical.py
│   ├── retrieve_semantic.py
│   ├── retrieve_hybrid.py
│   ├── evaluate.py
|   └── demo.ipynb        # Interactive IR interface
│            
│
├── data/
│   ├── queries.tsv
│   └── qrels.tsv
│
├── raw_data/
│   └── bbc/
│       ├── business/
│       ├── entertainment/
│       ├── politics/
│       ├── sport/
│       └── tech/
│
└── artifacts/
    ├── bbc_corpus.csv
    ├── bbc_chunks.jsonl
    ├── tfidf.pkl
    ├── bm25.pkl
    ├── embeddings.npy
    ├── doc_ids.json
    └── faiss.index
```

---

# 2. Installation

## 2.1 Create a virtual environment

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

## 2.2 Install dependencies

```bash
pip install -r requirements.txt
```

This installs:

* scikit-learn
* rank-bm25
* sentence-transformers
* faiss-cpu
* pandas, numpy
* ipywidgets
* notebook-related tools

---

# 3. Data Description

## 3.1 Raw BBC Dataset (`raw_data/bbc/`)

Contains 5 topic folders:

```
business/
entertainment/
politics/
sport/
tech/
```

Each folder contains plain-text news files.

Used as input for the preprocessing pipeline.

---

## 3.2 Processed Data (`artifacts/`)

| File               | Description                                          |
| ------------------ | ---------------------------------------------------- |
| `bbc_corpus.csv`   | Normalized corpus: article ID, title, text, category |
| `bbc_chunks.jsonl` | Sentence-aware chunks (~450–600 words, with overlap) |
| `tfidf.pkl`        | TF-IDF model + sparse matrix + doc IDs               |
| `bm25.pkl`         | BM25 tokenized corpus + doc IDs                      |
| `embeddings.npy`   | Semantic embeddings for each chunk                   |
| `doc_ids.json`     | Chunk ID list aligned with embeddings                |
| `faiss.index`      | Optional FAISS index for fast vector search          |

---

# 4. Build Pipeline

All build scripts live in `src/`.

## 4.1 Build corpus

```bash
python -m src.build_corpus \
    --input_dir raw_data/bbc \
    --output_csv artifacts/bbc_corpus.csv
```

## 4.2 Chunk corpus into sentence-aware segments

```bash
python -m src.build_chunks \
    --input_csv artifacts/bbc_corpus.csv \
    --output_jsonl artifacts/bbc_chunks.jsonl \
    --target_words 450 \
    --overlap_words 100 \
    --max_single 550
```

## 4.3 Create lexical IR indices (TF-IDF & BM25)

```bash
python -m src.build_lexical \
    --input_jsonl artifacts/bbc_chunks.jsonl \
    --out_dir artifacts
```

## 4.4 Create semantic embeddings (+ FAISS index)

```bash
python -m src.build_embeddings \
    --input_jsonl artifacts/bbc_chunks.jsonl \
    --out_dir artifacts \
    --model_name all-MiniLM-L6-v2 \
    --use_faiss
```

---

# 5. Retrieval API

## TF-IDF & BM25 (src/retrieve_lexical.py)

```python
from src.retrieve_lexical import TFIDFRetriever, BM25Retriever

tfidf = TFIDFRetriever.from_artifacts("artifacts")
bm25 = BM25Retriever.from_artifacts("artifacts")

results = tfidf.search("climate change impact", k=5)
```

---

## Semantic Retrieval (src/retrieve_semantic.py)

```python
from src.retrieve_semantic import SemanticRetriever

semantic = SemanticRetriever.from_artifacts("artifacts")
results = semantic.search("global warming economics", k=5)
```

---

## Hybrid RRF Retrieval (src/retrieve_hybrid.py)

```python
from src.retrieve_hybrid import HybridRetriever

hybrid = HybridRetriever(tfidf, bm25, semantic)
results = hybrid.search("economic impact of climate change", k=5)
```

---

# 6. Evaluation

Evaluate retrieval performance using Precision@k, Recall@k, and nDCG@k:

```bash
python -m src.evaluate \
    --queries data/queries.tsv \
    --qrels data/qrels.tsv \
    --artifacts_dir artifacts \
    --modes tfidf bm25 semantic hybrid \
    --k 5 10
```

Outputs a CSV under `artifacts/eval_results_*.csv`.

---

# 7. Interactive Notebook Demo

The main user-facing interface is:

```
src/demo.ipynb
```

It provides:

* A query input box
* Method selector (TF-IDF, BM25, Semantic, Hybrid, or All Methods)
* Adjustable top-k slider
* Optional snippet and score toggles
* Rich HTML-formatted search results
* Search statistics (only in comparison mode)

Launch the notebook:

```bash
jupyter notebook src/demo.ipynb
```

or open directly in VS Code.

---

# 8. Code Overview

| File                   | Purpose                                                              |
| ---------------------- | -------------------------------------------------------------------- |
| `build_corpus.py`      | Converts raw BBC text → structured CSV                               |
| `build_chunks.py`      | Splits articles into overlapping, sentence-aware chunks              |
| `build_lexical.py`     | Builds TF-IDF & BM25 lexical indices                                 |
| `build_embeddings.py`  | Generates semantic embeddings + FAISS index                          |
| `retrieve_lexical.py`  | TF-IDF / BM25 search classes                                         |
| `retrieve_semantic.py` | Embedding-based semantic search                                      |
| `retrieve_hybrid.py`   | Hybrid search using reciprocal rank fusion                           |
| `evaluate.py`          | Computes Precision@k, Recall@k, nDCG@k                               |
| `utils.py`             | JSONL utilities, normalization, sentence splitting, snippet creation |

Each file contains docstrings and comments for clarity.

---

# 9. Documentation

Additional files:

* `docs/API.md` — Retrieval API documentation
* `docs/DATA.md` — Dataset & artifacts explanation

---

# 10. Acknowledgements

* BBC dataset used for academic purposes
* SentenceTransformers for semantic embedding models
* FAISS for efficient vector search
* scikit-learn for TF-IDF
* rank-bm25 for BM25 scoring

