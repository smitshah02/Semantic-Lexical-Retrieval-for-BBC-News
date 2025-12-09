# API Documentation

This document describes the public classes, retrieval components, evaluation functions, and utilities used in the Semantic–Lexical Retrieval System for BBC News Articles.

---

## 1. Lexical Retrieval (`src/retrieve_lexical.py`)

### **TFIDFRetriever**
Retrieves documents using TF-IDF vectors and cosine similarity.

**Factory**
```python
TFIDFRetriever.from_artifacts(artifacts_dir: str) -> TFIDFRetriever
```

**Methods**

```python
search(query: str, k: int = 5) -> list[tuple[str, float]]
```

Returns ranked `(doc_id, score)` pairs.

---

### **BM25Retriever**

Retrieves documents using the BM25 ranking function.

**Factory**

```python
BM25Retriever.from_artifacts(artifacts_dir: str) -> BM25Retriever
```

**Methods**

```python
search(query: str, k: int = 5)
```

---

## 2. Semantic Retrieval (`src/retrieve_semantic.py`)

### **SemanticRetriever**

Semantic retrieval using SentenceTransformer embeddings and cosine similarity. Supports FAISS for fast nearest-neighbor search.

**Factory**

```python
SemanticRetriever.from_artifacts(artifacts_dir: str) -> SemanticRetriever
```

**Methods**

```python
search(query: str, k: int = 5)
```

---

## 3. Hybrid Retrieval (`src/retrieve_hybrid.py`)

### **HybridRetriever**

Combines TF-IDF, BM25, and semantic rankings using Reciprocal Rank Fusion (RRF).

**Constructor**

```python
HybridRetriever(tfidf, bm25, semantic)
```

**Methods**

```python
search(query: str, k: int = 5)
```

---

## 4. Evaluation (`src/evaluate.py`)

### **evaluate_system**

Computes IR metrics for one or more retrieval modes.

**Signature**

```python
evaluate_system(
    queries_path: str,
    qrels_path: str,
    artifacts_dir: str,
    modes: list[str],
    ks: list[int]
) -> pandas.DataFrame
```

Metrics:

* Precision@k
* Recall@k
* nDCG@k

Returns: Pandas DataFrame of scores per mode and per query.

---

## 5. Utilities (`src/utils.py`)

Common helper functions:

```python
read_jsonl(path)
write_jsonl(path, records)
normalize_text(text)
split_into_sentences(text)
create_text_snippet(text, max_chars=300)
load_queries(path)
load_qrels(path)
ensure_dir(path)
```

These support preprocessing, chunking, evaluation, and UI display.

---

## 6. Build Pipeline (`src/build_*.py`)

| Script                | Purpose                                            |
| --------------------- | -------------------------------------------------- |
| `build_corpus.py`     | Converts raw BBC text → normalized CSV             |
| `build_chunks.py`     | Splits text into overlapping sentence-aware chunks |
| `build_lexical.py`    | Builds TF-IDF & BM25 models                        |
| `build_embeddings.py` | Produces embeddings + FAISS index                  |

Each script includes a `main()` with CLI argument parsing.

---

## Notes

* All retrieval models return ranked document IDs with scores.
* All artifacts are reproducible from raw data.
* Notebook interface uses these APIs directly.