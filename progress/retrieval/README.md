# Retrieval Evaluation

## Dense Retrieval

Initial evaluation of only dense-vector retrieval on the current corpus.

### Setup

* **Corpus:** 50 PDFs / 1,877 chunks
* **Embedding:** `BAAI/bge-base-en-v1.5`
* **Vector DB:** Qdrant
* **Similarity:** Dot product (L2-normalized vectors, equivalent to cosine similarity)
* **Query:** Raw / unprocessed query
* **Initial retrieval:** Top-30

### Evaluation Dataset

30 evaluation questions were generated using an LLM from the extracted document chunks.

Each question is linked to its **exact target chunk ID**, allowing chunk-level retrieval evaluation.

### Dense Retrieval Results

| Metric    |             Result |
| --------- | -----------------: |
| Recall@1  |  **70.0% (21/30)** |
| Recall@3  |  **76.7% (23/30)** |
| Recall@5  |  **90.0% (27/30)** |
| Recall@10 |  **93.3% (28/30)** |
| Recall@30 | **100.0% (30/30)** |

**Recall@K** = whether the correct target chunk appears within the top K retrieved results.

### Dense + Semantic Reranker

The top-30 dense candidates were reranked using `BAAI/bge-reranker-base`.

| Metric   | Dense | + Reranker |
| -------- | ----: | ---------: |
| Recall@1 | 70.0% |  **76.7%** |
| Recall@3 | 76.7% |  **93.3%** |
| Recall@5 | 90.0% |  **96.7%** |

### Conclusion

Dense retrieval already provides a strong candidate pool, retrieving the target chunk within the top 30 for **100% of the evaluation questions**.

The reranker substantially improves ranking, especially at Top-3 and Top-5.

> **Note:** Recall@30 = 100% only means the correct chunk was present in the candidate pool. It does not guarantee that the retrieved context is sufficient to answer the question. That will be evaluated during the generation stage.
