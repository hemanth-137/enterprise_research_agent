# RAG Evaluation Report

**Benchmark:** Open-RAGBench  
**Evaluation sample:** 100 queries

> Results are based on the evaluated sample and should not be interpreted as full-benchmark performance.

### RAG Quality Results

| Metric | Result |
| -------- | -----: |
| Fully Correct | **67.0% (67/100)** |
| Correct or Partial | **80.0% (80/100)** |
| Fully Faithful | **84.0% (84/100)** |
| Sufficient Context | **90.0% (90/100)** |
| Average Correctness | **0.778** |
| Average Faithfulness | **0.895** |
| Average Context Sufficiency | **0.900** |

### Failure Analysis

| Outcome | Queries | Rate |
| -------- | -----: | ----: |
| Correct + Faithful + Sufficient Context | 66 | **66.0%** |
| Correct but Insufficient Context | 0 | **0.0%** |
| Incorrect despite Sufficient Context | 23 | **23.0%** |
| Insufficient Context | 10 | **10.0%** |
| Unfaithful | 16 | **16.0%** |

### Performance by Query Type

| Metric | extractive | abstractive |
| -------- | ---: | ---: |
| Queries | 42 | 58 |
| Fully Correct | 66.7% | 67.2% |
| Fully Faithful | 81.0% | 86.2% |
| Sufficient Context | 85.7% | 93.1% |

### Performance by Source

| Source | Queries | Fully Correct | Faithful | Sufficient Context |
| -------- | -----: | ------------: | -------: | -----------------: |
| text-table | 4 | 50.0% | 50.0% | 100.0% |
| text-table-image | 11 | 54.5% | 81.8% | 81.8% |
| text-image | 24 | 62.5% | 87.5% | 95.8% |
| text | 61 | 72.1% | 85.2% | 88.5% |