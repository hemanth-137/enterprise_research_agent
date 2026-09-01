# Enterprise RAG Agent

A local **Retrieval-Augmented Generation (RAG)** system for asking questions over PDF documents.

It combines:

* **Docling** for structure-aware PDF parsing, table extraction and chunking
* **BAAI/bge-base-en-v1.5** for embeddings
* **Qdrant** for local vector storage
* **BAAI/bge-reranker-base** for reranking
* **FastAPI + Streamlit** for the application interface

It also provides **source-grounded answers** with inline source citations, while the Streamlit interface lets you inspect the retrieved chunks and their metadata, including document name, page number, chunk ID, headings, and the full retrieved text.

The system also supports **conversational queries**, using recent chat history to determine whether a question can be answered directly or requires document retrieval.


## How It Works

### Document Ingestion

Documents are processed once before querying:

```text
PDF
  ↓
Docling
  ↓
Structure + Tables
  ↓
Contextualized Chunks
(Heading + Chunk Text)
  ↓
BGE Embeddings
  ↓
Qdrant
```

The ingestion pipeline is in:

`ingestion/unstructured/ingestion_main.py`

### Query Pipeline

```text
                    User Query + Chat History
                              │
                              ▼
                         Router LLM
                              │
                    ┌─────────┴─────────┐
                    │                   │
               No Retrieval        Retrieval Needed
                    │                   │
                    ▼            ┌──────┴──────┐
              Direct Answer      │             │
                                 │             │
                           Single Query   Need Sub-Queries
                                 │             │
                                 ▼             ▼
                           Vector Search   Generate up to
                              Top 30        3 Sub-Queries
                                 │             │
                                 ▼             ▼
                           BGE Reranker   Each Sub-Query
                                 │        → Vector Search
                                 ▼        → Rerank → Top 3
                              Top 5             │
                                 │             ▼
                                 │        Combine Results
                                 │             │
                                 │             ▼
                                 │      Remove Duplicate
                                 │         Chunk IDs
                                 │             │
                                 └──────┬──────┘
                                        │
                                        ▼
                               Retrieved Context
                                        +
                                 Chat History
                                        │
                                        ▼
                                  Gemini Answer
                                        │
                                        ▼
                              Inline Source Citations
                                        │
                                        ▼
                              Streamlit Source Viewer
```

## Project Structure

```text
.
├── app.py
├── main.py
│
├── generation/
│   ├── generate_answer.py
│   └── sys_prompts.py
│
├── ingestion/
│   └── unstructured/
│       ├── chunking.py
│       ├── doc_parser.py
│       ├── ingestion_main.py
│       ├── local_embedding.py
│       └── to_vector_db.py
│
├── retrieval/
│   ├── re_ranker.py
│   └── vector_search.py
│
└── evaluation/
    ├── calculations.py
    ├── get_answer.py
    ├── llm_judge.py
    ├── open_ragbench_eval.py
    └── rag_eval_report.md
```

## Setup

Create a Python 3.10 environment and install dependencies:

```bash
pip install -r requirements.txt
```

Run Qdrant locally using Docker and configure the required environment variables.

## Usage

### 1. Ingest documents

```bash
python ingestion/unstructured/ingestion_main.py
```

This parses the PDF file or PDFs in a folder, preserves document structure and tables, creates contextualized chunks, generates embeddings, and stores them in Qdrant.

### 2. Start the backend

```bash
uvicorn main:app --reload
```

### 3. Start the frontend

```bash
streamlit run app.py
```

## Evaluation

The system was evaluated on a **100-query sample from Open-RAGBench** using an LLM-based judge.
(answering LLm = local llama3.1 8B, judge = Gemini 3.5 Flash Lite)

Results included:

* **67% Fully Correct**
* **84% Fully Faithful**
* **90% Sufficient Context**

The evaluation implementation and detailed results are available in:

* `evaluation/open_ragbench_eval.py`
* `evaluation/llm_judge.py`
* `evaluation/rag_eval_report.md`

## Notes

OCR is disabled during PDF ingestion to keep processing practical on local hardware. This works well for digital PDFs but may reduce extraction quality for scanned or image-only documents.

## Further Details

For implementation details, see:

* `generation/generate_answer.py` — query routing, retrieval orchestration, conversational history, and answer generation
* `retrieval/` — vector search and reranking
* `ingestion/unstructured/` — PDF parsing, chunking, embeddings, and indexing
* `evaluation/` — benchmark evaluation and analysis