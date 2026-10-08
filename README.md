# AI in Practice — Labs 1-7

Complete implementation of all 7 labs from the AI-in-Practice Module 1 course.

---

## Project Overview

This project builds an end-to-end AI-powered **Policy Assistant** for a fictional insurance company (Aurora). Starting from raw ticket extraction, it progresses through prompt engineering, semantic search, RAG, diagnostics, tool-use with security, and finally a deployed service.

## Labs

| Lab | Topic | What It Does |
|-----|-------|--------------|
| **Lab 1** | The Reliable Extractor | Extracts structured records from messy support tickets using Pydantic schemas and LLM calls |
| **Lab 2** | The Prompt Lab | Experiment harness comparing prompt variants, model cascades, few-shot strategies with statistical significance testing |
| **Lab 3** | Semantic Search | Chunking strategies, dense/BM25/hybrid retrieval, cross-encoder and LLM reranking, ChromaDB indexing |
| **Lab 4** | RAG v1 | Retrieval-Augmented Generation with enforced citations, refusal detection, and LLM-as-judge evaluation |
| **Lab 5** | RAG v2 — Diagnosis | Failure mode classification and targeted fixes for the RAG pipeline |
| **Lab 6** | Tools & Red-Teaming | Budgeted tool loop with ToolGuard (allowlist, injection detection, call budget), red-team attacks |
| **Lab 7** | Ship It | FastAPI service with SSE streaming, Streamlit UI, observability dashboard, and CI regression gate |

## Tech Stack

- **LLM**: Google Gemini (via LiteLLM)
- **Embeddings**: Gemini text-embedding
- **Retrieval**: Dense (cosine), BM25, Hybrid (RRF), ChromaDB (HNSW)
- **Reranking**: Cross-encoder, LLM-based
- **Framework**: FastAPI, Streamlit
- **Evaluation**: LLM-as-judge (faithfulness, correctness), retrieval metrics (MRR, nDCG, hit rate)
- **Security**: ToolGuard (allowlist, schema validation, injection detection)
- **Observability**: JSONL tracing, cost tracking, latency percentiles

## Project Structure

```
.
├── aip/                  # Core library (LLM, retrieval, chunking, cost, tracing, guards)
├── labs/
│   ├── lab1/             # Ticket extraction
│   ├── lab2/             # Prompt engineering experiments
│   ├── lab3/             # Retrieval sweeps
│   ├── lab4/             # RAG pipeline + evaluation
│   ├── lab5/             # Diagnosis & fixes
│   ├── lab6/             # Tool agent + red-teaming
│   └── lab7/             # FastAPI service + gate + dashboard
├── data/                 # Corpus documents, eval golden set, attack payloads
├── CONCEPT_NOTE_LAB7.md  # Lab 7 concept note
├── EVALUATION_REPORT.md  # Evaluation report
└── presentation_lab7.html # Final presentation
```

## Running Lab 7 (the deployed service)

```bash
# Start the API server
uvicorn labs.lab7.service:app --reload --port 8000

# In another terminal, start the Streamlit UI
streamlit run labs/lab7/ui.py

# Run the regression gate
python labs/lab7/gate.py --config labs/lab7/thresholds.yml

# View the observability dashboard
streamlit run labs/lab7/dashboard.py
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # Add your GEMINI_API_KEY
```
