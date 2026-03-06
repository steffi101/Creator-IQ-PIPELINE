# CreatorIQ: Multi-Agent Ad Creative Pipeline

AI-powered system with 4 autonomous agents that generate, score, and optimize short-form video ad creatives. Built with FastAPI, ChromaDB, SQLite, and a 6-criteria hook evaluation model.

## Architecture

```
Browser (index.html)
    |
    v
FastAPI Server
    |
    |-- GET  /              --> Serves frontend
    |-- POST /rag/query     --> ChromaDB vector store (18 docs, cosine similarity)
    |-- POST /score/hooks   --> 6-criteria hook scoring model
    |-- POST /campaigns     --> SQLite database
    |-- POST /campaigns/:id/upload-csv  --> CSV ingestion + SQL analysis
    |-- POST /pipeline/run  --> 4-agent pipeline with LLM + RAG + scoring
    |-- GET  /campaigns/:id/compare     --> Cross-cycle comparison
    |-- GET  /api/info      --> Server info
    |-- GET  /health        --> Health check
```

## Components

### ChromaDB Vector Store
- 18 real ad trend documents with metadata
- Local TF-IDF embedding function (384-dim, hashed bigrams)
- Cosine similarity retrieval
- Agents receive retrieved docs as grounded context

### Hook Scoring Model
- 6 evaluation criteria: brevity, specificity, emotion, engagement, pattern interrupt, native feel
- Weighted composite score (0-100) with letter grades
- Script Deriver only receives top-scoring hooks

### SQLite Database
- 4 tables: campaigns, performance_data, cycles, hook_evaluations
- SQL aggregation by hook style, format, audience
- Full cycle history with comparison

### CSV Ingestion
- Upload real campaign performance data
- Performance Analyst uses actual CPI/CTR data instead of benchmarks

### Feedback Loop
- Cycle 1 analysis generates Next Cycle Brief
- All agents in Cycle 2 receive brief as context
- System improves measurably each cycle

## Quick Start

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key
uvicorn main:app --host 0.0.0.0 --port 8000
# Open http://localhost:8000
```

## Tech Stack

- **FastAPI** - Backend API server
- **ChromaDB** - Vector database for RAG
- **SQLite** - Persistent storage
- **Claude API** - LLM for agent reasoning
- **Vanilla JS** - Frontend (no build step needed)
