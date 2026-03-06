"""
CreatorIQ Backend API

FastAPI server with:
- ChromaDB vector store for RAG retrieval
- Hook scoring model
- SQLite database for persistence
- CSV ingestion for performance data
- LLM integration for agent pipeline
- Cycle management with feedback loop
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import json
import re
import os
import httpx

from services.vector_store import vector_store
from services.scoring import score_hook, score_hooks_batch, HookScore
from services.database import (
    create_campaign, get_campaign, insert_performance_data,
    get_performance_summary, save_cycle, get_cycle_history,
    get_latest_cycle_feedback, get_cycle_comparison
)
from services.csv_ingestion import parse_csv, generate_performance_summary_text


app = FastAPI(
    title="CreatorIQ API",
    description="Multi-Agent Ad Creative Pipeline with RAG, Scoring, and Feedback Loop",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Pydantic Models
# ============================================================

class CampaignCreate(BaseModel):
    app_name: str
    app_description: str = ""

class PipelineRequest(BaseModel):
    campaign_id: int
    cycle_number: int = 1

class HookScoreRequest(BaseModel):
    hooks: list[str]

class RAGQueryRequest(BaseModel):
    query: str
    n_results: int = 6
    category: Optional[str] = None


# ============================================================
# LLM Helper
# ============================================================

async def call_llm(prompt: str) -> str:
    """Call Claude API."""
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        return "[LLM call skipped - no API key. Set ANTHROPIC_API_KEY environment variable.]"

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            },
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 1000,
                "messages": [{"role": "user", "content": prompt}],
            }
        )
        data = response.json()
        return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")


# ============================================================
# Routes: Health & Info
# ============================================================

@app.get("/")
async def root():
    return FileResponse("static/index.html")

@app.get("/api/info")
def api_info():
    return {
        "name": "CreatorIQ API",
        "version": "1.0.0",
        "components": {
            "vector_store": "ChromaDB with cosine similarity",
            "scoring_model": "6-criteria hook evaluator",
            "database": "SQLite with persistent storage",
            "llm": "Claude Sonnet via Anthropic API",
        }
    }

@app.get("/health")
def health():
    stats = vector_store.get_stats()
    return {"status": "healthy", "vector_store": stats}


# ============================================================
# Routes: RAG / Vector Store
# ============================================================

@app.post("/rag/query")
def rag_query(req: RAGQueryRequest):
    """Query the vector store for relevant trend documents."""
    results = vector_store.query(
        query_text=req.query,
        n_results=req.n_results,
        category_filter=req.category,
    )
    return {
        "query": req.query,
        "results": results,
        "count": len(results),
    }

@app.get("/rag/stats")
def rag_stats():
    """Get vector store statistics."""
    return vector_store.get_stats()


# ============================================================
# Routes: Hook Scoring
# ============================================================

@app.post("/score/hooks")
def score_hooks(req: HookScoreRequest):
    """Score a batch of hooks using the evaluation model."""
    scored = score_hooks_batch(req.hooks)
    return {
        "hooks": [s.model_dump() for s in scored],
        "average_score": round(sum(s.composite for s in scored) / len(scored)) if scored else 0,
        "grade_distribution": {
            "A": sum(1 for s in scored if s.grade == "A"),
            "B": sum(1 for s in scored if s.grade == "B"),
            "C": sum(1 for s in scored if s.grade == "C"),
            "D": sum(1 for s in scored if s.grade == "D"),
        },
        "model_info": {
            "criteria": ["brevity", "specificity", "emotion", "engagement", "interrupt", "native"],
            "weights": {"brevity": 0.15, "specificity": 0.20, "emotion": 0.20, "engagement": 0.15, "interrupt": 0.15, "native": 0.15},
        }
    }

@app.post("/score/single")
def score_single(hook: str):
    """Score a single hook."""
    result = score_hook(hook)
    return result.model_dump()


# ============================================================
# Routes: Campaigns & Data
# ============================================================

@app.post("/campaigns")
def create_new_campaign(req: CampaignCreate):
    """Create a new campaign."""
    campaign_id = create_campaign(req.app_name, req.app_description)
    return {"campaign_id": campaign_id, "app_name": req.app_name}

@app.get("/campaigns/{campaign_id}")
def get_campaign_info(campaign_id: int):
    """Get campaign details."""
    campaign = get_campaign(campaign_id)
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")
    return campaign

@app.post("/campaigns/{campaign_id}/upload-csv")
async def upload_performance_csv(campaign_id: int, file: UploadFile = File(...)):
    """Upload a CSV file with campaign performance data."""
    content = await file.read()
    parsed = parse_csv(content.decode("utf-8"))

    if parsed.get("error"):
        raise HTTPException(status_code=400, detail=parsed["error"])

    insert_performance_data(campaign_id, parsed["rows"])
    summary = get_performance_summary(campaign_id)

    return {
        "rows_imported": parsed["row_count"],
        "headers": parsed["headers"],
        "summary": summary,
    }

@app.get("/campaigns/{campaign_id}/performance")
def get_performance(campaign_id: int):
    """Get performance data summary."""
    return get_performance_summary(campaign_id)


# ============================================================
# Routes: Pipeline Execution
# ============================================================

@app.post("/pipeline/run")
async def run_pipeline(req: PipelineRequest):
    """Execute one full pipeline cycle."""
    campaign = get_campaign(req.campaign_id)
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")

    app_name = campaign["app_name"]
    app_desc = campaign["app_description"]
    cycle_num = req.cycle_number

    # Get previous cycle feedback if cycle > 1
    previous_feedback = ""
    if cycle_num > 1:
        previous_feedback = get_latest_cycle_feedback(req.campaign_id) or ""

    # Get uploaded performance data if available
    perf_summary = get_performance_summary(req.campaign_id)
    perf_text = ""
    has_perf_data = perf_summary["overall"]["total_creatives"] and perf_summary["overall"]["total_creatives"] > 0
    if has_perf_data:
        perf_text = generate_performance_summary_text(perf_summary)

    feedback_context = f"\nPREVIOUS CYCLE BRIEF:\n{previous_feedback}" if previous_feedback else ""

    # ---- AGENT 1: Trend Scout with RAG ----
    rag_results = vector_store.query(f"{app_name} {app_desc}", n_results=6)
    rag_context = "\n".join([f"[Doc {i+1}, similarity={d['similarity']}] {d['content']}" for i, d in enumerate(rag_results)])

    trend_output = await call_llm(f"""You are an ad trend research agent. App: "{app_name}" - {app_desc}.

RETRIEVED FROM VECTOR STORE (RAG - {len(rag_results)} docs, ranked by cosine similarity):
{rag_context}
{feedback_context}
{perf_text}

Using the retrieved knowledge as your primary source, analyze trends for this app.
Return: TRENDING FORMATS (3, ranked), HOOK STYLES (3, with scroll-stop rates), FATIGUE BENCHMARKS.
{"Cycle " + str(cycle_num) + ": Adjust based on previous brief and real data." if cycle_num > 1 else ""}""")

    # ---- AGENT 2: Hook Generator ----
    hook_output = await call_llm(f"""You are a short-form video ad hook writer. App: "{app_name}" - {app_desc}.
Trend research: {trend_output}
{feedback_context}

Generate 10 scroll-stopping hooks under 15 words each.
Angles: Curiosity gap, Problem agitation, Social proof, Bold claim with number, POV, Before/after.
{"Cycle " + str(cycle_num) + ": Generate DIFFERENT hooks than before. Use previous brief." if cycle_num > 1 else ""}

Format: 1. [ANGLE] "[hook text]"  ...for all 10.""")

    # Extract and score hooks
    hook_lines = re.findall(r'"([^"]+)"', hook_output)
    if not hook_lines:
        hook_lines = [line.strip() for line in hook_output.split("\n") if line.strip() and line[0].isdigit()]
    scored_hooks = score_hooks_batch(hook_lines[:10])
    top_hooks = "\n".join([f"Score {s.composite}/100 ({s.grade}): \"{s.text}\"" for s in scored_hooks[:5]])

    # ---- AGENT 3: Script Deriver ----
    script_output = await call_llm(f"""You are a short-form video ad script writer. App: "{app_name}" - {app_desc}.
Top hooks (ranked by scoring model):
{top_hooks}

Write 3 filmable scripts:
SCRIPT 1 - UGC TESTIMONIAL (15 sec): hook, body (3 beats), CTA, text overlays.
SCRIPT 2 - SCREEN RECORDING (20 sec): hook, screen actions, CTA, text overlays.
SCRIPT 3 - PROBLEM-SOLUTION (20 sec): hook, agitation, solution, CTA.
{"Cycle " + str(cycle_num) + ": Adjust based on previous brief." if cycle_num > 1 else ""}""")

    # ---- AGENT 4: Performance Analyst ----
    hook_score_text = "\n".join([
        f'"{s.text}" - {s.composite}/100 ({s.grade}) | brevity:{s.brevity} specificity:{s.specificity} emotion:{s.emotion} engagement:{s.engagement} interrupt:{s.interrupt} native:{s.native}'
        for s in scored_hooks
    ])

    feedback_output = await call_llm(f"""You are a short-form video ad performance analyst. App: "{app_name}" - {app_desc}.

HOOK SCORES (from evaluation model):
{hook_score_text}

SCRIPTS: {script_output}
{perf_text if has_perf_data else "No real data uploaded. Use benchmarks."}
{feedback_context}

Provide:
1. TOP 3 PREDICTED PERFORMERS with hook score references
2. TESTING STRATEGY: Phase 1 (days 1-3), Phase 2 (4-7), Phase 3 (8-14)
3. FATIGUE SIGNALS with thresholds
4. {"DATA-DRIVEN INSIGHTS from uploaded campaign data" if has_perf_data else "BENCHMARK ESTIMATES: CPI range, budget, lifespan"}
5. NEXT CYCLE BRIEF: Specific instructions for Cycle {cycle_num + 1}.""")

    # Save to database
    outputs = {"trend": trend_output, "hook": hook_output, "script": script_output, "feedback": feedback_output}
    hook_score_dicts = [s.model_dump() for s in scored_hooks]
    rag_doc_dicts = [{"id": d["id"], "similarity": d["similarity"]} for d in rag_results]

    cycle_id = save_cycle(
        campaign_id=req.campaign_id,
        cycle_number=cycle_num,
        outputs=outputs,
        hook_scores=hook_score_dicts,
        rag_docs=rag_doc_dicts,
        used_perf_data=has_perf_data,
    )

    return {
        "cycle_id": cycle_id,
        "cycle_number": cycle_num,
        "outputs": outputs,
        "hook_scores": hook_score_dicts,
        "rag_documents_used": len(rag_results),
        "performance_data_used": has_perf_data,
        "average_hook_score": round(sum(s.composite for s in scored_hooks) / len(scored_hooks)) if scored_hooks else 0,
    }


# ============================================================
# Routes: Cycle History & Comparison
# ============================================================

@app.get("/campaigns/{campaign_id}/cycles")
def get_cycles(campaign_id: int):
    """Get all cycles for a campaign."""
    return get_cycle_history(campaign_id)

@app.get("/campaigns/{campaign_id}/compare")
def compare_cycles(campaign_id: int):
    """Compare hook scores across cycles."""
    return get_cycle_comparison(campaign_id)


# ============================================================
# Serve Frontend
# ============================================================

@app.get("/app")
async def serve_frontend():
    return FileResponse("static/index.html")


# ============================================================
# Startup
# ============================================================

@app.on_event("startup")
async def startup():
    """Initialize vector store on startup."""
    vector_store.initialize()
    print("CreatorIQ API ready")
    print(f"Vector store: {vector_store.get_stats()}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
