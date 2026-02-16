"""
FastAPI headless backend: exposes POST /run_trial and optional GET for last result.
Serves minimal frontend from / when frontend dir is present.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .run_trial import run_trial
from .run_trial_from_goal import run_trial_from_goal

_FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend"

app = FastAPI(
    title="Clinical Trial Simulator API",
    description="Run MCP-driven PK trials and retrieve results with reasoning traces.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory store for last run (optional; for minimal frontend)
_last_result: dict[str, Any] | None = None


class RunTrialRequest(BaseModel):
    drug: str = Field(default="warfarin", description="Drug name")
    dose_mg: float = Field(default=50.0, ge=0.1, description="Dose in mg")
    cohort_size: int = Field(default=100, ge=1, le=500)
    frequency: str = Field(default="every 24 hours", description="Dosing frequency")
    c_max_threshold_mg_L: float = Field(default=3.0, ge=0, description="C_max threshold for toxicity flag")
    phenotype_filters: list[dict[str, Any]] | None = Field(default=None, description="Optional phenotype distribution")


class RunTrialFromGoalRequest(BaseModel):
    goal: str = Field(..., description="Natural-language trial goal, e.g. 'Test 50mg Warfarin on 100 elderly with varying CYP2C9'")


@app.post("/run_trial")
def post_run_trial(req: RunTrialRequest) -> dict[str, Any]:
    """Run one trial: cohort generation → PK simulation (pk-sim-mcp) → outlier detection → Safety Auditor (PubMed MCP)."""
    global _last_result
    try:
        result = run_trial(
            dose_mg=req.dose_mg,
            frequency=req.frequency,
            cohort_size=req.cohort_size,
            c_max_threshold_mg_L=req.c_max_threshold_mg_L,
            drug=req.drug,
        )
        _last_result = result
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/run_trial_from_goal")
def post_run_trial_from_goal(req: RunTrialFromGoalRequest) -> dict[str, Any]:
    """LLM interprets goal → orchestrator runs trial → LLM summarizes result. Returns params_used, result, summary."""
    global _last_result
    try:
        out = run_trial_from_goal(req.goal)
        _last_result = out.get("result")
        return out
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/last_trial")
def get_last_trial() -> dict[str, Any]:
    """Return the result of the most recent POST /run_trial, or empty if none."""
    if _last_result is None:
        return {"message": "No trial run yet", "cohort_size": 0, "outliers": [], "per_patient_summary": []}
    return _last_result


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


if _FRONTEND_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(_FRONTEND_DIR)), name="frontend")

    @app.get("/")
    def index() -> FileResponse:
        index_file = _FRONTEND_DIR / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        raise HTTPException(status_code=404, detail="Frontend not found")
