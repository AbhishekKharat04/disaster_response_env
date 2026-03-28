"""
Disaster Response Coordination Environment — FastAPI Server App
Includes all required OpenEnv hackathon endpoints:
  /tasks    — list tasks + action schema
  /grader   — grader score after episode
  /baseline — run baseline agent, return scores
"""
import os, sys, asyncio, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Any, Dict, List, Optional

from openenv.core.env_server import create_app
from models import DisasterAction, DisasterObservation
from server.disaster_environment import DisasterResponseEnvironment
from server.scenarios import TASKS


def create_environment() -> DisasterResponseEnvironment:
    task_level = int(os.environ.get("TASK_LEVEL", "1"))
    return DisasterResponseEnvironment(task_level=task_level)


# Build the base OpenEnv FastAPI app
app = create_app(
    create_environment,
    DisasterAction,
    DisasterObservation,
    env_name="disaster_response_env",
)
@app.get("/")
def root():
    return {
        "name": "Disaster Response Coordination Environment",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
        "tasks": "/tasks",
        "baseline": "/baseline"
    }

# ─── /tasks endpoint ─────────────────────────────────────────────────────────

@app.get("/tasks", tags=["Tasks"])
def get_tasks():
    """
    Returns all tasks with their descriptions and the action schema
    (fields required for an action in a step).
    """
    tasks = []
    for t in TASKS:
        tasks.append({
            "task_id": f"task_{t['level']}",
            "name": t["name"],
            "level": t["level"],
            "difficulty": ["easy", "medium", "hard"][t["level"] - 1],
            "max_steps": t["max_steps"],
            "description": t["description"],
        })

    action_schema = {
        "response_plan": {
            "type": "string",
            "description": "Detailed narrative of your response strategy (min 50 words recommended)",
            "required": True,
        },
        "resource_allocations": {
            "type": "dict[str, int]",
            "description": "Map of resource name to integer count to deploy",
            "required": True,
            "example": {"ambulances": 3, "rescue_teams": 2, "fire_trucks": 1},
        },
        "priority_areas": {
            "type": "list[str]",
            "description": "Ordered list of area names — first = highest priority",
            "required": True,
        },
        "rationale": {
            "type": "string",
            "description": "Explanation of triage logic and trade-offs (min 30 words)",
            "required": False,
        },
    }

    return JSONResponse({
        "environment": "disaster_response_env",
        "total_tasks": len(tasks),
        "tasks": tasks,
        "action_schema": action_schema,
        "score_range": [0.0, 1.0],
        "scoring_note": "reward = step_score / 10, normalised to [0.0, 1.0]",
    })


# ─── /grader endpoint ────────────────────────────────────────────────────────

class GraderRequest(BaseModel):
    response_plan: str
    resource_allocations: Dict[str, int] = {}
    priority_areas: List[str] = []
    rationale: str = ""
    task_level: int = 1
    step: int = 1


@app.post("/grader", tags=["Grader"])
def grade_action(request: GraderRequest):
    """
    Grades a single action against a task without running a full episode.
    Returns score in 0.0–1.0 range.
    """
    from server.grader import grade_response

    if request.task_level < 1 or request.task_level > 3:
        raise HTTPException(status_code=400, detail="task_level must be 1, 2, or 3")

    task = TASKS[request.task_level - 1]
    situation = task["initial_situation"]

    action = DisasterAction(
        response_plan=request.response_plan,
        resource_allocations=request.resource_allocations,
        priority_areas=request.priority_areas,
        rationale=request.rationale,
    )

    raw_score, feedback, casualties = grade_response(
        task=task,
        action=action,
        step=request.step,
        situation=situation,
        history=[],
    )

    # Normalise to 0.0–1.0
    normalised = round(raw_score / 10.0, 4)

    return JSONResponse({
        "task_level": request.task_level,
        "task_name": task["name"],
        "score": normalised,           # 0.0–1.0
        "raw_score": raw_score,        # 0–10
        "casualties_prevented": casualties,
        "feedback": feedback,
        "score_range": [0.0, 1.0],
    })


# ─── /baseline endpoint ──────────────────────────────────────────────────────

def _run_baseline_for_task(task_level: int) -> Dict[str, Any]:
    """Run a single deterministic baseline episode and return results."""
    env = DisasterResponseEnvironment(task_level=task_level)
    task = TASKS[task_level - 1]
    obs = env.reset()

    # Deterministic baseline actions — one per task level
    BASELINE_ACTIONS = {
        1: DisasterAction(
            response_plan=(
                "Deploy rescue teams immediately to Area B (Floors 8-9) as it has the highest "
                "severity with 25 trapped including children and elderly. The elevator shaft fire "
                "means stairwell B is the only access — rescue teams are critical. Simultaneously "
                "send fire trucks to Area A to prevent spread upward. Stage ambulances at both "
                "zones. Area C gets one paramedic unit for street injuries."
            ),
            resource_allocations={"ambulances": 3, "rescue_teams": 2, "fire_trucks": 2, "paramedic_units": 1},
            priority_areas=["Area B — Floors 8-9", "Area A — Floors 3-5", "Area C — Surrounding Streets"],
            rationale=(
                "Area B has children, elderly, and no stairwell — rescue teams are irreplaceable here. "
                "Area A fire spread would worsen Area B. Area C injuries are ambulatory."
            ),
        ),
        2: DisasterAction(
            response_plan=(
                "Priority 1: District South gas leak — deploy heavy equipment and rescue teams "
                "immediately to prevent explosion. Priority 2: District West hospital — ambulances "
                "and medical units for ICU patient evacuation within the critical window. "
                "Priority 3: District North — rescue teams for trapped residents. "
                "District East gets medical units for displaced population triage."
            ),
            resource_allocations={"ambulances": 4, "rescue_teams": 3, "medical_units": 2, "heavy_equipment": 2},
            priority_areas=["District South", "District West — Hospital", "District North", "District East"],
            rationale=(
                "Gas leak in District South creates explosion risk affecting all nearby zones. "
                "Hospital ICU has a hard 12-hour window. North and East are accessible but less critical."
            ),
        ),
        3: DisasterAction(
            response_plan=(
                "Critical priorities: Zone 2 ammonia leak requires hazmat units immediately — "
                "failure means mass casualty event. Zone 1 nursing homes need boats and helicopters "
                "for elderly evacuation. Zone 3 hospital ICU evacuation starts with ambulances and "
                "field hospitals. Zone 5 bridge repair assigned to engineering crews — restoring it "
                "doubles future resource capacity. Zone 4 rescue teams for structural collapses."
            ),
            resource_allocations={
                "hazmat_units": 2, "boats": 3, "helicopters": 1, "ambulances": 4,
                "field_hospitals": 1, "rescue_teams": 3, "engineering_crews": 1, "heavy_equipment": 1,
            },
            priority_areas=["Zone 2 — Industrial District", "Zone 1 — Coastal Ward",
                          "Zone 3 — Central Hospital", "Zone 5 — Transport Hub", "Zone 4 — Residential North"],
            rationale=(
                "Ammonia leak is an immediate mass casualty risk. Nursing home elderly have no "
                "self-evacuation ability. Hospital ICU is time-critical. Bridge repair is a "
                "force multiplier for all future steps."
            ),
        ),
    }

    scores = []
    step = 0
    done = False

    while not done:
        step += 1
        action = BASELINE_ACTIONS.get(task_level, BASELINE_ACTIONS[1])
        obs, reward, done, info = env.step(action)
        scores.append(round(reward, 4))  # already 0.0–1.0

    final = round(sum(scores) / len(scores), 4)
    return {
        "task_level": task_level,
        "task_name": task["name"],
        "steps": step,
        "step_scores": scores,
        "final_score": final,   # 0.0–1.0
        "score_range": [0.0, 1.0],
    }


@app.get("/baseline", tags=["Baseline"])
def run_baseline(task_level: Optional[int] = None):
    """
    Runs the deterministic baseline agent against all 3 tasks (or one specific task).
    Returns reproducible scores in 0.0–1.0 range for each task.
    """
    try:
        if task_level is not None:
            if task_level < 1 or task_level > 3:
                raise HTTPException(status_code=400, detail="task_level must be 1, 2, or 3")
            result = _run_baseline_for_task(task_level)
            return JSONResponse({"results": [result], "score_range": [0.0, 1.0]})

        # Run all 3 tasks
        results = []
        for level in [1, 2, 3]:
            results.append(_run_baseline_for_task(level))

        overall = round(sum(r["final_score"] for r in results) / len(results), 4)
        return JSONResponse({
            "results": results,
            "overall_score": overall,
            "score_range": [0.0, 1.0],
            "note": "Deterministic baseline — scores are reproducible",
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))