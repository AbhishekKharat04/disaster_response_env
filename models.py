"""
Disaster Response Coordination Environment — Models
Action and Observation typed models for OpenEnv compliance.
"""
from typing import Optional, List, Dict, Any
from pydantic import Field
from openenv.core.env_server.types import Action, Observation


class DisasterAction(Action):
    """
    Action submitted by the agent each step.
    The agent must describe its response plan, allocate available resources,
    name priority areas, and justify its decisions.
    """
    response_plan: str = Field(
        ...,
        description=(
            "A clear, structured narrative of the response strategy for this step. "
            "Explain what you will do, why, and in what order."
        ),
    )
    resource_allocations: Dict[str, int] = Field(
        default_factory=dict,
        description=(
            "Resource allocation map. Keys match available resource names "
            "(e.g. 'ambulances', 'rescue_teams', 'firefighters', 'helicopters', "
            "'field_hospitals'). Values are integers."
        ),
    )
    priority_areas: List[str] = Field(
        default_factory=list,
        description=(
            "Ordered list of area names to prioritize this step. "
            "First element = highest priority."
        ),
    )
    rationale: str = Field(
        default="",
        description=(
            "Explanation of the triage logic used: why these areas got priority, "
            "why resources were split this way, what trade-offs were made."
        ),
    )


class DisasterObservation(Observation):
    """
    Observation returned to the agent after each step.
    Contains the updated disaster state, grader feedback, and reward signal.
    """
    situation_report: str = Field(
        default="",
        description="Detailed report of the current disaster situation.",
    )
    available_resources: Dict[str, int] = Field(
        default_factory=dict,
        description="Resources still available to allocate.",
    )
    affected_areas: List[Dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "List of affected area objects. Each has: "
            "name, severity (1-5), population_at_risk, needs (list), "
            "infrastructure_damage (str), estimated_casualties_if_ignored (int)."
        ),
    )
    time_step: int = Field(0, description="Current step index within the episode.")
    max_steps: int = Field(1, description="Total steps allowed for this task.")
    done: bool = Field(False, description="True when the episode is complete.")
    reward: float = Field(0.0, description="Step reward normalised to [0, 1].")
    step_score: float = Field(0.0, description="Raw grader score for this step (0–10).")
    cumulative_score: float = Field(
        0.0, description="Sum of step scores so far."
    )
    final_score: Optional[float] = Field(
        None, description="Final episode score (0–10), set when done=True."
    )
    feedback: str = Field(
        default="", description="Grader feedback explaining the score."
    )
    casualties_prevented: int = Field(
        0, description="Estimated number of casualties prevented so far."
    )
    task_level: int = Field(1, description="Task difficulty level: 1 (easy), 2 (medium), 3 (hard).")
    task_name: str = Field("", description="Human-readable name of the current task.")
    task_description: str = Field("", description="Full task briefing shown at episode start.")
