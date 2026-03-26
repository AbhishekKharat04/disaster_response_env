"""
Disaster Response Coordination Environment — Client
Typed EnvClient for connecting to the deployed environment server.
"""
from __future__ import annotations

from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import DisasterAction, DisasterObservation


class DisasterResponseEnv(EnvClient[DisasterAction, DisasterObservation, State]):
    """
    Client for the Disaster Response Coordination Environment.

    Usage (async):
        async with DisasterResponseEnv(base_url="https://your-space.hf.space") as env:
            obs = await env.reset()
            result = await env.step(DisasterAction(
                response_plan="Deploy rescue teams to highest-severity areas first...",
                resource_allocations={"ambulances": 3, "rescue_teams": 2},
                priority_areas=["Area B — Floors 8-9", "Area A — Floors 3-5"],
                rationale="Area B has children and elderly with no stairwell access.",
            ))
            print(f"Score: {result.observation.step_score}/10")
            print(result.observation.feedback)

    Usage (sync):
        with DisasterResponseEnv(base_url="https://your-space.hf.space").sync() as env:
            obs = env.reset()
            result = env.step(DisasterAction(...))
    """

    def _step_payload(self, action: DisasterAction) -> dict:
        return {
            "response_plan": action.response_plan,
            "resource_allocations": action.resource_allocations,
            "priority_areas": action.priority_areas,
            "rationale": action.rationale,
        }

    def _parse_result(self, payload: dict) -> StepResult[DisasterObservation]:
        obs_data = payload.get("observation", {})
        obs = DisasterObservation(**obs_data)
        return StepResult(
            observation=obs,
            reward=payload.get("reward", obs.reward),
            done=payload.get("done", obs.done),
        )

    def _parse_state(self, payload: dict) -> State:
        return State(
            episode_id=payload.get("episode_id", ""),
            step_count=payload.get("step_count", 0),
        )
