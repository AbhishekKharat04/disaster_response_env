"""
Disaster Response Coordination Environment — Core Environment
Implements the OpenEnv Environment interface with 3 progressive tasks.
"""
from __future__ import annotations

import copy
import logging
from uuid import uuid4
from typing import Any, Dict, List, Optional, Tuple

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

# Use relative-style imports that work both in Docker and standalone
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import DisasterAction, DisasterObservation
from server.scenarios import TASKS
from server.grader import grade_response

logger = logging.getLogger(__name__)


class DisasterResponseEnvironment(Environment):
    """
    Disaster Response Coordination Environment.

    An RL agent acts as an emergency response commander.
    Each episode presents one of three disaster scenarios (tasks):
      - Task 1: Apartment Building Fire      (1 step,  easy)
      - Task 2: Urban Earthquake             (3 steps, medium)
      - Task 3: Category 5 Hurricane        (5 steps, hard)

    The agent must allocate limited resources, prioritise affected areas,
    and justify decisions. Reward = step_score / 10 (normalised to [0,1]).
    Final episode score = average step score (0-10).
    """

    def __init__(self, task_level: int = 1):
        self._task_level: int = max(1, min(3, task_level))
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._task: Dict[str, Any] = {}
        self._situation: Dict[str, Any] = {}
        self._step: int = 0
        self._done: bool = False
        self._cumulative_score: float = 0.0
        self._casualties_prevented: int = 0
        self._history: List[Dict[str, Any]] = []

    # ─── OpenEnv Interface ────────────────────────────────────────────────────

    def reset(self, task_level: Optional[int] = None) -> DisasterObservation:
        """Initialise a new episode. Optionally override the task level."""
        if task_level is not None:
            self._task_level = max(1, min(3, task_level))

        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._step = 0
        self._done = False
        self._cumulative_score = 0.0
        self._casualties_prevented = 0
        self._history = []

        self._task = TASKS[self._task_level - 1]
        self._situation = copy.deepcopy(self._task["initial_situation"])

        obs = DisasterObservation(
            situation_report=self._situation["description"],
            available_resources=copy.deepcopy(self._situation["resources"]),
            affected_areas=copy.deepcopy(self._situation["affected_areas"]),
            time_step=0,
            max_steps=self._task["max_steps"],
            done=False,
            reward=0.0,
            step_score=0.0,
            cumulative_score=0.0,
            final_score=None,
            feedback=(
                f"Episode started. Task: {self._task['name']}.\n"
                f"{self._task['description']}"
            ),
            casualties_prevented=0,
            task_level=self._task_level,
            task_name=self._task["name"],
            task_description=self._task["description"],
        )
        logger.info(
            "Episode %s started. Task=%d (%s)",
            self._state.episode_id,
            self._task_level,
            self._task["name"],
        )
        return obs

    def step(
        self, action: DisasterAction
    ) -> Tuple[DisasterObservation, float, bool, Dict[str, Any]]:
        """Execute one action and return (observation, reward, done, info)."""
        if self._done:
            raise RuntimeError("Episode is done. Call reset() first.")

        self._state.step_count += 1
        self._step += 1

        # Record history
        self._history.append(
            {
                "step": self._step,
                "action": {
                    "response_plan": action.response_plan,
                    "resource_allocations": action.resource_allocations,
                    "priority_areas": action.priority_areas,
                    "rationale": action.rationale,
                },
            }
        )

        # Grade the action
        step_score, feedback, casualties = grade_response(
            task=self._task,
            action=action,
            step=self._step,
            situation=self._situation,
            history=self._history,
        )

        self._cumulative_score += step_score
        self._casualties_prevented += casualties

        # Advance situation
        self._situation = self._advance_situation(action)

        # Check termination
        max_steps = self._task["max_steps"]
        self._done = self._step >= max_steps

        # Normalised reward for RL training
        reward = step_score / 10.0

        # Compute final score when episode ends
        final_score: Optional[float] = None
        if self._done:
            final_score = round(self._cumulative_score / max_steps, 2)
            feedback += (
                f"\n\n{'='*50}\n"
                f"EPISODE COMPLETE\n"
                f"Final Score: {final_score:.2f}/10\n"
                f"Casualties Prevented: ~{self._casualties_prevented}\n"
                f"{'='*50}"
            )
            logger.info(
                "Episode %s complete. Final score=%.2f",
                self._state.episode_id,
                final_score,
            )

        obs = DisasterObservation(
            situation_report=self._situation["description"],
            available_resources=copy.deepcopy(self._situation.get("resources", {})),
            affected_areas=copy.deepcopy(self._situation.get("affected_areas", [])),
            time_step=self._step,
            max_steps=max_steps,
            done=self._done,
            reward=reward,
            step_score=step_score,
            cumulative_score=self._cumulative_score,
            final_score=final_score,
            feedback=feedback,
            casualties_prevented=self._casualties_prevented,
            task_level=self._task_level,
            task_name=self._task["name"],
            task_description=self._task["description"],
        )

        info = {
            "step_score": step_score,
            "cumulative_score": self._cumulative_score,
            "final_score": final_score,
            "casualties_prevented": self._casualties_prevented,
            "episode_id": self._state.episode_id,
        }

        return obs, reward, self._done, info

    @property
    def state(self) -> State:
        return self._state

    # ─── Internal helpers ─────────────────────────────────────────────────────

    def _advance_situation(self, action: DisasterAction) -> Dict[str, Any]:
        """Update situation description and resources after an action."""
        updates = self._task.get("situation_updates", {})
        next_step = self._step + 1

        # Get the next step's description
        new_description = updates.get(
            next_step,
            updates.get("default", self._situation.get("description", "")),
        )

        # Deduct allocated resources
        current_resources = copy.deepcopy(self._situation.get("resources", {}))
        allocations = action.resource_allocations or {}
        for resource, count in allocations.items():
            if resource in current_resources:
                current_resources[resource] = max(0, current_resources[resource] - count)

        # Areas remain (simplified — more complex simulation could update these)
        new_situation = {
            "description": new_description,
            "resources": current_resources,
            "affected_areas": copy.deepcopy(self._situation.get("affected_areas", [])),
        }

        return new_situation
