"""
Disaster Response Coordination Environment — Baseline Agent
A Claude/OpenAI-powered agent that demonstrates solving all 3 tasks.
Run this AFTER deploying the environment to Hugging Face Spaces.

Usage:
    pip install openenv-core openai
    export OPENAI_API_KEY=sk-...
    python baseline_agent.py --base_url https://YOUR-SPACE.hf.space --task_level 1
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import Any, Dict, List

try:
    from openai import AsyncOpenAI
except ImportError:
    print("❌ Please install openai: pip install openai")
    sys.exit(1)

try:
    from disaster_response_env import DisasterAction, DisasterObservation, DisasterResponseEnv
except ImportError:
    print("❌ Please install the env client: pip install -e .")
    sys.exit(1)


# ─── Agent system prompt ──────────────────────────────────────────────────────

AGENT_SYSTEM_PROMPT = """You are an expert Emergency Response Commander with 20 years of experience.
Your job is to coordinate disaster response operations optimally.

You will receive a situation report, available resources, and affected areas.
You must respond with a JSON action object ONLY — no extra text, no markdown fences.

The JSON must exactly match this schema:
{
    "response_plan": "Detailed narrative of your strategy (minimum 80 words)",
    "resource_allocations": {
        "resource_name": integer_count,
        ...
    },
    "priority_areas": ["highest priority area name", "second priority", ...],
    "rationale": "Your triage logic — why these areas, why these allocations, what trade-offs (minimum 40 words)"
}

Critical rules:
1. Never allocate more of any resource than what's available
2. Always address the highest-severity areas first
3. Never leave a severity-5 area completely unaddressed
4. Justify decisions based on population_at_risk and casualties_if_ignored
5. If children, elderly, or hazmat are mentioned — they always get elevated priority
"""


# ─── Agent ────────────────────────────────────────────────────────────────────

class DisasterResponseAgent:
    def __init__(self, model: str = "gpt-4o"):
        self.client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model = model
        self.conversation_history: List[Dict[str, Any]] = []

    def _format_observation(self, obs: DisasterObservation) -> str:
        areas_formatted = "\n".join(
            f"  [{i+1}] {a['name']}\n"
            f"      Severity: {a['severity']}/5 | At Risk: {a['population_at_risk']} people\n"
            f"      Needs: {', '.join(a.get('needs', []))}\n"
            f"      Damage: {a.get('infrastructure_damage','')}\n"
            f"      Casualties if ignored: ~{a.get('estimated_casualties_if_ignored',0)}"
            for i, a in enumerate(obs.affected_areas)
        )
        resources_formatted = "\n".join(
            f"  {k}: {v}" for k, v in obs.available_resources.items()
        )
        return (
            f"=== STEP {obs.time_step + 1} of {obs.max_steps} ===\n"
            f"TASK: {obs.task_name}\n\n"
            f"SITUATION REPORT:\n{obs.situation_report}\n\n"
            f"AVAILABLE RESOURCES:\n{resources_formatted}\n\n"
            f"AFFECTED AREAS:\n{areas_formatted}\n\n"
            f"Previous step feedback: {obs.feedback if obs.time_step > 0 else 'N/A (first step)'}"
        )

    async def act(self, obs: DisasterObservation) -> DisasterAction:
        obs_text = self._format_observation(obs)
        self.conversation_history.append({"role": "user", "content": obs_text})

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": AGENT_SYSTEM_PROMPT},
                *self.conversation_history,
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
        )

        raw_response = response.choices[0].message.content
        self.conversation_history.append(
            {"role": "assistant", "content": raw_response}
        )

        data = json.loads(raw_response)
        return DisasterAction(
            response_plan=data.get("response_plan", ""),
            resource_allocations=data.get("resource_allocations", {}),
            priority_areas=data.get("priority_areas", []),
            rationale=data.get("rationale", ""),
        )


# ─── Runner ───────────────────────────────────────────────────────────────────

async def run_episode(base_url: str, task_level: int, model: str) -> None:
    print(f"\n{'='*60}")
    print(f"🚨 DISASTER RESPONSE ENVIRONMENT — Task Level {task_level}")
    print(f"{'='*60}")
    print(f"Connecting to: {base_url}")
    print(f"Agent model:   {model}\n")

    agent = DisasterResponseAgent(model=model)

    async with DisasterResponseEnv(base_url=base_url) as env:
        # Reset — task level is set server-side via TASK_LEVEL env var
        obs = await env.reset()

        print(f"📋 Task: {obs.task_name}")
        print(f"📝 {obs.task_description}\n")
        print(f"Max steps: {obs.max_steps}")
        print("-" * 60)

        total_score = 0.0
        step = 0

        while not obs.done:
            step += 1
            print(f"\n🔄 STEP {step}/{obs.max_steps}")
            print(f"Situation:\n{obs.situation_report[:500]}...")
            print(f"Resources: {obs.available_resources}")

            # Agent decides action
            print("\n🤖 Agent thinking...")
            action = await agent.act(obs)

            print(f"\n📤 Agent action:")
            print(f"  Plan (first 200 chars): {action.response_plan[:200]}...")
            print(f"  Allocations: {action.resource_allocations}")
            print(f"  Priority areas: {action.priority_areas}")

            # Execute step
            result = await env.step(action)
            obs = result.observation

            print(f"\n📊 Step score: {obs.step_score:.1f}/10")
            print(f"💬 Feedback:\n{obs.feedback}")
            print(f"🏥 Casualties prevented so far: ~{obs.casualties_prevented}")
            total_score += obs.step_score

        print(f"\n{'='*60}")
        print(f"✅ EPISODE COMPLETE")
        print(f"   Final Score:          {obs.final_score:.2f}/10")
        print(f"   Total Steps:          {step}")
        print(f"   Casualties Prevented: ~{obs.casualties_prevented}")
        print(f"{'='*60}")


async def run_all_tasks(base_url: str, model: str) -> None:
    """Run all 3 tasks sequentially and report results."""
    results = []
    for level in [1, 2, 3]:
        print(f"\n\n{'#'*60}")
        print(f"# TASK {level}")
        print(f"{'#'*60}")
        await run_episode(base_url, level, model)
        results.append(level)

    print("\n\n" + "="*60)
    print("ALL TASKS COMPLETE")
    print("="*60)


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Disaster Response Baseline Agent")
    parser.add_argument(
        "--base_url",
        type=str,
        default="http://localhost:7860",
        help="Base URL of the deployed environment",
    )
    parser.add_argument(
        "--task_level",
        type=int,
        choices=[1, 2, 3, 0],
        default=1,
        help="Task level 1/2/3, or 0 to run all tasks",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="OpenAI model to use for the agent",
    )
    args = parser.parse_args()

    if "OPENAI_API_KEY" not in os.environ:
        print("❌ Set OPENAI_API_KEY environment variable first.")
        sys.exit(1)

    if args.task_level == 0:
        asyncio.run(run_all_tasks(args.base_url, args.model))
    else:
        asyncio.run(run_episode(args.base_url, args.task_level, args.model))
