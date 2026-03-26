---
title: Disaster Response Environment
emoji: 🚨
colorFrom: red
colorTo: yellow
sdk: docker
pinned: true
---

# 🚨 Disaster Response Coordination Environment

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compatible-blue)](https://github.com/meta-pytorch/OpenEnv)
[![HuggingFace](https://img.shields.io/badge/🤗-Hugging%20Face%20Space-yellow)](https://huggingface.co/spaces)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-green.svg)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **An OpenEnv-compatible reinforcement learning environment where AI agents coordinate emergency response operations across real-world disaster scenarios.**

---

## 🌟 What Makes This Unique

Most RL environments use abstract reward signals. This environment uses **real-world disaster triage logic** — the same frameworks used by actual emergency management agencies (ICS — Incident Command System). The agent must:

- **Think like a commander**, not just pick from discrete options
- **Allocate scarce resources** across competing life-or-death priorities  
- **Adapt** as situations evolve with new information each step
- **Communicate clearly** — feedback explicitly rewards command-quality briefings

The grader uses a **hybrid rule-based + LLM evaluation** system, rewarding both correct allocation decisions *and* the quality of reasoning — perfect for training language models via GRPO.

---

## 🎯 Tasks

| Task | Name | Difficulty | Steps | Resources | Scenarios |
|------|------|------------|-------|-----------|-----------|
| 1 | Apartment Building Fire | Easy | 1 | 4 types | 3 zones |
| 2 | Urban Earthquake — 4 Districts | Medium | 3 | 4 types | 4 districts |
| 3 | Category 5 Hurricane | Hard | 5 | 8 types | 6 zones |

### Task 1 — Apartment Building Fire (Easy)
A 10-storey building fire in downtown Bengaluru. Three areas need coverage:
- **Area A** (Floors 3-5): 12 trapped, accessible stairwell
- **Area B** (Floors 8-9): 25 trapped including children and elderly, elevator shaft on fire  
- **Area C** (Street): 8 minor injuries, gas main risk

One step to deploy all resources. Maximum lives saved wins.

### Task 2 — Urban Earthquake (Medium)
M6.8 earthquake, 4 districts, 3 steps. New information arrives each step — a gas leak update in Step 2, an aftershock in Step 3. The agent must adapt its strategy as the situation evolves.

### Task 3 — Category 5 Hurricane (Hard)
Full city disaster. Six zones ranging from a chemical plant ammonia leak to hospital evacuation to water treatment contamination. A second storm band grounds boats mid-episode. Repairing the railway bridge doubles resource capacity for later steps — testing whether the agent can think strategically across time.

---

## 🏗️ Architecture

```
disaster_response_env/
├── models.py                  # DisasterAction, DisasterObservation (Pydantic)
├── client.py                  # DisasterResponseEnv (OpenEnv EnvClient)
├── baseline_agent.py          # GPT-4o powered demonstration agent
├── pyproject.toml             # Package config
├── openenv.yaml               # OpenEnv manifest
└── server/
    ├── app.py                 # FastAPI app (create_app wrapper)
    ├── disaster_environment.py # Core env logic (Environment subclass)
    ├── scenarios.py           # All 3 task definitions
    ├── grader.py              # Hybrid rule-based + LLM grader
    ├── requirements.txt       # Server dependencies
    └── Dockerfile             # Container for HF Spaces
```

---

## 🚀 Quick Start

### 1. Install the client

```bash
pip install git+https://huggingface.co/spaces/YOUR_USERNAME/disaster_response_env
```

### 2. Use the environment (async)

```python
import asyncio
from disaster_response_env import DisasterAction, DisasterResponseEnv

async def main():
    async with DisasterResponseEnv(base_url="https://YOUR-SPACE.hf.space") as env:
        # Start Task 1 (Apartment Fire)
        obs = await env.reset()
        print(obs.situation_report)

        action = DisasterAction(
            response_plan=(
                "Immediate deployment: Area B is highest severity with 25 trapped including "
                "children on the crèche floor and elderly residents. Elevator shaft fire means "
                "aerial or stairwell B approach only. Prioritise 2 rescue teams to Area B "
                "immediately. Simultaneously deploy 2 fire trucks to Area A to contain spread "
                "and prevent Area B from worsening. Ambulances staged at both zones."
            ),
            resource_allocations={
                "ambulances": 2,       # Area B
                "rescue_teams": 2,     # Area B — all to highest severity
                "fire_trucks": 2,      # Area A — contain spread
                "paramedic_units": 1,  # Area C — street injuries
            },
            priority_areas=[
                "Area B — Floors 8-9",
                "Area A — Floors 3-5",
                "Area C — Surrounding Streets",
            ],
            rationale=(
                "Area B holds 25 lives including 4 children and 6 elderly — highest risk group. "
                "Stairwell A blocked, elevator shaft on fire — rescue teams are the only viable "
                "option. Area A gets fire trucks to prevent upward spread which would increase "
                "Area B casualties. Area C is ambulatory with minor injuries — 1 paramedic unit sufficient."
            ),
        )

        result = await env.step(action)
        print(f"Score: {result.observation.step_score}/10")
        print(result.observation.feedback)

asyncio.run(main())
```

### 3. Synchronous usage

```python
from disaster_response_env import DisasterAction, DisasterResponseEnv

with DisasterResponseEnv(base_url="https://YOUR-SPACE.hf.space").sync() as env:
    obs = env.reset()
    result = env.step(DisasterAction(
        response_plan="...",
        resource_allocations={"rescue_teams": 2, "ambulances": 3},
        priority_areas=["Area B — Floors 8-9"],
        rationale="...",
    ))
    print(f"Score: {result.observation.step_score}/10")
```

---

## 🤖 Running the Baseline Agent

```bash
export OPENAI_API_KEY=sk-...

# Run Task 1 (Fire)
python baseline_agent.py --base_url https://YOUR-SPACE.hf.space --task_level 1

# Run Task 2 (Earthquake)
python baseline_agent.py --base_url https://YOUR-SPACE.hf.space --task_level 2

# Run Task 3 (Hurricane)
python baseline_agent.py --base_url https://YOUR-SPACE.hf.space --task_level 3

# Run all 3 tasks sequentially
python baseline_agent.py --base_url https://YOUR-SPACE.hf.space --task_level 0
```

---

## 📊 Reward & Scoring

Each step returns:
- **`reward`**: Float in `[0, 1]` (step_score / 10) — for RL training loops
- **`step_score`**: Float in `[0, 10]` — detailed per-step score
- **`final_score`**: Float in `[0, 10]` — episode average (set when `done=True`)

### Grading criteria (per step)

| Criterion | Points |
|-----------|--------|
| Highest-severity area prioritised | 2.5 |
| Secondary areas addressed | 1.5 |
| No over-allocation | 2.0 |
| ≥50% resource utilisation | 1.5 |
| Substantive response plan (>50 words) | 1.5 |
| Clear rationale (>30 words) | 1.0 |
| **Total** | **10.0** |

With `OPENAI_API_KEY` set: rule-based score (40%) + GPT-4o-mini quality evaluation (60%).

---

## 🐳 Local Development

```bash
# Run server locally (no Docker)
pip install -e ".[server]"
cd server
uvicorn app:app --reload --port 7860

# Test with a quick action
python baseline_agent.py --base_url http://localhost:7860 --task_level 1
```

---

## 🤗 Deploying to Hugging Face Spaces

```bash
# Login to Hugging Face
pip install huggingface_hub
huggingface-cli login

# Install OpenEnv CLI
pip install openenv-core

# Push to HF Spaces
cd disaster_response_env
openenv push --repo-id YOUR_USERNAME/disaster_response_env
```

Or manually:
1. Create a new Space at huggingface.co/new-space
2. Select **Docker** as the SDK
3. Push this repo to the Space's git remote

Set these Space secrets (Settings → Repository Secrets):
- `OPENAI_API_KEY` — optional, enables LLM grading
- `TASK_LEVEL` — set to `1`, `2`, or `3`

---

## 🔗 Use with RL Frameworks

### TRL (GRPO)
```python
from trl import GRPOConfig, GRPOTrainer
from disaster_response_env import DisasterAction, DisasterResponseEnv

# The environment provides reward via result.observation.reward
# Compatible with TRL's OpenEnv integration
```

### Unsloth
Compatible with the [OpenEnv Unsloth notebook](https://colab.research.google.com/github/unslothai/notebooks)

---

## 📝 Observation Fields

| Field | Type | Description |
|-------|------|-------------|
| `situation_report` | str | Full disaster briefing for this step |
| `available_resources` | dict | Resource name → count available |
| `affected_areas` | list | Area objects with severity, population, needs |
| `time_step` | int | Current step (0-indexed) |
| `max_steps` | int | Total steps in this task |
| `done` | bool | True when episode ends |
| `reward` | float | Step reward [0,1] for RL |
| `step_score` | float | Raw step score [0,10] |
| `final_score` | float\|None | Episode score when done |
| `feedback` | str | Grader feedback on last action |
| `casualties_prevented` | int | Estimated lives saved |

---

## License

MIT License — see [LICENSE](LICENSE)

---

*Built for the OpenEnv Hackathon — March 2026. Solo entry.*
