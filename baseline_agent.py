"""
Disaster Response Coordination Environment — Baseline Agent
Rule-based deterministic agent. No API key required.
Usage: python baseline_agent.py --base_url https://YOUR-SPACE.hf.space
"""
import argparse, requests, json

BASE_ACTIONS = {
    1: {
        "response_plan": "Deploy rescue teams immediately to Area B (Floors 8-9) as it has the highest severity with 25 trapped including children and elderly. The elevator shaft fire means stairwell B is the only access. Simultaneously send fire trucks to Area A to prevent upward spread. Stage ambulances at both zones. Area C gets one paramedic unit for street injuries.",
        "resource_allocations": {"ambulances": 3, "rescue_teams": 2, "fire_trucks": 2, "paramedic_units": 1},
        "priority_areas": ["Area B — Floors 8-9", "Area A — Floors 3-5", "Area C — Surrounding Streets"],
        "rationale": "Area B has children and elderly with no stairwell access — rescue teams are irreplaceable. Area A fire spread would worsen Area B. Area C injuries are ambulatory and need only basic care."
    },
    2: {
        "response_plan": "Priority 1: District South gas leak — deploy heavy equipment and rescue teams immediately to prevent explosion. Priority 2: District West hospital — ambulances and medical units for ICU patient evacuation within the critical 12-hour window. Priority 3: District North — send rescue teams for trapped residents in collapsed buildings. District East gets medical units for displaced population triage.",
        "resource_allocations": {"ambulances": 4, "rescue_teams": 3, "medical_units": 2, "heavy_equipment": 2},
        "priority_areas": ["District South", "District West — Hospital", "District North", "District East"],
        "rationale": "Gas leak in District South creates explosion risk affecting all nearby zones. Hospital ICU has a hard 12-hour window for critical patients. North and East are accessible but less immediately life-threatening."
    },
    3: {
        "response_plan": "Critical priorities: Zone 2 ammonia leak requires hazmat units immediately — failure means mass casualty event. Zone 1 nursing homes need boats and helicopters for elderly evacuation — roads are flooded. Zone 3 hospital ICU evacuation starts with ambulances and field hospitals. Zone 5 bridge repair assigned to engineering crews — restoring it doubles future resource capacity. Zone 4 rescue teams for structural collapses. Zone 6 engineering crew to prevent water contamination.",
        "resource_allocations": {"hazmat_units": 2, "boats": 3, "helicopters": 1, "ambulances": 4, "field_hospitals": 1, "rescue_teams": 3, "engineering_crews": 1, "heavy_equipment": 1},
        "priority_areas": ["Zone 2 — Industrial District", "Zone 1 — Coastal Ward", "Zone 3 — Central Hospital", "Zone 5 — Transport Hub", "Zone 4 — Residential North"],
        "rationale": "Ammonia leak is an immediate mass casualty risk. Nursing home elderly have no self-evacuation. Hospital ICU is time-critical. Bridge repair is a force multiplier for all future steps."
    }
}

def run_episode(base_url: str, task_level: int):
    print(f"\n{'='*60}")
    print(f"DISASTER RESPONSE BASELINE AGENT — Task {task_level}")
    print(f"{'='*60}")

    # Reset
    r = requests.post(f"{base_url}/reset", json={"action": {}}, timeout=30)
    data = r.json()
    obs = data.get("observation", data)

    print(f"Task: {obs.get('task_name', '')}")
    print(f"Max steps: {obs.get('max_steps', 1)}")

    action = BASE_ACTIONS.get(task_level, BASE_ACTIONS[1])
    total_score = 0.0
    step = 0
    done = False

    while not done:
        step += 1
        print(f"\n--- Step {step} ---")
        print(f"Situation: {obs.get('situation_report','')[:200]}...")
        print(f"Resources: {obs.get('available_resources', {})}")
        print(f"\nAgent deploying: {action['resource_allocations']}")
        print(f"Priority: {action['priority_areas']}")

        r = requests.post(f"{base_url}/step", json={"action": action}, timeout=30)
        result = r.json()
        obs = result.get("observation", result)
        reward = result.get("reward", obs.get("reward", 0))
        done = result.get("done", obs.get("done", True))
        score = obs.get("step_score", reward * 10)
        total_score += reward

        print(f"\nStep score: {score:.1f}/10  (reward: {reward:.2f})")
        print(f"Feedback: {obs.get('feedback','')[:300]}")
        print(f"Casualties prevented: {obs.get('casualties_prevented', 0)}")

    final = obs.get("final_score", total_score / step)
    print(f"\n{'='*60}")
    print(f"EPISODE COMPLETE")
    print(f"Final Score: {final:.2f}/10")
    print(f"Normalised:  {final/10:.3f}/1.0")
    print(f"{'='*60}")
    return final

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_url", default="http://localhost:7860")
    parser.add_argument("--task_level", type=int, default=0)
    args = parser.parse_args()

    if args.task_level == 0:
        scores = []
        for level in [1, 2, 3]:
            score = run_episode(args.base_url, level)
            scores.append(score)
        print(f"\nOVERALL AVERAGE: {sum(scores)/len(scores):.2f}/10")
    else:
        run_episode(args.base_url, args.task_level)

if __name__ == "__main__":
    main()