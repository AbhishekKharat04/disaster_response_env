"""
Disaster Response Coordination Environment — Upgraded Grader
Semantic scoring, contradiction detection, consequence tracking.
No word-count gaming possible.
"""
from __future__ import annotations
import os, json, logging
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

try:
    from openai import OpenAI
    _openai_available = True
except ImportError:
    _openai_available = False


def grade_response(
    task: Dict[str, Any],
    action,
    step: int,
    situation: Dict[str, Any],
    history: List[Dict[str, Any]],
) -> Tuple[float, str, int]:
    score, feedback, casualties = _semantic_score(task, action, step, situation, history)

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if _openai_available and api_key:
        try:
            llm_score, llm_feedback = _llm_score(task, action, step, situation, history, score)
            final_score = round(0.4 * score + 0.6 * llm_score, 2)
            feedback = f"[Rule: {score}/10] [LLM: {llm_score}/10]\n\n{llm_feedback}"
        except Exception as exc:
            logger.warning("LLM grading failed: %s", exc)
            final_score = score
    else:
        final_score = score

    return final_score, feedback, casualties


def _semantic_score(
    task, action, step, situation, history
) -> Tuple[float, str, int]:
    """
    Semantic grader — checks WHAT the agent says, not HOW MUCH.
    Detects contradictions, checks specific resource/area mentions,
    penalises ignoring critical zones.
    """
    score = 0.0
    feedback_lines = []
    casualties = 0

    affected_areas: List[Dict] = situation.get("affected_areas", [])
    available: Dict[str, int] = situation.get("resources", {})
    allocations: Dict[str, int] = action.resource_allocations or {}
    priority_list: List[str] = [p.lower() for p in (action.priority_areas or [])]
    plan: str = (action.response_plan or "").lower()
    rationale: str = (action.rationale or "").lower()
    combined_text = plan + " " + rationale

    sorted_areas = sorted(affected_areas, key=lambda a: a.get("severity", 0), reverse=True)
    sev5_areas = [a for a in sorted_areas if a.get("severity", 0) >= 5]
    sev4_areas = [a for a in sorted_areas if a.get("severity", 0) == 4]
    total_available = sum(available.values())

    # ── CHECK 1: Severity-5 areas explicitly named in plan (not just generic) ──
    sev5_named = 0
    for area in sev5_areas:
        area_name = area["name"].lower()
        # Check if specific area name OR specific location keyword mentioned
        keywords = [w for w in area_name.replace("—", "").split() if len(w) > 3]
        if any(kw in combined_text for kw in keywords):
            sev5_named += 1
            casualties += area.get("estimated_casualties_if_ignored", 0) // 2

    if sev5_areas:
        if sev5_named == len(sev5_areas):
            score += 2.5
            feedback_lines.append(f"✅ All severity-5 areas explicitly named in plan. (+2.5)")
        elif sev5_named > 0:
            score += 1.2
            feedback_lines.append(f"⚠️  Only {sev5_named}/{len(sev5_areas)} severity-5 areas mentioned. (+1.2)")
        else:
            feedback_lines.append(f"❌ No severity-5 areas explicitly named — plan is too generic.")

    # ── CHECK 2: Contradiction detection ─────────────────────────────────────
    # Agent says area X is priority but allocates 0 resources matching its needs
    contradiction_found = False
    if priority_list and allocations:
        top_priority = priority_list[0] if priority_list else ""
        # Find the area object matching top priority
        top_area_obj = None
        for area in affected_areas:
            if any(w in top_priority for w in area["name"].lower().split() if len(w) > 3):
                top_area_obj = area
                break

        if top_area_obj:
            needed = top_area_obj.get("needs", [])
            allocated_to_needs = sum(allocations.get(r, 0) for r in needed)
            if allocated_to_needs == 0 and needed:
                contradiction_found = True
                feedback_lines.append(
                    f"❌ CONTRADICTION: '{priority_list[0]}' listed as top priority "
                    f"but needs {needed} — none allocated! (-1.5)"
                )
                score -= 1.5

    if not contradiction_found and priority_list:
        score += 1.5
        feedback_lines.append("✅ No contradiction between stated priorities and allocations. (+1.5)")

    # ── CHECK 3: Resource scarcity — agent must make real tradeoffs ───────────
    total_allocated = sum(allocations.values())
    over_allocated_resources = []
    for resource, count in allocations.items():
        if count > available.get(resource, 0):
            over_allocated_resources.append(f"{resource}({count}>{available.get(resource,0)})")

    if over_allocated_resources:
        score -= 1.0
        feedback_lines.append(f"❌ Over-allocated: {', '.join(over_allocated_resources)}. (-1.0)")
    else:
        utilisation = total_allocated / total_available if total_available > 0 else 0
        if utilisation >= 0.7:
            score += 2.0
            casualties += int(utilisation * 8)
            feedback_lines.append(f"✅ Strong resource utilisation: {utilisation:.0%}. (+2.0)")
        elif utilisation >= 0.4:
            score += 1.0
            feedback_lines.append(f"⚠️  Moderate resource utilisation: {utilisation:.0%}. (+1.0)")
        else:
            feedback_lines.append(f"❌ Low utilisation: {utilisation:.0%}. Deploy more resources.")

    # ── CHECK 4: Specific resource-to-area logic in rationale ────────────────
    # Check if rationale mentions WHY specific resources go to specific places
    resource_names = list(available.keys())
    area_keywords = []
    for area in affected_areas:
        area_keywords.extend([w for w in area["name"].lower().split() if len(w) > 3])

    resources_justified = sum(1 for r in resource_names if r.replace("_", " ") in combined_text)
    areas_justified = sum(1 for kw in set(area_keywords) if kw in combined_text)

    if resources_justified >= 2 and areas_justified >= 2:
        score += 2.0
        feedback_lines.append(
            f"✅ Rationale ties specific resources to specific areas "
            f"({resources_justified} resources, {areas_justified} area refs). (+2.0)"
        )
    elif resources_justified >= 1 or areas_justified >= 1:
        score += 1.0
        feedback_lines.append(
            f"⚠️  Partial justification — mention more specific resources/areas. (+1.0)"
        )
    else:
        feedback_lines.append(
            "❌ Rationale too generic — does not name specific resources or areas."
        )

    # ── CHECK 5: Temporal consequence awareness (steps 2+) ───────────────────
    if step > 1 and history:
        prev_action = history[-1].get("action", {}) if history else {}
        prev_priorities = [p.lower() for p in prev_action.get("priority_areas", [])]
        # Check if agent adapts when situation changes
        situation_desc = situation.get("description", "").lower()
        change_keywords = ["new", "update", "aftershock", "critical", "now", "collapsed", "worsening"]
        situation_changed = any(kw in situation_desc for kw in change_keywords)

        if situation_changed:
            # Check if agent mentions the new development
            new_dev_mentioned = any(kw in combined_text for kw in change_keywords)
            if new_dev_mentioned:
                score += 1.5
                feedback_lines.append("✅ Agent adapted to new situation developments. (+1.5)")
            else:
                feedback_lines.append(
                    "⚠️  Situation changed but agent didn't acknowledge new developments."
                )
        else:
            score += 1.5
            feedback_lines.append("✅ Situation tracking consistent. (+1.5)")
    elif step == 1:
        # For step 1: check if agent mentions vulnerability groups (children, elderly, hazmat)
        vulnerability_keywords = ["child", "elderly", "hazmat", "ammonia", "icu", "infant", "crèche", "creche", "nursing"]
        vulnerabilities_mentioned = [kw for kw in vulnerability_keywords if kw in combined_text]
        if vulnerabilities_mentioned:
            score += 1.5
            feedback_lines.append(
                f"✅ Vulnerable groups acknowledged: {vulnerabilities_mentioned}. (+1.5)"
            )
        else:
            feedback_lines.append(
                "⚠️  No mention of vulnerable groups (elderly, children, hazmat). "
                "Triage should prioritise these."
            )

    score = max(0.0, min(10.0, round(score, 2)))
    feedback = "\n".join(feedback_lines)
    return score, feedback, casualties


_GRADER_SYSTEM_PROMPT = """You are an expert emergency response evaluator.
Score the agent's disaster response action from 0-10.
Be rigorous — high scores require specific, justified, non-contradictory decisions.

Respond ONLY with valid JSON:
{
  "score": <float 0-10>,
  "strengths": ["..."],
  "weaknesses": ["..."],
  "summary": "..."
}"""


def _llm_score(task, action, step, situation, history, rule_score) -> Tuple[float, str]:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    user_prompt = f"""
TASK: {task['name']} (Level {task['level']}, Step {step}/{task['max_steps']})
SITUATION: {situation.get('description', '')[:800]}
AVAILABLE: {json.dumps(situation.get('resources', {}))}
AREAS: {json.dumps(situation.get('affected_areas', []), indent=2)[:600]}

AGENT ACTION:
Plan: {action.response_plan}
Allocations: {json.dumps(action.resource_allocations)}
Priority: {action.priority_areas}
Rationale: {action.rationale}

Rule-based score: {rule_score}/10

Evaluate: Are priorities correct? Any contradictions? Are vulnerable groups addressed?
Are resource tradeoffs justified? Does reasoning match allocations?
"""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": _GRADER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1, max_tokens=500,
    )
    raw = resp.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    parsed = json.loads(raw)
    score = max(0.0, min(10.0, float(parsed.get("score", rule_score))))
    lines = [f"📊 {score}/10 — {parsed.get('summary', '')}"]
    if parsed.get("strengths"):
        lines.append("✅ " + " | ".join(parsed["strengths"]))
    if parsed.get("weaknesses"):
        lines.append("⚠️  " + " | ".join(parsed["weaknesses"]))
    return score, "\n".join(lines)