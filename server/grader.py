"""
Disaster Response Coordination Environment — Grader
Hybrid grader: rule-based scoring + optional LLM refinement via OpenAI.
Returns a score (0-10), detailed feedback, and estimated casualties prevented.
"""
from __future__ import annotations

import os
import json
import logging
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

# ─── Try to import OpenAI (optional — falls back gracefully) ─────────────────
try:
    from openai import OpenAI  # openai >= 1.0
    _openai_available = True
except ImportError:
    _openai_available = False


# ─── Public entry point ───────────────────────────────────────────────────────

def grade_response(
    task: Dict[str, Any],
    action,
    step: int,
    situation: Dict[str, Any],
    history: List[Dict[str, Any]],
) -> Tuple[float, str, int]:
    """
    Grade the agent's action.

    Returns
    -------
    score : float
        Step score in [0, 10].
    feedback : str
        Detailed grader feedback.
    casualties_prevented : int
        Estimated casualties prevented by this action.
    """
    rubric = task.get("scoring_rubric", {})
    affected_areas: List[Dict] = situation.get("affected_areas", [])
    available: Dict[str, int] = situation.get("resources", {})
    allocations: Dict[str, int] = action.resource_allocations or {}

    # --- Rule-based component (always runs) ---
    rule_score, rule_feedback, casualties = _rule_based_score(
        task, action, step, affected_areas, available, allocations, rubric
    )

    # --- LLM refinement (runs only if OPENAI_API_KEY is set) ---
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if _openai_available and api_key:
        try:
            llm_score, llm_feedback = _llm_score(
                task, action, step, situation, history, rule_score
            )
            # Blend: 40% rule-based, 60% LLM (LLM has more nuance)
            final_score = round(0.4 * rule_score + 0.6 * llm_score, 2)
            final_feedback = f"[Rule-based: {rule_score}/10] [LLM: {llm_score}/10]\n\n{llm_feedback}"
        except Exception as exc:
            logger.warning("LLM grading failed, using rule-based only: %s", exc)
            final_score = rule_score
            final_feedback = rule_feedback
    else:
        final_score = rule_score
        final_feedback = rule_feedback

    return final_score, final_feedback, casualties


# ─── Rule-based scorer ────────────────────────────────────────────────────────

def _rule_based_score(
    task, action, step, affected_areas, available, allocations, rubric
) -> Tuple[float, str, int]:
    score = 0.0
    feedback_lines = []
    casualties = 0
    total_rubric_points = sum(rubric.values()) if rubric else 10.0

    # Sort areas by severity descending
    sorted_areas = sorted(affected_areas, key=lambda a: a.get("severity", 0), reverse=True)
    highest_severity_area = sorted_areas[0]["name"] if sorted_areas else ""
    second_severity_area = sorted_areas[1]["name"] if len(sorted_areas) > 1 else ""

    total_allocated = sum(allocations.values())
    total_available = sum(available.values())

    # ── Check 1: Highest-severity area is in priority_areas ──────────────────
    priority_list = [p.lower() for p in (action.priority_areas or [])]
    top_area_prioritised = any(
        highest_severity_area.lower() in p or p in highest_severity_area.lower()
        for p in priority_list
    )
    if top_area_prioritised:
        pts = min(2.5, total_rubric_points * 0.25)
        score += pts
        casualties += sorted_areas[0].get("estimated_casualties_if_ignored", 0) // 2
        feedback_lines.append(
            f"✅ Correctly prioritised highest-severity area '{highest_severity_area}'. (+{pts:.1f})"
        )
    else:
        feedback_lines.append(
            f"⚠️  Highest-severity area '{highest_severity_area}' not explicitly prioritised. "
            "Triage logic may be suboptimal."
        )

    # ── Check 2: Second area not completely ignored ───────────────────────────
    if second_severity_area:
        second_in_priority = any(
            second_severity_area.lower() in p or p in second_severity_area.lower()
            for p in priority_list
        )
        if second_in_priority or len(priority_list) >= 2:
            pts = min(1.5, total_rubric_points * 0.15)
            score += pts
            casualties += sorted_areas[1].get("estimated_casualties_if_ignored", 0) // 3
            feedback_lines.append(
                f"✅ Secondary area '{second_severity_area}' also addressed. (+{pts:.1f})"
            )
        else:
            feedback_lines.append(
                f"⚠️  Secondary area '{second_severity_area}' appears neglected."
            )

    # ── Check 3: Allocation doesn't exceed availability ───────────────────────
    over_allocated = False
    for resource, count in allocations.items():
        available_count = available.get(resource, 0)
        if count > available_count:
            over_allocated = True
            feedback_lines.append(
                f"❌ Over-allocated '{resource}': requested {count}, only {available_count} available."
            )
    if not over_allocated and total_allocated > 0:
        pts = min(2.0, total_rubric_points * 0.20)
        score += pts
        feedback_lines.append(
            f"✅ Resource allocation is within available limits. (+{pts:.1f})"
        )

    # ── Check 4: At least 50% of available resources deployed ────────────────
    if total_available > 0:
        utilisation = total_allocated / total_available
        if utilisation >= 0.5:
            pts = min(1.5, total_rubric_points * 0.15)
            score += pts
            casualties += int(utilisation * 5)
            feedback_lines.append(
                f"✅ Good resource utilisation: {utilisation:.0%} of available resources deployed. (+{pts:.1f})"
            )
        else:
            feedback_lines.append(
                f"⚠️  Low resource utilisation: only {utilisation:.0%} deployed. "
                "Consider deploying more resources."
            )

    # ── Check 5: Response plan is substantive (>50 words) ────────────────────
    plan_words = len((action.response_plan or "").split())
    if plan_words >= 50:
        pts = min(1.5, total_rubric_points * 0.15)
        score += pts
        feedback_lines.append(
            f"✅ Response plan is substantive ({plan_words} words). (+{pts:.1f})"
        )
    elif plan_words >= 20:
        pts = min(0.75, total_rubric_points * 0.075)
        score += pts
        feedback_lines.append(
            f"⚠️  Response plan is brief ({plan_words} words). More detail recommended. (+{pts:.1f})"
        )
    else:
        feedback_lines.append(
            f"❌ Response plan too brief ({plan_words} words). Elaborate on your strategy."
        )

    # ── Check 6: Rationale provided ──────────────────────────────────────────
    rationale_words = len((action.rationale or "").split())
    if rationale_words >= 30:
        pts = min(1.0, total_rubric_points * 0.10)
        score += pts
        feedback_lines.append(
            f"✅ Clear rationale provided ({rationale_words} words). (+{pts:.1f})"
        )
    else:
        feedback_lines.append(
            "⚠️  Rationale is weak or missing. Explain your triage logic."
        )

    score = min(10.0, round(score, 2))
    feedback = "\n".join(feedback_lines)
    return score, feedback, casualties


# ─── LLM scorer ──────────────────────────────────────────────────────────────

_GRADER_SYSTEM_PROMPT = """You are an expert emergency response evaluator.
You will score an AI agent's disaster response action on a scale of 0-10.
Be rigorous: high scores (8-10) require near-optimal resource allocation,
clear rationale, correct triage priorities, and anticipation of downstream consequences.

Always respond with ONLY valid JSON in this exact format:
{
  "score": <float 0-10>,
  "strengths": ["..."],
  "weaknesses": ["..."],
  "suggestions": ["..."],
  "summary": "..."
}"""


def _llm_score(
    task, action, step, situation, history, rule_score
) -> Tuple[float, str]:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    affected_areas_str = json.dumps(
        situation.get("affected_areas", []), indent=2
    )
    available_str = json.dumps(situation.get("resources", {}), indent=2)

    user_prompt = f"""
TASK: {task['name']} (Level {task['level']}, Step {step}/{task['max_steps']})
TASK DESCRIPTION: {task['description']}

CURRENT SITUATION:
{situation.get('description', '')}

AVAILABLE RESOURCES:
{available_str}

AFFECTED AREAS:
{affected_areas_str}

AGENT'S ACTION:
Response Plan: {action.response_plan}
Resource Allocations: {json.dumps(action.resource_allocations)}
Priority Areas: {action.priority_areas}
Rationale: {action.rationale}

RULE-BASED SCORE (for reference): {rule_score}/10

Please evaluate this action. Consider:
1. Are the highest-severity areas correctly prioritised?
2. Is the resource allocation optimal given the constraints?
3. Is the rationale evidence-based and logical?
4. Does the plan account for downstream consequences?
5. Is the communication clear enough for field commanders?

Respond ONLY with the JSON format specified.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": _GRADER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens=600,
    )

    raw = response.choices[0].message.content.strip()
    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    parsed = json.loads(raw)

    score = float(parsed.get("score", rule_score))
    score = max(0.0, min(10.0, score))

    lines = [f"📊 LLM Score: {score}/10", f"\n{parsed.get('summary', '')}"]
    if parsed.get("strengths"):
        lines.append("\n✅ Strengths:")
        lines.extend(f"  • {s}" for s in parsed["strengths"])
    if parsed.get("weaknesses"):
        lines.append("\n⚠️  Areas for improvement:")
        lines.extend(f"  • {w}" for w in parsed["weaknesses"])
    if parsed.get("suggestions"):
        lines.append("\n💡 Suggestions:")
        lines.extend(f"  • {s}" for s in parsed["suggestions"])

    return score, "\n".join(lines)
