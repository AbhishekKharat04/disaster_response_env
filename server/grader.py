"""
Disaster Response Coordination Environment — Final Grader v3
Checks:
  1. Allocation-priority alignment (does top priority get most resources?)
  2. Hard contradiction penalties (lower zone over-served vs higher)
  3. Semantic area-resource justification
  4. Vulnerability group acknowledgment
  5. Temporal adaptation (step 2+)
  6. State mutation tracking (ignored zones worsen)
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


# Tracks severity increases from ignored zones across steps
_severity_memory: Dict[str, Dict[str, int]] = {}


def grade_response(
    task: Dict[str, Any],
    action,
    step: int,
    situation: Dict[str, Any],
    history: List[Dict[str, Any]],
) -> Tuple[float, str, int]:

    score, feedback, casualties = _core_grade(task, action, step, situation, history)

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if _openai_available and api_key:
        try:
            llm_score, llm_fb = _llm_score(task, action, step, situation, history, score)
            final = round(0.4 * score + 0.6 * llm_score, 2)
            feedback = f"[Rule:{score}/10] [LLM:{llm_score}/10]\n\n{llm_fb}"
        except Exception as e:
            logger.warning("LLM grading failed: %s", e)
            final = score
    else:
        final = score

    return final, feedback, casualties


def _get_area_resources(area: Dict, allocations: Dict[str, int]) -> int:
    """How many resources matching this area's needs were allocated."""
    needs = area.get("needs", [])
    return sum(allocations.get(r, 0) for r in needs)


def _core_grade(task, action, step, situation, history) -> Tuple[float, str, int]:
    score = 0.0
    lines = []
    casualties = 0

    affected: List[Dict] = situation.get("affected_areas", [])
    available: Dict[str, int] = situation.get("resources", {})
    allocations: Dict[str, int] = action.resource_allocations or {}
    priority_list: List[str] = action.priority_areas or []
    plan = (action.response_plan or "").lower()
    rationale = (action.rationale or "").lower()
    combined = plan + " " + rationale

    episode_id = task.get("name", "default")
    if episode_id not in _severity_memory:
        _severity_memory[episode_id] = {}

    # Apply severity mutations from previous ignored zones
    for area in affected:
        name = area["name"]
        if name in _severity_memory[episode_id]:
            area["severity"] = min(5, area["severity"] + _severity_memory[episode_id][name])

    sorted_areas = sorted(affected, key=lambda a: a.get("severity", 0), reverse=True)
    total_available = max(1, sum(available.values()))
    total_allocated = sum(allocations.values())

    # ── CHECK 1: ALLOCATION-PRIORITY ALIGNMENT ─────────────────────────────
    # Top priority area must receive MORE relevant resources than lower areas.
    # This is the core check — not keyword presence.
    if len(sorted_areas) >= 2:
        top_area = sorted_areas[0]
        second_area = sorted_areas[1]
        top_resources = _get_area_resources(top_area, allocations)
        second_resources = _get_area_resources(second_area, allocations)

        if top_resources > second_resources:
            score += 3.0
            casualties += top_area.get("estimated_casualties_if_ignored", 0) // 2
            lines.append(
                f"✅ ALIGNMENT: Highest-severity area '{top_area['name']}' "
                f"received {top_resources} matching resources vs "
                f"{second_resources} for '{second_area['name']}'. (+3.0)"
            )
        elif top_resources == second_resources and top_resources > 0:
            score += 1.5
            lines.append(
                f"⚠️  PARTIAL: Top area tied with second area on resources. "
                f"Should prioritise more strongly. (+1.5)"
            )
        else:
            # Hard penalty: lower-priority zone over-served
            score -= 1.0
            lines.append(
                f"❌ CONTRADICTION: '{second_area['name']}' (severity {second_area['severity']}) "
                f"received {second_resources} resources but "
                f"'{top_area['name']}' (severity {top_area['severity']}) "
                f"only got {top_resources}. Hard penalty. (-1.0)"
            )

    # ── CHECK 2: STATED PRIORITY vs ACTUAL ALLOCATION ─────────────────────
    # Agent's top stated priority must match the area getting most resources.
    if priority_list and affected:
        stated_top = priority_list[0].lower()
        # Find which area got the most matching resources
        area_scores_map = {
            a["name"]: _get_area_resources(a, allocations)
            for a in affected
        }
        actual_top_name = max(area_scores_map, key=area_scores_map.get)

        stated_matches_actual = any(
            w in stated_top
            for w in actual_top_name.lower().split()
            if len(w) > 3
        ) or any(
            w in actual_top_name.lower()
            for w in stated_top.split()
            if len(w) > 3
        )

        if stated_matches_actual:
            score += 2.0
            lines.append(
                f"✅ CONSISTENCY: Stated priority '{priority_list[0]}' "
                f"matches actual top-allocated area '{actual_top_name}'. (+2.0)"
            )
        else:
            score -= 1.5
            lines.append(
                f"❌ INCONSISTENCY: Stated priority '{priority_list[0]}' "
                f"but most resources went to '{actual_top_name}'. Hard penalty. (-1.5)"
            )

    # ── CHECK 3: OVER-ALLOCATION CHECK ────────────────────────────────────
    over = [f"{r}({allocations[r]}>{available.get(r,0)})"
            for r in allocations if allocations[r] > available.get(r, 0)]
    if over:
        score -= 1.5
        lines.append(f"❌ Over-allocated resources: {', '.join(over)}. (-1.5)")
    else:
        utilisation = total_allocated / total_available
        if utilisation >= 0.65:
            score += 1.5
            casualties += int(utilisation * 6)
            lines.append(f"✅ Resource utilisation: {utilisation:.0%}. (+1.5)")
        elif utilisation >= 0.35:
            score += 0.75
            lines.append(f"⚠️  Moderate utilisation: {utilisation:.0%}. (+0.75)")
        else:
            lines.append(f"❌ Low utilisation: {utilisation:.0%}. Deploy more.")

    # ── CHECK 4: RATIONALE JUSTIFIES SPECIFIC ALLOCATION ──────────────────
    # Check that rationale mentions BOTH the area AND the resource going there.
    resource_names = [r.replace("_", " ") for r in available.keys()]
    area_keywords = []
    for area in affected:
        area_keywords.extend([w for w in area["name"].lower().split() if len(w) > 3])

    r_hits = sum(1 for r in resource_names if r in combined)
    a_hits = sum(1 for kw in set(area_keywords) if kw in combined)

    if r_hits >= 2 and a_hits >= 2:
        score += 2.0
        lines.append(
            f"✅ Rationale references specific resources ({r_hits}) "
            f"and areas ({a_hits}). (+2.0)"
        )
    elif r_hits >= 1 and a_hits >= 1:
        score += 1.0
        lines.append(f"⚠️  Partial justification. Mention more specific resources+areas. (+1.0)")
    else:
        lines.append("❌ Rationale too generic — no specific resource-to-area reasoning.")

    # ── CHECK 5: VULNERABILITY GROUPS / TEMPORAL ADAPTATION ───────────────
    if step == 1:
        vuln_kw = ["child", "elderly", "hazmat", "ammonia", "icu",
                   "creche", "crèche", "nursing", "infant", "trapped"]
        found = [kw for kw in vuln_kw if kw in combined]
        if found:
            score += 1.5
            lines.append(f"✅ Vulnerable groups acknowledged: {found}. (+1.5)")
        else:
            lines.append("⚠️  No mention of vulnerable groups. Triage should flag these.")
    else:
        # Temporal: does agent adapt to situation changes?
        sit_desc = situation.get("description", "").lower()
        change_kw = ["new", "update", "aftershock", "critical", "collapsed",
                     "worsening", "now", "confirmed", "reported", "additional"]
        if any(kw in sit_desc for kw in change_kw):
            adapted = any(kw in combined for kw in change_kw)
            if adapted:
                score += 1.5
                lines.append("✅ Agent acknowledged and adapted to new developments. (+1.5)")
            else:
                lines.append("❌ Situation changed but agent ignored new developments.")
        else:
            score += 1.5
            lines.append("✅ Situation tracking consistent. (+1.5)")

    # ── STATE MUTATION: track ignored zones ───────────────────────────────
    # Zones that received 0 matching resources get +1 severity next step
    _severity_memory[episode_id] = {}
    for area in affected:
        name = area["name"]
        res = _get_area_resources(area, allocations)
        if res == 0 and area.get("severity", 0) >= 3:
            _severity_memory[episode_id][name] = 1
            lines.append(
                f"⚠️  STATE: '{name}' ignored this step — "
                f"severity will increase next step."
            )
        else:
            casualties += min(res, 3)

    score = max(0.0, min(10.0, round(score, 2)))
    return score, "\n".join(lines), casualties


_SYSTEM = """You are a strict emergency response evaluator.
Score 0-10. Be harsh. Only give 8+ for near-optimal decisions.
Reply ONLY with JSON: {"score": float, "strengths": [...], "weaknesses": [...], "summary": "..."}"""


def _llm_score(task, action, step, situation, history, rule_score):
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    prompt = f"""
TASK: {task['name']} Step {step}/{task['max_steps']}
SITUATION: {situation.get('description','')[:600]}
RESOURCES AVAILABLE: {json.dumps(situation.get('resources',{}))}
AREAS: {json.dumps(situation.get('affected_areas',[]),indent=2)[:500]}

AGENT:
Plan: {action.response_plan[:400]}
Allocations: {json.dumps(action.resource_allocations)}
Priority: {action.priority_areas}
Rationale: {action.rationale[:300]}

Rule score: {rule_score}/10

Check: Does top priority actually get most resources?
Any contradictions between stated plan and allocations?
Are vulnerable groups (children/elderly/hazmat) addressed?
"""
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"system","content":_SYSTEM},{"role":"user","content":prompt}],
        temperature=0.1, max_tokens=400,
    )
    raw = r.choices[0].message.content.strip()
    if "```" in raw:
        raw = raw.split("```")[1].lstrip("json").strip()
    p = json.loads(raw)
    s = max(0.0, min(10.0, float(p.get("score", rule_score))))
    fb = f"📊 {s}/10 — {p.get('summary','')}"
    if p.get("strengths"):
        fb += "\n✅ " + " | ".join(p["strengths"])
    if p.get("weaknesses"):
        fb += "\n⚠️  " + " | ".join(p["weaknesses"])
    return s, fb