"""
Disaster Response Coordination Environment — Scenarios
Three carefully designed tasks at increasing difficulty.
"""
from typing import List, Dict, Any

# ─────────────────────────────────────────────────────────────────────────────
# TASK 1 — Single-Incident Fire Response (Easy, 1 step)
# Goal: Correctly triage a burning apartment building with limited resources.
# ─────────────────────────────────────────────────────────────────────────────
TASK_1 = {
    "level": 1,
    "name": "Apartment Building Fire",
    "description": (
        "A 10-storey apartment building in downtown Bengaluru is on fire. "
        "Flames have engulfed floors 3-5 and 8-9. Two groups of residents are trapped. "
        "You have ONE step to deploy all available resources. "
        "Your goal is to maximise lives saved and prevent fire spread."
    ),
    "max_steps": 1,
    "initial_situation": {
        "description": (
            "INCIDENT REPORT — 14:32 hrs\n"
            "Location: Sunrise Apartments, 10-storey residential block.\n"
            "Fire reported on floors 3-5 (fully engulfed) and floor 8-9 (spreading).\n\n"
            "AREA A — Floors 3-5 (Lower Fire Zone):\n"
            "  • ~12 residents trapped, smoke inhalation risk\n"
            "  • Fire spread rate: HIGH\n"
            "  • Stairwell B still accessible\n"
            "  • 3 residents confirmed with injuries\n\n"
            "AREA B — Floors 8-9 (Upper Fire Zone):\n"
            "  • ~25 residents trapped, including 6 elderly and 4 children\n"
            "  • Floor 8 has a crèche — children unaccounted for\n"
            "  • Elevator shaft acting as chimney, accelerating fire\n"
            "  • Stairwell A cut off by smoke\n\n"
            "AREA C — Surrounding Streets:\n"
            "  • Pedestrians injured by debris (est. 8 minor injuries)\n"
            "  • Gas main rupture risk on the east side\n\n"
            "You have 1 step to deploy all resources."
        ),
        "resources": {
            "ambulances": 5,
            "fire_trucks": 3,
            "rescue_teams": 2,
            "paramedic_units": 2,
        },
        "affected_areas": [
            {
                "name": "Area A — Floors 3-5",
                "severity": 4,
                "population_at_risk": 12,
                "needs": ["fire_trucks", "rescue_teams", "ambulances"],
                "infrastructure_damage": "Stairwell B accessible, structure unstable",
                "estimated_casualties_if_ignored": 6,
            },
            {
                "name": "Area B — Floors 8-9",
                "severity": 5,
                "population_at_risk": 25,
                "needs": ["rescue_teams", "ambulances", "paramedic_units"],
                "infrastructure_damage": "Elevator shaft on fire, Stairwell A blocked",
                "estimated_casualties_if_ignored": 15,
            },
            {
                "name": "Area C — Surrounding Streets",
                "severity": 2,
                "population_at_risk": 8,
                "needs": ["ambulances", "paramedic_units"],
                "infrastructure_damage": "Gas main rupture risk",
                "estimated_casualties_if_ignored": 2,
            },
        ],
    },
    "situation_updates": {},
    "optimal_allocation": {
        "ambulances": {"Area B — Floors 8-9": 2, "Area A — Floors 3-5": 2, "Area C — Surrounding Streets": 1},
        "fire_trucks": {"Area A — Floors 3-5": 2, "Area B — Floors 8-9": 1},
        "rescue_teams": {"Area B — Floors 8-9": 2},
        "paramedic_units": {"Area B — Floors 8-9": 1, "Area C — Surrounding Streets": 1},
    },
    "scoring_rubric": {
        "area_b_prioritized": 3.0,       # Area B (highest severity) gets ≥2 rescue resources
        "area_a_not_ignored": 2.0,       # Area A gets ≥1 fire truck and ≥1 ambulance
        "area_c_addressed": 1.0,         # Area C gets ≥1 medical resource
        "no_over_allocation": 2.0,       # Does not try to allocate more than available
        "clear_rationale": 1.5,          # Rationale explicitly mentions severity, children, elderly
        "systematic_approach": 0.5,      # Plan is structured and coherent
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# TASK 2 — Earthquake Multi-District Response (Medium, 3 steps)
# Goal: Manage evolving earthquake aftermath across 4 districts over 3 steps.
# ─────────────────────────────────────────────────────────────────────────────
TASK_2 = {
    "level": 2,
    "name": "Urban Earthquake — 4 Districts",
    "description": (
        "A magnitude 6.8 earthquake has struck the city at 06:15 hrs. "
        "Four districts report casualties and structural collapses. "
        "You must coordinate response over 3 time-steps (each = 2 hrs of real time). "
        "New information arrives each step — adapt accordingly. "
        "Maximise lives saved with limited resources across all 3 steps."
    ),
    "max_steps": 3,
    "initial_situation": {
        "description": (
            "EARTHQUAKE RESPONSE — Step 1 of 3 (06:15-08:15 hrs)\n\n"
            "Seismic event M6.8 — epicentre 12 km east of city centre.\n\n"
            "DISTRICT NORTH — Residential blocks:\n"
            "  • 3 buildings collapsed, ~40 trapped\n"
            "  • Roads accessible, rescue possible\n"
            "  • Hospital operating normally nearby\n\n"
            "DISTRICT SOUTH — Commercial area:\n"
            "  • 1 office tower partially collapsed (floors 1-4)\n"
            "  • ~60 office workers trapped, many injured\n"
            "  • Gas leak reported — explosion risk\n\n"
            "DISTRICT EAST — Slum settlement:\n"
            "  • ~80 informal structures damaged or collapsed\n"
            "  • Very high density, est. 200 displaced\n"
            "  • No confirmed death toll yet\n\n"
            "DISTRICT WEST — City hospital:\n"
            "  • Hospital structurally damaged, 120 patients need evacuation\n"
            "  • ICU patients require ventilator-equipped transport\n"
            "  • Backup generators running but fuel low\n\n"
            "Deploy wisely — more info will arrive in Step 2."
        ),
        "resources": {
            "ambulances": 10,
            "rescue_teams": 6,
            "medical_units": 4,
            "heavy_equipment": 2,
        },
        "affected_areas": [
            {
                "name": "District North",
                "severity": 3,
                "population_at_risk": 40,
                "needs": ["rescue_teams", "ambulances"],
                "infrastructure_damage": "3 collapsed buildings, roads clear",
                "estimated_casualties_if_ignored": 12,
            },
            {
                "name": "District South",
                "severity": 5,
                "population_at_risk": 60,
                "needs": ["rescue_teams", "heavy_equipment", "ambulances"],
                "infrastructure_damage": "Partial collapse + gas leak",
                "estimated_casualties_if_ignored": 25,
            },
            {
                "name": "District East",
                "severity": 3,
                "population_at_risk": 200,
                "needs": ["rescue_teams", "medical_units"],
                "infrastructure_damage": "Dense informal housing collapse",
                "estimated_casualties_if_ignored": 20,
            },
            {
                "name": "District West — Hospital",
                "severity": 4,
                "population_at_risk": 120,
                "needs": ["ambulances", "medical_units"],
                "infrastructure_damage": "Hospital evacuation needed, ICU patients critical",
                "estimated_casualties_if_ignored": 18,
            },
        ],
    },
    "situation_updates": {
        2: (
            "EARTHQUAKE RESPONSE — Step 2 of 3 (08:15-10:15 hrs)\n\n"
            "UPDATE: Gas leak in District South has been contained (if rescue team deployed).\n"
            "NEW INFO: Search teams in District East report 35 confirmed trapped — "
            "higher than initially estimated. 5 children found alive under debris.\n"
            "NEW INFO: District West hospital generator fuel critically low — "
            "30 ICU patients at risk within 2 hours.\n"
            "District North rescue: 18 of 40 already extracted (if resources were sent).\n\n"
            "Remaining resources after Step 1 deployments. Reassign as needed."
        ),
        3: (
            "EARTHQUAKE RESPONSE — Step 3 of 3 (10:15-12:15 hrs)\n\n"
            "CRITICAL: Aftershock M5.1 struck 09:45 hrs.\n"
            "District North: 2 more buildings partially collapsed — 15 new trapped.\n"
            "District South: Office tower now fully evacuated (if resources sent in Steps 1-2).\n"
            "District East: 28 rescued so far. 7 still trapped, 3 critical.\n"
            "District West: Hospital successfully evacuated or — fuel crisis continues.\n\n"
            "This is your final allocation. Make it count."
        ),
        "default": "Situation continues to evolve. Reassess and reallocate.",
    },
    "scoring_rubric": {
        "south_priority": 2.5,       # District South (gas leak + trapped) gets heavy equipment + rescue
        "west_hospital": 2.0,        # Hospital gets ambulances + medical units in Step 1 or 2
        "adaptive_reallocation": 2.0, # Agent updates priorities when new info arrives in Step 2
        "aftershock_response": 1.5,  # Agent responds to District North aftershock in Step 3
        "east_children": 1.0,        # Children in East get explicit mention/priority in Step 2
        "rationale_quality": 1.0,    # Structured, evidence-based justification each step
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# TASK 3 — Hurricane City Disaster (Hard, 5 steps)
# Goal: Full city-level coordination over 5 time-steps, resource scarcity,
#       conflicting priorities, dynamic evolving state.
# ─────────────────────────────────────────────────────────────────────────────
TASK_3 = {
    "level": 3,
    "name": "Category 5 Hurricane — City-Scale Response",
    "description": (
        "Hurricane Veera (Category 5) made landfall 2 hours ago. "
        "Six districts face simultaneous catastrophic damage. "
        "You are the Emergency Response Commander. "
        "Coordinate over 5 time-steps (each = 6 hours of real time). "
        "Resources are scarce — every allocation is a trade-off. "
        "Infrastructure damage worsens if not addressed quickly. "
        "Communicate clearly: your briefings will be read by field commanders."
    ),
    "max_steps": 5,
    "initial_situation": {
        "description": (
            "HURRICANE VEERA — DISASTER COMMAND BRIEFING\n"
            "T+2hrs — Initial Assessment\n\n"
            "ZONE 1 — COASTAL WARD (Highest Flood Risk):\n"
            "  • Storm surge 4.5m, 300 residents stranded on rooftops\n"
            "  • 2 nursing homes, est. 80 elderly residents, no evacuation yet\n"
            "  • Roads flooded — only boats/helicopters accessible\n"
            "  • Severity: CRITICAL (5/5)\n\n"
            "ZONE 2 — INDUSTRIAL DISTRICT:\n"
            "  • Chemical plant roof collapse — ammonia leak suspected\n"
            "  • 50 workers trapped, hazmat risk\n"
            "  • Adjacent residential area (200 people) needs evacuation\n"
            "  • Severity: CRITICAL (5/5)\n\n"
            "ZONE 3 — CENTRAL HOSPITAL:\n"
            "  • Main hospital, 400 patients, basement flooded\n"
            "  • ICU on ground floor — must be evacuated within 12 hrs\n"
            "  • 3 field hospital sites identified 5km away\n"
            "  • Severity: HIGH (4/5)\n\n"
            "ZONE 4 — RESIDENTIAL NORTH:\n"
            "  • 15 buildings structurally damaged, ~120 displaced\n"
            "  • 3 confirmed collapses, ~25 trapped under rubble\n"
            "  • Accessible by road\n"
            "  • Severity: HIGH (4/5)\n\n"
            "ZONE 5 — TRANSPORT HUB (Airport + Railway):\n"
            "  • Airport flooded, 600 stranded travellers\n"
            "  • Railway bridge damaged — city cut off from aid corridor\n"
            "  • Restoring bridge = massive multiplier on resource delivery\n"
            "  • Severity: MEDIUM (3/5)\n\n"
            "ZONE 6 — WATER TREATMENT PLANT:\n"
            "  • Main pump flooded — city water supply at risk\n"
            "  • 48-hr window before full contamination\n"
            "  • 2 engineers trapped on site\n"
            "  • Severity: MEDIUM (3/5)\n\n"
            "Deploy strategically. Situation will update every 6 hours."
        ),
        "resources": {
            "ambulances": 12,
            "rescue_teams": 8,
            "helicopters": 3,
            "field_hospitals": 2,
            "hazmat_units": 2,
            "boats": 5,
            "heavy_equipment": 3,
            "engineering_crews": 2,
        },
        "affected_areas": [
            {
                "name": "Zone 1 — Coastal Ward",
                "severity": 5,
                "population_at_risk": 380,
                "needs": ["boats", "helicopters", "ambulances"],
                "infrastructure_damage": "Roads flooded, nursing homes isolated",
                "estimated_casualties_if_ignored": 80,
            },
            {
                "name": "Zone 2 — Industrial District",
                "severity": 5,
                "population_at_risk": 250,
                "needs": ["hazmat_units", "rescue_teams", "ambulances"],
                "infrastructure_damage": "Chemical plant collapse, ammonia leak",
                "estimated_casualties_if_ignored": 60,
            },
            {
                "name": "Zone 3 — Central Hospital",
                "severity": 4,
                "population_at_risk": 400,
                "needs": ["ambulances", "field_hospitals", "rescue_teams"],
                "infrastructure_damage": "Basement flooded, ICU at risk",
                "estimated_casualties_if_ignored": 100,
            },
            {
                "name": "Zone 4 — Residential North",
                "severity": 4,
                "population_at_risk": 145,
                "needs": ["rescue_teams", "heavy_equipment", "ambulances"],
                "infrastructure_damage": "3 collapses, roads passable",
                "estimated_casualties_if_ignored": 25,
            },
            {
                "name": "Zone 5 — Transport Hub",
                "severity": 3,
                "population_at_risk": 600,
                "needs": ["engineering_crews", "heavy_equipment"],
                "infrastructure_damage": "Railway bridge damaged, airport flooded",
                "estimated_casualties_if_ignored": 5,
            },
            {
                "name": "Zone 6 — Water Treatment",
                "severity": 3,
                "population_at_risk": 500000,
                "needs": ["engineering_crews", "rescue_teams"],
                "infrastructure_damage": "Main pump flooded, contamination risk",
                "estimated_casualties_if_ignored": 3,
            },
        ],
    },
    "situation_updates": {
        2: (
            "HURRICANE VEERA — T+8hrs\n\n"
            "Zone 1: 140 coastal residents rescued (if boats deployed). "
            "Nursing homes still unreached — elderly deteriorating. "
            "New: Second storm band approaching — boats may be grounded in 6 hrs.\n\n"
            "Zone 2: Ammonia cloud confirmed. 200m exclusion zone established. "
            "If hazmat units not deployed: 12 workers dead, civilian evacuation failing.\n\n"
            "Zone 3: Hospital ICU evacuation 30% complete (if resources sent). "
            "Basement now 1.5m deep — ground floor access lost.\n\n"
            "Zone 4: 15 of 25 trapped rescued (if teams sent). "
            "New collapse reported — 8 additional people buried.\n\n"
            "Zone 5: Railway bridge repair crew reports 12-hr window to restore. "
            "Restoring bridge will DOUBLE resource capacity next step.\n\n"
            "Zone 6: Engineers still trapped. Water contamination at 20% threshold."
        ),
        3: (
            "HURRICANE VEERA — T+14hrs\n\n"
            "CRITICAL UPDATE: Second storm band hit. Boats grounded for 6 hrs.\n"
            "Zone 1: Nursing home roof collapsed — 20 elderly at extreme risk. "
            "Helicopters are the only option now.\n\n"
            "Zone 2: Hazmat neutralised (if addressed). Rescue ongoing.\n\n"
            "Zone 3: Hospital evacuation must complete this step or 50 ICU patients lost.\n\n"
            "Zone 4: Search complete, all survivors found. Medical treatment now priority.\n\n"
            "Zone 5: If bridge repaired — you now have +4 ambulances, +2 rescue teams available.\n\n"
            "Zone 6: Water contamination at 45%. City-wide boil water advisory issued."
        ),
        4: (
            "HURRICANE VEERA — T+20hrs\n\n"
            "Storm passing. Boats operational again.\n"
            "Zone 1: If nursing home not addressed — 35 casualties confirmed.\n"
            "Zone 3: Hospital evacuation complete or partial based on prior steps.\n"
            "Zone 6: Water contamination at 70% — civil unrest beginning.\n\n"
            "Priority now: stabilise all zones, transition from rescue to recovery.\n"
            "Assess resource reallocation for final push."
        ),
        5: (
            "HURRICANE VEERA — T+26hrs — FINAL STEP\n\n"
            "Storm has passed. Recovery phase begins.\n"
            "This is your final coordination window.\n\n"
            "Remaining threats:\n"
            "  • Any unrescued coastal residents face 6-hr survival window\n"
            "  • Water contamination crisis if plant not restored\n"
            "  • Displaced persons (est. 800) need shelter coordination\n\n"
            "Write your final situation report and resource deployment for hand-off to recovery teams."
        ),
        "default": "Situation continues to evolve. Reassess and reallocate.",
    },
    "scoring_rubric": {
        "hazmat_priority": 2.0,           # Hazmat units sent to Zone 2 early (Steps 1-2)
        "hospital_evacuation": 2.0,       # Hospital ICU addressed within Steps 1-3
        "nursing_home_rescue": 1.5,       # Helicopters/boats to Zone 1 nursing homes
        "bridge_repair_foresight": 1.0,   # Bridge repair prioritised as a force multiplier
        "water_treatment": 1.0,           # Zone 6 addressed before contamination threshold
        "adaptive_storm_response": 1.0,   # Boat redeployment when grounded in Step 3
        "communication_quality": 1.5,     # Clear, commander-ready briefings each step
    },
}

# All tasks in order
TASKS = [TASK_1, TASK_2, TASK_3]
