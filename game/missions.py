"""Workshop missions: a scenario + a learning objective + a rival field."""
from __future__ import annotations

MISSIONS = [
    {
        "id": "m1_first_portfolio",
        "title": "Mission 1 · Your first AI portfolio",
        "scenario": "config/base.yaml",
        "months": 60,
        "rivals": "heuristics",
        "brief": "Five implementation firms compete for 60 enterprise AI use cases across customer operations, "
                 "software engineering and regulated workflows. Build a portfolio that survives and earns.",
        "objectives": ["qualify", "feasibility", "unit_economics", "momentum"],
    },
    {
        "id": "m2_regulated",
        "title": "Mission 2 · Winning regulated customers",
        "scenario": "config/base.yaml",
        "months": 72,
        "rivals": "rl",
        "brief": "Regulated enterprises pay the most but demand reliability. Rival firms are trained RL agents. "
                 "Choose models and timing so pilots clear compliance.",
        "objectives": ["compliance", "root_cause", "pilot_design"],
    },
    {
        "id": "m3_compute_crunch",
        "title": "Mission 3 · The compute crunch",
        "scenario": "config/scenarios/compute_crunch.yaml",
        "months": 84,
        "rivals": "mixed",
        "brief": "Fab shortages throttle compute. Prices spike and service degrades, so customers churn. Manage "
                 "unit economics and runway through supply shocks.",
        "objectives": ["unit_economics", "cash", "scale"],
    },
    {
        "id": "m4_full_cycle",
        "title": "Mission 4 · Full 10-year market cycle",
        "scenario": "config/base.yaml",
        "months": 120,
        "rivals": "rl",
        "brief": "A full decade: model breakthroughs, entrants, bankruptcies. Compete against the trained policy "
                 "and compare with what the AI coach would have done in your seat.",
        "objectives": ["portfolio", "scale", "sunk_cost", "cash"],
    },
]


def get_mission(mid: str) -> dict:
    for m in MISSIONS:
        if m["id"] == mid:
            return m
    raise KeyError(mid)
