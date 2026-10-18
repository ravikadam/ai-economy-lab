"""Coaching rules: turn each decision into implementation-team lessons.

Every rule reads only what the player could see (signals, public prices, own
state) — never hidden difficulty — so feedback is about decision quality given
information, not about luck. Each lesson maps to an implementation playbook
principle used in debriefs.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from agents.heuristics import estimate
from economy.actions import Action, Verb
from entities.firms import Stage

PRINCIPLES = {
    "qualify": "Qualify the use case: value × reachable customers before build",
    "feasibility": "Check the capability gap before committing a team",
    "compliance": "Reliability/compliance requirements are gates, not nice-to-haves",
    "unit_economics": "Unit economics: inference + support must fit inside the price",
    "pilot_design": "Pilot where the customer is ready; confidence gates adoption",
    "momentum": "Keep momentum: ready projects lose value while idle",
    "scale": "Scale only after the product proves retention and margin",
    "sunk_cost": "Cut losers early; sunk cost is not a reason to continue",
    "root_cause": "After a failure, fix the root cause before retrying",
    "cash": "Protect runway: a dead firm delivers nothing to customers",
    "portfolio": "Balance the portfolio across risk and sectors",
}


@dataclass
class Lesson:
    code: str
    principle: str
    severity: str      # info | warn | critical | good
    message: str

    def to_dict(self) -> dict:
        return asdict(self)


def _L(code, sev, msg):
    return Lesson(code, code, sev, msg)


def review_decision(world, firm, action: Action) -> list[Lesson]:
    out: list[Lesson] = []
    v = action.verb
    b = world.cfg.builders
    if v == Verb.START:
        o = world.opps[action.opportunity_id]
        e = estimate(world, firm, o.opportunity_id, action.model_id, action.tier)
        sec = world.cfg.demand.sectors[o.sector]
        req = sec.reliability_floor + o.reliability_req
        rel = world.public["model_reliability"][action.model_id]
        if e.p_tech < 0.35:
            out.append(_L("feasibility", "warn",
                          f"Long shot: estimated technical success ≈{e.p_tech:.0%} (capability gap {e.cap_gap:+.2f}). "
                          "A higher budget tier, stronger model or easier use case would de-risk it."))
        if rel < req:
            out.append(_L("compliance", "critical",
                          f"{world.models[action.model_id].name} reliability {rel:.2f} is below the {req:.2f} this "
                          f"{sec.name} use case needs — the pilot will fail compliance unless the model improves."))
        if e.margin <= 0:
            out.append(_L("unit_economics", "critical",
                          f"Each customer loses ≈${-e.margin:.1f}k/month: inference + support exceed the price."))
        if e.ev < 0 and e.p_tech >= 0.35 and e.margin > 0:
            out.append(_L("qualify", "warn", "Expected value is negative: the market (value × customers) is too "
                                              "small for the development cost and time left."))
        if world.public["sector_confidence"][o.sector] < 0.3:
            out.append(_L("pilot_design", "info", f"{sec.name} enterprises are skeptical (confidence "
                                                   f"{world.public['sector_confidence'][o.sector]:.2f}); expect "
                                                   "pilot deferrals and slow adoption."))
        sectors = [world.opps[p.opportunity_id].sector for p in firm.projects]
        if sectors and all(s == o.sector for s in sectors) and len(sectors) >= 2:
            out.append(_L("portfolio", "info", "All projects in one sector: a confidence shock there hits everything."))
        if not out:
            out.append(_L("qualify", "good", f"Well-qualified start: p_tech≈{e.p_tech:.0%}, margin ≈${e.margin:.1f}k/"
                                             "customer/month, positive expected value."))
    elif v == Verb.SCALE:
        p = firm.projects[action.slot]
        if p.trailing_profit() <= 0 or p.customers < 4:
            out.append(_L("scale", "warn", "Scaling before retention and margin are proven multiplies losses."))
        else:
            out.append(_L("scale", "good", "Scaling a profitable product with traction."))
    elif v == Verb.RETRY:
        p = firm.projects[action.slot]
        if p.failure_kind == "commercial":
            opp = world.opps[p.opportunity_id]
            sec = world.cfg.demand.sectors[opp.sector]
            req = sec.reliability_floor + opp.reliability_req
            if world.public["model_reliability"][action.model_id] < req:
                out.append(_L("root_cause", "critical", "Retrying the pilot with a model that still misses the "
                                                         "reliability requirement repeats the failure."))
        elif p.failure_kind == "technical" and action.model_id == p.model_id and action.tier <= p.tier:
            out.append(_L("root_cause", "warn", "Same model and no extra budget: the retry only gains the "
                                                 "knowledge bonus from the last attempt."))
    elif v == Verb.ABANDON:
        p = firm.projects[action.slot]
        if p.is_live and p.trailing_profit() > 0:
            out.append(_L("sunk_cost", "warn", "Shutting down a profitable product — its customers churn."))
        else:
            out.append(_L("sunk_cost", "good", "Cutting a non-performing project frees budget and a slot."))
    return out


def review_state(world, firm) -> list[Lesson]:
    out: list[Lesson] = []
    rw = world.runway(firm)
    if rw < 3:
        out.append(_L("cash", "critical", f"Runway {rw:.1f} months — bankruptcy risk. Pause or cut projects."))
    elif rw < firm.profile.runway_target / 2:
        out.append(_L("cash", "warn", f"Runway {rw:.1f} months is below half your {firm.profile.runway_target:.0f}-"
                                      "month target."))
    for p in firm.projects:
        if p.stage in (Stage.PROTOTYPE_READY, Stage.PILOT_READY) and p.stage_age >= 2:
            out.append(_L("momentum", "warn", f"Use case #{p.opportunity_id} has been ready for {p.stage_age} "
                                              "months without moving to the next stage."))
        if p.is_live and len(p.profit_hist) >= 6 and p.trailing_profit(6) < 0:
            out.append(_L("sunk_cost", "warn", f"Use case #{p.opportunity_id} has lost money for 6 months."))
    return out
