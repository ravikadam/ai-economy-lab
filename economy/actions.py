"""The world's action contract and the single source of truth for legality.

Policies, heuristics and human players all submit `Action`s. Masks
(agents/masks.py) and the human game UI are both derived from the functions
here, so "what is legal" can never diverge between them.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING

from entities.firms import DEV_STAGES, LIVE_STAGES, RESERVE_MONTHS, FirmState, Stage

if TYPE_CHECKING:
    from economy.world import World


class Verb(IntEnum):
    NOOP = 0
    START = 1      # open a project on an opportunity with a model + budget tier
    PILOT = 2      # take a PROTOTYPE_READY project to paying pilot customers
    LAUNCH = 3     # take a PILOT_READY project to general availability
    SCALE = 4      # invest in sales/support capacity for a DEPLOYED product
    ABANDON = 5    # close a project (live products lose their customers)
    RETRY = 6      # re-attempt a FAILED project (may switch model / tier)
    PAUSE = 7      # toggle pause on an in-development project (stops burn)


N_VERBS = len(Verb)
SLOT_VERBS = {Verb.PILOT, Verb.LAUNCH, Verb.SCALE, Verb.ABANDON, Verb.RETRY, Verb.PAUSE}
SPEND_VERBS = {Verb.START, Verb.PILOT, Verb.LAUNCH, Verb.SCALE, Verb.RETRY}


@dataclass
class Action:
    verb: Verb = Verb.NOOP
    opportunity_id: int = -1   # START
    slot: int = -1             # project-slot verbs
    model_id: int = 0          # START / RETRY
    tier: int = 1              # START / RETRY
    reserve: int | None = None  # new reserve level (applies from next month)

    def to_dict(self) -> dict:
        return {"verb": Verb(self.verb).name, "opportunity_id": self.opportunity_id, "slot": self.slot,
                "model_id": self.model_id, "tier": self.tier, "reserve": self.reserve}

    @staticmethod
    def from_dict(d: dict) -> "Action":
        v = d.get("verb", "NOOP")
        verb = Verb[v] if isinstance(v, str) else Verb(int(v))
        return Action(verb, int(d.get("opportunity_id", -1)), int(d.get("slot", -1)),
                      int(d.get("model_id", 0)), int(d.get("tier", 1)), d.get("reserve"))


def slot_ok(firm: FirmState, verb: Verb, slot: int) -> bool:
    if slot < 0 or slot >= len(firm.projects):
        return False
    st = firm.projects[slot].stage
    if verb == Verb.PILOT:
        return st == Stage.PROTOTYPE_READY
    if verb == Verb.LAUNCH:
        return st == Stage.PILOT_READY
    if verb == Verb.SCALE:
        return st == Stage.DEPLOYED
    if verb == Verb.ABANDON:
        return True
    if verb == Verb.RETRY:
        return st == Stage.FAILED
    if verb == Verb.PAUSE:
        return st in DEV_STAGES or st == Stage.PAUSED
    return False


def spend_cost(world: "World", firm: FirmState, action: Action) -> float:
    b = world.cfg.builders
    v = action.verb
    if v == Verb.START:
        return b.start_cost + b.tier_burn[action.tier]
    if v == Verb.PILOT:
        return b.pilot_cost
    if v == Verb.LAUNCH:
        return b.launch_cost
    if v == Verb.SCALE:
        return b.scale_cost
    if v == Verb.RETRY:
        return b.retry_cost + b.tier_burn[action.tier]
    return 0.0


def spending_capacity(world: "World", firm: FirmState) -> float:
    """Cash plus undrawn credit, minus the firm's own reserve floor."""
    cash = world.ledger.balance(firm.account)
    credit = max(0.0, world.cfg.builders.credit_limit - firm.debt)
    floor = RESERVE_MONTHS[firm.reserve_level] * world.monthly_burn(firm)
    return cash + credit - floor


def startable_opportunities(world: "World", firm: FirmState) -> list[int]:
    owned = {p.opportunity_id for p in firm.projects}
    claimed = world.claimed_opportunities() if not world.cfg.ablation.competition else set()
    return [o.opportunity_id for o in world.available_opportunities()
            if o.opportunity_id not in owned and o.opportunity_id not in claimed]


def check_action(world: "World", firm: FirmState, action: Action) -> tuple[bool, str]:
    b = world.cfg.builders
    v = action.verb
    if action.reserve is not None and not (0 <= int(action.reserve) < len(RESERVE_MONTHS)):
        return False, "reserve level out of range"
    if v == Verb.NOOP:
        return True, ""
    if v in (Verb.START, Verb.RETRY):
        if not (0 <= action.model_id < world.cfg.n_models):
            return False, "unknown model"
        if not (0 <= action.tier < world.cfg.n_tiers):
            return False, "unknown tier"
    if v == Verb.START:
        if len(firm.projects) >= b.max_projects:
            return False, "project slots full"
        if action.opportunity_id not in startable_opportunities(world, firm):
            return False, "opportunity not available to this firm"
    elif v in SLOT_VERBS:
        if not slot_ok(firm, v, action.slot):
            return False, f"slot {action.slot} not eligible for {v.name}"
    if v in SPEND_VERBS and spend_cost(world, firm, action) > spending_capacity(world, firm) + 1e-9:
        return False, "insufficient funds above reserve floor"
    return True, ""


def legal_verbs(world: "World", firm: FirmState) -> list[bool]:
    """Is there at least one legal parameterisation of each verb?"""
    b = world.cfg.builders
    cap = spending_capacity(world, firm)
    out = [False] * N_VERBS
    out[Verb.NOOP] = True
    min_tier_cost = min(b.tier_burn)
    out[Verb.START] = (len(firm.projects) < b.max_projects and cap >= b.start_cost + min_tier_cost
                       and bool(startable_opportunities(world, firm)))
    for v in SLOT_VERBS:
        if any(slot_ok(firm, v, s) for s in range(len(firm.projects))):
            if v in SPEND_VERBS:
                probe = Action(v, tier=int(min(range(world.cfg.n_tiers), key=lambda t: b.tier_burn[t])))
                out[v] = spend_cost(world, firm, probe) <= cap
            else:
                out[v] = True
    return out


def legal_tiers(world: "World", firm: FirmState, verb: Verb) -> list[bool]:
    cap = spending_capacity(world, firm)
    return [spend_cost(world, firm, Action(verb, tier=t)) <= cap for t in range(world.cfg.n_tiers)]


__all__ = ["Action", "Verb", "N_VERBS", "SLOT_VERBS", "SPEND_VERBS", "check_action", "legal_verbs",
           "legal_tiers", "slot_ok", "spend_cost", "spending_capacity", "startable_opportunities",
           "LIVE_STAGES"]
