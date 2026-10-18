"""Encode/decode between policy head indices and world `Action`s."""
from __future__ import annotations

from economy.actions import SLOT_VERBS, Action, Verb

# index tuple: (verb, candidate_idx, slot, model, tier, reserve)
IndexAction = tuple[int, int, int, int, int, int]


def decode(idx: IndexAction, cand: list[int]) -> Action:
    v, c, s, m, t, r = (int(x) for x in idx)
    verb = Verb(v)
    a = Action(verb=verb, reserve=r)
    if verb == Verb.START:
        a.opportunity_id = cand[c] if c < len(cand) else -1
        a.model_id, a.tier = m, t
    elif verb in SLOT_VERBS:
        a.slot = s
        if verb == Verb.RETRY:
            a.model_id, a.tier = m, t
    return a


def encode(a: Action, cand: list[int], current_reserve: int) -> IndexAction | None:
    """Map a world Action (e.g. from a human or heuristic) to head indices; None if START target not in candidates."""
    v = int(a.verb)
    c = 0
    if a.verb == Verb.START:
        if a.opportunity_id not in cand:
            return None
        c = cand.index(a.opportunity_id)
    s = a.slot if a.verb in SLOT_VERBS else 0
    r = current_reserve if a.reserve is None else int(a.reserve)
    return (v, c, max(0, s), a.model_id, a.tier, r)
