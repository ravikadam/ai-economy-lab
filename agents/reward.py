"""Reward (spec §7, §8).

raw reward   = Δ net worth / REWARD_SCALE, where net worth = cash − debt + PV(live
               products). Operating cash flow, investment spend, interest and
               bankruptcy losses all flow through Δcash, so nothing is double
               counted; the sum over an episode telescopes to terminal value −
               initial value (spec terminal value definition).
shaped reward = raw − downside_aversion·max(0, −raw) − financing distress
               − bankruptcy penalty − illegal-action penalty.

Raw (economic) and shaped (training) rewards are reported separately so reward
gaming cannot masquerade as economic success.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from economy.world import World
    from entities.firms import FirmState

ILLEGAL_PENALTY = 0.01
DISTRESS_SCALE = 0.02


def compute_rewards(world: "World", f: "FirmState", bankrupt: bool, illegal: int) -> tuple[float, float]:
    from economy.world import REWARD_SCALE
    nw = world.net_worth(f)
    raw = (nw - f.last_net_worth) / REWARD_SCALE
    f.last_net_worth = nw
    prof = f.profile
    shaped = raw - prof.downside_aversion * max(0.0, -raw)
    if not bankrupt:
        runway = world.runway(f)
        shaped -= DISTRESS_SCALE * prof.bankruptcy_aversion * max(0.0, 1.0 - runway / max(prof.runway_target, 1.0))
        if f.debt > 0:
            shaped -= DISTRESS_SCALE * 0.5 * f.debt / world.cfg.builders.credit_limit
    else:
        shaped -= world.cfg.builders.bankruptcy_penalty * prof.bankruptcy_aversion
    shaped -= ILLEGAL_PENALTY * illegal
    f.cum_raw_reward += raw
    f.cum_shaped_reward += shaped
    return raw, shaped
