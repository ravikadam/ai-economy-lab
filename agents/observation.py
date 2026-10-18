"""Observation encoding (spec §7): own state + project slots + candidate set + lagged market.

Only public or firm-private information enters: hidden difficulty, competitors'
private budgets/actions, future shocks and enterprise-specific realised value
are never encoded. Belief estimates come from agents.heuristics.estimate, which
uses the firm's own noisy signals.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from agents.heuristics import estimate
from entities.firms import N_STAGES, RESERVE_MONTHS, FirmState, Stage

if TYPE_CHECKING:
    from economy.config import ScenarioConfig
    from economy.world import World


@dataclass(frozen=True)
class ObsLayout:
    own: int
    slot: int
    cand: int
    market: int
    n_slots: int
    n_cand: int

    @property
    def total(self) -> int:
        return self.own + self.n_slots * self.slot + self.n_cand * self.cand + self.market


def layout(cfg: "ScenarioConfig") -> ObsLayout:
    ns, nm, nt = cfg.n_sectors, cfg.n_models, cfg.n_tiers
    own = 7 + 7 + len(RESERVE_MONTHS) + ns + 3
    slot = 1 + N_STAGES + 2 + nm + nt + 2 + ns + 1 + 2 + 4
    cand = 1 + ns + 2 + 8 + 4 * nm + 1
    market = 6 + nm * ns + nm + nm + ns * 3 + 4
    return ObsLayout(own, slot, cand, market, cfg.builders.max_projects, cfg.builders.candidates)


def _onehot(i: int, n: int) -> list[float]:
    v = [0.0] * n
    if 0 <= i < n:
        v[i] = 1.0
    return v


def _t(x: float, s: float) -> float:
    return float(np.tanh(x / s))


def build_observation(world: "World", f: FirmState, cand: list[int]) -> np.ndarray:
    cfg = world.cfg
    b = cfg.builders
    ns, nm, nt = cfg.n_sectors, cfg.n_models, cfg.n_tiers
    L = layout(cfg)
    out: list[float] = []
    # ---- own
    cash = world.cash(f)
    profit = f.monthly_profit_hist[-1] if f.monthly_profit_hist else 0.0
    out += [cash / b.initial_cash, f.debt / b.credit_limit, world.runway(f) / 36.0,
            f.age(world.month) / cfg.episode.months, 1.0 - world.month / cfg.episode.months,
            f.skill, f.reputation]
    out += f.profile.vector()
    out += _onehot(f.reserve_level, len(RESERVE_MONTHS))
    out += _onehot(f.sector_affinity, ns)
    out += [len(f.projects) / b.max_projects, _t(profit, 100.0),
            _t(sum(p.customers for p in f.projects), 40.0)]
    # ---- slots
    for s in range(b.max_projects):
        if s < len(f.projects):
            p = f.projects[s]
            opp = world.opps[p.opportunity_id]
            e = estimate(world, f, p.opportunity_id, p.model_id, p.tier, retries=p.retries)
            out += [1.0] + _onehot(int(p.stage), N_STAGES)
            out += [p.progress / max(opp.duration, 1), min(p.stage_age, 24) / 12.0]
            out += _onehot(p.model_id, nm) + _onehot(p.tier, nt)
            out += [p.customers / max(opp.need_count, 1), _t(p.trailing_profit(), 50.0)]
            out += _onehot(opp.sector, ns)
            out += [p.retries / 3.0, float(p.failure_kind == "technical"), float(p.failure_kind == "commercial")]
            out += [e.p_tech, e.p_comm, _t(e.margin, 10.0), len(p.pilot_ids) / max(b.pilot_customers, 1)]
        else:
            out += [0.0] * L.slot
    # ---- candidates
    live_by_opp: dict[int, int] = {}
    for g in world.alive_firms():
        if g.firm_id == f.firm_id:
            continue
        for p in g.projects:
            if p.is_live:
                live_by_opp[p.opportunity_id] = live_by_opp.get(p.opportunity_id, 0) + 1
    for k in range(b.candidates):
        if k < len(cand):
            o = world.opps[cand[k]]
            out += [1.0] + _onehot(o.sector, ns)
            out += [world.difficulty_signal(f, o.opportunity_id), f.knowledge.get(o.opportunity_id, 1.0)]
            out += [o.value_signal / 30.0, o.need_count / 40.0, o.duration / 5.0, o.integration_burden,
                    o.reliability_req * 10.0, o.inference_units / 400.0, live_by_opp.get(o.opportunity_id, 0) / 3.0,
                    min(world.month - o.discovered_month, 24) / 24.0]
            for m in range(nm):
                e = estimate(world, f, o.opportunity_id, m, 1)
                out += [e.p_tech, e.p_comm, _t(e.margin, 10.0), e.cap_gap]
            out += [float(o.sector == f.sector_affinity)]
        else:
            out += [0.0] * L.cand
    # ---- market (lagged public indicators)
    pub = world.public
    pc = cfg.compute_price
    out += [pub["compute_price"] / pc.ceiling, pub["utilization"], pub["served_frac"], pub["fab_shock"],
            min(3.0, pub["fab_backlog"] / max(pub["fab_capacity"], 1.0)) / 3.0,
            pub["chip_price"] / max(np.mean(cfg.chips.base_price), 1e-6) / 3.0]
    for m in range(nm):
        out += list(pub["model_capability"][m])
    out += list(pub["model_reliability"])
    out += [u * 100.0 for u in pub["model_unit_price"]]
    out += list(pub["sector_confidence"])
    sizes = [s.enterprises for s in cfg.demand.sectors]
    out += [a / max(n, 1) for a, n in zip(pub["sector_adoption"], sizes)]
    out += [_t(r, 300.0) for r in pub["sector_revenue"]]
    out += [pub["n_firms"] / b.max_firms, pub["hhi"], pub["available_opps"] / cfg.opportunities.count,
            len(pub["launches"]) / 5.0]
    arr = np.asarray(out, dtype=np.float32)
    assert arr.shape[0] == L.total, (arr.shape, L.total)
    return np.nan_to_num(arr, nan=0.0, posinf=5.0, neginf=-5.0)
