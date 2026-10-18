"""Enterprise customers, stored struct-of-arrays for vectorised monthly updates."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EnterprisePopulation:
    sector: np.ndarray            # [E] int
    budget: np.ndarray            # [E] monthly AI budget ($k)
    spend: np.ndarray             # [E] current committed monthly subscriptions ($k)
    hurdle: np.ndarray            # [E] required ROI
    risk_tolerance: np.ndarray    # [E] in [0,1]
    confidence: np.ndarray        # [E] in [0,1]
    procurement_ready: np.ndarray # [E] month from which they can sign a new deal
    need: np.ndarray              # [E, O] bool — enterprise has this use case
    benefit: np.ndarray           # [E, O] hidden realised benefit multiplier (lognormal)
    provider: np.ndarray          # [E, O] int firm_id serving it, -1 none
    price_paid: np.ndarray        # [E, O] monthly subscription ($k)
    piloting: np.ndarray          # [E, O] int firm_id piloting, -1 none

    @property
    def size(self) -> int:
        return int(self.sector.shape[0])


def generate_enterprises(cfg, opps, rng: np.random.Generator) -> EnterprisePopulation:
    secs = cfg.demand.sectors
    sector = np.concatenate([np.full(s.enterprises, i, dtype=np.int64) for i, s in enumerate(secs)])
    E, O = sector.shape[0], len(opps)
    budget_mean = np.array([s.monthly_budget_mean for s in secs])[sector]
    budget = budget_mean * rng.lognormal(0.0, 0.4, size=E)
    hurdle = np.array([s.hurdle_roi for s in secs])[sector] * rng.uniform(0.7, 1.3, size=E)
    risk_tol = rng.beta(2.0, 2.0, size=E)
    conf = np.array([s.initial_confidence for s in secs])[sector] + rng.normal(0, 0.05, size=E)
    need = np.zeros((E, O), dtype=bool)
    for o in opps:
        idx = np.flatnonzero(sector == o.sector)
        chosen = rng.choice(idx, size=min(o.need_count, idx.size), replace=False)
        need[chosen, o.opportunity_id] = True
    benefit = rng.lognormal(-0.5 * cfg.demand.benefit_dispersion ** 2, cfg.demand.benefit_dispersion, size=(E, O))
    return EnterprisePopulation(
        sector=sector,
        budget=budget,
        spend=np.zeros(E),
        hurdle=hurdle,
        risk_tolerance=risk_tol,
        confidence=np.clip(conf, 0.02, 0.98),
        procurement_ready=np.zeros(E, dtype=np.int64),
        need=need,
        benefit=benefit,
        provider=np.full((E, O), -1, dtype=np.int64),
        price_paid=np.zeros((E, O)),
        piloting=np.full((E, O), -1, dtype=np.int64),
    )
