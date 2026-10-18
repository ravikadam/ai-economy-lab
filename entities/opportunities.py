"""Use-case opportunities: finite, partially observable, sector-specific."""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np


class OppStatus(IntEnum):
    UNDISCOVERED = 0
    AVAILABLE = 1


@dataclass
class Opportunity:
    opportunity_id: int
    sector: int
    latent_difficulty: float     # hidden
    value: float                 # true mean monthly benefit per customer ($k), hidden
    value_signal: float          # public noisy estimate
    capability_threshold: float  # minimum capability for the product to be usable at all
    duration: int                # months per development stage at tier speed 1.0
    integration_burden: float
    reliability_req: float       # added to sector reliability floor
    inference_units: float       # monthly inference units per customer
    need_count: int              # number of enterprises with this need
    status: OppStatus = OppStatus.UNDISCOVERED
    discovered_month: int = -1

    def observable_features(self) -> dict:
        return {
            "opportunity_id": self.opportunity_id,
            "sector": self.sector,
            "value_signal": self.value_signal,
            "duration": self.duration,
            "integration_burden": self.integration_burden,
            "reliability_req": self.reliability_req,
            "inference_units": self.inference_units,
            "need_count": self.need_count,
        }


def generate_opportunities(cfg, sector_sizes: list[int], rng: np.random.Generator) -> list[Opportunity]:
    oc = cfg.opportunities
    n_sec = len(sector_sizes)
    opps = []
    for i in range(oc.count):
        s = int(rng.integers(n_sec))
        d01 = rng.beta(*oc.difficulty_beta)
        d = oc.difficulty_scale[0] + d01 * (oc.difficulty_scale[1] - oc.difficulty_scale[0])
        v = float(rng.lognormal(*oc.value_lognormal))
        v_sig = v * float(np.exp(rng.normal(0, oc.value_signal_noise)))
        share = rng.uniform(*oc.need_share)
        opps.append(Opportunity(
            opportunity_id=i,
            sector=s,
            latent_difficulty=float(d),
            value=v,
            value_signal=v_sig,
            capability_threshold=float(max(0.0, d - 0.25)),
            duration=int(rng.integers(oc.duration[0], oc.duration[1] + 1)),
            integration_burden=float(rng.uniform(*oc.integration_burden)),
            reliability_req=float(rng.uniform(*oc.reliability_req)),
            inference_units=float(rng.uniform(*oc.inference_units)),
            need_count=max(2, int(round(share * sector_sizes[s]))),
        ))
    return opps
