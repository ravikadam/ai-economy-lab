"""Domain randomisation: draw a fresh market regime for every training episode.

Each episode perturbs supply, entry, technology and difficulty parameters around the
base scenario. The ranges are fixed in advance and deliberately stop short of the named
regimes in config/scenarios/ (compute_crunch, entrant_wave, fast_tech, slow_tech,
harsh_failures), so those stay genuine out-of-distribution tests. RANGES records each
range next to the held-out value it must not reach; `check_holdout()` asserts it.
"""
from __future__ import annotations

import numpy as np

from economy.config import ScenarioConfig

# name: (low, high, held-out value(s) that must lie outside [low, high])
RANGES = {
    "fab_shortage_prob": (0.005, 0.04, [0.06]),
    "shortage_severity_lo": (0.2, 0.4, [0.5]),
    "dc_capacity_scale": (0.75, 1.25, [0.6]),
    "install_delay": (2, 4, [6]),
    "capex_lead_time": (14, 21, [24]),
    "entry_start_month": (9, 18, [6]),
    "entry_base_prob": (0.02, 0.08, [0.15]),
    "entry_max_prob": (0.25, 0.45, [0.6]),
    "release_lo": (5, 9, [4, 14]),
    "release_hi": (13, 21, [8, 24]),
    "capability_step_lo": (0.02, 0.04, [0.01, 0.05]),
    "capability_step_hi": (0.06, 0.09, [0.04, 0.10]),
    "breakthrough_prob": (0.08, 0.22, [0.05, 0.3]),
    "difficulty_lo": (0.1, 0.25, [0.3]),
    "difficulty_hi": (0.95, 1.1, [1.15]),
    "signal_noise": (0.16, 0.28, [0.32]),
    "success_bias": (0.2, 0.6, [0.0]),
    "confidence_pilot_failure": (0.02, 0.06, [0.08]),
}
INT_KEYS = {"install_delay", "capex_lead_time", "entry_start_month", "release_lo", "release_hi"}


def check_holdout() -> None:
    for k, (lo, hi, held) in RANGES.items():
        for h in held:
            assert not (lo <= h <= hi), f"{k}: held-out value {h} inside training range [{lo}, {hi}]"


def draw(rng: np.random.Generator) -> dict:
    out = {}
    for k, (lo, hi, _) in RANGES.items():
        out[k] = int(rng.integers(lo, hi + 1)) if k in INT_KEYS else float(rng.uniform(lo, hi))
    return out


def randomized_config(base: ScenarioConfig, rng: np.random.Generator) -> ScenarioConfig:
    p = draw(rng)
    d = base.model_dump()
    d["name"] = f"{base.name}+dr"
    d["shocks"]["fab_shortage_prob"] = p["fab_shortage_prob"]
    d["shocks"]["fab_shortage_severity"] = (p["shortage_severity_lo"], p["shortage_severity_lo"] + 0.3)
    d["datacenters"]["initial_capacity"] = [c * p["dc_capacity_scale"] for c in d["datacenters"]["initial_capacity"]]
    d["datacenters"]["install_delay"] = p["install_delay"]
    d["fab"]["capex_lead_time"] = p["capex_lead_time"]
    d["builders"]["entry_start_month"] = p["entry_start_month"]
    d["builders"]["entry_base_prob"] = p["entry_base_prob"]
    d["builders"]["entry_max_prob"] = p["entry_max_prob"]
    d["technology"]["release_interval"] = (p["release_lo"], p["release_hi"])
    d["technology"]["capability_step"] = (p["capability_step_lo"], p["capability_step_hi"])
    d["technology"]["breakthrough_prob"] = p["breakthrough_prob"]
    d["technology"]["bias"] = p["success_bias"]
    d["opportunities"]["difficulty_scale"] = (p["difficulty_lo"], p["difficulty_hi"])
    d["opportunities"]["signal_noise"] = p["signal_noise"]
    d["demand"]["confidence_pilot_failure"] = p["confidence_pilot_failure"]
    return ScenarioConfig.model_validate(d)


def config_for_seed(base: ScenarioConfig, seed: int) -> ScenarioConfig:
    """Deterministic randomised regime for a market seed (used for training episodes and dev markets)."""
    return randomized_config(base, np.random.default_rng([seed, 4242]))


check_holdout()
