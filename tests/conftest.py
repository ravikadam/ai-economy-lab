import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.heuristics import HEURISTICS  # noqa: E402
from economy.config import load_config  # noqa: E402
from economy.world import World  # noqa: E402


@pytest.fixture
def base_cfg():
    return load_config(ROOT / "config/base.yaml")


def cfg_with(**overrides):
    base = load_config(ROOT / "config/base.yaml").model_dump()

    def merge(a, b):
        for k, v in b.items():
            if isinstance(v, dict) and isinstance(a.get(k), dict):
                merge(a[k], v)
            else:
                a[k] = v
        return a
    return load_config(None, overrides=merge(base, overrides))


def run_world(cfg, seed=1, policy="best_npv", months=None):
    w = World(cfg, seed)
    rng = np.random.default_rng(seed)
    for _ in range(months or cfg.episode.months):
        acts = {f.firm_id: HEURISTICS[policy](w, f, rng) for f in w.alive_firms()}
        r = w.step(acts)
        if r.done:
            break
    return w
