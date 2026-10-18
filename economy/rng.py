"""Seeded, named random streams.

Each subsystem gets its own numpy Generator spawned from one SeedSequence, so
adding a draw in one subsystem never shifts the draws of another. Random draws
that matter for audit (stage gates) are logged by the caller.
"""
from __future__ import annotations

import numpy as np

STREAMS = ("world", "supply", "tech", "demand", "gates", "entry", "opps", "signals", "alloc", "policy")


class RngStreams:
    def __init__(self, seed: int):
        self.seed = int(seed)
        children = np.random.SeedSequence(self.seed).spawn(len(STREAMS))
        self._gens = {name: np.random.default_rng(ss) for name, ss in zip(STREAMS, children)}

    def __getitem__(self, name: str) -> np.random.Generator:
        return self._gens[name]

    def state(self) -> dict:
        return {k: g.bit_generator.state for k, g in self._gens.items()}

    def set_state(self, state: dict) -> None:
        for k, s in state.items():
            self._gens[k].bit_generator.state = s
