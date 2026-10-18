"""Upstream value chain: fabricator -> chip designers -> data centers -> model providers."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Fabricator:
    wafer_capacity: float
    yield_rate: float
    backlog: float = 0.0              # wafers ordered, not yet produced
    shock_factor: float = 1.0         # 1.0 = no disruption
    shock_until: int = -1
    pending_capacity: float = 0.0     # capacity under construction
    capex_total: float = 0.0
    output_last: float = 0.0
    utilization: float = 0.0
    account: str = "fab:0"

    @property
    def effective_capacity(self) -> float:
        return self.wafer_capacity * self.shock_factor


@dataclass
class ChipDesigner:
    designer_id: int
    efficiency: float                 # compute units per chip
    base_price: float
    unit_price: float
    orders: dict[int, float] = field(default_factory=dict)   # dc_id -> chips outstanding

    @property
    def account(self) -> str:
        return f"chip:{self.designer_id}"


@dataclass
class DataCenter:
    dc_id: int
    capacity: float                   # compute units / month
    pending_capacity: float = 0.0
    chips_on_order: float = 0.0       # in compute-unit terms
    served_last: float = 0.0
    util_hist: list[float] = field(default_factory=list)

    @property
    def account(self) -> str:
        return f"dc:{self.dc_id}"

    def trailing_util(self, n: int = 3) -> float:
        h = self.util_hist[-n:]
        return sum(h) / len(h) if h else 0.0


@dataclass
class ModelProvider:
    model_id: int
    name: str
    open_weight: bool
    capability: list[float]           # per sector
    reliability: float
    compute_per_unit: float
    markup: float
    next_release: int = 0
    releases: int = 0
    load_last: float = 0.0            # inference units served last month

    @property
    def account(self) -> str:
        return f"model:{self.model_id}"

    def unit_price(self, compute_price: float) -> float:
        """Price a builder pays per inference unit (API, or self-hosting cost for open weights)."""
        return self.compute_per_unit * compute_price * self.markup
