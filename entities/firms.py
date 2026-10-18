"""Use-case builder firms (the learning agents) and their projects."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


class Stage(IntEnum):
    RESEARCH = 0
    PROTOTYPE = 1
    PROTOTYPE_READY = 2   # technical gate passed, awaiting PILOT
    PILOT = 3
    PILOT_READY = 4       # commercial gate passed, awaiting LAUNCH
    DEPLOYED = 5
    SCALED = 6
    FAILED = 7            # technical or commercial failure; retryable
    PAUSED = 8


N_STAGES = len(Stage)
DEV_STAGES = (Stage.RESEARCH, Stage.PROTOTYPE, Stage.PILOT)
LIVE_STAGES = (Stage.DEPLOYED, Stage.SCALED)


@dataclass
class RiskProfile:
    name: str
    bankruptcy_aversion: float
    downside_aversion: float
    loss_sensitivity: float
    uncertainty_aversion: float
    exploration: float
    runway_target: float
    regret_weight: float

    def vector(self) -> list[float]:
        return [self.bankruptcy_aversion / 3.0, self.downside_aversion, self.loss_sensitivity,
                self.uncertainty_aversion, self.exploration, self.runway_target / 12.0, self.regret_weight]


@dataclass
class Project:
    firm_id: int
    opportunity_id: int
    model_id: int
    tier: int
    stage: Stage = Stage.RESEARCH
    progress: float = 0.0          # progress within current stage, in months-equivalent
    age: int = 0
    stage_age: int = 0
    investment: float = 0.0
    retries: int = 0
    failure_kind: str = ""         # "technical" | "commercial" | ""
    paused_from: Stage | None = None
    customers: int = 0
    pilot_ids: list[int] = field(default_factory=list)
    revenue_last: float = 0.0
    cost_last: float = 0.0
    profit_hist: list[float] = field(default_factory=list)
    outcome: str = ""              # final outcome when closed
    price: float = 0.0             # monthly subscription per customer

    @property
    def is_live(self) -> bool:
        return self.stage in LIVE_STAGES

    def trailing_profit(self, n: int = 3) -> float:
        h = self.profit_hist[-n:]
        return sum(h) / len(h) if h else 0.0


@dataclass
class FirmState:
    firm_id: int
    birth_month: int
    profile: RiskProfile
    skill: float
    sector_affinity: int
    debt: float = 0.0
    reputation: float = 0.5
    reserve_level: int = 1                     # index into RESERVE_MONTHS
    projects: list[Project] = field(default_factory=list)
    knowledge: dict[int, float] = field(default_factory=dict)   # opp_id -> signal noise multiplier
    alive: bool = True
    death_month: int | None = None
    controller: str = "policy"                 # policy | heuristic:<name> | human | frozen:<tag>
    launches: int = 0
    tech_failures: int = 0
    commercial_failures: int = 0
    deferrals: int = 0
    illegal_actions: int = 0
    monthly_profit_hist: list[float] = field(default_factory=list)
    last_net_worth: float = 0.0
    cum_raw_reward: float = 0.0
    cum_shaped_reward: float = 0.0

    @property
    def account(self) -> str:
        return f"firm:{self.firm_id}"

    def age(self, month: int) -> int:
        return month - self.birth_month


RESERVE_MONTHS = (2.0, 6.0, 12.0)
