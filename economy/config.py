"""Scenario configuration: typed, validated, YAML-backed.

Every numeric value is an illustrative, configurable default (spec §12) — not an
empirical claim. Money is in $k, time in months, compute in GPU-equivalent units.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "1.0"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EpisodeCfg(_Strict):
    months: int = Field(120, ge=6)


class FabCfg(_Strict):
    wafer_capacity: float = 400.0          # wafers / month
    yield_rate: float = Field(0.85, gt=0, le=1)
    chips_per_wafer: float = 20.0
    wafer_price: float = 3.0               # $k / wafer (base)
    capex_per_wafer_capacity: float = 40.0  # $k per +1 wafer/month capacity
    capex_lead_time: int = Field(18, ge=1)  # months until new capacity is online
    invest_backlog_ratio: float = 1.2       # invest when backlog > ratio * monthly output
    invest_step_frac: float = 0.25          # each investment adds this fraction of capacity
    max_capacity: float = 4000.0


class ChipDesignerCfg(_Strict):
    count: int = Field(2, ge=1)
    efficiency: list[float] = [10.0, 8.0]   # compute units of monthly capacity per chip
    base_price: list[float] = [1.6, 1.2]    # $k per chip
    price_sensitivity: float = 0.5          # price response to fab utilisation


class DataCenterCfg(_Strict):
    count: int = Field(2, ge=1)
    initial_capacity: list[float] = [30000.0, 22000.0]  # compute units / month
    install_delay: int = Field(3, ge=0)     # months from chip delivery to capacity online
    target_utilization: float = 0.75
    order_trigger_utilization: float = 0.80
    order_step_frac: float = 0.15           # order chips worth this fraction of capacity
    power_cost_per_unit: float = 0.002      # $k per served compute unit
    depreciation: float = 0.005             # monthly capacity retirement
    initial_cash: float = 50000.0


class ComputePriceCfg(_Strict):
    initial: float = 0.006                  # $k per compute unit
    floor: float = 0.003
    ceiling: float = 0.03
    adjust_rate: float = 0.6                # price elasticity to (util - target)


class ModelCfg(_Strict):
    name: str
    open_weight: bool
    capability: list[float]                 # per sector, ~[0,1.2]
    reliability: float = Field(gt=0, le=1)
    compute_per_unit: float                 # compute units per unit of inference demand
    markup: float                           # API price = compute cost * markup (open: self-host overhead)


class TechnologyCfg(_Strict):
    models: list[ModelCfg] = [
        ModelCfg(name="frontier", open_weight=False, capability=[0.55, 0.60, 0.50],
                 reliability=0.90, compute_per_unit=1.0, markup=2.2),
        ModelCfg(name="open", open_weight=True, capability=[0.42, 0.45, 0.35],
                 reliability=0.82, compute_per_unit=0.7, markup=1.25),
    ]
    release_interval: tuple[int, int] = (6, 18)
    capability_step: tuple[float, float] = (0.03, 0.08)
    open_catchup: float = 0.35              # open model closes this share of the gap at its release
    reliability_step: float = 0.01
    breakthrough_prob: float = 0.15         # chance a release is a breakthrough (x2.5 step)
    alpha: float = 6.0                      # success logit weights (spec §5)
    beta: float = 1.0
    gamma: float = 1.0
    delta: float = 1.5
    bias: float = 0.4
    p_min: float = 0.03
    p_max: float = 0.97


class SectorCfg(_Strict):
    name: str
    enterprises: int
    monthly_budget_mean: float              # $k / month AI budget per enterprise
    hurdle_roi: float                       # required benefit/cost - 1
    price_sensitivity: float                # weight on cost in net value
    capability_weight: float                # how much benefit scales with capability margin
    reliability_weight: float               # how much benefit scales with reliability margin
    reliability_floor: float                # sector-wide compliance floor (added to opp req)
    initial_confidence: float = Field(ge=0, le=1)


class DemandCfg(_Strict):
    sectors: list[SectorCfg] = [
        SectorCfg(name="customer_ops", enterprises=110, monthly_budget_mean=40.0, hurdle_roi=0.3,
                  price_sensitivity=1.4, capability_weight=0.6, reliability_weight=0.6,
                  reliability_floor=0.70, initial_confidence=0.55),
        SectorCfg(name="software_eng", enterprises=100, monthly_budget_mean=60.0, hurdle_roi=0.2,
                  price_sensitivity=0.8, capability_weight=1.6, reliability_weight=0.5,
                  reliability_floor=0.70, initial_confidence=0.60),
        SectorCfg(name="regulated", enterprises=90, monthly_budget_mean=80.0, hurdle_roi=0.4,
                  price_sensitivity=0.9, capability_weight=0.8, reliability_weight=2.0,
                  reliability_floor=0.84, initial_confidence=0.40),
    ]
    benefit_dispersion: float = 0.35        # lognormal sigma of enterprise-specific benefit
    adoption_steepness: float = 4.0
    procurement_delay: tuple[int, int] = (1, 4)
    sales_capacity_deployed: int = 3        # max new customers / product / month
    sales_capacity_scaled: int = 9
    churn_base: float = 0.01
    churn_bad_value: float = 0.25
    confidence_pilot_success: float = 0.03
    confidence_pilot_failure: float = 0.04
    confidence_churn: float = 0.004
    confidence_bankruptcy: float = 0.03
    confidence_recovery: float = 0.02       # monthly drift back toward initial
    confidence_diffusion: float = 0.3       # enterprise confidence -> sector confidence rate
    price_share: float = 0.32               # subscription = share * value signal
    competition_discount: float = 0.12
    integration_cost_months: float = 1.5    # one-off integration cost, in months of subscription


class OpportunityCfg(_Strict):
    count: int = Field(60, ge=4)
    initially_available: int = 24
    discovery_rate: float = 0.04            # monthly chance an undiscovered opp is revealed
    difficulty_beta: tuple[float, float] = (2.2, 2.2)
    difficulty_scale: tuple[float, float] = (0.15, 1.05)
    value_lognormal: tuple[float, float] = (2.6, 0.5)   # monthly benefit per customer, $k
    need_share: tuple[float, float] = (0.08, 0.35)       # share of sector enterprises needing it
    duration: tuple[int, int] = (2, 5)                    # months per development stage
    integration_burden: tuple[float, float] = (0.0, 0.6)
    reliability_req: tuple[float, float] = (0.0, 0.08)
    inference_units: tuple[float, float] = (80.0, 400.0)  # monthly inference units per customer
    signal_noise: float = 0.22             # initial sd of difficulty signal
    value_signal_noise: float = 0.25        # multiplicative sd of value signal


class BuilderCfg(_Strict):
    initial_firms: int = Field(5, ge=1)
    max_firms: int = Field(12, ge=1)
    initial_cash: float = 1500.0
    credit_limit: float = 400.0
    interest_rate: float = 0.012            # monthly
    overhead: float = 25.0                  # $k / month base team cost
    max_projects: int = 3
    candidates: int = 12
    tier_burn: list[float] = [12.0, 25.0, 50.0]   # $k / month per in-dev project
    tier_effort: list[float] = [0.0, 0.5, 1.0]
    tier_speed: list[float] = [0.75, 1.0, 1.5]    # stage progress per month
    start_cost: float = 20.0
    pilot_cost: float = 40.0
    pilot_customers: int = 3
    pilot_months: int = 2
    launch_cost: float = 30.0
    scale_cost: float = 150.0
    support_cost_per_customer: float = 0.8   # $k / month
    retry_cost: float = 30.0
    retry_effort_bonus: float = 0.35
    research_signal_shrink: float = 0.5      # research stage multiplies signal noise by this
    skill_range: tuple[float, float] = (0.3, 0.8)
    skill_gain_per_launch: float = 0.03
    reputation_gain: float = 0.05
    reputation_loss: float = 0.05
    entry_start_month: int = 12
    entry_base_prob: float = 0.04
    entry_profit_sensitivity: float = 0.002   # per $k of trailing mean monthly profit
    entry_max_prob: float = 0.35
    replace_bankrupt: bool = False
    bankruptcy_penalty: float = 2.0           # in reward units (see agents.reward)


class RiskProfileCfg(_Strict):
    name: str
    bankruptcy_aversion: float
    downside_aversion: float
    loss_sensitivity: float
    uncertainty_aversion: float
    exploration: float
    runway_target: float
    regret_weight: float


class RiskCfg(_Strict):
    profiles: list[RiskProfileCfg] = [
        RiskProfileCfg(name="cautious", bankruptcy_aversion=3.0, downside_aversion=0.6, loss_sensitivity=0.4,
                       uncertainty_aversion=0.6, exploration=0.05, runway_target=12, regret_weight=0.1),
        RiskProfileCfg(name="balanced", bankruptcy_aversion=1.5, downside_aversion=0.3, loss_sensitivity=0.2,
                       uncertainty_aversion=0.3, exploration=0.10, runway_target=8, regret_weight=0.2),
        RiskProfileCfg(name="aggressive", bankruptcy_aversion=0.7, downside_aversion=0.1, loss_sensitivity=0.05,
                       uncertainty_aversion=0.1, exploration=0.20, runway_target=4, regret_weight=0.3),
    ]
    profile_weights: list[float] = [1.0, 1.0, 1.0]


class ShockCfg(_Strict):
    fab_shortage_prob: float = 0.02         # monthly
    fab_shortage_months: tuple[int, int] = (3, 9)
    fab_shortage_severity: tuple[float, float] = (0.3, 0.6)   # capacity loss fraction
    demand_shock_prob: float = 0.0
    demand_shock_size: tuple[float, float] = (-0.3, 0.3)


class AblationCfg(_Strict):
    """Switch mechanisms off one at a time (spec §9 ablations)."""
    competition: bool = True
    technical_failure: bool = True
    confidence_feedback: bool = True
    supply_bottlenecks: bool = True
    delayed_capex: bool = True
    risk_heterogeneity: bool = True


class ScenarioConfig(_Strict):
    schema_version: str = SCHEMA_VERSION
    name: str = "base"
    episode: EpisodeCfg = EpisodeCfg()
    fab: FabCfg = FabCfg()
    chips: ChipDesignerCfg = ChipDesignerCfg()
    datacenters: DataCenterCfg = DataCenterCfg()
    compute_price: ComputePriceCfg = ComputePriceCfg()
    technology: TechnologyCfg = TechnologyCfg()
    demand: DemandCfg = DemandCfg()
    opportunities: OpportunityCfg = OpportunityCfg()
    builders: BuilderCfg = BuilderCfg()
    risk: RiskCfg = RiskCfg()
    shocks: ShockCfg = ShockCfg()
    ablation: AblationCfg = AblationCfg()

    @model_validator(mode="after")
    def _check(self) -> "ScenarioConfig":
        if len(self.chips.efficiency) != self.chips.count or len(self.chips.base_price) != self.chips.count:
            raise ValueError("chips.efficiency/base_price length must equal chips.count")
        if len(self.datacenters.initial_capacity) != self.datacenters.count:
            raise ValueError("datacenters.initial_capacity length must equal datacenters.count")
        n_sec = len(self.demand.sectors)
        for m in self.technology.models:
            if len(m.capability) != n_sec:
                raise ValueError(f"model {m.name} capability must have {n_sec} entries")
        if self.builders.initial_firms > self.builders.max_firms:
            raise ValueError("initial_firms > max_firms")
        if self.opportunities.initially_available > self.opportunities.count:
            raise ValueError("initially_available > opportunity count")
        if len(self.risk.profile_weights) != len(self.risk.profiles):
            raise ValueError("risk.profile_weights length mismatch")
        if len(self.builders.tier_burn) != len(self.builders.tier_effort) or \
                len(self.builders.tier_burn) != len(self.builders.tier_speed):
            raise ValueError("tier arrays must have equal length")
        return self

    @property
    def n_sectors(self) -> int:
        return len(self.demand.sectors)

    @property
    def n_models(self) -> int:
        return len(self.technology.models)

    @property
    def n_tiers(self) -> int:
        return len(self.builders.tier_burn)


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> ScenarioConfig:
    """Load a scenario YAML. A YAML may say `extends: other.yaml` to inherit and override."""
    data: dict[str, Any] = {}
    if path is not None:
        path = Path(path)
        raw = yaml.safe_load(path.read_text()) or {}
        parent = raw.pop("extends", None)
        if parent:
            data = load_config(path.parent / parent).model_dump()
        data = _deep_merge(data, raw)
    if overrides:
        data = _deep_merge(data, overrides)
    return ScenarioConfig.model_validate(data)
