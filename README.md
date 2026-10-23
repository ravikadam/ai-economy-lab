# AI Economy Lab

A reproducible multi-agent simulation of a stylised AI industry over ten years: chip fab, chip designers,
data centres, frontier and open-weight model providers, AI product companies and 300 enterprise customers
in three sectors. AI product companies learn to compete with reinforcement learning (behaviour cloning, then
shared masked PPO), trained on an Apple-Silicon Mac. A live viewer shows the trained agents and rule-based
rivals playing a full 120-month market on their own.

All parameters are illustrative, configurable defaults. This is not a forecast of real AI markets.

## Setup

```bash
uv venv --python 3.12 .venv && VIRTUAL_ENV=.venv uv pip install -e ".[dev]"
```

## Commands

| What | Command |
|---|---|
| Acceptance tests (spec §15) | `.venv/bin/python -m pytest` |
| Headless simulation + replay + PNG dashboard | `.venv/bin/python -m scripts.simulate --config config/base.yaml --seed 42 --plots` |
| Throughput and CPU vs MPS benchmark | `.venv/bin/python -m scripts.benchmark` |
| Train one agent (imitation, then shared masked PPO on varied markets; update step on MPS) | `.venv/bin/python -m training.train --config config/train_v2.yaml --seed 0` |
| One-time test on fresh markets (paper) | `.venv/bin/python -m scripts.test_v2 --runs "runs/v2-seed*" "runs/v2nodr-seed*" "runs/v2pure-seed*"` |
| Train with PFAR (paired-fork advantage against a rule) | `.venv/bin/python -m training.train --config config/train_pfar.yaml --seed 0` (or `train_pfar_pure.yaml` from scratch) |
| PFAR test on fresh markets 700-799 and hypotheses | `.venv/bin/python -m scripts.test_v2 --runs "runs/v2-seed*" "runs/v2pure-seed*" "runs/pfar-seed*" "runs/pfarpure-seed*" --first-seed 700 --out results/v2/test_pfar.json && .venv/bin/python -m scripts.pfar_eval` |
| PFAR shared-luck noise diagnostic | `.venv/bin/python -m scripts.pfar_diag` |
| Market-findings robustness and Holm correction | `.venv/bin/python -m scripts.review_r2` |
| Switch-off tests (paper) | `.venv/bin/python -m scripts.ablate_v2 --runs "runs/v2-seed*"` |
| Evaluate on unseen seeds, held-out regimes, ablations | `.venv/bin/python -m training.evaluate --checkpoint runs/latest --seeds 100:129 --scenarios "config/scenarios/*.yaml" --ablations` |
| Paper behaviour and market statistics | `.venv/bin/python -m scripts.experiments --checkpoint results/v2/seed0/best.pt --seeds 200:229 --out runs/paper_v2 && PAPER_DIR=runs/paper_v2 .venv/bin/python scripts/paper_stats.py` |
| Live viewer: watch RL agents play on their own, replays, dashboards | `.venv/bin/streamlit run viewer/app.py` |

## Bundled results

`results/v2/` holds everything behind the paper: the five trained agents (`seed0`..`seed4`), the comparison
arms (`control_nodr_seed0..4` trained without varied markets, `control_pure_seed0..4` trained without imitation,
`imitation_only_seed0..4` the imitation-trained networks before any reinforcement learning), the
test (`test.json`), its confidence intervals (`intervals.json`, from `python -m scripts.ci_v2`), the
switch-off tests (`ablations.json`) and the market statistics (`stats.json`). Files ending
`_before_scoring_fix.json` hold the same games scored before a bankrupt company's leftover project value was
removed from its final score (see the paper's audit-trail appendix).
`pfar_seed0..4` and `pfar_pure_seed0..4` are the PFAR agents (imitation start and from scratch); `test_pfar.json`
and `pfar_eval.json` hold their one-time test on fresh markets 700-799 and the pre-registered hypotheses,
`pfar_diag.json` the shared-luck noise diagnostic, and `review_r2.json` the market-findings robustness checks.
`review_r1.json` holds the review analyses on fresh markets 500-549 (`python -m scripts.review_r1`): the oracle
yardstick, the sensitivity checks and the compute-crunch bankruptcy diagnostics.
The test plan was written before training: `paper/PROTOCOL_v2.md`. A fresh clone can watch the agents play
without training: when `runs/latest` does not exist, the code falls back to `results/v2/seed0`.
Older results from the first version of the study (`results/policy`, `results/policy_seeds`, `results/multiseed.json`)
are kept for reference. The arXiv paper source, figure script and submission guide are in `paper/`.

## Layout

```
config/      base.yaml, train.yaml, train_v2*.yaml, tiny.yaml, scenarios/ (held-out regimes)
economy/     world.py (9-phase monthly tick), supply, technology, demand, clearing, accounting (ledger), events, actions (legality)
entities/    firms, opportunities, suppliers, enterprises (struct-of-arrays)
agents/      observation, masks, actions, policy (autoregressive masked actor-critic), heuristics, reward, runner
envs/        multi_env (seat-based dynamic population), parallel_env (PettingZoo ParallelEnv + Gymnasium adapter)
training/    train (PPO), bc (behaviour cloning warm start), vec (subprocess worlds), evaluate, checkpoints
game/        session (human seat, coach, what-if, ghost run, debrief), coach (playbook lessons), missions
analysis/    replay logs, metrics frames, PNG plots
viewer/      Streamlit app
docs/        concept.html (how the game runs)
scripts/     simulate, benchmark, experiments, paper_stats, test_v2, ablate_v2
```

## Design decisions (documented per spec §18)

- **Device:** benchmarked, not assumed. PPO updates are ~1.5× faster on MPS, while small per-step inference
  batches are faster on CPU. So `device: auto` trains on MPS and `rollout_device: cpu` runs rollouts.
- **Behaviour-cloning warm start:** pure PPO from scratch collapsed to "do nothing" (sparse, delayed,
  sequential payoff). The policy is cloned from `best_npv`, then fine-tuned. Set `bc_teacher: null` to disable.
- **Reward:** Δ(cash − debt + PV of live products + stage-weighted book value of projects in development).
  This telescopes to terminal value − initial value. The shaped training reward is reported separately.
- **Ablation "no competition"** = opportunities become exclusive to the first firm that starts them.
- **Fixed seat controllers** apply only to a seat's first occupant; entrants draw from the controller mix.
