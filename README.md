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
| Train (behaviour cloning, then shared masked PPO; update step on MPS) | `.venv/bin/python -m training.train --config config/train.yaml` |
| Evaluate on unseen seeds, held-out regimes, ablations | `.venv/bin/python -m training.evaluate --checkpoint runs/latest --seeds 100:129 --scenarios "config/scenarios/*.yaml" --ablations` |
| Paper experiments and statistics | `.venv/bin/python -m scripts.experiments --seeds 200:229 && .venv/bin/python scripts/paper_stats.py` |
| Live viewer: watch RL agents play on their own, replays, dashboards | `.venv/bin/streamlit run viewer/app.py` |

## Bundled results

`results/` holds the trained policy (`results/policy/best.pt`) and the numbers behind the paper, so a fresh
clone can watch the agents play without training: when `runs/latest` does not exist, the code falls back to
`results/policy`. `results/policy/metrics_pure_ppo_failed.jsonl` is the first, failed pure-PPO run.
`results/policy_seeds/seed1..4` are four more agents trained with `--seed 1..4`; `results/multiseed.json` compares
all five against the best rule on 100 unseen markets (`python -m scripts.multiseed_eval --runs <run dirs>`).
The arXiv paper source, figure script and submission guide are in `paper/`.

## Layout

```
config/      base.yaml, train.yaml, tiny.yaml, scenarios/ (held-out regimes)
economy/     world.py (9-phase monthly tick), supply, technology, demand, clearing, accounting (ledger), events, actions (legality)
entities/    firms, opportunities, suppliers, enterprises (struct-of-arrays)
agents/      observation, masks, actions, policy (autoregressive masked actor-critic), heuristics, reward, runner
envs/        multi_env (seat-based dynamic population), parallel_env (PettingZoo ParallelEnv + Gymnasium adapter)
training/    train (PPO), bc (behaviour cloning warm start), vec (subprocess worlds), evaluate, checkpoints
game/        session (human seat, coach, what-if, ghost run, debrief), coach (playbook lessons), missions
analysis/    replay logs, metrics frames, PNG plots
viewer/      Streamlit app
docs/        concept.html (how the game runs)
scripts/     simulate, benchmark, experiments, paper_stats
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
