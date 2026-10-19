# Protocol for the v2 training run (fixed before training)

Written 2026-10-19, before any v2 agent was trained or tested. Nothing below changes after test results are seen.

## What changes in training (config/train_v2.yaml)
1. **Varied markets (domain randomisation).** 75% of training and imitation episodes draw a fresh regime from
   `training/randomize.py`: shortage frequency and severity, data-centre capacity, install and fab lead times,
   entry timing and rates, model release cadence and step size, breakthrough odds, use-case difficulty, signal
   noise, success bias and pilot-failure confidence hit. Every range stops short of the five named regimes in
   `config/scenarios/`, enforced by `check_holdout()`. Those regimes are therefore out-of-distribution tests.
2. **Larger selection set.** Checkpoints are scored every 20 iterations on 40 dev markets: base seeds 1000–1019
   and randomised seeds 2000–2019. Sampled play averages 2 draws; greedy play is scored too. The best score is
   kept, together with the mode that achieved it (sampled or greedy). Early stop after 8 evaluations without
   improvement; cap at 1,000,000 world-months.
3. **Exploration floor.** The entropy bonus rises automatically (up to 0.05) whenever policy entropy falls
   below 0.25.
4. **Wider opponent pool.** Training rivals: 50% shared policy, 15% best_npv, 10% easy_first,
   10% cash_preserving, 5% random, 10% frozen past snapshots (pool of 10).
5. **Five seeds** (0–4), each with its own training and imitation markets.

## Test (run once, after all five seeds finish)
- **Fresh markets never used before:** base scenario seeds 400–499 (100 markets). Also the five named regimes
  (compute_crunch, entrant_wave, fast_tech, slow_tech, harsh_failures), each on seeds 400–449.
- **Protocol:** the focal seat is played by the agent, using the mode chosen on dev. In sampled mode the result
  is the average of 3 draws. The rival background is fixed and rule-based. best_npv plays the same seat in the
  same markets, giving a paired difference.
- **Primary endpoint:** for each of the five seeds, the mean paired difference (agent − best_npv) on base
  400–499 with a 95% CI. "Significantly better" means the CI lies above zero.
- **Success criterion:** all five seeds significantly better on base, and the across-seed mean difference at
  least +3.0 $M.
- **Secondary:** each named regime (mean difference, number of seeds significantly better or worse), survival.
- **Reporting:** all five seeds are reported, including any failures. The test set is not reused for tuning.
  If v2 fails the criterion, it is reported as a failure.

## Control arms (added 2026-10-19 while seed 0 trained, before any test evaluation)
Two controls isolate what the fixes contribute. Both use the same selection, exploration floor, opponents and
test as v2:
- **No randomisation** (`config/train_v2_nodr.yaml`): `randomize: 0`. Seeds 0–2.
- **No imitation** (`config/train_v2_pure.yaml`): pure PPO from a random network. Seed 0.

The test runs once, after all arms finish, on the markets listed above.

## Analyses fixed in advance
- Behaviour and market-dynamics analysis uses v2 seed 0, chosen before training, and analysis markets 200–229.
- Ablations: all five v2 seeds, each mechanism switched off, on seeds 400–449.

## Amendments after the test (added 2026-10-19, 21:40 IST onwards; not part of the plan above)
- **Scoring correction.** A bankrupt company's final value still counted its unfinished projects and recently
  live products. Fixed in `economy/world.py` (`net_worth`: a bankrupt firm is worth cash − debt). The test games,
  switch-off games and analysis markets were replayed once with identical decisions and re-scored; pre-fix results
  are kept as `results/v2/*_before_scoring_fix.json`. Survival and every main-agent classification were unchanged.
- **Additional intervals.** `scripts/ci_v2.py` adds percentile-bootstrap intervals per agent and a two-level
  (training runs × markets) bootstrap for the five-agent average. The per-agent normal-approximation interval above
  remains the one that decides the success criterion.
- **Provenance.** This file was first committed in `68bf412`, together with the results; its local creation time
  (01:19:37 IST) precedes the first training run (01:20:49 IST) but is not independent evidence.
