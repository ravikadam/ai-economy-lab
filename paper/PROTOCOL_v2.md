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

## Revision plan R1 (written 2026-10-22 before any of these runs; pushed to GitHub before they start)
In response to review. Nothing below changes the original test or its results.

1. **More comparison runs.** Train `config/train_v2_nodr.yaml` with seeds 3–4 and `config/train_v2_pure.yaml`
   with seeds 1–4, unchanged, so each comparison arm has 5 runs. Test them exactly as the original runs (seeds
   400–499 base, 400–449 per named condition). These markets were already used once to test other agents; no
   agent or setting is tuned on them. Compare arms with the two-level bootstrap in `scripts/ci_v2.py` and report
   the smallest difference detectable with 80% power at the 5% level.
2. **Imitation-only baseline.** The imitation-trained starting network of each of the 5 full-recipe runs
   (`checkpoints/bc_init.pt`, before any PPO update), with its action mode (sampled or greedy) chosen on the 40
   practice markets exactly as for the trained agents, tested like item 1.
3. **Strength of the yardstick.** On fresh markets 500–549 (base and compute crunch): the oracle rule (best expected
   value given the true hidden difficulty), all four rules and the five agents, in the focal seat.
4. **Sensitivity.** On fresh markets 500–549, base scenario, the five agents and best expected value with one
   parameter changed at a time: competition discount 0.06 and 0.24 (base 0.12); monthly credit cost 0.006 and 0.024
   (base 0.012); price share 0.24 and 0.40 (base 0.32); overhead 15 and 40 $k/month (base 25). Report the lead and
   its two-level interval for each.
5. **Crunch bankruptcies.** On fresh markets 500–549, compute crunch: per-month projects running, budget tier,
   model choice, runway and debt for the agents and best expected value in the focal seat, comparing companies that
   went bankrupt with those that survived.

## Revision plan R2 (written 2026-10-22 15:55 IST before running; pushed before the runs start)
In response to a second review. Nothing below changes earlier results.

1. **Robustness of the market findings.** On fresh markets 600–629, with the v2 seed-0 agent in the same mixed
   population as the analysis markets (`scripts/experiments.py`), vary one structural parameter at a time, low and
   high: adoption steepness 2 and 8 (base 4); competition discount 0.06 and 0.24 (0.12); entry base probability 0.02
   and 0.08 (0.04); compute price ceiling 0.015 and 0.06 (0.03); confidence diffusion 0.15 and 0.6 (0.3); fab lead
   time 12 and 24 months (18). Each variant runs in the base scenario and the compute crunch with the same change.
   Report for each: S-curve shape (months to half of final adoption, and whether monthly new adoptions peak in the
   middle of the decade), regulated-sector adoption relative to the others, the first launcher's share against an
   even split, founding companies' value relative to entrants, and the crunch's compute-price peak and value
   created relative to the matching base variant.
2. **Multiple comparisons.** From the saved test results only: the success bar (five per-agent tests in normal
   markets) is the single primary endpoint. For the 30 per-agent tests across six conditions, and the six pooled
   five-agent tests, report how many leads stay clear after a Holm correction for the family-wise error rate.

## Revision plan R3: PFAR (written 2026-10-22 before implementation; pushed before any PFAR run)
**Method.** Paired-Fork Advantage against a Rule. During PPO training, at a sampled share (25%) of decisions
where the learning company's action and the best-expected-value rule's action differ and are not both "hold",
the worker copies the market twice. Copy A applies the agent's action, copy B the rule's action; every other
company's action that month is identical. Both copies then run 12 months with all learning and frozen seats
acting by the current policy (a sampling generator seeded identically in both copies) and all random streams
shared (common random numbers). The fork advantage is
  A = sum_{k<12} 0.99^k (r^A_k - r^B_k)/s + 0.99^12 (V(s^A_12) - V(s^B_12)),
with shaped rewards r, the running reward scale s, the current value function V, and no bootstrap after
bankruptcy. It replaces the GAE advantage for that sample. Q(s, rule(s)) does not depend on the agent's action,
so it is a valid baseline. Everything else is identical to `config/train_v2.yaml` (and `train_v2_pure.yaml`).

**Arms.** PFAR with imitation start (seeds 0-4) and PFAR from scratch (seeds 0-4), compared with the existing
full-recipe runs (v2, seeds 0-4) and no-imitation runs (v2pure, seeds 0-4). Same practice-market selection.

**Test (once, after all PFAR runs finish).** Fresh markets 700-799 (base) and 700-749 for each named condition;
all 20 agents and the rules in the focal seat; sampled agents averaged over 3 plays.

**Hypotheses (decided now).**
- P1: PFAR's five-agent lead over the rule in normal markets exceeds v2's; supported if the two-level bootstrap
  interval of (PFAR - v2) lies above zero.
- P2: PFAR's average survival in the compute crunch exceeds v2's; supported if the interval of the difference lies
  above zero.
- P3: PFAR from scratch succeeds (clearly better than the rule in normal markets) in at least 4 of 5 runs, against
  3 of 5 for v2pure.
- Diagnostic D: on 200 sampled decisions from markets 800-819, the variance of the fork-advantage estimate with
  shared randomness versus independent randomness in the two copies (8 repeats each). Reported whatever it shows.
Negative or null results will be reported as such.
