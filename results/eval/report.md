# AI Economy Lab — evaluation report

Checkpoint: `runs/latest`  ·  seeds 100–129 (n=30, unseen in training)  ·  generated 2026-10-18 11:48

Protocol: focal seat 0 is the candidate; seats 1–4 and entrants are a fixed rule-based background. Same seed ⇒ same latent world for every candidate (paired comparison). Raw return = Δ net worth in $M (cash − debt + PV of live products).

## Base scenario: `base`

| controller | raw return $M (±95% CI) | survival | disc. profit $k | p10 return | launches | tech-fail rate | comm-fail rate | policy − this |
|---|---|---|---|---|---|---|---|---|
| policy | 17.21 ± 3.55 | 0.97 | 11801 | 6.16 | 2.93 | 1.07 | 0.16 | — |
| random | -1.81 ± 0.04 | 0.00 | 2 | -1.90 | 0.07 | 0.08 | 0.00 | +19.02 ± 3.55 |
| easy_first | 1.79 ± 1.73 | 0.53 | 2957 | -1.77 | 3.03 | 0.26 | 0.16 | +15.41 ± 3.73 |
| best_npv | 13.12 ± 3.90 | 0.93 | 10073 | 1.51 | 2.80 | 0.46 | 0.00 | +4.09 ± 4.81 |
| cash_preserving | 5.79 ± 3.23 | 0.70 | 4919 | -1.81 | 2.37 | 0.38 | 0.02 | +11.42 ± 5.29 |

## Held-out scenario: `compute_crunch` (config/scenarios/compute_crunch.yaml)

| controller | raw return $M (±95% CI) | survival | disc. profit $k | p10 return | launches | tech-fail rate | comm-fail rate | policy − this |
|---|---|---|---|---|---|---|---|---|
| policy | 4.58 ± 2.70 | 0.53 | 2469 | -1.70 | 2.93 | 1.07 | 0.16 | — |
| random | -1.82 ± 0.05 | 0.00 | 5 | -1.90 | 0.07 | 0.09 | 0.00 | +6.39 ± 2.72 |
| easy_first | -1.37 ± 0.47 | 0.03 | -3828 | -1.90 | 2.97 | 0.26 | 0.16 | +5.95 ± 2.69 |
| best_npv | 7.03 ± 2.61 | 0.80 | 6636 | -1.29 | 2.93 | 0.44 | 0.00 | -2.45 ± 3.46 |
| cash_preserving | 3.74 ± 2.72 | 0.60 | 3692 | -1.77 | 2.40 | 0.36 | 0.02 | +0.83 ± 3.78 |

## Held-out scenario: `entrant_wave` (config/scenarios/entrant_wave.yaml)

| controller | raw return $M (±95% CI) | survival | disc. profit $k | p10 return | launches | tech-fail rate | comm-fail rate | policy − this |
|---|---|---|---|---|---|---|---|---|
| policy | 15.12 ± 3.79 | 0.90 | 10542 | 0.57 | 2.87 | 1.03 | 0.18 | — |
| random | -1.85 ± 0.04 | 0.00 | 20 | -1.90 | 0.10 | 0.09 | 0.00 | +16.97 ± 3.78 |
| easy_first | 1.09 ± 1.39 | 0.53 | 2491 | -1.82 | 3.03 | 0.25 | 0.17 | +14.03 ± 4.08 |
| best_npv | 14.37 ± 3.77 | 0.93 | 10735 | 1.94 | 2.80 | 0.49 | 0.00 | +0.75 ± 5.02 |
| cash_preserving | 6.57 ± 2.16 | 0.93 | 5417 | -0.73 | 2.73 | 0.34 | 0.01 | +8.55 ± 4.64 |

## Held-out scenario: `fast_tech` (config/scenarios/fast_tech.yaml)

| controller | raw return $M (±95% CI) | survival | disc. profit $k | p10 return | launches | tech-fail rate | comm-fail rate | policy − this |
|---|---|---|---|---|---|---|---|---|
| policy | 18.93 ± 3.96 | 0.97 | 12900 | 5.93 | 2.93 | 0.52 | 0.12 | — |
| random | -1.80 ± 0.04 | 0.00 | 71 | -1.90 | 0.20 | 0.05 | 0.00 | +20.73 ± 3.96 |
| easy_first | 1.74 ± 1.56 | 0.67 | 2970 | -1.73 | 3.03 | 0.13 | 0.07 | +17.19 ± 4.06 |
| best_npv | 14.07 ± 3.60 | 0.97 | 10692 | 3.00 | 2.90 | 0.34 | 0.00 | +4.86 ± 5.12 |
| cash_preserving | 8.10 ± 3.38 | 0.83 | 6152 | -1.58 | 2.83 | 0.15 | 0.02 | +10.84 ± 4.49 |

## Held-out scenario: `harsh_failures` (config/scenarios/harsh_failures.yaml)

| controller | raw return $M (±95% CI) | survival | disc. profit $k | p10 return | launches | tech-fail rate | comm-fail rate | policy − this |
|---|---|---|---|---|---|---|---|---|
| policy | 15.54 ± 4.58 | 0.80 | 10461 | -1.08 | 2.67 | 1.52 | 0.09 | — |
| random | -1.85 ± 0.03 | 0.00 | 34 | -1.90 | 0.07 | 0.14 | 0.00 | +17.39 ± 4.58 |
| easy_first | 2.73 ± 1.85 | 0.57 | 3289 | -1.78 | 2.83 | 0.74 | 0.07 | +12.81 ± 5.03 |
| best_npv | 13.95 ± 4.18 | 0.80 | 10390 | -1.64 | 2.53 | 0.80 | 0.03 | +1.59 ± 5.66 |
| cash_preserving | 4.38 ± 2.28 | 0.70 | 4192 | -1.81 | 2.17 | 0.48 | 0.03 | +11.16 ± 4.76 |

## Held-out scenario: `slow_tech` (config/scenarios/slow_tech.yaml)

| controller | raw return $M (±95% CI) | survival | disc. profit $k | p10 return | launches | tech-fail rate | comm-fail rate | policy − this |
|---|---|---|---|---|---|---|---|---|
| policy | 17.26 ± 3.55 | 0.93 | 11766 | 5.18 | 2.93 | 1.32 | 0.17 | — |
| random | -1.83 ± 0.05 | 0.00 | 22 | -1.90 | 0.13 | 0.10 | 0.00 | +19.09 ± 3.56 |
| easy_first | 1.72 ± 1.44 | 0.67 | 3028 | -1.67 | 3.00 | 0.26 | 0.19 | +15.54 ± 3.54 |
| best_npv | 13.36 ± 3.98 | 0.93 | 10204 | 2.50 | 2.80 | 0.50 | 0.03 | +3.90 ± 5.31 |
| cash_preserving | 5.73 ± 2.36 | 0.77 | 4870 | -1.69 | 2.43 | 0.38 | 0.00 | +11.53 ± 4.67 |

## Ablations (one mechanism removed at a time)

| ablation | policy raw return | best_npv raw return | policy − best_npv |
|---|---|---|---|
| no competition | 18.27 | 23.48 | -5.21 |
| no technical_failure | 24.49 | 12.21 | +12.28 |
| no confidence_feedback | 16.43 | 12.99 | +3.44 |
| no supply_bottlenecks | 16.51 | 11.40 | +5.11 |
| no delayed_capex | 18.87 | 13.72 | +5.15 |
| no risk_heterogeneity | 17.24 | 15.31 | +1.93 |

## Reading this report

* Results are from a stylised simulator with illustrative parameters — not evidence about real AI markets.
* The RL policy is not required to beat baselines everywhere; regret vs the best heuristic is reported as-is.
* Shaped (training) reward is reported separately in JSON; the table shows raw economic return.
* Wall time: 334s.