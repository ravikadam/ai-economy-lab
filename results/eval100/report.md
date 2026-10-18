# AI Economy Lab — evaluation report

Checkpoint: `runs/ppo-20261018-113906`  ·  seeds 300–399 (n=100, unseen in training)  ·  generated 2026-10-18 12:26

Protocol: focal seat 0 is the candidate; seats 1–4 and entrants are a fixed rule-based background. Same seed ⇒ same latent world for every candidate (paired comparison). Raw return = Δ net worth in $M (cash − debt + PV of live products).

## Base scenario: `base`

| controller | raw return $M (±95% CI) | survival | disc. profit $k | p10 return | launches | tech-fail rate | comm-fail rate | policy − this |
|---|---|---|---|---|---|---|---|---|
| policy | 19.91 ± 2.18 | 0.98 | 13436 | 6.01 | 2.97 | 0.65 | 0.13 | — |
| random | -1.83 ± 0.02 | 0.00 | 50 | -1.90 | 0.16 | 0.08 | 0.00 | +21.73 ± 2.19 |
| easy_first | 3.06 ± 0.97 | 0.64 | 3529 | -1.67 | 2.95 | 0.20 | 0.10 | +16.85 ± 2.00 |
| best_npv | 14.04 ± 1.58 | 0.99 | 10523 | 4.47 | 2.97 | 0.42 | 0.04 | +5.87 ± 2.49 |
| cash_preserving | 7.16 ± 1.77 | 0.77 | 5482 | -1.75 | 2.50 | 0.36 | 0.00 | +12.75 ± 2.37 |

## Held-out scenario: `compute_crunch` (config/scenarios/compute_crunch.yaml)

| controller | raw return $M (±95% CI) | survival | disc. profit $k | p10 return | launches | tech-fail rate | comm-fail rate | policy − this |
|---|---|---|---|---|---|---|---|---|
| policy | 7.75 ± 1.91 | 0.69 | 4295 | -1.87 | 2.97 | 0.65 | 0.12 | — |
| random | -1.83 ± 0.02 | 0.00 | 24 | -1.90 | 0.16 | 0.09 | 0.00 | +9.58 ± 1.91 |
| easy_first | -1.65 ± 0.11 | 0.04 | -4388 | -1.90 | 2.96 | 0.19 | 0.10 | +9.40 ± 1.90 |
| best_npv | 6.22 ± 1.37 | 0.79 | 5772 | -1.76 | 3.08 | 0.41 | 0.03 | +1.53 ± 2.13 |
| cash_preserving | 3.94 ± 1.36 | 0.63 | 3633 | -1.81 | 2.49 | 0.35 | 0.01 | +3.81 ± 2.26 |

## Reading this report

* Results are from a stylised simulator with illustrative parameters — not evidence about real AI markets.
* The RL policy is not required to beat baselines everywhere; regret vs the best heuristic is reported as-is.
* Shaped (training) reward is reported separately in JSON; the table shows raw economic return.
* Wall time: 251s.