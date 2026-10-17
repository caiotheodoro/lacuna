# Confirmatory result

Pre-registered in `docs/PRE-REGISTRATION.md`, frozen in commit `6ea27e2`
before this run existed. Tested on two loss configurations held out of every
exploratory run and of the estimator selection in D27, so the estimator was
not chosen on the data that tests it.

21 of 24 episodes completed; three were lost to transient provider errors and
the run continued past them (D30). N = 828 actions. Headline linkage is
effect-signature, what an auditor without ground truth can compute.

## The four hypotheses

Source: `artifacts/hypotheses.json`, from `artifacts/thinning/confirm.json`.

| | claim | result |
|---|---|---|
| **H1** recovery | `ll_bic` covers N in ≥ 90% of replicates | **PASS** — 0.940 |
| **H2** informativeness | `ll_bic` detects the gap in > 90% | **PASS** — 1.00 |
| **H3** dependence | `ll_independence` loses ≥ 10 points under coupling, `ll_bic` < 5 | **PASS** — 95.0 and 1.0 points |
| **H4** floor | `ll_bic` beats `union_count` on gap detection by ≥ 0.50 | **PASS** — 1.00 |

## What the numbers say

**`fin_mid`** (both pipes losing half their lines, env diff restricted to
financial ops): 63.15 of 828 actions reached no witness. The union of all
three logs undercounts by exactly that and covers the truth in 0% of
replicates. `ll_bic` covers in 94% with an interval 42.0 wide, and in every
replicate its lower end exceeds the union — it asserts the logs are
incomplete, which the trivial bound never does.

**`fin_mid_rho100`** (the same, with gateway and trace failing together):
122.79 actions hidden. `ll_independence` collapses to 0.0 coverage,
undercounting by 93.63 with an interval only 24.1 wide: confident and wrong,
the exact failure a reader should fear. `ll_bic` holds at 0.93 by fitting the
interaction, at 73.0 wide. H3 was the hypothesis that could have shown the
coupling experiment does not bite; it bit hard.

## Where it is weak

- **`lc_gated` fails under coupling**: 0.99 coverage but bias −65.62 and
  width 357.8, and its gap detection falls to 0.91. It stays nominally
  covered by being very wide, not by being right.
- **`anchor_g4` costs an order of magnitude in width** (545.0) for 3 points
  of coverage over `ll_bic`.
- **The study is 21 episodes, not the 24 planned.** The stopping rule said
  analyse what completed and report it as incomplete rather than top it up,
  so N is 828 rather than the ~950 the power calculation targeted, and the
  relative interval width is 5.1% rather than 5%.
- **Oracle and signature linkage agree** (0.940 against 0.940 at `fin_mid`),
  so linkage error is not carrying the result.

## Scope

One agent (`deepseek-v4.1-flash`), one environment, benign logging loss only.
A passing H1 is **not** evidence for the adversarial claims that motivate
`docs/PLAN.md`; no live adversarial run exists (D10).

## Table (signature linkage)

Source: `artifacts/thinning/confirm.json` — 21 episodes, N=828, 100 loss replicates per configuration, episodes resampled with replacement per replicate, seed 4242, signature linkage, nominal 95%.

**`fin_mid`** — gateway drops 0.5, trace drops 0.5, env view `financial`. Mean actions hidden from all three witnesses: **63.15**.

| estimator | coverage ± MCSE | detects gap | unbounded | mean width | bias | RMSE |
|---|---|---|---|---|---|---|
| `trust_trace` | 0.000 ± 0.000 | 0.0 | 0.00 | 0.0 | -412.92 | 413.3 |
| `union_count` | 0.000 ± 0.000 | 0.0 | 0.00 | 0.0 | -63.15 | 63.67 |
| `union_bound` | 1.000 ± 0.000 | 0.0 | 1.00 | — | -63.15 | 63.67 |
| `ll_independence` | 0.950 ± 0.022 | 1.0 | 0.00 | 38.63 | -1.77 | 10.34 |
| `ll_bic` | 0.940 ± 0.024 | 1.0 | 0.00 | 42.04 | 3.39 | 12.63 |
| `ll_average` | 0.970 ± 0.017 | 1.0 | 0.00 | 57.52 | 4.57 | 12.7 |
| `anchor_g1` | 0.940 ± 0.024 | 1.0 | 0.00 | 45.35 | 7.18 | 13.3 |
| `anchor_g4` | 0.940 ± 0.024 | 1.0 | 0.00 | 310.71 | 7.18 | 13.3 |
| `lc_gated` | 0.990 ± 0.010 | 1.0 | 0.00 | 138.89 | 18.31 | 23.76 |
| `strat_lc_gated` | 1.000 ± 0.000 | 1.0 | 0.00 | 361.52 | 0.35 | 12.42 |

**`fin_mid_rho100`** — gateway drops 0.5, trace drops 0.5, env view `financial`. Mean actions hidden from all three witnesses: **122.79**.

| estimator | coverage ± MCSE | detects gap | unbounded | mean width | bias | RMSE |
|---|---|---|---|---|---|---|
| `trust_trace` | 0.000 ± 0.000 | 0.0 | 0.00 | 0.0 | -414.97 | 415.45 |
| `union_count` | 0.000 ± 0.000 | 0.0 | 0.00 | 0.0 | -122.79 | 123.33 |
| `union_bound` | 1.000 ± 0.000 | 0.0 | 1.00 | — | -122.79 | 123.33 |
| `ll_independence` | 0.000 ± 0.000 | 1.0 | 0.00 | 24.15 | -93.63 | 94.4 |
| `ll_bic` | 0.930 ± 0.025 | 1.0 | 0.00 | 72.99 | 1.53 | 19.53 |
| `ll_average` | 0.950 ± 0.022 | 1.0 | 0.63 | 73.76 | 2.92 | 19.83 |
| `anchor_g1` | 0.930 ± 0.025 | 1.0 | 0.00 | 72.99 | 1.53 | 19.53 |
| `anchor_g4` | 0.960 ± 0.020 | 1.0 | 0.00 | 544.96 | 1.53 | 19.53 |
| `lc_gated` | 0.990 ± 0.010 | 0.91 | 0.09 | 357.76 | -65.62 | 82.54 |
| `strat_lc_gated` | 1.000 ± 0.000 | 0.01 | 1.00 | — | None | None |

