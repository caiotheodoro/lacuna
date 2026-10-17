# Live findings: can capture-recapture count what no log shows?

Three witnesses: W1 the agent's own trace, W2 the gateway log, W3 the
end-of-shift env diff plus network sink. The LLM auditor is dropped (D20).
Ground truth is the environment kernel's ledger, which no witness reads.

Episodes are recorded with complete logs; loss is applied afterwards, 100
replicates per configuration. Headline numbers use the bootstrap run, which
resamples the 15 episodes with replacement each replicate so the interval
includes between-shift variation as well as logging faults (D23).

## 1. The method recovers what the logs lost

| hidden from all 3 | `union_count` coverage | `ll_independence` coverage | width | detects gap |
|---|---|---|---|---|
| 0.06 (`light`) | 0.940 | 0.940 | 0.1 | 0.00 |
| 0.94 (`fin_light`) | 0.360 | 0.940 | 2.44 | 0.00 |
| 5.1 (`sampled`) | 0.010 | 0.920 | 9.28 | 0.69 |
| 48.77 (`fin_sampled`) | 0.000 | 0.970 | 36.94 | 1.00 |

In the hardest configuration 48.8 of 589 actions (8.3%) reach no witness on average. The
union of all three logs undercounts by exactly that and never covers.

## 2. Coverage alone would not settle it

`union_bound` covers 1.000 in every configuration by being unbounded above in
every replicate. A wide interval can pass a coverage test while saying
nothing, so each estimator also reports **detects gap**: the share of
replicates whose lower end exceeds the union of all witnesses, which is the
claim an auditor needs — *your logs are missing actions, at 95%*.

At `fin_sampled` `ll_independence` makes that claim in 100% of replicates.
`union_bound` makes it in 0%, by construction. That is the difference between
a useful interval and a safe one.

## 3. Breaking the assumption breaks the estimator, as predicted

The method's success could have been circular: the hidden actions came from
independent coin flips, and `ll_independence` assumes independence. So the
assumption was broken on purpose (D22, predictions written first) by coupling
the gateway and trace losses, as one outage takes both pipes down.

| coupling | `ll_independence` coverage | its bias | `ll_bic` coverage | its bias |
|---|---|---|---|---|
| none | 0.970 | +0.48 | 0.950 | +1.21 |
| half | 0.920 | -4.64 | 0.930 | -2.47 |
| full | 0.780 | -10.82 | 0.960 | -0.51 |

The independence model undercovers and undercounts once its assumption is
false, and the model-selection variant recovers by fitting the W1:W2
interaction that is now really there. The pre-set falsifier — coverage
holding at 0.95 under full coupling — did not occur. `lc_gated` never reports
a confident wrong answer, and `ll_average` goes unbounded in every replicate
at full coupling rather than answer.

## 4. What is weak

- **Under a shared outage only the model-selecting variant holds up.**
  `ll_bic` stays at 0.960, while `ll_independence` falls to 0.780 and
  undercounts by 10.8. An auditor who picked the simpler model because it won
  under independent loss would be misled by exactly the failure that matters.
  Which model to run cannot be settled from the data alone, and this project
  does not yet say how to choose.
- **The advantage appears only where intervals widen.** At `light` the method
  ties `union_count` (0.940 both) and detects nothing, because there is
  nothing to detect.
- **`strat_lc_gated` is dominated**: it refuses 43% of `fin_sampled`
  replicates and averages 325 wide when it answers. Stratification needs a
  better design than summing per-op intervals (D17).
- **`anchor_g1` undercovers at `fin_light`** (0.860 in the fixed-episode run; see that artifact),
  because restricting the env view makes its ψ=1 assumption wrong.

## 5. Scope

One agent (`deepseek-v4-flash`), one environment, 15 episodes, benign loss
only. No adversarial rung has been run live (D10): every L1/L2 number in this
repo is from the parametric simulator. Signature linkage — what an auditor
without ground-truth ids can do — matched oracle linkage throughout
(`fin_sampled`: coverage 0.970 against 0.950, width 36.3 against 36.8).

Bootstrapping over episodes barely moved any number, against the expectation
that it would widen them: between-shift variation is small next to logging
loss here, because the estimator is applied to a pooled table that scales
with the sample.

## Tables (bootstrap run)

Source: `artifacts/thinning/long-30.bootstrap.json` — 15 episodes, N=589, 100 loss replicates per configuration, episodes resampled with replacement per replicate, seed 4242, oracle linkage, nominal 95%.

**`light`** — gateway drops 0.05, trace drops 0.1, env view `full`. Mean actions hidden from all three witnesses: **0.06**.

| estimator | coverage ± MCSE | detects gap | unbounded | mean width | bias | RMSE |
|---|---|---|---|---|---|---|
| `trust_trace` | 0.000 ± 0.000 | 0.0 | 0.00 | 0.0 | -59.49 | 59.91 |
| `union_count` | 0.940 ± 0.024 | 0.0 | 0.00 | 0.0 | -0.06 | 0.24 |
| `union_bound` | 1.000 ± 0.000 | 0.0 | 1.00 | — | -0.06 | 0.24 |
| `ll_independence` | 0.940 ± 0.024 | 0.0 | 0.00 | 0.1 | 0.03 | 0.25 |
| `ll_bic` | 0.940 ± 0.024 | 0.0 | 0.00 | 0.12 | 0.04 | 0.25 |
| `ll_average` | 1.000 ± 0.000 | 0.0 | 0.00 | 1.02 | -0.06 | 0.24 |
| `anchor_g1` | 0.940 ± 0.024 | 0.0 | 0.00 | 0.26 | 0.04 | 0.26 |
| `anchor_g4` | 0.990 ± 0.010 | 0.0 | 0.00 | 1.24 | 0.04 | 0.26 |
| `lc_gated` | 1.000 ± 0.000 | 0.0 | 0.00 | 5.19 | 0.37 | 0.9 |
| `strat_lc_gated` | 1.000 ± 0.000 | 0.0 | 0.05 | 25.92 | 1.13 | 1.63 |

**`sampled`** — gateway drops 0.9, trace drops 0.3, env view `full`. Mean actions hidden from all three witnesses: **5.1**.

| estimator | coverage ± MCSE | detects gap | unbounded | mean width | bias | RMSE |
|---|---|---|---|---|---|---|
| `trust_trace` | 0.000 ± 0.000 | 0.0 | 0.00 | 0.0 | -176.34 | 176.68 |
| `union_count` | 0.010 ± 0.010 | 0.0 | 0.00 | 0.0 | -5.1 | 5.66 |
| `union_bound` | 1.000 ± 0.000 | 0.0 | 1.00 | — | -5.1 | 5.66 |
| `ll_independence` | 0.920 ± 0.027 | 0.69 | 0.00 | 9.28 | 0.01 | 2.83 |
| `ll_bic` | 0.920 ± 0.027 | 0.69 | 0.00 | 9.95 | 0.32 | 4.07 |
| `ll_average` | 0.910 ± 0.029 | 0.25 | 0.01 | 14.35 | 0.21 | 4.77 |
| `anchor_g1` | 0.930 ± 0.025 | 0.69 | 0.00 | 9.26 | 0.04 | 2.85 |
| `anchor_g4` | 0.980 ± 0.014 | 0.69 | 0.00 | 34.88 | 0.04 | 2.85 |
| `lc_gated` | 0.970 ± 0.017 | 0.39 | 0.45 | 24.51 | 1.58 | 4.3 |
| `strat_lc_gated` | 1.000 ± 0.000 | 0.11 | 0.27 | 60.91 | 2.04 | 3.84 |

**`fin_light`** — gateway drops 0.05, trace drops 0.1, env view `financial`. Mean actions hidden from all three witnesses: **0.94**.

| estimator | coverage ± MCSE | detects gap | unbounded | mean width | bias | RMSE |
|---|---|---|---|---|---|---|
| `trust_trace` | 0.000 ± 0.000 | 0.0 | 0.00 | 0.0 | -60.66 | 61.22 |
| `union_count` | 0.360 ± 0.048 | 0.0 | 0.00 | 0.0 | -0.94 | 1.36 |
| `union_bound` | 1.000 ± 0.000 | 0.0 | 1.00 | — | -0.94 | 1.36 |
| `ll_independence` | 0.940 ± 0.024 | 0.0 | 0.00 | 2.44 | 0.0 | 1.02 |
| `ll_bic` | 0.940 ± 0.024 | 0.0 | 0.00 | 2.46 | 0.01 | 1.08 |
| `ll_average` | 0.970 ± 0.017 | 0.0 | 0.00 | 3.56 | -0.07 | 1.13 |
| `anchor_g1` | 0.890 ± 0.031 | 0.0 | 0.00 | 2.67 | -0.09 | 1.18 |
| `anchor_g4` | 0.960 ± 0.020 | 0.0 | 0.00 | 9.34 | -0.09 | 1.18 |
| `lc_gated` | 0.980 ± 0.014 | 0.02 | 0.05 | 9.58 | 1.06 | 2.06 |
| `strat_lc_gated` | 0.990 ± 0.010 | 0.04 | 0.16 | 71.77 | 3.31 | 4.66 |

**`fin_sampled`** — gateway drops 0.9, trace drops 0.3, env view `financial`. Mean actions hidden from all three witnesses: **48.77**.

| estimator | coverage ± MCSE | detects gap | unbounded | mean width | bias | RMSE |
|---|---|---|---|---|---|---|
| `trust_trace` | 0.000 ± 0.000 | 0.0 | 0.00 | 0.0 | -175.65 | 175.97 |
| `union_count` | 0.000 ± 0.000 | 0.0 | 0.00 | 0.0 | -48.77 | 49.16 |
| `union_bound` | 1.000 ± 0.000 | 0.0 | 1.00 | — | -48.77 | 49.16 |
| `ll_independence` | 0.970 ± 0.017 | 1.0 | 0.00 | 36.94 | 0.48 | 9.71 |
| `ll_bic` | 0.950 ± 0.022 | 0.98 | 0.00 | 38.82 | 1.21 | 15.04 |
| `ll_average` | 0.970 ± 0.017 | 0.98 | 0.00 | 62.26 | 1.42 | 13.97 |
| `anchor_g1` | 0.960 ± 0.020 | 1.0 | 0.00 | 37.92 | 0.86 | 10.11 |
| `anchor_g4` | 0.990 ± 0.010 | 1.0 | 0.00 | 233.18 | 0.86 | 10.11 |
| `lc_gated` | 1.000 ± 0.000 | 0.85 | 0.15 | 69.18 | 1.68 | 8.95 |
| `strat_lc_gated` | 1.000 ± 0.000 | 0.93 | 0.37 | 278.36 | -5.23 | 19.5 |

**`fin_sampled_rho50`** — gateway drops 0.9, trace drops 0.3, env view `financial`. Mean actions hidden from all three witnesses: **51.17**.

| estimator | coverage ± MCSE | detects gap | unbounded | mean width | bias | RMSE |
|---|---|---|---|---|---|---|
| `trust_trace` | 0.000 ± 0.000 | 0.0 | 0.00 | 0.0 | -176.87 | 177.16 |
| `union_count` | 0.000 ± 0.000 | 0.0 | 0.00 | 0.0 | -51.17 | 51.56 |
| `union_bound` | 1.000 ± 0.000 | 0.0 | 1.00 | — | -51.17 | 51.56 |
| `ll_independence` | 0.920 ± 0.027 | 1.0 | 0.00 | 35.54 | -4.64 | 9.27 |
| `ll_bic` | 0.930 ± 0.025 | 0.97 | 0.00 | 37.15 | -2.47 | 11.88 |
| `ll_average` | 0.980 ± 0.014 | 0.9 | 0.00 | 115.29 | -2.66 | 11.9 |
| `anchor_g1` | 0.980 ± 0.014 | 1.0 | 0.00 | 39.02 | 0.46 | 8.62 |
| `anchor_g4` | 1.000 ± 0.000 | 1.0 | 0.00 | 242.01 | 0.46 | 8.62 |
| `lc_gated` | 1.000 ± 0.000 | 0.56 | 0.44 | 122.9 | -2.35 | 13.31 |
| `strat_lc_gated` | 1.000 ± 0.000 | 0.46 | 0.59 | 207.03 | -26.9 | 29.7 |

**`fin_sampled_rho100`** — gateway drops 0.9, trace drops 0.3, env view `financial`. Mean actions hidden from all three witnesses: **54.85**.

| estimator | coverage ± MCSE | detects gap | unbounded | mean width | bias | RMSE |
|---|---|---|---|---|---|---|
| `trust_trace` | 0.000 ± 0.000 | 0.0 | 0.00 | 0.0 | -176.62 | 177.0 |
| `union_count` | 0.000 ± 0.000 | 0.0 | 0.00 | 0.0 | -54.85 | 55.45 |
| `union_bound` | 1.000 ± 0.000 | 0.0 | 1.00 | — | -54.85 | 55.45 |
| `ll_independence` | 0.780 ± 0.041 | 1.0 | 0.00 | 34.16 | -10.82 | 14.55 |
| `ll_bic` | 0.960 ± 0.020 | 1.0 | 0.00 | 40.56 | -0.51 | 10.37 |
| `ll_average` | 0.990 ± 0.010 | 1.0 | 1.00 | — | 0.27 | 10.37 |
| `anchor_g1` | 0.960 ± 0.020 | 1.0 | 0.00 | 40.52 | -0.57 | 10.4 |
| `anchor_g4` | 0.990 ± 0.010 | 1.0 | 0.00 | 253.56 | -0.57 | 10.4 |
| `lc_gated` | 1.000 ± 0.000 | 1.0 | 0.00 | 129.47 | -7.03 | 16.94 |
| `strat_lc_gated` | 1.000 ± 0.000 | 0.0 | 1.00 | — | None | None |

