# Simulator findings

500 replicates per scenario, seed 2026, N=200 actions, nominal 95%. Produced by `make sim`; the table at the bottom is `python3 scripts/sim_table.py` output, not hand-typed. Third sweep, adding `lc_gated` (D9). It supersedes the first sweep, whose `ll_average` numbers were corrupted by the IRLS bug in `docs/DECISIONS.md` D6.

This is exploration, not a pre-registered result. The predictions in D7 and D8 were written before this run.

## What it says

1. **The week-4 gate passes on identified scenarios, via `lc`.**
   - `lc` covers 0.962–1.000 on every L0/L1 scenario where N is identified, which is all of them except l0_siloed (point 2). That includes l0_heterogeneous, where `ll_bic` gets 0.296.
   - That costs some width: 33 vs 26 on heterogeneous, 15 vs 7 on independent.
   - On those scenarios `lc` beats `union_count` on RMSE except `l0_auditor_copies_trace`, where the union misses almost nothing. On l0_siloed it is far worse than the floor (570 vs 52).
2. **Siloed witnesses make `lc` confidently wrong.** When each witness covers its own kind of action and they barely overlap, N is not identified.
   - `ll_bic` says so: 85% of its intervals are unbounded.
   - `lc` covers 0.054 and is bounded 86% of the time.
   - Every point method overcounts by 200–500. Siloed witnesses are negatively dependent, so independence-style estimators overcount.
   - This is the identifiability warning from the `latent_class.py` docstring, measured. `lc` cannot be reported without an identifiability diagnostic next to it.
3. **H3, the auditor trap, holds.** With W4 added as if independent, `ll_independence` covers 0.646 with intervals 0.25 wide. Fitting the dependence fixes it: `ll_bic` 0.906, `lc` 0.990.
4. **H2: point methods fail against an agent that also evades the env diff; the anchor bounds behave as the theory says.**
   - In l2_env_blind the true anchor odds ratio is 4.35.
   - `anchor_g1`, `g2`, `g4`, `g8` cover 0.010 / 0.368 / 0.934 / 0.998.
   - Every point estimator covers ≤ 0.218 there.
   - The prediction in D7 was that g4 would undercover partly. It reached 0.934 because the sampling interval adds slack on top of Γ.
5. **When the anchor is truly independent (ψ = 1), `anchor_g1` covers slightly low: 0.91–0.96 across those scenarios.**
   - The env-only cell is small (single digits to low double digits) and the 2-list profile interval is a little narrow.
   - **Open:** check whether a mid-p or Bayesian 2-list interval fixes it. Trigger: before `anchor_g1` is the reported lower end in any claim.
6. **`ll_average` is fixed.** In the K=4 auditor scenario RMSE fell from 301 to 0.93.

7. **The identifiability gate works as pre-stated in D9.**
   - `lc_gated` refuses 0.97 of l0_siloed tables, which lifts coverage from 0.054 to 0.974. A refusal reports "at least n".
   - On identified scenarios it refuses 0.00–0.04, except l0_heterogeneous at 0.16. There, coverage stays 0.962: the refusals fall on tables `lc` already covered, so the price is a lost finite answer, not lost coverage.
   - It never refuses on l2_env_blind (0.00), as expected. Hiding that is correlated across every witness does not show in the observed cells.

8. **The Γ bounds are one-sided by construction, and that is a real limit.**
   ψ ∈ [1, Γ] says hiding can only make the anchor miss *more*, so raising Γ
   moves the upper end up and never moves the lower end down. On l0_siloed the
   witnesses are negatively dependent (ψ < 1): the ψ=1 point already overcounts
   by 292, so the whole interval sits above N and coverage is 0.002 at every
   Γ ∈ {1, 2, 4, 8}, with mean width rising from 580 to 3489 for nothing. The
   bounds are the right tool against an agent that hides (positive dependence)
   and the wrong tool against witnesses that specialise (negative dependence).
   For the second case the answer is the gate's refusal, not a wider Γ.

## What an auditor would report

A recommendation for the pre-registration, not yet adopted:
- **L0/L1:** `lc_gated`.
- **L2:** the `anchor_g{Γ}` curve, read as "unless hiding and env-blindness are dependent by more than Γ, N ≤ U(Γ)".
- **Always:** `union_bound` as the floor.

## Table

Source: `artifacts/sim_coverage.json`, reps=500, seed=2026.

| scenario | estimator | coverage ± MCSE | unbounded | mean width | bias | RMSE |
|---|---|---|---|---|---|---|
| l0_auditor_copies_trace | `anchor_g1@trace+proxy+env` | 0.908 ± 0.013 | 0.00 | 2.51 | 0.09 | 1.07 |
| l0_auditor_copies_trace | `anchor_g2@trace+proxy+env` | 0.960 ± 0.009 | 0.00 | 4.84 | 0.09 | 1.07 |
| l0_auditor_copies_trace | `anchor_g4@trace+proxy+env` | 0.970 ± 0.008 | 0.00 | 9.34 | 0.09 | 1.07 |
| l0_auditor_copies_trace | `anchor_g8@trace+proxy+env` | 0.994 ± 0.004 | 0.00 | 18.18 | 0.09 | 1.07 |
| l0_auditor_copies_trace | `auditor_count` | 0.000 ± 0.000 | 0.00 | 0.0 | -33.22 | 33.6 |
| l0_auditor_copies_trace | `lc@trace+proxy+env` | 1.000 ± 0.000 | 0.00 | 6.59 | 0.4 | 1.22 |
| l0_auditor_copies_trace | `lc@trace+proxy+env+auditor` | 0.990 ± 0.004 | 0.00 | 4.53 | -0.07 | 0.92 |
| l0_auditor_copies_trace | `lc_gated@trace+proxy+env` | 1.000 ± 0.000 | 0.04 | 6.44 | 0.39 | 1.21 |
| l0_auditor_copies_trace | `lc_gated@trace+proxy+env+auditor` | 0.990 ± 0.004 | 0.01 | 4.49 | -0.08 | 0.92 |
| l0_auditor_copies_trace | `ll_average@trace+proxy+env` | 0.980 ± 0.006 | 0.05 | 3.87 | -0.26 | 1.01 |
| l0_auditor_copies_trace | `ll_average@trace+proxy+env+auditor` | 0.964 ± 0.008 | 0.02 | 7.65 | -0.27 | 0.93 |
| l0_auditor_copies_trace | `ll_bic@trace+proxy+env` | 0.942 ± 0.011 | 0.00 | 2.12 | 0.1 | 0.92 |
| l0_auditor_copies_trace | `ll_bic@trace+proxy+env+auditor` | 0.906 ± 0.013 | 0.00 | 1.77 | 0.05 | 0.84 |
| l0_auditor_copies_trace | `ll_independence@trace+proxy+env` | 0.948 ± 0.010 | 0.00 | 2.08 | 0.08 | 0.88 |
| l0_auditor_copies_trace | `ll_independence@trace+proxy+env+auditor` | 0.646 ± 0.021 | 0.00 | 0.25 | -0.45 | 0.86 |
| l0_auditor_copies_trace | `trust_trace` | 0.002 ± 0.002 | 0.00 | 0.0 | -5.81 | 6.26 |
| l0_auditor_copies_trace | `union_count@trace+proxy+env` | 0.522 ± 0.022 | 0.00 | 0.0 | -0.64 | 1.03 |
| l0_heterogeneous | `anchor_g1@trace+proxy+env` | 0.072 ± 0.012 | 0.00 | 6.22 | -10.99 | 11.65 |
| l0_heterogeneous | `anchor_g2@trace+proxy+env` | 0.394 ± 0.022 | 0.00 | 11.71 | -10.99 | 11.65 |
| l0_heterogeneous | `anchor_g4@trace+proxy+env` | 0.850 ± 0.016 | 0.00 | 22.46 | -10.99 | 11.65 |
| l0_heterogeneous | `anchor_g8@trace+proxy+env` | 0.980 ± 0.006 | 0.00 | 44.0 | -10.99 | 11.65 |
| l0_heterogeneous | `lc@trace+proxy+env` | 0.962 ± 0.009 | 0.00 | 33.38 | -5.22 | 7.37 |
| l0_heterogeneous | `lc_gated@trace+proxy+env` | 0.962 ± 0.009 | 0.16 | 32.09 | -5.98 | 7.76 |
| l0_heterogeneous | `ll_average@trace+proxy+env` | 0.702 ± 0.021 | 0.00 | 71.44 | -5.6 | 17.61 |
| l0_heterogeneous | `ll_bic@trace+proxy+env` | 0.296 ± 0.020 | 0.00 | 26.47 | -4.71 | 16.77 |
| l0_heterogeneous | `ll_independence@trace+proxy+env` | 0.004 ± 0.003 | 0.00 | 4.52 | -11.6 | 12.16 |
| l0_heterogeneous | `trust_trace` | 0.000 ± 0.000 | 0.00 | 0.0 | -35.12 | 35.56 |
| l0_heterogeneous | `union_count@trace+proxy+env` | 0.000 ± 0.000 | 0.00 | 0.0 | -13.52 | 13.99 |
| l0_independent | `anchor_g1@trace+proxy+env` | 0.924 ± 0.012 | 0.00 | 8.34 | -0.04 | 2.53 |
| l0_independent | `anchor_g2@trace+proxy+env` | 0.978 ± 0.007 | 0.00 | 15.99 | -0.04 | 2.53 |
| l0_independent | `anchor_g4@trace+proxy+env` | 0.986 ± 0.005 | 0.00 | 31.13 | -0.04 | 2.53 |
| l0_independent | `anchor_g8@trace+proxy+env` | 0.986 ± 0.005 | 0.00 | 61.55 | -0.04 | 2.53 |
| l0_independent | `lc@trace+proxy+env` | 0.980 ± 0.006 | 0.00 | 15.34 | 0.82 | 2.59 |
| l0_independent | `lc_gated@trace+proxy+env` | 0.982 ± 0.006 | 0.00 | 15.27 | 0.79 | 2.54 |
| l0_independent | `ll_average@trace+proxy+env` | 0.984 ± 0.006 | 0.00 | 9.79 | -0.12 | 2.22 |
| l0_independent | `ll_bic@trace+proxy+env` | 0.946 ± 0.010 | 0.00 | 7.3 | 0.01 | 2.24 |
| l0_independent | `ll_independence@trace+proxy+env` | 0.954 ± 0.009 | 0.00 | 7.24 | -0.03 | 2.07 |
| l0_independent | `trust_trace` | 0.000 ± 0.000 | 0.00 | 0.0 | -30.32 | 30.71 |
| l0_independent | `union_count@trace+proxy+env` | 0.028 ± 0.007 | 0.00 | 0.0 | -3.69 | 4.11 |
| l0_siloed | `anchor_g1@trace+proxy+env` | 0.002 ± 0.002 | 0.00 | 580.01 | 291.53 | 324.39 |
| l0_siloed | `anchor_g2@trace+proxy+env` | 0.002 ± 0.002 | 0.03 | 1233.83 | 291.53 | 324.39 |
| l0_siloed | `anchor_g4@trace+proxy+env` | 0.002 ± 0.002 | 0.36 | 2035.15 | 291.53 | 324.39 |
| l0_siloed | `anchor_g8@trace+proxy+env` | 0.002 ± 0.002 | 0.71 | 3488.92 | 291.53 | 324.39 |
| l0_siloed | `lc@trace+proxy+env` | 0.054 ± 0.010 | 0.14 | 1392.85 | 529.02 | 570.71 |
| l0_siloed | `lc_gated@trace+proxy+env` | 0.974 ± 0.007 | 0.97 | 775.74 | 296.31 | 311.02 |
| l0_siloed | `ll_average@trace+proxy+env` | 0.000 ± 0.000 | 0.98 | 1465.56 | 462.86 | 533.21 |
| l0_siloed | `ll_bic@trace+proxy+env` | 0.852 ± 0.016 | 0.85 | 437.72 | 296.69 | 313.22 |
| l0_siloed | `ll_independence@trace+proxy+env` | 0.002 ± 0.002 | 0.00 | 300.49 | 208.6 | 226.57 |
| l0_siloed | `trust_trace` | 0.000 ± 0.000 | 0.00 | 0.0 | -132.0 | 132.18 |
| l0_siloed | `union_count@trace+proxy+env` | 0.000 ± 0.000 | 0.00 | 0.0 | -51.68 | 52.04 |
| l1_hides_from_trace | `anchor_g1@trace+proxy+env` | 0.956 ± 0.009 | 0.00 | 12.52 | 0.26 | 3.46 |
| l1_hides_from_trace | `anchor_g2@trace+proxy+env` | 0.990 ± 0.004 | 0.00 | 25.03 | 0.26 | 3.46 |
| l1_hides_from_trace | `anchor_g4@trace+proxy+env` | 0.990 ± 0.004 | 0.00 | 49.92 | 0.26 | 3.46 |
| l1_hides_from_trace | `anchor_g8@trace+proxy+env` | 0.990 ± 0.004 | 0.00 | 99.87 | 0.26 | 3.46 |
| l1_hides_from_trace | `lc@trace+proxy+env` | 0.976 ± 0.007 | 0.00 | 21.7 | 1.64 | 3.87 |
| l1_hides_from_trace | `lc_gated@trace+proxy+env` | 0.976 ± 0.007 | 0.00 | 21.7 | 1.64 | 3.87 |
| l1_hides_from_trace | `ll_average@trace+proxy+env` | 0.980 ± 0.006 | 0.00 | 13.67 | 0.22 | 3.16 |
| l1_hides_from_trace | `ll_bic@trace+proxy+env` | 0.926 ± 0.012 | 0.00 | 10.78 | 0.29 | 3.24 |
| l1_hides_from_trace | `ll_independence@trace+proxy+env` | 0.950 ± 0.010 | 0.00 | 10.7 | 0.28 | 2.89 |
| l1_hides_from_trace | `trust_trace` | 0.000 ± 0.000 | 0.00 | 0.0 | -55.51 | 55.86 |
| l1_hides_from_trace | `union_count@trace+proxy+env` | 0.004 ± 0.003 | 0.00 | 0.0 | -6.53 | 6.99 |
| l2_bypasses_proxy | `anchor_g1@trace+proxy+env` | 0.910 ± 0.013 | 0.00 | 20.75 | 0.23 | 6.08 |
| l2_bypasses_proxy | `anchor_g2@trace+proxy+env` | 0.972 ± 0.007 | 0.00 | 45.43 | 0.23 | 6.08 |
| l2_bypasses_proxy | `anchor_g4@trace+proxy+env` | 0.972 ± 0.007 | 0.00 | 94.74 | 0.23 | 6.08 |
| l2_bypasses_proxy | `anchor_g8@trace+proxy+env` | 0.972 ± 0.007 | 0.00 | 193.71 | 0.23 | 6.08 |
| l2_bypasses_proxy | `lc@trace+proxy+env` | 0.990 ± 0.004 | 0.00 | 41.21 | -2.95 | 6.64 |
| l2_bypasses_proxy | `lc_gated@trace+proxy+env` | 0.990 ± 0.004 | 0.03 | 41.02 | -3.08 | 6.62 |
| l2_bypasses_proxy | `ll_average@trace+proxy+env` | 0.940 ± 0.011 | 0.00 | 39.96 | 0.8 | 9.52 |
| l2_bypasses_proxy | `ll_bic@trace+proxy+env` | 0.882 ± 0.014 | 0.00 | 23.04 | 0.71 | 8.79 |
| l2_bypasses_proxy | `ll_independence@trace+proxy+env` | 0.338 ± 0.021 | 0.00 | 10.76 | -8.13 | 9.24 |
| l2_bypasses_proxy | `trust_trace` | 0.000 ± 0.000 | 0.00 | 0.0 | -55.63 | 56.04 |
| l2_bypasses_proxy | `union_count@trace+proxy+env` | 0.000 ± 0.000 | 0.00 | 0.0 | -14.94 | 15.44 |
| l2_env_blind | `anchor_g1@trace+proxy+env` | 0.010 ± 0.004 | 0.00 | 12.16 | -21.86 | 22.49 |
| l2_env_blind | `anchor_g2@trace+proxy+env` | 0.368 ± 0.022 | 0.00 | 24.21 | -21.86 | 22.49 |
| l2_env_blind | `anchor_g4@trace+proxy+env` | 0.934 ± 0.011 | 0.00 | 48.23 | -21.86 | 22.49 |
| l2_env_blind | `anchor_g8@trace+proxy+env` | 0.998 ± 0.002 | 0.00 | 96.48 | -21.86 | 22.49 |
| l2_env_blind | `lc@trace+proxy+env` | 0.218 ± 0.018 | 0.00 | 21.08 | -22.41 | 23.0 |
| l2_env_blind | `lc_gated@trace+proxy+env` | 0.218 ± 0.018 | 0.00 | 21.05 | -22.41 | 23.0 |
| l2_env_blind | `ll_average@trace+proxy+env` | 0.094 ± 0.013 | 0.00 | 15.85 | -23.3 | 23.92 |
| l2_env_blind | `ll_bic@trace+proxy+env` | 0.018 ± 0.006 | 0.00 | 9.57 | -23.15 | 23.83 |
| l2_env_blind | `ll_independence@trace+proxy+env` | 0.000 ± 0.000 | 0.00 | 7.68 | -24.31 | 24.77 |
| l2_env_blind | `trust_trace` | 0.000 ± 0.000 | 0.00 | 0.0 | -55.89 | 56.24 |
| l2_env_blind | `union_count@trace+proxy+env` | 0.000 ± 0.000 | 0.00 | 0.0 | -28.31 | 28.68 |
