# MNAR robustness: stratifying on what the auditor can see

**Status: exploratory.** Built after the confirmatory run, like D31/D32. Nothing
here can change the H1-H4 verdicts in `docs/CONFIRMATORY.md`. The `heldout` run
below is a post-hoc regression check for a new estimator on configurations whose
results were already public; it is not a confirmatory test of `strat_bic`.

## The question

D32 found that under the `mnar` pool (both pipes drop `update_customer` and
`write_file` at 1.8x the rate, and the `financial` env view cannot see those ops
either) `ll_bic` covers only by widening to ~2x N, and covers nothing once a
correlated outage is added.

That pool's missingness depends on `op`, and every captured record carries its
op. So the loss is MAR given an observed covariate, not MNAR in the strict
sense: inside one op the drops are independent coin flips at a constant rate.
A model fitted per op is then correctly specified even where a pooled model is
not. The question is how much of quantification that buys back, and where it
stops: when the loss is keyed on something no record carries.

## Design (written before any run)

### Estimator: `strat_bic` (`src/lacuna/estimators/stratified.py`)

Per op stratum:

1. Drop witnesses that saw nothing in the stratum (W3 under the `financial`
   view, in non-financial strata).
2. Require at least two live witnesses and discordance (some action seen by a
   proper subset of them, D17). Otherwise the stratum is **unidentified**.
3. Fit every log-linear model in `model_space`, take the BIC-best converged
   one, and evaluate its profile likelihood on an integer grid of N_s (dense
   up to 300 points, otherwise a geometric grid interpolated onto the
   integers), over the region within a chi-square drop of 30 of the peak.
4. Normalise it into a density: a flat prior on N_s >= n_s. If the
   likelihood is still non-negligible at the search cap (20 n_s, as for every
   log-linear estimator here), extrapolate the tail as a power law, which is
   its large-N shape (two witnesses: L ~ N^-m, m the doubly-seen count). Tail
   mass sits at +inf. An exponent <= 1 is an improper posterior, and the
   stratum is **unbounded**.
5. Convolve the identified strata's densities and report the total's median
   and central 95% interval. A quantile the finite mass cannot reach is +inf.

Unidentified and unbounded strata are left out of the convolution. Each adds
its observed count to the lower end (a valid floor whatever it missed) and
sends the point and upper end to +inf. Empty strata are skipped: an auditor
cannot know an op exists that no witness recorded, and under oracle linkage the
stratum list comes from the ledger, which no estimator may read.

How this differs from `strat_lc_gated` (D17, D27): that summed per-stratum
interval endpoints at a Bonferroni level, which spends coverage once per
stratum and was dominated (D27). `strat_bic` spends it once,
on the distribution of the total. `strat_lc_gated` stays in `estimate_all` for
comparison.

### Boundary pool: `mnar_hidden` (`src/lacuna/thinning.py`)

`Config` gains `hidden_mult` and `hidden_env_blind`. An action is in the hidden
class when its ledger id is a multiple of 3: a key no witness record shows, and
uncorrelated with op. For the hidden class both pipes drop at 1.8x the rate and
W3 never sees it; the env view is `full`, so W3 is live in every stratum. This
is the `mnar` mechanism with the key moved from the op to something the auditor
cannot observe.

- `full_mid_hidden`: both pipes 0.50 (0.90 for the hidden class), rho=0.
- `full_mid_hidden_rho100`: the same with one shared outage per action.

The rule draws no extra random numbers, so every other configuration replays
exactly as before.

### Runs

All `--bootstrap-episodes --reps 100 --seed 4242`, offline:

```
PYTHONPATH=src uv run python scripts/thin_episodes.py --run confirm --pool mnar        ... --out confirm-mnar-strat.json
PYTHONPATH=src uv run python scripts/thin_episodes.py --run confirm --pool mnar_hidden ... --out confirm-mnar-hidden.json
PYTHONPATH=src uv run python scripts/thin_episodes.py --run confirm --pool heldout     ... --out confirm-heldout-strat.json
PYTHONPATH=src uv run python scripts/thin_episodes.py --run long-30 --pool exploratory ... --out long-30-strat.json
```

Tables and the held/failed column are produced by `scripts/mnar_tables.py`.

## Predictions (committed before the runs)

Committed in `267367a`, before any of the four runs started.

What I expect, and why. These are written from the design, the op counts in
the confirm episodes (update_customer 227, write_file 21, refunds 193, cancels
98, emails 289; W3 misses 30 update_customer actions and nothing else), and
small synthetic tables; no run of the four above existed.

**Reasoning.**

- *Under `mnar` at rho=0 the pooled model is not as wrong as D32 reads.* W3 is
  dead on exactly the two biased ops and complete on the others, so its
  capture status separates the strata, and the pooled model with W1:W3 and
  W2:W3 terms is correctly specified: it is Lincoln-Petersen on the W3-unseen
  slice, where both biased ops share one drop rate. `ll_bic`'s ~2x N width
  (`artifacts/thinning/confirm-mnar.json`) is the information in the ~2.3
  doubly-seen update_customer actions that 227 x 0.1 x 0.1 implies. Stratifying
  cannot add information; it can only stop borrowing it.
- *A flat prior is not innocuous on thin strata.* With m doubly seen, the
  two-witness posterior tail falls like N^-m. On a 20/20/m table the 97.5%
  point is past the cap for m <= 3, where the profile-likelihood interval is
  still finite. `strat_bic` will report unbounded where `ll_bic` does not.
- *`write_file` is a small stratum* (21 actions in confirm, 16 in long-30).
  Wherever its capture is low it has almost no doubly-seen action and makes
  the whole total unbounded.
- *Two-witness strata take W1 independent of W2 as given, untested.* Under a
  coupled outage with unequal drop rates (`fin_sampled_rho100`: gateway 0.9,
  trace 0.3) the gateway's survivors are a subset of the trace's, so the
  gateway-only cell is empty and Lincoln-Petersen reads "nothing missed".
  Trace-only actions exist, so the discordance check passes. The pooled model
  can estimate W1:W2 in the financial strata, where W3 is live, and the
  no-three-way assumption that carries it across strata holds there.
- *Under `mnar_hidden` W3 is blind to the hidden class*, so inside an op the
  W3-unseen slice holds the hidden class plus the visible class's own W3
  misses (update_customer only: 15 visible-class actions in confirm). Per op
  the hidden class is small (7 to 89 actions, 0.07 to 0.9 doubly seen
  expected), so BIC prefers models without the W3 interactions and the fit
  carries the visible class's 0.5 capture over to the hidden class. Pooled
  data is enough to select those interactions, but the update_customer
  contamination adds high-capture actions to the slice and biases it low.
- *Under `mnar_hidden` with rho=1* the two pipes agree on every action, and W3
  supplies the discordance, so every stratum passes the identifiability
  check. The W1:W2 model extrapolates the hidden-to-seen ratio from the
  visible class (1:1) to the hidden one (9:1).

**Predictions** (signature linkage, the headline linkage of D28; each is
checked mechanically by `scripts/mnar_tables.py checks`):

- **P0.** Every estimator that existed before this change reproduces its
  numbers exactly in the new artifacts: `confirm-mnar-strat` against
  `confirm-mnar`, `confirm-heldout-strat` against `confirm`, `long-30-strat`
  against `long-30.bootstrap`.
- **P1.** `fin_mid_mnar`: `strat_bic` coverage >= 0.93.
- **P2.** `fin_mid_mnar`: `strat_bic` unbounded in >= 0.80 of replicates.
- **P3.** `fin_mid_mnar`: `strat_bic` detects the gap in 0.45 to 0.85 of
  replicates (whenever update_customer has two or more doubly-seen actions).
- **P4.** `fin_mid_mnar_rho100`: `strat_bic` refuses: unbounded >= 0.99,
  coverage >= 0.99, detects gap <= 0.05. The hidden actions all sit in strata
  with no discordance; refusing is the correct output where `ll_bic` has no
  coverage at all (`confirm-mnar.json`).
- **P5.** `full_mid_hidden`: `strat_bic` does not restore quantification:
  coverage <= 0.50, and where finite, bias <= -100.
- **P6.** `full_mid_hidden`: `ll_bic` coverage <= 0.85, bias between -160 and
  -20.
- **P7.** `full_mid_hidden_rho100`: both confidently wrong. `strat_bic`
  coverage <= 0.10, unbounded <= 0.10, bias <= -150; `ll_bic` coverage <= 0.10,
  bias <= -150.
- **P8.** Mean hidden mass matches the design arithmetic from the ledger,
  233.0 at rho=0 and 262.2 at rho=1, within 10%.
- **P9.** `fin_mid`: `strat_bic` coverage >= 0.93, mean width >= 1.5x
  `ll_bic`'s, unbounded in 0.05 to 0.50 of replicates (write_file).
- **P10.** `fin_mid_rho100`: `strat_bic` refuses (unbounded >= 0.99, detects
  gap <= 0.05) while `ll_bic` keeps coverage >= 0.90. This is the price of not
  pooling: here the cross-stratum assumption is true and pooling uses it.
- **P11.** `light`, `sampled`, `fin_light`: `strat_bic` coverage >= 0.85.
  (`light` is near-degenerate, almost nothing hidden, and `ll_bic` already
  undercovers there in `long-30.bootstrap.json`.)
- **P12.** `fin_sampled`: `strat_bic` coverage >= 0.93, unbounded >= 0.50.
- **P13.** `fin_sampled_rho100`: `strat_bic` coverage <= 0.80 and bias <= 0
  where finite, while `ll_bic` keeps >= 0.90.
- **P14.** `fin_sampled_rho50`: `strat_bic` bias <= 0 where finite.

If P1-P4 hold and P5-P7 hold, the claim is: stratifying on op restores a valid
answer when the loss is MAR given op (wide, often unbounded, never confidently
wrong in that pool), and cannot when the loss is keyed on something no record
carries. If P13 holds as well, stratifying trades one untestable assumption
(homogeneity across strata) for another (independence inside two-witness
strata), and neither estimator dominates.

## Results

The four runs above, after the predictions commit. Every figure in this
section is `scripts/mnar_tables.py` output pasted verbatim, or names its
artifact. Signature linkage throughout, as the predictions were stated;
`tables oracle` prints the oracle version, and the oracle prediction checks
are at the end of the next section.

**P0, reproduction.** Every estimator that existed before this change
reproduces its earlier artifact exactly, so the hidden rule and the new
estimator changed nothing else:

| new artifact | reference | estimators compared | max abs difference |
|---|---|---|---|
| `confirm-mnar-strat` | `confirm-mnar` | 11 | 0.0 |
| `confirm-heldout-strat` | `confirm` | 11 | 0.0 |
| `long-30-strat` | `long-30.bootstrap` | 11 | 0.0 |

Source: `artifacts/thinning/confirm-mnar-strat.json` (pool `mnar`, run `confirm`), 21 episodes, N=828, 100 replicates, episodes resampled: True, seed 4242, signature linkage.

| config | hidden | estimator | coverage | detects gap | unbounded | width | bias |
|---|---|---|---|---|---|---|---|
| `fin_mid_mnar` | 200.21 | `union_count` | 0.000 | 0.00 | 0.00 | 0.0 | -200.21 |
|  |  | `ll_bic` | 0.940 | 0.94 | 0.13 | 1978.4 | +152.45 |
|  |  | `lc_gated` | 0.910 | 0.22 | 0.78 | 262.9 | -134.36 |
|  |  | `strat_lc_gated` | 1.000 | 0.92 | 0.89 | 461.6 | -134.97 |
|  |  | `strat_bic` | 1.000 | 0.66 | 1.00 | — | — |
| `fin_mid_mnar_rho100` | 221.2 | `union_count` | 0.000 | 0.00 | 0.00 | 0.0 | -221.20 |
|  |  | `ll_bic` | 0.000 | 1.00 | 0.00 | 27.6 | -196.23 |
|  |  | `lc_gated` | 0.610 | 0.99 | 0.00 | 337.1 | -164.65 |
|  |  | `strat_lc_gated` | 1.000 | 0.01 | 1.00 | — | — |
|  |  | `strat_bic` | 1.000 | 0.00 | 1.00 | — | — |

Source: `artifacts/thinning/confirm-mnar-hidden.json` (pool `mnar_hidden`, run `confirm`), 21 episodes, N=828, 100 replicates, episodes resampled: True, seed 4242, signature linkage.

| config | hidden | estimator | coverage | detects gap | unbounded | width | bias |
|---|---|---|---|---|---|---|---|
| `full_mid_hidden` | 239.38 | `union_count` | 0.000 | 0.00 | 0.00 | 0.0 | -239.38 |
|  |  | `ll_bic` | 0.600 | 0.99 | 0.02 | 455.1 | -101.41 |
|  |  | `lc_gated` | 0.560 | 0.63 | 0.37 | 209.4 | -189.01 |
|  |  | `strat_lc_gated` | 1.000 | 0.81 | 0.98 | 291.1 | -194.75 |
|  |  | `strat_bic` | 0.510 | 0.99 | 0.52 | 46.8 | -211.53 |
| `full_mid_hidden_rho100` | 265.83 | `union_count` | 0.000 | 0.00 | 0.00 | 0.0 | -265.83 |
|  |  | `ll_bic` | 0.010 | 0.99 | 0.01 | 31.4 | -234.55 |
|  |  | `lc_gated` | 0.430 | 0.99 | 0.00 | 279.0 | -217.39 |
|  |  | `strat_lc_gated` | 1.000 | 0.01 | 0.98 | 1280.2 | -198.50 |
|  |  | `strat_bic` | 0.000 | 1.00 | 0.00 | 39.0 | -230.00 |

Source: `artifacts/thinning/confirm-heldout-strat.json` (pool `heldout`, run `confirm`), 21 episodes, N=828, 100 replicates, episodes resampled: True, seed 4242, signature linkage.

| config | hidden | estimator | coverage | detects gap | unbounded | width | bias |
|---|---|---|---|---|---|---|---|
| `fin_mid` | 63.15 | `union_count` | 0.000 | 0.00 | 0.00 | 0.0 | -63.15 |
|  |  | `ll_bic` | 0.940 | 1.00 | 0.00 | 42.0 | +3.39 |
|  |  | `lc_gated` | 0.990 | 1.00 | 0.00 | 138.9 | +18.31 |
|  |  | `strat_lc_gated` | 1.000 | 1.00 | 0.00 | 361.5 | +0.35 |
|  |  | `strat_bic` | 0.980 | 1.00 | 0.09 | 96.6 | +12.46 |
| `fin_mid_rho100` | 122.79 | `union_count` | 0.000 | 0.00 | 0.00 | 0.0 | -122.79 |
|  |  | `ll_bic` | 0.930 | 1.00 | 0.00 | 73.0 | +1.53 |
|  |  | `lc_gated` | 0.990 | 0.91 | 0.09 | 357.8 | -65.62 |
|  |  | `strat_lc_gated` | 1.000 | 0.01 | 1.00 | — | — |
|  |  | `strat_bic` | 1.000 | 0.00 | 1.00 | — | — |

Source: `artifacts/thinning/long-30-strat.json` (pool `exploratory`, run `long-30`), 15 episodes, N=589, 100 replicates, episodes resampled: True, seed 4242, signature linkage.

| config | hidden | estimator | coverage | detects gap | unbounded | width | bias |
|---|---|---|---|---|---|---|---|
| `light` | 0.14 | `union_count` | 0.860 | 0.00 | 0.00 | 0.0 | -0.14 |
|  |  | `ll_bic` | 0.870 | 0.00 | 0.00 | 0.2 | -0.02 |
|  |  | `lc_gated` | 1.000 | 0.00 | 0.00 | 6.5 | +0.62 |
|  |  | `strat_lc_gated` | 1.000 | 0.00 | 0.09 | 26.4 | +1.07 |
|  |  | `strat_bic` | 1.000 | 0.00 | 0.03 | 1.2 | -0.14 |
| `sampled` | 5.68 | `union_count` | 0.000 | 0.00 | 0.00 | 0.0 | -5.68 |
|  |  | `ll_bic` | 0.940 | 0.68 | 0.00 | 9.2 | -0.60 |
|  |  | `lc_gated` | 0.990 | 0.27 | 0.50 | 27.2 | +0.42 |
|  |  | `strat_lc_gated` | 1.000 | 0.07 | 0.31 | 56.3 | +0.84 |
|  |  | `strat_bic` | 1.000 | 0.80 | 0.01 | 12.1 | +0.04 |
| `fin_light` | 0.8 | `union_count` | 0.450 | 0.00 | 0.00 | 0.0 | -0.80 |
|  |  | `ll_bic` | 0.970 | 0.00 | 0.00 | 2.7 | +0.23 |
|  |  | `lc_gated` | 0.980 | 0.02 | 0.05 | 11.9 | +1.49 |
|  |  | `strat_lc_gated` | 1.000 | 0.00 | 0.15 | 70.4 | +2.62 |
|  |  | `strat_bic` | 0.990 | 0.00 | 0.10 | 4.0 | +0.11 |
| `fin_sampled` | 49.08 | `union_count` | 0.000 | 0.00 | 0.00 | 0.0 | -49.08 |
|  |  | `ll_bic` | 0.930 | 0.98 | 0.00 | 36.3 | -1.54 |
|  |  | `lc_gated` | 1.000 | 0.79 | 0.21 | 90.2 | +1.36 |
|  |  | `strat_lc_gated` | 1.000 | 0.88 | 0.44 | 412.9 | -0.02 |
|  |  | `strat_bic` | 0.880 | 0.97 | 0.75 | 217.6 | +26.11 |
| `fin_sampled_rho50` | 52.79 | `union_count` | 0.000 | 0.00 | 0.00 | 0.0 | -52.79 |
|  |  | `ll_bic` | 0.870 | 0.98 | 0.00 | 111.1 | +3.01 |
|  |  | `lc_gated` | 1.000 | 0.56 | 0.44 | 122.2 | -0.69 |
|  |  | `strat_lc_gated` | 1.000 | 0.52 | 0.61 | 249.6 | -26.01 |
|  |  | `strat_bic` | 0.950 | 0.86 | 0.68 | 150.6 | -24.06 |
| `fin_sampled_rho100` | 54.21 | `union_count` | 0.000 | 0.00 | 0.00 | 0.0 | -54.21 |
|  |  | `ll_bic` | 0.970 | 1.00 | 0.00 | 40.8 | +0.55 |
|  |  | `lc_gated` | 1.000 | 1.00 | 0.00 | 142.8 | -3.94 |
|  |  | `strat_lc_gated` | 1.000 | 0.00 | 1.00 | — | — |
|  |  | `strat_bic` | 0.810 | 0.36 | 0.50 | 115.4 | -48.94 |

**Post-hoc mechanism checks** (not predicted; `scripts/mnar_diagnostics.py`):

Source: `artifacts/thinning/mnar-diagnostics.json`, BIC-best model per stratum, 20 replicates of the confirm episodes, seed 99, signature linkage.

| config | stratum | BIC-best model (count) |
|---|---|---|
| `full_mid_hidden` | `cancel_order` | independence (12), W1:W3 (4), W2:W3 (3), W1:W3+W2:W3 (unbounded) (1) |
| `full_mid_hidden` | `issue_refund` | independence (16), W1:W3+W2:W3 (unbounded) (2), W1:W3 (2) |
| `full_mid_hidden` | `send_email` | independence (12), W1:W2+W1:W3+W2:W3 (unbounded) (3), W1:W2+W1:W3 (2), W1:W2+W2:W3 (1), W1:W3 (1), W1:W2 (1) |
| `full_mid_hidden` | `update_customer` | independence (18), W1:W3+W2:W3 (1), W1:W3 (1) |
| `full_mid_hidden` | `write_file` | independence (16), W1:W3 (3), W2:W3 (1) |
| `fin_mid_mnar` | `cancel_order` | independence (20) |
| `fin_mid_mnar` | `issue_refund` | independence (20) |
| `fin_mid_mnar` | `send_email` | W1:W2 (20) |
| `fin_mid_mnar` | `update_customer` | independence (13), independence (unbounded) (7) |
| `fin_mid_mnar` | `write_file` | independence (unbounded) (16), unidentified (3), independence (1) |

Source: same file, `strat_bic` misses by side, 40 replicates of the long-30 episodes resampled, seed 7.

| config | covers | lo above N | hi below N | unbounded |
|---|---|---|---|---|
| `fin_sampled` | 35 | 5 | 0 | 28 |
| `fin_sampled_rho100` | 31 | 0 | 9 | 21 |

## Predictions against outcomes

Signature linkage. Held/failed is computed here, not judged by hand.

| id | prediction | observed | outcome |
|---|---|---|---|
| P1 | fin_mid_mnar: strat_bic covers | coverage 1.000 | held |
| P2 | fin_mid_mnar: strat_bic mostly unbounded | unbounded_rate 1.000 | held |
| P3 | fin_mid_mnar: strat_bic detects gap in [0.45, 0.85] | detects_gap 0.660 | held |
| P4 | fin_mid_mnar_rho100: strat_bic refuses | unbounded_rate 1.000; coverage 1.000; detects_gap 0.000 | held |
| P5 | full_mid_hidden: strat_bic fails | coverage 0.510; bias -211.53 | **failed** |
| P6 | full_mid_hidden: ll_bic undercovers, biased low | coverage 0.600; bias -101.410 | held |
| P7 | full_mid_hidden_rho100: both confidently wrong | coverage 0.000; unbounded_rate 0.000; bias -230.00; coverage 0.010; bias -234.55 | held |
| P8 | hidden mass matches the design arithmetic (233.0, 262.2) within 10% | hidden 239.38; hidden 265.83 | held |
| P9 | fin_mid: strat_bic covers, wider than ll_bic, sometimes unbounded | coverage 0.980; width 96.6 vs 42.0; unbounded_rate 0.090 | held |
| P10 | fin_mid_rho100: strat_bic refuses, ll_bic holds | unbounded_rate 1.000; detects_gap 0.000; coverage 0.930 | held |
| P11 | light, sampled, fin_light: strat_bic covers >= 0.85 | coverage 1.000; coverage 1.000; coverage 0.990 | held |
| P12 | fin_sampled: strat_bic covers, mostly unbounded | coverage 0.880; unbounded_rate 0.750 | **failed** |
| P13 | fin_sampled_rho100: strat_bic undercovers low, ll_bic holds | coverage 0.810; bias -48.94; coverage 0.970 | **failed** |
| P14 | fin_sampled_rho50: strat_bic biased low where finite | bias -24.06 | held |

Twelve of fifteen held by the letter (P0 to P14). The three that failed:

- **P5 failed by its coverage bound and held in substance.** `strat_bic`
  covers no more often than it is unbounded, which is what finite intervals
  that almost never cover would produce. Its finite intervals are narrow and
  far too low (the width and bias in the `full_mid_hidden` row). The diagnostic table shows why: in
  every op stratum BIC most often picks the independence model, which lends
  the visible class's 0.5 capture to the hidden class. The only way out is
  the W1:W3+W2:W3 model, and that one is unbounded. The miss on the bound is
  inside one Monte Carlo SE (`coverage_mcse` 0.05 in
  `confirm-mnar-hidden.json`).
- **P12 failed, and for a reason I did not predict.** At `fin_sampled` (rho=0,
  where the two-witness strata are correctly specified) `strat_bic`
  undercovers and its bias is positive. The diagnostic shows every miss is
  the lower end above N, never the upper end below it. The flat prior on a
  heavy-tailed stratum likelihood drags the whole posterior up, its 2.5%
  point included. So the flat-prior central interval does not just widen on
  thin strata, it is miscalibrated there. That is a fault in the method as
  specified, not in the data.
- **P13 failed by its coverage bound and held in substance.** The point is
  low, `ll_bic` holds, and the diagnostic shows every miss is the upper end
  below N: the two-witness strata read a coupled outage as nothing missed.
  The signature-linkage coverage sits 0.01 above the bound. The
  oracle-linkage coverage in `long-30-strat.json`, 0.77, is inside it.

Under oracle linkage (not the stated linkage, shown for robustness), twelve
of fifteen also hold, but not the same twelve. P13 holds, and P9 fails
alongside P5 and P12. `strat_bic`'s bias at `fin_mid` is positive under both
linkages (`confirm-heldout-strat.json`). That fits the same lower-end push as
P12, though I did not check which side the P9 misses fall on.

Oracle linkage. Held/failed is computed here, not judged by hand.

| id | prediction | observed | outcome |
|---|---|---|---|
| P1 | fin_mid_mnar: strat_bic covers | coverage 0.990 | held |
| P2 | fin_mid_mnar: strat_bic mostly unbounded | unbounded_rate 1.000 | held |
| P3 | fin_mid_mnar: strat_bic detects gap in [0.45, 0.85] | detects_gap 0.640 | held |
| P4 | fin_mid_mnar_rho100: strat_bic refuses | unbounded_rate 1.000; coverage 1.000; detects_gap 0.000 | held |
| P5 | full_mid_hidden: strat_bic fails | coverage 0.620; bias -193.21 | **failed** |
| P6 | full_mid_hidden: ll_bic undercovers, biased low | coverage 0.760; bias -81.430 | held |
| P7 | full_mid_hidden_rho100: both confidently wrong | coverage 0.000; unbounded_rate 0.000; bias -221.90; coverage 0.000; bias -226.23 | held |
| P8 | hidden mass matches the design arithmetic (233.0, 262.2) within 10% | hidden 232.88; hidden 262.12 | held |
| P9 | fin_mid: strat_bic covers, wider than ll_bic, sometimes unbounded | coverage 0.920; width 91.7 vs 39.2; unbounded_rate 0.110 | **failed** |
| P10 | fin_mid_rho100: strat_bic refuses, ll_bic holds | unbounded_rate 1.000; detects_gap 0.000; coverage 0.950 | held |
| P11 | light, sampled, fin_light: strat_bic covers >= 0.85 | coverage 1.000; coverage 0.930; coverage 0.990 | held |
| P12 | fin_sampled: strat_bic covers, mostly unbounded | coverage 0.840; unbounded_rate 0.710 | **failed** |
| P13 | fin_sampled_rho100: strat_bic undercovers low, ll_bic holds | coverage 0.770; bias -49.42; coverage 0.960 | held |
| P14 | fin_sampled_rho50: strat_bic biased low where finite | bias -21.10 | held |

## The claim, made precise

1. **Loss MAR given an observed covariate (`fin_mid_mnar`).** Stratifying on
   that covariate makes the answer *valid*. `strat_bic` covers, and at rho=1
   it refuses where `ll_bic` is confidently wrong: the unidentified strata
   are exactly where the hidden actions are. It does not make the answer
   *finite*: `strat_bic` is unbounded in every replicate. The strata the loss
   falls on hold two witnesses at about 0.1 capture, and no stratified
   estimate gets a finite upper end from that. Quantification is restored
   only as far as the worst stratum carries information, and here it carries
   none. Gap detection survives at a lower rate than under `ll_bic`. Where
   the strata do carry information it quantifies: at `fin_mid`
   (`confirm-heldout-strat.json`) it covers with a finite interval in most
   replicates, about twice as wide as `ll_bic`'s.
2. **Loss keyed on something no record carries (`mnar_hidden`).**
   Stratification does not help. At rho=0 `strat_bic` covers only about as
   often as it is unbounded, and its finite intervals sit far below N. At
   rho=1 it covers 0.000. W3 supplies
   the discordance, so every stratum passes the identifiability check while
   the no-three-way assumption fails inside it. The pooled `ll_bic` is also
   wrong, less so at rho=0 because pooling gives BIC enough data to choose
   the W3 interactions. The check this repo has (two live witnesses plus
   discordance) is necessary for identification, not sufficient.
3. **Neither estimator dominates.** Pooling relies on a cross-stratum
   assumption (no three-way interaction). Stratifying relies on independence
   inside two-witness strata. Each fails where the other holds:
   - `fin_mid_rho100` and `fin_sampled_rho100`: the cross-stratum assumption
     is true. `ll_bic` covers. `strat_bic` refuses, or reads the coupled
     outage as nothing missed.
   - `fin_mid_mnar_rho100`: the cross-stratum assumption is false.
     `strat_bic` refuses correctly and `ll_bic` covers 0.000.

   Nothing in the observed table separates the two cases.

What an auditor can take from this. When `strat_bic` refuses and `ll_bic`
reports a finite interval, the finite answer rests entirely on an untestable
cross-stratum assumption. That disagreement is worth reporting. Agreement is
not reassurance: under `full_mid_hidden_rho100` both are finite, both agree,
and both are wrong.

## Weaknesses

- **The flat prior is miscalibrated on thin strata.** It pushes the lower end
  above N (P12) and opens the upper end where the profile-likelihood
  interval is finite. A 1/N prior, or calibrating per-stratum intervals by
  simulation before convolving, might fix this. Neither has been tried.
- **One small stratum decides the total.** `write_file` (21 actions in
  confirm, 16 in long-30) makes `strat_bic` unbounded whenever its own
  capture is low. Merging strata that share a witness-visibility pattern
  would recover the pooled answer in `fin_mid_mnar`, but it reintroduces a
  homogeneity assumption.
- **Two-witness strata take independence on faith** (P13). Within such a
  stratum it cannot be tested.
- **BIC is weak in thin strata.** It prefers a simpler, misspecified model
  that gives a finite, confident answer (diagnostic table).
- **Reviewer findings, not fixed** (a fresh-context review of
  `stratified.py`; it found no correctness bugs):
  - The tail exponent comes from the secant slope at the cap, which
    approaches the asymptotic exponent from below. Strata whose posterior is
    proper can be declared improper. The effect is conservative on the
    upper end, but the stratum's lower end then falls back to its observed
    count, which can cost gap detection.
  - A single discordant action passes the identifiability gate. A table
    where one action is W1-only and twenty are seen by both gives a
    one-point interval. `ll_bic` does the same.
  - The support can run slightly past the cap carrying negligible mass.

  Fixing any of these changes `strat_bic`'s numbers, so a fix needs a re-run
  declared as such, and the results above were already seen.
- **The hidden key is uncorrelated with op by construction.** A real hidden
  driver (load, time of day) may be partly correlated with op. That middle
  ground is untested. The hidden class is also exactly the class W3 misses,
  so W3's capture status reveals it except where W3 has misses of its own
  (update_customer). That is why the pooled model is only half wrong at
  rho=0.
- **Scope.** Everything here is exploratory. One set of 21 confirm episodes
  serves the `mnar`, `mnar_hidden` and `heldout` runs. At 100 replicates the
  coverage MCSE is up to 0.05, so the P5 and P13 bound misses are within
  noise.
