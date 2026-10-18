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

_Pending: generated by `scripts/mnar_tables.py` after the runs._
