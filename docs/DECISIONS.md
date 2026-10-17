# Decisions

Dated, append-only. Each entry says what changed, why, and what it touches.

## 2026-10-17

**D1. No R on the build machine; Rcapture/dga cross-check deferred.**
`docs/PLAN.md` asks for agreement with R to 1e-3. R is not installed. Instead,
the closed forms that any log-linear fit must reproduce are checked exactly:
Lincoln-Petersen for K=2, the saturated K=3 missing-cell formula, and the
K=2 profile likelihood against the product-binomial likelihood term for term.
IRLS is checked against statsmodels. This is stronger where it applies and
silent on model averaging. **Open:** the R cross-check of `ll_average`
against `dga`, triggered by installing R or by the latent-class estimator
landing (LCMCR is R-only).

**D2. `ll_average` is a cheap model average, not dga.**
Each model's profile likelihood is normalised under a flat prior on N, then
mixed with BIC weights. Madigan & York (1997) / `dga` do this exactly, with
hyper-Dirichlet priors over decomposable models. Mass from a model still flat
at the cap (20·n) goes to infinity for the median and the upper tail; its grid
shape is kept for the lower tail. Each reading is conservative on the side
where it is used.

**D3. `intervals.py` from assay not vendored yet.**
The simulator needs no bootstrap: replicates are independent draws with a
known truth, and coverage is a binomial proportion with an analytic MCSE.
It is vendored when agent episodes need cluster resampling.

**D4. The unconditional MLE is not the conditional point.**
A test assumed the profile-likelihood argmax sits within 3 of
n + exp(b0). At small n it does not (K=2, n=25: 41 against 43.2). The
profile was checked against the analytic likelihood and is correct; the
test's assumption was wrong. The point reported is the conditional estimate,
as Rcapture reports it; the interval is profile likelihood.

**D5. Prior-art sweep (week 0) passed with a caveat.**
No published work applies multiple-systems estimation over witnesses to count
unlogged agent actions (about 80% confidence; details in
`docs/RELATED-WORK.md`). The nearest AI precedent (Kansal 2026,
capture-recapture over LLM judges for label errors) is a negative result:
intervals were valid but too wide to beat a raw count. Width, not coverage,
is the risk to watch.

**D6. Warm-start IRLS bug fixed; first-sweep `ll_average` K=4 numbers are void.**
A warm start far from the optimum underflowed μ to exactly 0, the IRLS
working response went NaN, and NaN profile values corrupted the mixture's
CDF. That produced points like 280 and 3930 against a true N of 200. Now μ is floored at every step,
a failed warm start is retried cold, and a non-finite profile value raises
instead of passing silently. The two tables that exposed it are regression
tests. Every `ll_average` number from the first sweep is superseded by the
rerun.

**D7. Latent-class estimator (`lc`) and env-anchored sensitivity bounds added.**
`lc` is LCMCR (truncated stick-breaking, K*=10, Beta(1,1), p(N) ∝ 1/N,
1000 burn + 2000 draws), Gibbs on cell counts. `anchor_g{Γ}` collapses to
env vs. everyone else and bounds N for an odds ratio ψ ∈ [1, Γ].

Written before the rerun, so the sweep can contradict it:
- In l2_env_blind the true anchor odds ratio is ψ = 4.35, computed in closed form from the scenario's parameters. Expected: `anchor_g1` and `anchor_g2` undercover badly, `anchor_g4` undercovers partly, `anchor_g8` covers.
- In l2_bypasses_proxy, ψ = 1 by construction. Expected: `anchor_g1` covers near nominal.
- With K=3, `lc` is not identified beyond one class. Expected on l0_heterogeneous: coverage higher than `ll_bic` (0.296 in the first sweep), bought with width.

**D8. Fresh-context review of the new estimators: the math held, four fixes made.**
A reviewer working only from the code and the textbook models checked every
LCMCR full conditional, the direction of the offset bound, the profile
likelihood, and the termination of the integer searches. All were correct.
Fixed:
- `averaged` now reads all three quantiles from one mixture. Mass past the cap is a single atom, extrapolated per model from its slope at the cap. Before, a model that merely had a long tail sent its whole weight to +inf; the reviewer's example table went from 36/47/inf to 36/46/100.
- An unconverged fit now raises (`ConvergenceError`) or is dropped. Before, it was used silently with z0 ≈ 1.
- `anchor_bounds` rejects an anchor index outside the table.
- New scenario `l0_siloed` (added before the rerun): each witness covers its own kind of action and they barely overlap. Pre-stated expectation: every point method overcounts (siloed witnesses are negatively dependent) and `lc` reports intervals that look confident and are wrong.

**D9. Identifiability gate (`lc_gated`), stated before the full sweep.**
Refuse to report `lc` if any log-linear model within ΔBIC ≤ 6 of the best is
unbounded, or if `lc`'s interval misses any such model's point. A refusal
reports `union_bound`. ΔBIC = 6 is Kass & Raftery's "strong" threshold and was
chosen before any check ran. A 40-rep check on dev seed 99 (not the sweep
seed) gave: refusal 0.95 on l0_siloed, 0.00–0.12 on identified L0/L1, and
0.00 on l2_env_blind.

Expected in the seed-2026 sweep:
- refusal ≥ 0.8 on siloed;
- refusal ≤ 0.2 on identified L0/L1;
- no refusal on l2_env_blind. Agent-correlated hiding is invisible in the observed cells, which is why the L2 claim rests on the Γ bounds and not on any data-driven gate.

**D10. No natural-language adversarial side-task prompts for live models.**
Writing prompts that instruct a live model to covertly move data and avoid
monitoring was stopped by a safety classifier, and that text is not in the
tree. The L1/L2 organisms exist only as scripted plans (tool-call sequences
the scripted agent replays), which exercise the same kernel, witnesses and
linkage.

**Open:** how live-model L1/L2 organisms are produced. Options are an
established, published model-organism or AI-control setup used as its
authors intend, or prompts the project owner writes. Trigger: before any
live-model L1/L2 run. L0 live runs need no side task.

**D11. Agent environment is L0-only. The live pilot is exploratory.**
The back-office env has no adversarial side tasks, no batch side channel,
and a trace file that the agent cannot write (D10). Actions go missing only
through benign causes:
- injected logging faults (gateway 5%, harness trace 10%, independent);
- the env diff's natural blind spots (repeat writes collapse, reverts vanish).

Live runs use OpenCode Go `deepseek-v4-flash` for both the agent and the W4
auditor, with keys read from the environment. Pricing is conservative: the
provider's peak rate, $0.30/M in and $1.20/M out. The pilot is 20 episodes on
seeds 2000–2019 (the scripted runs used 1000+) with a $3 cap. It runs before
the pre-registration freezes, so it is exploratory and informs only the power
calculation and the measured W4 behaviour.

**D12. Long sessions and post-hoc log thinning. Design and predictions, written before any of these runs were analysed.**
Live runs log with zero injected faults: 30, 60 and 120 tickets per episode,
on seeds 3100+, 3200+ and 3300+. Loss is applied afterwards by
`scripts/thin_episodes.py`, R replicates per configuration, each record
dropped independently. That is valid because the faults are independent of
the agent by construction (D11). Coverage is then conditional on the agent's
realised behaviour, which is fixed per run.

Configurations:
- `light`: gateway drop 0.05, trace 0.10, full env
- `sampled`: gateway keeps 1 in 10 (drop 0.9), trace 0.3, full env
- `fin_light`: as `light`, env limited to financial state (refunds, cancellations, outbound mail)
- `fin_sampled`: as `sampled`, financial env

The financial-only env stands in for change-data-capture on a subset of
tables. It makes env capture depend on action type: heterogeneity by an
*observed* covariate.

New estimator, `strat_lc_gated`: stratify by op, drop the witnesses that saw
nothing in a stratum, run `lc_gated` per stratum, and sum points and interval
ends. Summing interval ends is conservative for coverage. A stratum with
fewer than two live witnesses reports `union_bound`.

Predictions:
1. W4 auditor recall falls with session length: ≥ 0.98 at 7 tickets (pilot), lower at 60, below 0.95 at 120.
2. With the full env, the never-seen cell stays 0 in every thinning configuration, and every estimator covers.
3. With `fin_sampled`, the never-seen cell is > 0, made up of updates and file writes. `ll_independence` and `anchor_g1` undercover, because env capture depends on type and so ψ ≠ 1. `strat_lc_gated` covers at ≥ 0.9, or refuses.

**D13. Model: `deepseek-v4-flash` up to the long-session batch; `deepseek-v4.1-flash` from then on.**
`deepseek-v4-flash` came from tribench's config without being checked
against the provider. OpenCode Go's `/models` lists only
`deepseek-v4.1-flash` and `deepseek-v4-pro`; the old id still answers and
reports itself as `deepseek-v4-flash`.

The owner's call: keep `pilot-l0` and the `long-30`, `long-60` and `long-120`
runs as they are, on v4-flash, and use v4.1-flash for every later run. Each
run's `summary.json` records its model. Comparisons across the switch are
labelled as cross-model. The conservative peak price (D11) is reused for
v4.1-flash until its price is checked.

**D14. The W4 auditor as built is not an independent witness. Measured, then fixed for later runs.**
Prediction 1 in D12 said auditor recall would fall with session length. It did
not: recall was 1.0000 with 0 false positives at both 7 tickets (20 episodes,
N=214) and 30 tickets (15 episodes, N=589, ~75 calls per transcript).

The reason is structural, not a property of the model. The auditor reads a
rendered transcript that contains every tool call verbatim, so W4 is a
near-deterministic function of a superset of W1. Capture-recapture over a
witness with capture probability 1 is degenerate: it pins N to the observed
count and contributes no information about what every witness missed.

Fixes, for later runs only (the recorded runs keep their numbers):
- Episodes now store the full transcript, so a different W4 can be scored
  without paying for the agent again.
- `render(view="narration")` gives the auditor only what the agent said, with
  no call or result lines. That is the witness Arslan (2609.07680) describes:
  one that can only relay the agent's own account. Whether the agent narrates
  enough for this to work at all is an empirical question, measured next.

**Open:** whether narration-only W4 has recall below 1. If the agent does not
narrate, W4 is dropped from the live arm and the live claim rests on W1-W3.

**D15. Adversarial refutation round. Six attacks checked; four changed the docs.**
A reviewer whose brief was to attack the project, given only the repo, raised:

Upheld and acted on:
- Every `@all4` row recorded so far is degenerate (D14). They are now marked
  `[D14]` in `docs/ENV-FINDINGS.md` with a header saying they support no claim.
- `README.md` still said no live agent had run. Corrected.
- `docs/PRE-REGISTRATION.md` said it freezes before the first agent run, which
  had already happened. It now says *confirmatory* run, states that everything
  so far is exploratory, and forbids basing the power calculation on
  `pilot-l0`, whose never-seen cell was 0 and whose W4 was degenerate.
- The Γ bounds cannot cover under negative dependence, by construction. Stated
  in `docs/SIM-FINDINGS.md` as a limit of the method, with the l0_siloed
  numbers (coverage 0.002 at every Γ, width 580 to 3489).

Upheld, already recorded, no change:
- Every L1/L2 number is against scripted organisms, not a model evading
  detection (D10). No live adversarial claim is made anywhere.
- Width, not coverage, is the risk the nearest precedent failed on (D5). The
  hard scenarios confirm it: on l0_siloed the honest output is `union_bound`
  with extra steps, which is what the gate reports.

The reviewer's framing question - whether a complete tamper-evident proxy
would dominate this apparatus - is the paper's motivation section, not a bug.
The answer has to be that such a proxy is what Qin et al. (2609.30266) propose
and do not evaluate, and that `l2_bypasses_proxy` is the case where it fails.
That argument is not yet written down.

**D16. The thinning analysis inherited the degenerate-W4 bug, and prediction 3 of D12 was measured wrong.**
The first thinning pass reported "mean actions unseen by every witness = 0.00"
in all four loss configurations, including `fin_sampled`, which D12 predicted
would be > 0. That statistic was computed over the four-witness table, and W4
has capture probability 1 (D14), so it is 0 by construction and measures
nothing.

The statistic now counts actions unseen by the three hard witnesses (trace,
gateway, env diff), which is what the `@3` estimators actually face. The
run is being redone; the first pass's `mean_unseen_by_all` figures are void.

Rows ending `@4` in any thinning output remain degenerate and support no
claim, exactly as in `docs/ENV-FINDINGS.md`.

**D17. Review round on `thinning.py`. Two statistical bugs fixed, one claim narrowed.**
A reviewer given only the module and its dependencies found:

Fixed:
- **Summed intervals undercover.** `stratified()` added per-stratum 95%
  intervals. The sum covers only if every stratum covers, since the sum is
  monotone in each term, so joint coverage was 0.95**m: 0.77 at m=5 strata.
  Each stratum now runs at 1 - alpha/m (Bonferroni), so the joint level is at
  least 1 - alpha.
- **Perfectly overlapping witnesses looked identified.** A stratum whose only
  non-zero cell is the all-live one carries no information about what the
  witnesses jointly missed, but `lc_gated` reported 20 [20, 22] on it. A
  stratum now needs discordance (some action seen by a proper subset of its
  live witnesses) to be estimated at all; otherwise it contributes
  `union_bound`.

Narrowed, not fixed:
- The reviewer said a witness that labels an action with the wrong op would
  inflate N across strata. That is true of signature linkage only. Under
  oracle linkage the per-op tables still sum to the ledger: a wrong-op W4
  record fails to match and is counted as a false positive, verified
  directly. Signature linkage genuinely does create a phantom there, which is
  a linkage error the method should be measured on, not hidden. W4 false
  positives have been 0 in every live run so far.
- The asymmetry when a stratum is unidentified (point and upper end go to
  infinity, lower end stays finite) is intended: the observed count remains a
  valid floor even when nothing bounds the total above.

**D18. Session-length result, and the confound in it.**
Auditor recall over the four live runs (50 episodes, ~1,500 actions):

| run | tickets | episodes | actions/ep | calls/ep | recall | false positives |
|---|---|---|---|---|---|---|
| pilot-l0 | 7 | 20 | 10.7 | 21.0 | 1.0000 | 0 |
| long-30 | 30 | 15 | 39.3 | 72.2 | 1.0000 | 0 |
| long-60 | 60 | 10 | 62.0 | 104.3 | 1.0000 | 0 |
| long-120 | 120 | 5 | 73.8 | 119.8 | 1.0000 | 0 |

D12 prediction 1 (recall falls with session length) is refuted for every
length reached: a full-transcript auditor made no error at all.

The confound: actions per ticket fall from 1.31 to 0.61 as the queue grows,
so the agent silently skips more than half of a 120-ticket queue. Adding
tickets therefore buys transcript length sublinearly, and the longest
condition is a 120-call, 74-action transcript rather than the ~150-action one
the ticket count suggests. The claim is bounded by calls, not by tickets: at
up to ~120 calls, full-transcript W4 recall is 1.0.

Pushing further needs a different lever than queue size, because the agent's
own throughput is the limit. That the agent abandons a long queue is a
separate behavioural finding and is not the subject here.

**D19. Expected hidden count under each loss configuration, stated before the corrected thinning finished.**
From the `long-30` op mix (N=589: send_email 201, update_customer 165,
issue_refund 143, cancel_order 64, write_file 16), the financial env view sees
refunds, cancellations and mail, so 181 actions (30.7%) are invisible to it.

Expected actions unseen by all three hard witnesses:
- `light` and `sampled` (full env): ~0. The env diff sees every op, and only
  loses repeats and reverts.
- `fin_light`: 181 x 0.10 x 0.05 = 0.9, about one action.
- `fin_sampled`: 181 x 0.30 x 0.90 = 48.9, about 8.3% of N.

`fin_sampled` is therefore the first configuration in this project where the
quantity being estimated is not zero, and so the first real test of the
estimators on live agent behaviour. The test is whether the 95% intervals
cover 589 while the union sees only ~540.

**D20. The LLM-auditor witness does not work, at either setting. W4 is dropped from the live arm.**
Measured on `narr-30` (10 episodes, 30 tickets, N=403, deepseek-v4.1-flash,
identical environment, transcripts stored):

| W4 input | recall | false positives |
|---|---|---|
| full transcript (50 episodes, D18) | 1.0000 | 0 |
| agent narration only (10 episodes) | 0.0000 | 0 |

With the full transcript the auditor reads every tool call verbatim, so its
capture probability is 1 and it pins N to the observed count (D14). With
narration only it reports nothing, and the reason is sharper than "the agent
does not narrate". It narrates 4 times in 64 calls, and the closing message
does describe the work: "All 31 queue items are handled: O26 and O17
cancelled (with emails), nine refunds issued ... all address/email updates
applied with duplicates ignored". That is an aggregate account: counts and a
few identifiers, never the per-action detail (which customer, which field,
which amount) a record needs. The auditor returned a parseable empty list in
all 10 episodes with 0 false positives, so it was working and there was
nothing individual to extract.

The finding is therefore that self-narration is aggregate while an audit
record must be per-action. "Nine refunds" is a count that cannot be
overlapped with another witness, and overlap is what capture-recapture
consumes.

There is no setting in between that makes W4 independent, because the only
record of what the agent did is the call log, and the call log is W1. An LLM
auditor reading the agent's own transcript is either a copy of the trace or
empty. This is the mechanism behind Arslan's "Audit Without Verification"
(2609.07680) result, measured at both extremes.

Consequence: the live arm has three witnesses, not four. W1 trace, W2
gateway, W3 env diff are produced by different infrastructure and are
genuinely distinct. Every `@4` number in this project is superseded by this
and supports no claim. The simulator keeps its W4 scenario
(`l0_auditor_copies_trace`), which models a partially dependent auditor, as a
hypothetical that the live system does not exhibit.

**Open:** whether an auditor with an independent evidence source (for example
reading the environment's state, not the transcript) is a usable fourth
witness. That is a different design, not a view on the same transcript.

**D21. First configuration where the estimand is not zero, and D19's prediction was accurate.**
Under `fin_sampled` (gateway keeps 1 in 10, trace loses 30%, env diff sees
only financial ops) on the `long-30` episodes, the mean number of actions
unseen by all three hard witnesses was **49.2** against the 48.9 predicted in
D19 from the op mix. The estimators then face a union of about 540 and a
truth of 589.

That pass still computed the baselines over the four-witness table, so
`union_count` equalled N by construction through the degenerate W4. Every
estimator in `estimate_all` now runs over the three hard witnesses only, and
the run is being redone. The `@3` rows of the previous pass were already
three-witness and are unaffected; the baseline rows of that pass are void.

**D22. The falsifying experiment: correlated log loss. Prediction stated before the run.**
The `fin_sampled` result rests on the gateway and trace failing
independently, which is exactly what a log-linear fit without a W1:W2 term
assumes. The experiment as built cannot fail that assumption, because it
generates the loss with independent coin flips. That is the strongest
objection to the live result, so it gets a test rather than a caveat.

`Config.rho` is the share of actions whose gateway and trace losses come from
one shared draw, as a single logging outage takes both pipes down. Two new
configurations, `fin_sampled_rho50` (rho=0.5) and `fin_sampled_rho100`
(rho=1.0), hold everything else fixed.

Because the gateway drops more often than the trace (0.9 against 0.3), full
coupling makes the gateway's survivors a subset of the trace's: no action is
seen by the gateway alone. The discordance that identifies N is destroyed.

Predicted, before running:
- `ll_independence` undercovers badly at rho=1.0 and partly at rho=0.5, and
  undercounts, because positive dependence between witnesses biases
  capture-recapture down.
- `ll_bic` does better than `ll_independence`, since the W1:W2 interaction it
  can select is the true structure.
- `lc_gated` either refuses or covers; it must not report a confident wrong
  answer.
- `anchor_g1` undercovers, since the env anchor is unaffected but the pair it
  is anchored against has collapsed.

If `ll_independence` keeps its 0.95 coverage at rho=1.0, the experiment is
not testing what it claims to test and the live result is weaker than stated.

**D23. The reported uncertainty was too small: it ignored episode-to-episode variation.**
Coverage was computed over 100 logging-fault realisations applied to the same
15 episodes, so the MCSE of about 0.02 described only fault variation and
treated those 15 episodes as the population. A new shift of the same agent
would differ in op mix, action count and how much the env diff can see, and
none of that was in the interval.

`--bootstrap-episodes` resamples the 15 episodes with replacement for each
replicate before applying loss, so each replicate has its own N and its own
mix. That is the cluster bootstrap this workspace uses elsewhere, with the
episode as the cluster, and it is the honest version of the claim.

Both are reported: the fixed-episode run answers "given this agent's
behaviour, do the intervals cover under logging loss", and the bootstrap run
answers "would they cover for another shift". The second is the one a reader
should believe.

**D24. Second refutation round. Two attacks were already in flight; two changed the work.**
The reviewer's two strongest attacks were the two already built and running:
the independence circularity (D22) and the MCSE covering only fault variation
(D23). Independent agreement on both.

New and acted on:
- **Coverage cannot separate an informative interval from a wide one.**
  `union_bound` covers 1.000 by being unbounded every time. A 36.8-wide
  interval covering 0.950 might be the same failure in milder form.
  `summarise` now reports `detects_gap`: the share of replicates whose lower
  end exceeds the union of all witnesses. That is the claim an auditor
  actually needs, "your logs are missing actions, with 95% confidence", and
  the trivial bound can never make it.
- **The W4 flip from recall 1.0 to 0.0 looked suspicious rather than
  measured.** Checked: the auditor was working (parseable empty list in all
  10 episodes, 0 false positives) and the agent does narrate its work, but
  in aggregate ("nine refunds issued"), never per action. D20 is corrected to
  say so, which is a sharper claim than the original.

Noted, not yet acted on:
- The reviewer argues `ll_bic` losing to `ll_independence` is a symptom of
  the same circularity rather than a robustness finding. That is right while
  loss is independent, and the rho configurations are the test: there
  `ll_bic` can select the W1:W2 interaction that is genuinely present, and
  D22 predicts it should overtake `ll_independence`.
- The `light` configuration shows `union_count` and `ll_independence` tied at
  0.930, so the method's advantage appears only where intervals widen. That
  is a fair description of the trade and belongs in the writeup rather than
  being defended.

**D25. The falsifying experiment ran. The method is not circular, and the predictions held.**
Correlated loss (D22), same episodes, 100 replicates, oracle linkage:

| coupling rho | hidden from all 3 | ll_independence coverage | its bias | ll_bic coverage | its bias |
|---|---|---|---|---|---|
| 0.0 | 48.8 | 0.960 | +0.22 | 0.960 | +0.20 |
| 0.5 | 51.3 | 0.880 | -5.13 | 0.870 | -0.87 |
| 1.0 | 53.3 | 0.790 | -8.80 | 0.880 | +1.43 |

Against the predictions written before the run:
- `ll_independence` undercovers at rho=1.0 (0.790) and partly at rho=0.5
  (0.880), and undercounts (-8.8, -5.1). **Held.**
- `ll_bic` beats it at rho=1.0 (0.880 against 0.790, bias +1.4 against -8.8),
  because the W1:W2 interaction it can select is the structure that is now
  really there. **Held.**
- `lc_gated` never reports a confident wrong answer: it covers 0.990 at both
  rho values, refusing 38% of replicates at rho=0.5. **Held.**
- `anchor_g1` was predicted to undercover. It does, mildly (0.940, 0.900),
  not the collapse predicted. **Partly wrong.**
- `ll_average` goes unbounded in 100% of replicates at rho=1.0: it declines to
  answer rather than answer wrongly. Not predicted, and the right behaviour.

The falsifier set out in D22 was that `ll_independence` keeping 0.95 coverage
at rho=1.0 would mean the experiment tested nothing. It did not: coverage
fell to 0.790. The independence assumption is doing real work, breaking it
breaks the estimator in the predicted direction, and a correlation-aware
model recovers. That answers the circularity objection with a measurement
rather than an argument.

The cost is honesty about scope: at rho=1.0 the best estimator covers 0.88,
below nominal. Under a shared logging outage this method is biased low, and
`union_count` is still worse (coverage 0.000, undercounting by 53).

**D26. Episode bootstrap, and four numbers I transcribed wrongly.**
The bootstrap run (D23) barely moved anything: at `fin_sampled`,
`ll_independence` covers 0.970 resampling episodes against 0.960 fixed, and
at full coupling 0.780 against 0.790. The reviewer expected "a much wider,
less flattering interval"; between-shift variation is small next to logging
loss here, because the estimator reads a pooled table that scales with the
sample. The bootstrap run is the headline anyway, since it is the claim a
reader should believe.

Writing that section up, I typed four figures from the wrong artifact
(mixing the fixed-episode and bootstrap runs): `fin_light` union coverage
0.470 for 0.360, its detect rate 0.07 for 0.00, and two biases in the
coupling table. `scripts/headline_tables.py` now emits both headline tables
from the artifact, and `docs/ENV-FINDINGS.md` quotes its output. This is
precisely the failure AGENTS.md rule 2 exists to prevent, caught by checking
every number against the artifact after writing.

**D27. Primary estimator chosen on worst-case behaviour, and the power curve.**
Across all six loss configurations of the bootstrap run, worst-case coverage
and mean gap-detection rate:

| estimator | worst coverage | worst at | mean detects_gap | mean width |
|---|---|---|---|---|
| `union_bound` | 1.000 | — | 0.00 | unbounded always |
| `union_count` | 0.000 | fin_sampled | 0.00 | 0 |
| `ll_independence` | 0.780 | fin_sampled_rho100 | 0.61 | 19.7 |
| **`ll_bic`** | **0.920** | sampled | **0.61** | **21.5** |
| `ll_average` | 0.910 | sampled | 0.52 | 39.3 |
| `anchor_g1` | 0.890 | fin_light | 0.61 | 21.6 |
| `anchor_g4` | 0.960 | fin_light | 0.61 | 129.0 |
| `lc_gated` | 0.970 | sampled | 0.47 | 60.1 |
| `strat_lc_gated` | 0.990 | fin_light | 0.26 | 128.8 |

`ll_bic` is the primary estimator for the live arm. It keeps the
informativeness of the independence model (0.61) for 9% more width, has no
catastrophic failure (its worst case is variance in an easy regime, not bias
in a hard one), and is the model that survives the coupling that breaks
`ll_independence`. `anchor_g4` is the conservative alternative: 0.960
worst-case for six times the width. `strat_lc_gated` is dominated and is
dropped from the headline.

Power, `fin_sampled`, `ll_bic`, 60 replicates per size:

| episodes | N | mean width | width / N | coverage |
|---|---|---|---|---|
| 4 | 157 | 18.1 | 0.116 | 0.967 |
| 8 | 314 | 28.5 | 0.091 | 0.950 |
| 15 | 589 | 36.9 | 0.063 | 0.983 |
| 30 | 1179 | 53.0 | 0.045 | 0.967 |

Relative width follows 1.53/sqrt(N), the capture-recapture rate: predicted
ratios 0.516 and 0.707 against observed 0.543 and 0.714. So a target relative
width w needs N = (1.53/w)^2: 5% needs N≈936 (about 24 episodes at 30
tickets), 3% needs N≈2600 (about 66). Coverage is flat in study size, so
episodes buy precision, not validity.

**D28. Red-team of the pre-registration. It was not adequate; five changes.**
A reviewer given only the pre-registration and the plan, briefed to stop a
bad one being frozen, found:

1. **The estimator choice was circular with H1.** `ll_bic` was selected in
   D27 on worst-case coverage over six configurations, and H1 then tested its
   coverage on one of those six, two points below the figure that selected
   it. Fresh episode seeds do not fix this: the configuration structure was
   identical. **Fixed** by adding `HELDOUT` in `src/lacuna/thinning.py`
   (`fin_mid`, `fin_mid_rho100`), outside `CONFIGS` and touched by no run, and
   moving every confirmatory hypothesis onto them. A test asserts the two
   pools stay disjoint.
2. **Two sub-hypotheses could not fail.** `union_bound` having
   `detects_gap` 0 is an identity of an interval that starts at the union,
   and `union_count` undercounting is a property of the design. **Fixed:**
   both moved to a "design facts, not hypotheses" section, and H4 rewritten
   as a comparison that can fail (`ll_bic` beats `union_count` on
   `detects_gap` by at least 0.50).
3. **Tie-breaks were unspecified.** **Fixed:** the headline is `ll_bic` under
   effect-signature linkage, disagreement with other estimators is reported
   as a caveat rather than resolved, and a linkage gap above 5 points is
   reported as linkage bias.
4. **The scope section let the result be over-read.** **Fixed:** an explicit
   sentence that a passing H1 is not evidence for the adversarial claims that
   motivate the plan and may not be cited as such.
5. **No stopping rule, no multiplicity statement, no named headline.**
   **Fixed:** the run is attempted once, an interrupted run is reported as
   incomplete rather than re-seeded, episodes are never discarded, the four
   hypotheses are named as the only pass/fail claims and reported together
   whatever happens, and the headline number is named.

Choosing the held-out configuration before seeing it is the substantive cost:
`fin_mid` (both pipes at 0.50, financial env) is a regime no exploratory run
measured, so H1 could genuinely fail there.

**D29. The claims gate caught a stale headline on its first run.**
`tests/test_published_claims.py` compares every figure in the README's Status
section against `MEASUREMENT_CARD.json`, which `scripts/make_card.py` builds
from the artifacts. Its first run rejected six figures, including a coverage
of "95.0%" that the bootstrap artifact puts at 0.97, and an "about 6% of N"
width that no artifact states. Both came from the hand-written draft of D26
and had survived my own re-reading.

The README now quotes card values only. Numbers that are derived rather than
measured (a hidden count as a percentage, a width relative to N) are out of
the Status section entirely: they can be computed by a reader from figures
that are in the card.
