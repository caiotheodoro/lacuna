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

**D30. The first confirmatory attempt died at episode 7 on a transient 400. Re-run on the same seeds.**
The run stopped after 6 of 24 episodes with a 400 Bad Request from the
provider. Seed 7006, the episode that failed, runs fine in isolation, so the
error was transient and not a property of the episode.

Two harness faults, both fixed:
- `ChatClient` retried 429 and 5xx but not 400, and discarded the response
  body, so the run reported a status code with no reason. The body is now in
  the exception and 400 is retried.
- One failed episode aborted the whole study. A failure is now recorded in
  `summary.json` under `failed_episodes` and the run continues.

The pre-registration's stopping rule covers a run interrupted by budget or
network, which is reported as incomplete rather than re-run. This was neither:
it was a harness bug. The study is re-run on the **same seeds** (7000+), no
hypothesis having been computed from the partial data, and the partial run is
discarded rather than pooled. Re-running on different seeds, or keeping both
and choosing, would be seed-shopping; neither is done.

**D31. The loss model was MCAR; the real threat is MNAR. Test added, predictions first.**
A reviewer's strongest attack on the confirmatory result: every loss
configuration so far deletes log lines by coin flips that do not depend on
the action. The `rho` family correlates the *witnesses* with each other, never
the missingness with the actions. Real logging loss is MNAR — bursts, load
spikes, sampling rules keyed to volume — and a pooled capture-recapture model
has no defence against missingness that concentrates on particular actions.

`Config.op_bias` multiplies both pipes' drop probability for named ops. The
new pool `MNAR` raises the drop rate 1.8x on `update_customer` and
`write_file`, which are exactly the ops the financial env view cannot see.
All three witnesses are then blind to the same actions: the worst case the
design can express without an adversary.

This is **exploratory**: it was built after the confirmatory run and is not
covered by the frozen pre-registration. It cannot change the H1-H4 verdicts.

Predicted, before running:
- Hidden mass rises well above the 63.15 of `fin_mid`, since the ops that
  W3 misses now also lose more log lines.
- `ll_bic` undercovers and undercounts. The hidden actions concentrate in a
  stratum where every witness is weak, which is heterogeneity, and the
  simulator already showed log-linear models failing there
  (`l0_heterogeneous`, coverage 0.296).
- `lc_gated` does better than `ll_bic`, because latent classes are the
  textbook answer to heterogeneity. It has been dominated in every
  configuration so far, and this is where it should earn its place.
- `union_count` stays at 0.000 coverage, worse than anything.

If `ll_bic` keeps 0.93 coverage here, the MNAR objection is weaker than the
reviewer argues and I will say so.

**D32. Under MNAR loss the method can still detect incompleteness but can no longer measure it.**
Results on the MNAR pool (exploratory, after the confirmatory run; signature
linkage, same 21 episodes, N=828):

| config | hidden | estimator | coverage | detects gap | width | bias |
|---|---|---|---|---|---|---|
| `fin_mid_mnar` | 200.21 | `union_count` | 0.000 | 0.00 | 0 | -200.21 |
| | | `ll_independence` | 0.000 | 1.00 | 18.1 | -183.61 |
| | | `ll_bic` | 0.940 | 0.94 | **1978.4** | +152.45 |
| | | `lc_gated` | 0.910 | 0.22 | 262.9 | -134.36 |
| `fin_mid_mnar_rho100` | 221.2 | `ll_bic` | **0.000** | 1.00 | 27.6 | -196.23 |
| | | `lc_gated` | 0.610 | 0.99 | 337.1 | -164.65 |

Against the predictions in D31:
- Hidden mass rises far above `fin_mid`'s 63.15, to 200.21. **Held.**
- `ll_bic` fails. **Held in substance, and worse than predicted.** At
  `fin_mid_mnar` its 0.940 coverage is not a result: the interval is 1978
  wide against N=828, so it covers by spanning more than twice the
  population. Added to coupling it collapses to 0.000.
- `lc_gated` rescues it. **Wrong.** It is better than `ll_bic` under coupling
  (0.610 against 0.000) but is biased -134 and its gap detection falls to
  0.22, so it usually cannot even say the logs are incomplete.
- `union_count` at 0.000 everywhere. **Held.**

The falsifier in D31 was that `ll_bic` keeping 0.93 coverage would weaken the
objection. By the letter it kept 0.940 — but on a 1978-wide interval, which
is why width must be read with coverage and why `detects_gap` alone is not
enough. The objection stands.

**What this does to the claim.** The confirmatory result is unchanged on its
own terms: under loss that does not depend on the action, the method
recovers N. Under loss that concentrates on the actions no witness sees, it
separates into two claims with different strength:
- **Detection survives**: `ll_bic` still asserts the logs are incomplete in
  94% of replicates, and `union_count` never does.
- **Quantification does not**: the interval becomes too wide to be a
  measurement, and adding correlated outage removes coverage entirely.

An auditor could use this to say "your logs are missing actions" and not to
say "they are missing about this many", unless the loss mechanism is known
to be action-independent. That is a much narrower claim than
`docs/ENV-FINDINGS.md` and `docs/CONFIRMATORY.md` make on their own, and both
now point here.

**D33. Two of the three lost confirmatory episodes were a harness defect, not transient errors. D30 was wrong about that.**
The dataset build surfaced the stored error strings. Seeds 7009 and 7022
failed with a provider 400: "The `reasoning_content` in the thinking mode must
be passed back to the API." Seed 7012 was a read timeout. `ChatClient` echoed
reasoning with `if reply.get("reasoning_content"):`, which drops an empty
string; it now tests key presence, with a test that an empty value survives.
The first, aborted attempt's 400 at seed 7006 was probably the same defect,
but its body was discarded (D30), so that cannot be shown.

D30 called the losses transient because seed 7006 ran cleanly on its own.
That inference was invalid: a defect triggered by one trajectory passes on a
fresh one, so a clean rerun does not separate a transient fault from a
behaviour-dependent bug. It also weakens the attrition argument in D31 and
`docs/CONFIRMATORY.md`. Which episodes were lost may depend on how the agent
reasoned, not only on ticket mix, and the ticket-mix check cannot see that.

The registered verdict stands on its 21 episodes. Its hypotheses were already
computed and read, so adding episodes now would be exactly the
outcome-dependent topping-up the stopping rule forbids. Instead, as a
**labelled post-hoc supplement**, the three lost seeds are rerun on the fixed
harness (`artifacts/episodes/confirm-supp`). They are kept out of the verdict,
and the analysis reports whether a pooled 24-episode table would change any of
H1-H4.

Predicted before the supplement ran: no verdict changes. The supplement adds
about an eighth of N, and coverage is flat in study size (D27). If any
hypothesis flips, the confirmatory result is fragile to its own attrition, and
the paper says so.

The transfer worker, which branched before the fix, was told to apply it
before any paid episode, or to report its failures as they stand if its run
had already begun.

**D34. The reasoning 400 is trajectory-dependent and not reproducible on demand. D33's mechanism is withdrawn.**
After the key-presence change of D33, seed 7009 failed again with the same
provider 400 in the supplement. Rerun with request and reply shapes recorded,
it completed. A further batch of six (7009, 7012, 7022, 7006, 7000, 7001)
all completed. On the changed client 2 of 10 episodes failed; on the original
client 3 of 24 had. The empty-string mechanism D33 named is not established,
and the code comment no longer claims it.

What the instrumentation did establish: every tool-call reply carries the
`reasoning_content` key, and its value is sometimes an empty string. That
empty string, passed back, has been accepted. The trigger is open. The next
failure will record the shape of every assistant turn it sent.

The scientific point does not depend on the mechanism. The same seed fails on
one trajectory and completes on another, so the confirmatory losses were
behaviour-dependent, which is what the D31 ticket-mix check could not see.

The D33 supplement is completed under a rule fixed here, before any analysis:
each lost seed contributes its **first** successful completion on the
changed client, in chronological order. That is 7022 from `confirm-supp`,
7009 from `diag-7009` and 7012 from `diag-batch`. The 24-episode pooled
analysis is post hoc and reported beside the registered 21-episode verdict,
never in place of it. The prediction in D33 stands: no verdict changes.

**D35. With the three lost episodes restored, H1 fails. The prediction in D33 was wrong.**
On the post-hoc 24-episode pool (D34's rule, N=949, seed 4242, signature
linkage, `artifacts/hypotheses-supplement.json`):
- **H1 fails**: `ll_bic` covers 0.860 ± 0.035 at `fin_mid`, below the frozen
  0.90. On the registered 21 episodes it was 0.940 ± 0.024.
- H2, H3 and H4 still pass.
- `ll_bic`'s bias at `fin_mid` rose from +3.39 to +9.76, and its RMSE from
  12.63 to 21.31.

The registered verdict is unchanged: `hypotheses.json` is byte-identical
(sha a4cfe4839b74), and the pre-registration names the 21-episode analysis.
It is still not honest to report that verdict without this one beside it. The
H1 pass was fragile to the run's own attrition, and that attrition was
behaviour-dependent (D34).

Not yet separated: how much of the 0.08 drop is Monte Carlo noise, since the
episode count changes every bootstrap draw. The combined standard error is
about 0.042, so the drop is about 1.9 of them. Measured next by rerunning both
pools at loss seeds 1, 2 and 3 as well as 4242, with **every seed reported**.
Decision rule, fixed now: if the 24-episode pool's mean coverage across the
four seeds is below 0.90, H1's failure under full attrition is called real;
if the 21-episode pool's mean is also within one MCSE of 0.90, the
registered pass is called marginal.

**D36. H1's pass was marginal, and its failure under full attrition is real. The headline changes.**
`ll_bic` coverage at `fin_mid`, signature linkage, 100 replicates per seed
(`artifacts/supplement-variance.json`):

| pool | seed 4242 | seed 1 | seed 2 | seed 3 | mean |
|---|---|---|---|---|---|
| 21 episodes, registered | 0.94 | 0.92 | 0.91 | 0.89 | 0.915 |
| 24 episodes, lost seeds restored | 0.86 | 0.88 | 0.91 | 0.86 | 0.877 |

Applying the rule written in D35 before these ran:
- The 24-episode mean, 0.877, is below 0.90, so **H1 fails under full
  attrition**.
- The 21-episode mean, 0.915, is within one Monte Carlo SE (~0.027) of 0.90,
  so **the registered pass is marginal**.

The registered seed gave the highest of the four coverages, and seed 3 would
have failed H1 on the registered episodes themselves. The design flaw is in
the pre-registration: a single loss seed of 100 replicates has an MCSE of
~0.024, against a result only 0.04 above its threshold. That should have been
caught at the red-team (D28) and was not.

`ll_bic` overcounts consistently at `fin_mid` (bias +3.4 to +9.8 across
seeds), so it sits below nominal there. Nothing else in the verdict moves:
- H2 (gap detection 1.00) and H4 pass on every seed;
- H3's coupling contrast holds;
- `union_count` covers 0.000 throughout.

The claim that survives is stated without the 0.940: `ll_bic` covers about
0.88 to 0.92, below its nominal 0.95, while counting the union never covers.
It asserts that the logs are incomplete in every replicate, and it breaks
visibly when its independence assumption does.

README, `docs/CONFIRMATORY.md`, the paper and the dataset card are updated to
lead with this.

**D37. A stratified estimator and an unobserved-key boundary, predictions first.** (branch `mnar`, merged)
Exploratory, after the confirmatory run. `strat_bic` fits the BIC-best
log-linear model per operation, turns each profile likelihood into a
flat-prior density, convolves the strata, and reports the median and central
95%. A stratum that is unidentified (fewer than two live witnesses, or no
discordance) adds its observed count to the lower end and sends the upper end
to infinity. A new pool, `mnar_hidden`, keys the D31 loss on the ledger id
modulo 3, which no record carries, with the full env view.
Fifteen predictions were committed in `267367a` before any run; 12 held. The
three misses are listed in `docs/MNAR-ROBUSTNESS.md`. Every estimator that
existed before reproduces its earlier artifact exactly.

**D38. Stratifying on the operation restores validity under MAR-given-op, not a finite answer. An unobserved key breaks both.**
- `fin_mid_mnar`: `strat_bic` covers by refusing (unbounded in every
  replicate). The affected strata hold two witnesses at about 0.1 capture,
  and no finite interval is honest there.
- `fin_mid_mnar_rho100`: it refuses again, where `ll_bic` covers 0.000.
- `mnar_hidden`, loss keyed on something no record shows: both are
  confidently wrong (`ll_bic` 0.600, `strat_bic` 0.510; about 0 at rho=1).
  W3 supplies discordance, so the identifiability gate passes while the
  no-three-way assumption fails.

That is the boundary the paper states: quantification is recoverable when the
loss depends on something the auditor can observe and condition on, and not
otherwise.

**D39. Neither the pooled nor the stratified estimator dominates. Their disagreement is reported, not resolved.**
- Pooling assumes no three-way interaction across strata. Stratifying assumes
  independence inside two-witness strata.
- The coupled exploratory configurations favour `ll_bic`; `fin_mid_mnar_rho100`
  favours `strat_bic`. The data cannot tell these cases apart.
- `ll_bic` stays the headline and `strat_bic` is reported beside it. Where
  `strat_bic` refuses and `ll_bic` is finite, the finite answer rests on an
  assumption the data cannot test, and is reported that way.
- Agreement between them is not evidence: both are wrong at
  `full_mid_hidden_rho100`.

**D40. Known faults in `strat_bic`, deferred.**
- The flat prior is miscalibrated on thin strata. At `fin_sampled` every miss
  is the lower end above N.
- One small stratum (`write_file`, 16 to 21 actions) can make the whole total
  unbounded.
- A fresh-context reviewer found three risks, none fixed because results had
  been seen: the tail exponent at the cap is underestimated, so some proper
  strata are called unbounded; a single discordant action passes the gate;
  the support can overrun the cap by a negligible mass.

A fix needs a declared rerun.

**D41. Gap detection has never been tested for false positives. Test added, predictions first.**
A hostile review (score 3/10) argued that "detects the gap" is nearly vacuous.
`ll_independence` detects it in every replicate even when it covers 0.000, and
no configuration so far had zero hidden actions with dependent witnesses. The
precise question: when nothing is hidden, how often does an estimator claim the
logs are incomplete?

New loss mode `Config.anti`: gateway and trace losses are mutually exclusive
(one shared draw; the gateway drops if u < p_gw and the trace drops if
u >= 1 - p_tr, so with p_gw + p_tr <= 1 an action never loses both). Every
action then reaches W1 or W2, so the true hidden count is exactly 0, and W1
and W2 are negatively dependent: two logs that specialise. Pool `falsepos`:
- `light`: independent and nearly nothing hidden;
- `anti_full`: specialised logs, full env view;
- `anti_fin`: specialised logs, financial env view, so W3 cannot correct the
  non-financial actions.

The confirm episodes, 100 replicates, seed 4242, bootstrap over episodes.
Gap detection here is a false-positive rate.

Predicted, before running:
- `light`: every estimator's false-positive rate is about 0.
- `anti_full`: low, because W3 sees nearly everything and pins the table.
- `anti_fin`: `ll_independence` claims a gap in about every replicate, with a
  large positive bias. Within the W3-blind actions, 40% are trace-only, 40%
  gateway-only and 20% both, so independence implies a phantom of about 0.8
  times their count. `ll_bic` also has a high false-positive rate, because the
  negative W1:W2 dependence lives only among W3-blind actions and the pooled
  table cannot isolate it. `strat_bic` is confidently wrong too: its
  non-financial strata are two-witness Lincoln-Petersen problems with real
  discordance, so they pass its gate.

If `anti_fin` gives a false-positive rate near 0 for `ll_bic`, the review's
point is weaker than it looks and I will say so.

**D42. Gap detection's false-positive control depends on linkage. Under the headline linkage, specialised logs produce a false gap every time.**
`artifacts/thinning/confirm-falsepos.json` (confirm episodes, N=828, 100
replicates, seed 4242). The share of replicates claiming a gap, when the true
hidden count is 0 by construction under `anti`:

| config | estimator | oracle linkage | signature linkage |
|---|---|---|---|
| `light` | all | 0.00 | 0.00 |
| `anti_full` | `ll_independence` | 0.94 | 0.93 |
| `anti_full` | `ll_bic` | 0.00 | 0.00 |
| `anti_fin` | `ll_independence` | 1.00 (bias +59.3) | 1.00 (bias +58.1) |
| `anti_fin` | `ll_bic` | **0.00** (covers 1.000) | **1.00** (covers 0.000, bias +17.1) |
| `anti_fin` | `strat_bic` | 1.00 (bias +236.0) | 1.00 (bias +237.3) |
| `anti_fin` | `lc_gated` | 0.00 | 0.98 |

Against D41:
- `light` about 0. **Held.**
- `anti_full` low for all. **Wrong for `ll_independence`** (0.93-0.94); held
  for `ll_bic`.
- `anti_fin`, `ll_independence` claims a gap every time with a large bias.
  **Held.**
- `anti_fin`, `ll_bic` high. **Held under signature linkage, wrong under
  oracle linkage**: with exact matching it fits the negative W1:W2 dependence
  and claims nothing.
- `strat_bic` confidently wrong. **Held** under both.

The falsifier was "`ll_bic` near 0 means the review's point is weaker". That
happened under oracle linkage only. The pre-registered headline linkage is
signature, and there the point stands.

Mechanism: when two logs specialise, an action is often seen by only one of
them. Signature linkage pairs records in order within a signature, so two
different actions sharing a signature (two emails to one customer) get merged
into one "seen by both". That fakes overlap in some places and loses distinct
actions in others, and the model reads the distortion as hidden mass.
Unseen-by-all under signature linkage was 0.78 at `anti_fin`, all of it a
linkage artefact; `ll_bic` reported about 17 actions.

What changes in the claim: gap detection is trustworthy when the logs fail
independently, or when records can be matched exactly. When logs specialise
and matching is by signature, it asserts gaps that do not exist. The
confirmatory configurations did not exercise this (oracle and signature
agreed there at 0.940), which is why it went unseen until now.

**D43. Hostile review of the paper (3/10). What changed and what did not.**
Accepted and changed:
- **The abstract led with "passed all four".** It now leads with when the
  count is needed at all, reports the marginal pass (D36), and gives the
  failure map most of its length.
- **Relevance.** If every action passes one channel that numbers its entries
  before sampling, the gaps give N exactly. Added to §1 and the limits. In this
  environment every action passes the gateway, so the live runs calibrate the
  estimator and do not show it is needed there.
- **The selection rule was misreported.** `lc_gated` (0.970) and `anchor_g4`
  (0.960) have better worst-case coverage than `ll_bic` (0.920). The choice was
  a width trade, and now says so.
- **Width against the hidden count.** At `fin_mid` the 42.0-wide interval
  spans about two thirds of the 63.15 actions it estimates. Now stated.
- **The 48.9 prediction** checks the loss arithmetic and the harness, not the
  estimator. Now stated.
- **The coupling test's scope.** It shows the independence assumption carries
  weight, and that model selection recovers a dependence the model class
  contains. It does not show robustness to dependence in general.
- **False positives of gap detection.** Measured (D41, D42) and added to §7.
- **The three failed MNAR predictions are named. Two self-reproaching lines
  are cut**; first-person receipts stay, as the author's register.

Not done, and listed as limits instead:
- a live adversarial agent (D10);
- loss from an unauthored mechanism on a live system;
- a second model family;
- canary and sequence-number baselines run head to head.

Rejected: "H3 is tautological" and "detection is vacuous". Both are answered
by measurement rather than wording. H3's contrast does not appear under a
dependence the model cannot represent (§7). Gap detection has a 0.00
false-positive rate with independent logs and fails, measurably, only when
logs specialise and are matched by signature.

**D44. Fourth review round: the confirmatory shortfall is linkage, not the estimator. Three paper claims withdrawn.**
A review and a refuter (both Sonnet, given the repo and not my notes) ran
against the paper at 4364e0e. The refuter's checks are now repo scripts with
artifacts under `artifacts/robustness/`. Every figure below comes from them.
All of it is post hoc and cannot change `hypotheses.json`.

*Precision* (`scripts/precision.py`, `precision.json`: 30 fresh loss seeds ×
100 replicates, both linkages on the same random numbers). `ll_bic` at `fin_mid`:

| pool | oracle | signature | `ll_independence` (signature) |
|---|---|---|---|
| registered, 21 episodes | 0.948 ± 0.004 | 0.893 ± 0.006 | 0.948 |
| restored, 24 episodes | 0.945 ± 0.004 | 0.862 ± 0.006 | 0.949 |

- The four-seed mean of D35/D36 (0.915) was a high draw. The registered
  analysis covers about 0.89 in expectation: at the bar, below nominal.
- The whole shortfall is effect-signature linkage. With exact matching
  `ll_bic` is nominal on both pools.
- Under signature linkage BIC selects the trace–gateway interaction in 0.358
  (registered) and 0.517 (restored) of replicates, against 0.010 and 0.008
  under oracle linkage. That branch covers 0.858 / 0.847 with bias +12.71 /
  +13.41.
- Signature collisions fake trace–gateway overlap. 0.223 of confirm actions
  share a signature with another action in the same episode
  (`scripts/collisions.py`).
- Under the pre-registered tie-break (oracle and signature differ by more than
  5 points), this is linkage bias. The paper's "matching did not move coverage
  here" was true at seed 4242 only, and is withdrawn.
- `ll_independence`, the true model at `fin_mid`, is nominal throughout. H1
  therefore says nothing for BIC selection.
- `fin_mid_rho100` is identical under both linkages: with ρ=1 and equal drop
  rates the trace and gateway are the same log, so in-order pairing reproduces
  the oracle table. `ll_bic` covers 0.942 there, `ll_independence` 0.000. That
  is a recoverable dependence, not a robustness test.

*MNAR dose response* (`scripts/mnar_dose.py`, `mnar-dose.json`, 300
replicates per dose). The paper showed only 1.8×, the dose where `ll_bic`
looks least wrong.
- Op-keyed loss:
  - `ll_bic` covers 0.647 at 1.1× and 0.187 at 1.2× (oracle), with finite
    intervals. It recovers at 1.6–1.8× only by widening. Oracle and signature
    agree, so this is the estimator, not linkage.
  - `strat_bic` covers 0.92–0.98 at every dose. Its finite intervals cover
    0.93–0.98 up to 1.4×, while the refusal rate rises from 0.067 to 1.00.
- The hidden-key failure is W3's blindness to the keyed third of actions
  (heterogeneity on an unobserved key), not the extra log loss:
  - W3 blind at 1.0×: `ll_bic` 0.16, `strat_bic` 0.467;
  - W3 seeing them, logs ×1.8: 0.767 and 0.84.
- My strat_bic claim "refusal is diagnostic" is withdrawn. It also refuses in
  1.00 of replicates at benign `fin_mid_rho100`
  (`thinning/confirm-heldout-strat.json`), where `ll_bic` is right.

*False gaps* (`scripts/falsegap_battery.py`, `falsegap-battery.json`):
exclusive loss at 0.2–0.5 × both env views × four pools × both linkages, plus
three nulls.
- `ll_bic` claims no false gap under oracle linkage in any cell, and none
  under signature linkage with the full env view (at most 0.01).
- Under signature linkage with the financial view it claims one at every rate
  and in every pool (0.15–1.00), with overcounts up to +438.
- So D42 understated it: this is a regime, not one cell.

*The live setting.* Before thinning, trace and gateway each hold every action
in every live run (`collisions.json`: confirm 828/828), so all hidden mass is
post-hoc thinning plus the injected financial view. The gateway also logs reads
and errors: 1480 calls for 828 actions in confirm. A numbered gateway recovers
N only if it numbers actions, not calls.

What changes in the paper:
- H1 is reported as at the bar under the registered linkage and nominal under
  exact matching.
- The linkage bias is named, with the BIC-selection mechanism.
- §7 shows the dose response.
- The false-gap regime is stated as a regime.
- The abstract says the loss is simulated and each log was complete before it.
- The exploratory tables show both linkages. The estimator was chosen on
  oracle rows. Under signature linkage its worst case is 0.87, at `light`.

Accepted from the reviewer as well:
- refusal rates in the confirm, mnar and worst tables;
- `trust_trace` and `anchor_g1` restored to the confirm table, as the
  pre-registration lists them;
- the auditor runs mixed two model versions;
- forged records are untested;
- the power constant is 1.53, not 1.54 (rounding);
- spend is 3.95 USD with diagnostics (3.66 without).

**D45. A second environment, infra/ops, behind an `Env` bundle (branch `transfer`).**
`src/lacuna/envs/infra.py`, `infra_tasks.py`, `infra_tools.py`: ten services,
five on-call teams, replicas, versions and config, the back-office kernel's
contract (hidden ledger, every action ledgered once, reads never, end-state
diff plus a network sink). W3 keeps its full view; tickets make an operator
do work the diff cannot see (deploys rolled back after a failed smoke check,
load-test scaling, debug toggles, double edits). The harness gained an `Env`
bundle (tools, handlers, action ops, call normaliser, W3 view); `None` is
the original back-office path, and `tests/test_infra.py` pins a golden hash
over 240 back-office episodes taken before the bundle existed. Review fixes:
`Env.value_ops` was dead and is dropped (`linkage.VALUE_OPS` stays one union
over envs, safe because op names are disjoint); `ENV_VIEWS` names back-office
ops only, so `financial` blinds W3 completely on infra; a fleet override for
density checks was shadowed by a local and is fixed and tested.

**D46. The pipeline check predicted the failure; the environment was not changed; the verdict depends on density.**
Before the freeze, 24 free scripted episodes showed signature-linkage
`ll_bic` far below oracle at `transfer_mid` (`transfer-scripted-24.json`).
Making signatures unique would have tested an easier domain, so the finding
went into the predictions. Post hoc (`artifacts/robustness/transfer-density.json`,
scripted, 24 episodes per cell): at 10 services signature coverage is 0.860,
0.750, 0.450 and 0.155 at 10, 20, 30 and 60 tickets per episode; at 30
services, 0.905 at 30 tickets and 0.545 at 60. Oracle stays 0.91-0.96. The
registered failure is a property of 30 tickets over 10 services, not of infra
work in general.

**D47. Transfer result: T1 fails, and the failure is the linkage, which is part of the method.**
23 of 24 episodes, N=1036, $1.97 (`transfer_hypotheses.json`). T1 FAIL at
0.300. At 20 fresh loss seeds the expectation is 0.401, per-seed 0.28-0.48
(`transfer-precision.json`), so 0.300 is a low draw; the scripted run gives
0.420 against 0.942 oracle. Oracle `ll_bic` (0.946 at 20 seeds; registered
0.920 ± 0.027, not distinguishable from 0.90) is descriptive: no auditor can
compute it. "The method transfers" is withdrawn.
- T2 and T4 pass at 1.00 but are uninformative under signature linkage: with
  nothing hidden (`anti_full`) the same estimator claims a gap in 1.00 of
  replicates (`transfer-falsegap.json`).
- T3 passes by the letter. Under signature linkage `ll_bic`'s bias stays at
  −34.25 / −36.35 while its interval doubles, 55.82 to 108.34 with 0.07
  unbounded (`transfer.json`). Under oracle linkage it shows H3's pattern:
  `ll_bic` 0.946 to 0.945, `ll_independence` 0.948 to 0.000.
- Mechanism (`transfer-mechanism.json`): exact matching for signature groups
  W3 misses in whole or in part restores 0.930; for groups W3 sees whole it
  changes nothing (0.383 against 0.438). Undo tickets alone give 0.904, all
  other tickets 0.851. Of the −32.22 bias, 0.573 is observed actions lost to
  merging (−18.45) and the rest is extrapolation from overstated
  trace-and-gateway overlap (277.47 against 259.69). Page and runbook digests
  in the signature do not help (0.395).
- The agent is irrelevant: 1035 of 1036 actions are in the scripted plans
  (`transfer-behaviour.json`).
- Predictions: 12 of 12 that could fail held. Four rows (the 0.25 ×
  W3-invisible ratio, the doubling under coupling, `union_count` at 0.000
  twice) are identities or true by construction and no longer count; the
  read bands were added after the freeze, in `0748430`.
- The failed episode's `diag`: no `reasoning_content` key at turns 74 and 81,
  present at 77, and 27 of 706 completed calls lacked it too, so absence does
  not explain the 400.
- Spend in `derived.json` is now 5.92 USD with diagnostics, the transfer run
  included; the paper's total follows it. `docs/PREREG-TRANSFER.md` cites
  D34, committed twelve minutes after it froze; left as is.

**D48. A false gap needs a W3 blind spot plus in-order pairing on W3-visible ops whose text the signature drops. This corrects the paper's §7.**
Under `anti_full` (nothing hidden) on the infra episodes
(`transfer-falsegap.json`): signature matching on `send_page` alone claims a
false gap in 0.98 of replicates; on every op except pages and incidents it
claims none and undercounts instead (bias −28.48); with W3 handed every action
it missed, signature matching everywhere claims none. On the back-office
confirm episodes under `anti_fin`, signature on `send_email` alone claims one
in 1.00, signature on everything else in 0.00. My earlier line in
`docs/TRANSFER.md`, that transient work repeating signatures is enough for a
false gap, is refuted and withdrawn. For the paper's §7: the interaction it
says was not isolated is isolated (the emails, W3-visible and matched without
their text, read as hidden mass once W3 is blind elsewhere), and "an env diff
that sees every operation" is not enough: the infra diff sees every op type
and the false gap is still there. What the diff must see is every action.

**D49. Final review round: better linkage fixes back-office and every false gap; bursts break the estimator itself.**
A final review and refuter (Sonnet) ran on the full paper at e03dd13. Their
checks are now `scripts/linkage_arms.py` and `scripts/burst_loss.py`, writing
`artifacts/robustness/linkage-arms.json` and `burst-loss.json`. All post hoc.

*Linkage arms.* Five ways to build the capture table:
- `signature`: as pre-registered;
- `digest`: the signature, keeping the content digest every witness record
  already stores for email and file bodies;
- `callid`: trace and gateway linked exactly by the provider's tool-call id,
  which the harness writes into both logs; W3 attached by signature, in order;
- `callid_digest`: the same, with W3 attached by digest;
- `oracle`: the ledger id.

All arms reproduce N exactly on unthinned logs. `ll_bic` coverage, 2000
replicates per held-out cell, 1000 per exclusive-loss cell:

| cell | signature | digest | callid | oracle |
|---|---|---|---|---|
| back-office `fin_mid`, registered pool | 0.875 | 0.942 | 0.946 | 0.946 |
| back-office `fin_mid`, restored pool | 0.852 | 0.944 | 0.945 | 0.945 |
| infra `transfer_mid` | 0.406 | 0.379 | 0.859 | 0.944 |
| infra `transfer_mid_rho100` | 0.803 | 0.803 | 0.803 | 0.944 |

- **Back-office.** The shortfall was a design choice. The signature dropped
  digests the logs carry, a choice made for the LLM auditor dropped in D20.
  Either the digest or the call id restores nominal coverage.
- **Infra.**
  - The call id fixes the trace–gateway half; the env diff, which has no call
    id, is still attached by signature.
  - At ρ=1 the trace and gateway are the same log, so the call id adds nothing,
    and all of the remaining error is W3 mis-linking.
  - The digest does not help in infra: the collisions there are identical
    repeated invisible actions.
- **False gaps** (exclusive loss, nothing hidden):
  - `callid`: no false gap and coverage 1.000 in every cell, in both
    environments.
  - `digest`: no false gap in back-office, but it undercounts by about one
    action on near-zero-width intervals, so coverage is 0.29–0.81. In infra it
    does worse than the signature.
- **Mechanisms differ by environment.**
  - Back-office: BIC fits a trace–gateway interaction that the collisions
    fake (D44).
  - Infra: BIC selects that interaction rarely. The bias is merged clusters
    plus W3 mis-linking (D47); the independence model is as wrong as BIC there.
- The call id is unforgeable only while the harness writes both logs. An agent
  that writes its own trace could reuse or invent ids. Untested.

*Bursts* (`burst-loss.json`, oracle linkage, 2000 replicates). Each block of w
consecutive ledger ids is dropped with probability 0.5, independently per log
or jointly. `ll_bic` coverage at w = 1, 2, 3, 6, 12:
- confirm, financial view, independent: 0.941, 0.848, 0.792, 0.679, 0.559;
- long-30: 0.949 → 0.507;
- infra: 0.948 → 0.802;
- joint outages: 0.950 → 0.683 (confirm, financial view);
- with the full env view the fall is milder: 0.941 → 0.83.

Bias stays near zero and width barely moves. This is overdispersion: the
profile-likelihood interval assumes per-record loss. Exact matching does not
help, so this is the estimator's own limit, the second after action-dependent
loss.

*Also accepted:*
- §7's "an env diff that sees every operation" is wrong (D48: it must see every
  action).
- The abstract must say the agent was benign, the loss was simulated, and
  canary/sequence-number baselines were not compared.
- "The estimator transfers" contradicts D47: the method as registered missed
  its bar in both environments.
- `strat_bic` is calibrated where it answers, up to 1.4x; beyond that its
  coverage is mostly refusal. At 1.6x only about 8 of 300 intervals are finite.
- The hidden-key table and the 1.8x table differ because they use different
  replicate counts; say so.

The paper is restructured around this:
- one section on linkage, covering both environments and the arms;
- one on where the estimator itself fails (action-dependent loss, bursts);
- the superseded per-seed supplement moved to an appendix.

**D50. Review round on the restructured paper: bursts are an interval problem, partly fixable; the call id fixes linkage only where W3 attachment is trivial.**
A review and refuter (Sonnet) ran at 783b535. The reviewer's verdict was
"accept with minor revisions". Its findings were abstract overclaims:
- "no bias";
- "both missed their bar" stated as fact;
- an unscoped stratification sentence;
- call id not labelled post hoc;
- exact matching not labelled an oracle.

The refuter's findings changed two framings.

*Bursts are the interval, not the estimator*
(`scripts/overdispersion.py`, `overdispersion.json`; 1000 replicates per row,
exact matching unless noted).
- New in `loglinear.py`: `dispersion` (Pearson X²/df, floored at 1) and two
  quasi-Poisson intervals that scale the likelihood-ratio cutoff by it.
- `quasi_independence` (ll_independence_qp) takes the dispersion from the
  independence fit.
  - Independent bursts at w=12: 0.910 (confirm), 0.893 (long-30), 0.917
    (infra), against 0.611/0.603/0.676 plain.
  - Per-record loss: 0.959 against 0.950, width +13%.
  - Under coupling the factor counts the real dependence as dispersion and the
    interval explodes: about 920 wide, coverage 1.000. The size of the factor is
    itself the warning.
- `quasi_bic` (ll_bic_qp) was first built with the independence-fit dispersion.
  It exploded the same way under coupling (factor about 277), so it now takes
  the dispersion from the model BIC selected.
  - Safe under coupling: 0.947, width 73.3.
  - Only partly fixes bursts: 0.685 / 0.588 / 0.852 at w=12.
  - Lifts signature-linked `fin_mid` from 0.886 to 0.937.
- Joint bursts (one outage for both logs): neither variant helps (0.685).
- Op-keyed MNAR: partial (0.252 → 0.464 at 1.2x). The bias remains.
- So the claim "bursts break the estimator" is withdrawn. The point estimate is
  nearly unbiased for the independence model (`ll_bic` picks spurious
  interactions, bias up to +8.7). Burst loss is fixable when the logs are
  otherwise independent. Combining a dispersion correction with dependence
  selection is not solved here.

*Call-id linkage, scoped* (`linkage-arms.json` rerun with full-view
back-office cells and a last-writer arm).
- Under the registered financial view, `callid` equals exact matching in every
  cell, because W3 sees every email, refund and cancellation, so attaching it
  is trivial.
- With the full view W3 must be attached for repeated updates too, and `callid`
  falls 0.01–0.03 short of exact matching (refuter: mean 0.926 vs 0.940 over 16
  cells).
- The false-gap result holds by construction whenever the trace and gateway
  jointly see every action.
- A last-writer rule (`callid_lastwriter`) attaches W3's single end-state row to
  the latest logged call on that (op, resource) if its value matches. It
  recovers about 40% of the infra gap.
- The rest is identical repeated actions, which a diff row cannot be attached
  to without a per-row id or timestamp.

*Also accepted:*
- finite-interval counts in the dose table (86 of 300 at 1.4x, 8 at 1.6x);
- D40's deferred faults named in §9;
- the 0.893 and 0.875 cells come from different seed sets, noted;
- "exact" is equated with "oracle", which no auditor has;
- stale README, card and CONFIRMATORY lines.

**D51. Episode-level dispersion fixes bursts and coupling together; the last-writer rule used the ledger order.**
A review and refuter (Sonnet) ran on the D50 round at 66188ac. The refuter
refuted D50's "not solved here".

*Episode-level dispersion* (`loglinear.homogeneity_dispersion`,
`quasi_bic_episodes` → `ll_qbic_hom`; `scripts/episode_dispersion.py`,
`episode-dispersion.json`; 1000 replicates per row, exact matching unless
noted).
- **Method.**
  - The dispersion is the Pearson X²/df of the episodes × observed-cells
    table, floored at 1. It needs no log-linear model, so dependence between
    logs is not counted as dispersion.
  - The model is chosen by QBIC on that scale, and the profile cutoff is
    scaled by it.
  - This is the QAIC recipe of Lebreton et al. (1992) and Burnham & Anderson
    (2002), with the dispersion from replication across episodes. With three
    witnesses the global model has 0 residual df, so the usual source is
    empty.
- **Where it holds.**
  - Per-record loss: 0.939 (back-office) and 0.954 (infra), at the plain
    width (40.0 vs 39.1).
  - Coupled loss: 0.935 at ρ=0.5, 0.965 at ρ=1, and 0.948 in infra at ρ=1,
    at the plain width.
  - Bursts: 0.952 at w=3 and 0.984 / 0.966 / 0.988 at w=12, at about twice
    the width.
  - Joint outage: 0.982.
  - Signature-linked infra bursts: 0.943.
- **Where it does not.**
  - Action-dependent loss: 0.257 at 1.2x. That is not dispersion.
  - Signature mis-linkage under per-record loss: 0.400 in infra, where φ_ep is
    1.04.
  - Small audits under bursts: it keeps coverage by widening, and gap
    detection falls to 0.61 at 8 episodes and 0.14 at 4.
- Needs visible episode boundaries.
- The D50 variants move to an appendix. Their φ is not an alarm: it reads 3.77
  in coupled infra, where coverage is nominal, and 1.0 under joint bursts.

*Last-writer order.*
- `callid_lastwriter` sorted calls by `Record.oracle`, the ledger id: true
  execution order, which no auditor has.
- It now merges the gateway and trace line orders, which are auditor-visible
  and tied together by shared call ids (`linkage_arms.merged_order`).
- Infra `transfer_mid`: 0.916 (the ledger sort gave 0.888). Coupled: 0.839,
  unchanged. Back-office full view: 0.920 / 0.924.
- D50's "about 40% of the infra gap" was 34% under the ledger sort. The merged
  order closes 67%: (0.916 − 0.859) / (0.944 − 0.859).

*Also fixed from the review:*
- the abstract states the QP result only for the variant it holds for;
- "dispersion as an alarm" is dropped;
- the full-view shortfall is 1 to 3.5 points;
- "five" linkage arms, not four;
- the QP references are cited (Wedderburn 1974; McCullagh & Nelder 1989;
  Lebreton et al. 1992; Anderson, Burnham & White 1994; Burnham & Anderson
  2002);
- the D40 faults are stated as possible, not certain;
- the appendix value ops now include infra.
