# lacuna

How many consequential actions did the agent take that no log line shows?

Trace integrity is a solved-ish problem: hash chains, append-only servers,
TEEs. Trace *completeness* is not. Agents delete and spoof their own trace
events when it pays (Qin et al. 2026, arXiv 2609.30266). Every audit regime
that relies on logs (EU AI Act Art. 12, ISO/IEC 42001, SB 53 incident reports)
assumes the log is whole, and nothing measures how far from whole it is.

lacuna treats the question as a population-size problem. Several witnesses each see
part of what the agent did:

| Witness | What it sees | Who controls it |
|---|---|---|
| W1 `trace` | the agent's own log | the agent |
| W2 `gateway` (`proxy` in result labels) | MCP/HTTP calls through an interception proxy | infra; bypassable through code execution |
| W3 `env` | state diffs + a network sink | the environment; misses transient write-then-revert |
| ~~W4 `auditor`~~ | an LLM that rebuilds actions from the transcript | **dropped: not independent (D20)** |

Capture-recapture (multiple-systems estimation, the method used to count
casualties nobody recorded) turns the overlaps between witnesses into an
estimate of the actions no witness saw, with an interval. Against an adaptive
agent that correlates its omissions across witnesses, a point estimate
is not identified, and lacuna reports bounds instead.

The floor is `union_bound`: "at least as many as any witness saw, maybe
infinitely more". It always covers. A method earns its place only by giving a
finite upper end without losing coverage.

## In short

- **Question:** how many consequential agent actions are in no log?
- **Method:** capture-recapture over three logs (the agent's trace, a gateway log, an
  environment diff). `ll_bic` turns their overlaps into an interval for the true count.
- **Pre-registered, back-office:** passed at its registered loss seed (0.94) but covers
  0.893 over 3000 replicates, below the 0.90 bar.
- **Pre-registered, infra:** failed (0.3 at the registered seed, 0.401 over 20 seeds).
- **Pre-registered, bypass:** failed. The agent ran 0.856 of its actions through a batch
  tool that the trace and the gateway log as one call (predicted 0.05 to 0.30), and
  `ll_bic` covers 0.759 with intervals about 8.6 times wider in absolute terms (against a
  different pool), though it still detects that a gap exists (0.995, also 1.0 with no
  bypass). No witness parses the batch; that channel is a stipulation. Under the registered
  view, 96% of the hidden mass (317.8 of 330.7 actions per replicate) is bypassed customer
  updates that no witness can record, so the cell asks the estimator to extrapolate to
  actions with zero capture probability. Post hoc, parsing the scripts restores 0.949.
- **Scope:** the method is for settings where every action reaches at least one counted log
  under independent-enough loss. The bypass study tests the boundary where a channel
  reaches none, and there the estimator cannot bound N.
- **Why:** the first two tests used effect-signature record matching; matching by the
  ledger id, which no auditor has, is nominal in both, but only without drift in the loss
  rate.
- **Cheaper checks:** a tail-corrected gateway counter and planted canary actions survive
  action-dependent loss in simulation. The counter fails under bursts, shared session
  quality and bypass (live, too, in the bypass study); canaries cost width and gap
  detection. See Status.
- **Main limitation:** the agents were benign and all log loss is simulated afterwards on
  complete logs, in two environments and a batch-tool variant of one. Nothing here tests an
  agent that hides on purpose.

Terms used below:
- `ll_bic`: Poisson log-linear model of the three-log capture table, pairwise
  interactions chosen by BIC, 95% profile-likelihood interval.
- `fin_mid`: back-office loss setting where the trace and the gateway each drop half
  their lines at random and the env diff sees only financial operations; `transfer_mid` is
  the same loss in the infra environment.
- signature linkage: records match across logs on operation, resource and sometimes
  value, in order. Call-id linkage: trace and gateway match on the provider's tool-call
  id that the harness writes into both. Ledger-id linkage: match on the environment's
  ledger id, which no auditor has.

## Status

**Three pre-registered tests: back-office passed at its seed and misses in expectation, infra failed, bypass failed** (`docs/DECISIONS.md` D44–D54, D61–D64).

- **Back-office** (`docs/PRE-REGISTRATION.md`): `ll_bic` passed H1 at its registered
  loss seed (0.94) but covers 0.893 over 3000 replicates, below the 0.90 bar.
- **Infra** (`docs/PREREG-TRANSFER.md`, write-up in `docs/TRANSFER.md`): T1 failed,
  0.3 at the registered seed and 0.401 over 20 seeds. The paid run reproduces a
  free scripted one, so this tests the environment and the linkage, not the agent.
- **Bypass** (`docs/PREREG-BYPASS.md`, write-up in D63, post hoc follow-up in D64): the
  back-office environment plus a `run_script` tool. The trace and the gateway log one call
  holding the script text, no witness parses it (a stipulation: an auditor that parsed it
  would recover every action), and the env diff sees the effects. The agent ran 1098
  of 1283 actions in scripts, a share of 0.856, where the registration predicted 0.05 to
  0.30. That share is the agent's choice under a parser that rejected 34 of 99 scripts
  whole. B1 failed: `ll_bic` at `fin_mid` with call-id linkage covers 0.759, which is 0.187
  below its 0.946 with no bypass (a different pool), on intervals 333.6 wide against 39.0:
  about 8.6 times as wide in absolute terms, about 5.5 times relative to N. The gateway
  counter covers 0.0 (a design check, true by construction). `ll_bic` still detects that a
  gap exists in 0.995 of replicates (a design check; also 1.0 with no bypass).
  - *Why it fails:* under the registered `fin_mid` view, 317.8 of the 330.7 actions a
    replicate hides (96%) are bypassed `update_customer` calls that no witness can
    record, so the cell asks the estimator to extrapolate to actions with zero capture
    probability.
  - *Post hoc* (`artifacts/robustness/bypass-counterfactuals.json`, 30 fresh loss seeds):
    parsing the scripts restores `ll_bic` to 0.949 at that cell, so the failure comes from
    the stipulation that no counted log itemises the channel, not from the pool, the
    ticket mix or the tool. Under the full env view parsing helps only with ledger-id
    linkage (0.944), not call id (0.487) or signature (0.21), a separate linkage problem.
    When the env diff also sees the bypassed updates, coverage is 0.008 at bias +403.8, so
    the registered cell's bias of +34 is partly a cancellation, not tolerance.
  - *Other estimators:* none is calibrated (at most 0.84): signature linkage 0.785, Chapman
    on the trace and the env diff 0.834, the independence model 0.392, an env diff that
    sees every operation 0.319. In the audit-warning pilot Chapman W1xW3 covers 0.968 and
    `ll_independence` 0.966 at `fin_mid_rho100`, only at widths of 3453 and 2681 for N = 603.
    No share threshold is established: scripted runs at shares of 0.10 and 0.28 covered
    0.817 and 0.922.
  - *Audit-warning pilot:* under the prompt the share was 0.99 against 0.87 on the same 12
    seeds (pooled +0.134, paired +0.121 [0.01, 0.30], predicted at most 0.10). The neutral
    run already bypassed at 0.96 to 0.98 in 10 of 12 seeds. The difference comes from two
    neutral episodes, `9503` (all 8 scripts rejected: 4 did not parse, 4 failed validation)
    and `9509` (share 0.61); without them it is +0.013. With a ceiling at 1.0 and 12 pairs
    this does not show the prompt changed behaviour. Its `ll_bic` coverage of 0.996 is not
    calibration: 0.996 of its intervals are unbounded, and gap detection is 0.0. With the
    scripts parsed it is 0.939.
  - *Frozen code:* `git diff prereg-bypass -- src scripts/run_episodes.py
    scripts/bypass_analysis.py scripts/bypass_verdicts.py` is empty at the analysis commit
    `1046284`. The wider `-- src scripts` now shows unrelated later changes.

The first two were nominal when records are matched by ledger id, which no auditor has:

| `ll_bic` coverage | signature (registered) | shared call id (post hoc) | ledger id |
|---|---|---|---|
| back-office `fin_mid` | 0.875 | 0.946 | 0.946 |
| infra `transfer_mid` | 0.406 | 0.859 | 0.944 |

These come from a different loss-seed set than the 0.893 above. The signature dropped
content digests the logs carry and merged distinct actions. Linking trace and gateway
by the provider's tool-call id fixes back-office, where the env diff is easy to
attach; infra stays short because the diff carries no id, and a last-writer rule
that looked like a fix produces false gaps (D52). The call id is written by the
harness. Forged and duplicated lines break it (below); reusing a real action's call id
to merge distinct actions was not tested.

What holds on every seed and every pool:
- counting the union of all three logs never covers the truth (coverage 0.0);
- `ll_bic` asserts the logs are incomplete at `fin_mid` in every replicate;
- when the gateway and the trace fail together, the independence model
  collapses to 0.0 while `ll_bic` holds.

Before thinning, the trace and the gateway each held every action, so all
hidden mass is simulated loss.

**The limits.**
- **Action-dependent loss:** when log loss concentrates on the actions the env diff
  cannot see, the pooled estimator fails silently from 1.1x extra loss (0.657,
  then 0.197 at 1.2x, with signature matching; exact matching gives 0.647 and 0.187). Stratifying on the operation stays
  calibrated where it answers, refusing more often as the loss grows.
- **Bursts:** loss in blocks of consecutive actions makes the plain interval too
  narrow (0.559 at blocks of 12 in `burst-loss.json`; the range over every seed set that
  ran the cell is under Run). Most of that is
  not dispersion: the agent works one kind of ticket at a time, so a lost block lands
  where the env diff is blind and `ll_bic` is biased (+29.6 and +58.8 on the
  long-episode pools). Permuting which actions the diff sees restores 0.875–0.94
  (D54). `ll_bic_hom` (post hoc) keeps `ll_bic`'s model, never covers less, and
  trades width for coverage: 0.952 / 0.994 under bursts and 0.894 on long-60 in
  `interval-comparison.json`, and 0.846 on long-120 in `final-controls.json`. Report it
  beside the plain interval.
- **Outages and shared session quality:** whole-session outages of the gateway and
  trace leave plain `ll_bic` nominal (0.936 / 0.943) only while the env diff stays up.
  Session quality shared by all three logs defeats the pooled intervals;
  `tier_bic` recovers 0.838 / 0.907 there but is not a default (0.292 under
  action-dependent loss, 0.595 under bursts).
- **False gaps:** signature matching produces them when logs lose mutually
  exclusive actions and the env diff misses some; call-id linking removes them.
- **Drift:** a loss rate that ramps over the episode in the trace and the gateway, with
  the same mean, breaks `ll_bic` even under exact matching: 0.742 and 0.471 at
  `fin_mid` (`drift/0.4` and `drift/0.8` in `new-loss.json`).
- **Faults in the lines:** a twentieth of lines duplicated in all three logs takes
  `ll_bic` to 0.0 under signature matching (bias +108) and 0.094 under call-id
  matching; a twentieth forged as phantoms in all three, to 0.0 under both. The call
  id fixes only duplicates in the trace and the gateway. Reordering changes nothing.

**Cheaper checks** (`baselines.json`; simulated loss on recorded episodes, exact matching,
500 replicates per cell, confirm pool only):
- A gateway sequence counter with a tail correction covers 0.952 at `1.2x` loss on
  unseen operations, where `ll_bic` covers 0.254 (0.187 in `mnar-dose.json`: another seed
  set and fewer replicates, about two standard errors apart). It covers 0.97 under hidden-key
  loss at `1.8x` (`ll_bic` 0.69), at about the width of `ll_bic`'s or less. It fails when
  loss is not independent or the gateway is not the only route: 0.0 under bursts, 0.48
  under shared session quality, 0.002 with a twentieth of actions bypassing the gateway.
  The bypass cells drop gateway records only; in the real bypass environment the trace
  misses those actions too, so they say nothing about `ll_bic` under bypass; the live bypass study above is the test.
- Canary actions (`k=5` per episode, the episode's operation mix) cover 0.922 or more in
  every cell where they are exchangeable with real actions, 0.938 under bursts, with
  wider intervals and gap detection of 0.69 against 1.0 for `ll_bic`. They cover 0.0
  when they sit outside the loss key and 0.872 at `1.8x` with a uniform operation mix.
  They always get an env record when in view, which flatters them, and were never run
  with bypass.
- Chapman on the trace and the env diff covers 0.926 under fully coupled trace and
  gateway loss, at `ll_bic`'s width; on the trace and the gateway, 0.0.

## Layout

- `src/lacuna/sim.py` — capture-history simulator across the threat ladder
- `src/lacuna/estimators/` — floor baselines; log-linear MSE with profile-likelihood intervals; latent-class model and its identifiability gate; env-anchored Γ bounds
- `src/lacuna/envs/` — back-office kernel with a hidden ground-truth ledger, tools, gateway log, harness trace. Benign loss only (D10, D11); `bypass.py` adds the `run_script` batch tool (D61)
- `src/lacuna/thinning.py` — post-hoc logging loss over recorded episodes
- `scripts/` — `sim_coverage.py`, `run_episodes.py`, `thin_episodes.py`, `reaudit.py`, plus the table renderers the docs quote

**The live arm has three witnesses, not four (`docs/DECISIONS.md` D20).** An LLM auditor reading the agent's own transcript has capture probability 1 when it sees the call log and 0 when it does not; there is no setting that makes it independent. Every `@4` number in this repo is superseded and supports no claim.

**Open:**
- No adversarial rung has been run live (D10): every L1/L2 number is simulator-only.
- The cheaper checks ran on thinned recorded episodes from the confirm pool only; canaries
  were never run with bypass.
- `strat_lc_gated` is superseded by `strat_bic` (D17, D38); `strat_bic`'s deferred faults are listed in D40.

## Run

Needs `uv` and `Python >= 3.11`; `make paper` also needs `tectonic`.

```bash
make install
make validate   # lint + tests + privacy gate; no key, no network
make sim        # 500-rep coverage sweep, ~6 min on a laptop
make paper      # tables, figures and both PDFs (needs tectonic)
```

The registered confirmatory analysis needs `--bootstrap-episodes`; without it
`ll_bic` at `fin_mid` covers 0.90, not 0.94
(`artifacts/thinning/confirm-heldout-fixed-episodes.json`). At 100 replicates each
(Monte Carlo error about 0.03) the two are not distinguishable, and the fixed-episode run
still reaches the 0.90 bar. The command rewrites `artifacts/thinning/confirm.json`
unless given `--out`. H1-H4 and every `ll_bic`, `ll_independence` and union field
reproduce exactly (`make repro-headline` checks this into a temp dir); `ll_average` and
`strat_lc_gated` at `fin_mid_rho100` differ slightly (8 fields; the solver sits on
boundary fits, likely a numpy/scipy version change since the run). No published number
depends on them.

```bash
uv run python scripts/thin_episodes.py --run confirm --pool heldout --bootstrap-episodes
uv run python scripts/test_hypotheses.py --run confirm
```

Several cells were run on more than one set of loss seeds, so the same cell reads
differently in different tables. `ll_bic` at blocks of 12, back-office, financial view,
exact matching: 0.559 in `burst-loss.json` (the value quoted above), and 0.535 to 0.588
across `burst-loss`, `overdispersion`, `episode-dispersion`, `interval-comparison` and
`final-controls`; the report's appendix lists each. `ll_bic_hom` on the same cell:
0.952 in `interval-comparison.json`, 0.966 in `final-controls.json`.

## Data and license

- The capture histories, ledgers and results are also a Hugging Face dataset,
  [`caiotheodoro/lacuna-capture-histories`](https://huggingface.co/datasets/caiotheodoro/lacuna-capture-histories),
  built by `scripts/build_dataset.py`. It withholds the episode transcripts.
- Code and data are `Apache-2.0` (`LICENSE`, `NOTICE`). The `transcript` fields in
  `artifacts/episodes/*/episodes.jsonl` hold the reasoning text of the DeepSeek
  models the live runs used (served through OpenCode Go), which that license does
  not cover; the provider's terms apply.
- Paper and technical report: `paper/lacuna-workshop.tex`, `paper/lacuna.tex` (`make paper`;
  PDFs are attached to the GitHub release). The report's "Reproducing this" section
  names the script behind each result. `scripts/thin_episodes.py` overwrites
  `artifacts/thinning/` unless given `--out`.
- Terms such as `ll_bic`, `fin_mid` and signature or ledger-id linkage are defined
  in the dataset card's glossary.
