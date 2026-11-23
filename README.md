# lacuna

How many consequential actions did an AI agent take that no log shows? lacuna
estimates that count with capture-recapture (the method for counting casualties nobody
recorded): each log sees part of what the agent did, and the overlaps between logs
estimate what none of them saw.

## Status: what holds

*Coverage* is the share of simulated replicates whose 95% interval contains the true
count; the bar is 0.90. In study 5 nothing is simulated: coverage is a share of rig runs
and the bar is on the lower end of its Wilson interval.

| Claim | Evidence | Status |
|---|---|---|
| Counting unlogged actions works in the back-office environment | Pre-registered. Passed at its registered loss seed (coverage 0.94) but covers 0.893 over 3000 replicates, below the 0.90 bar. A second model (`qwen3.8-flash`) on plain back-office covers 0.942; loss is simulated, so that tests another action mix, not a model family | scoped |
| The registered record matching transfers to a second (infra) environment | Pre-registered, predicted to fail. T1 failed: 0.3 at the registered seed, 0.401 over 20 seeds. Linking the trace and the gateway by a shared call id lifts back-office to 0.946 and infra to 0.859 | fail |
| The count is bounded when a channel is itemised by no counted log | Pre-registered bypass study. The agent ran 0.856 of its actions through a batch tool that the trace and the gateway log as one call (predicted 0.05 to 0.30). Coverage 0.759 against 0.946 with no bypass. The estimator still detects a gap in 0.995 of replicates (also 1.0 with no bypass, so it does not say which channel is unseen); counting the union covers 0.0. Post hoc, one record per bypassed action, taken from the ledger, restores 0.949. The second model ran 0.97 of its actions through the batch tool and covered 0.20 | fail |
| Itemising a channel is enough | No. An auditor that parses the scripts but loses a script's lines together covers 0.469 on the neutral pool (the same records lost one by one: 0.952), 0.61 under the adversary prompt (0.94) and 0.335 on the second model (0.941). Against not parsing, that is worse on the neutral pool (0.655) and better on the second model (0.20). The intervals are too narrow, not biased: the lines of one script are not independent. No dispersion correction was tried on it | fail |
| It survives loss that depends on the action, comes in bursts or drifts | Fails silently: 0.647 at a tenth more loss on the operations the env diff cannot see, 0.187 at a fifth more; 0.559 under bursts; 0.742 under a drifting loss rate | fail |
| In one rig, the interval covers when a real telemetry pipeline loses the records and two originators share no counter | Pre-registered study 5 (`docs/PREREG-STUDY5.md`; D92, corrected by D93), one rig, no loss simulated. `ll_sa_hom` covered 269 of the 277 runs left (Wilson 0.944 to 0.985) with no interval unbounded, so the criterion as registered is met; the registered prediction that it would not be was wrong. 23 of the 300 planned runs were set aside on integrity, and with those counted as not covered it is 269 of 300 (0.857 to 0.926), which straddles the 0.90 bar: not shown. The point was closer to the truth than a sequence-gap count in 223 of the 277 runs and than the union in 261: a pass against both, the second weak by construction (below). Without the end state (`chapman_sa`) 10 of 277 are covered: missed | scoped |

The thesis the first five rows support: capture-recapture bounds the unlogged actions
when every channel is itemised by at least two logs and records are lost action by
action; when a channel is itemised by none, no log-based count can see it, and only
design review, not the estimator, detects that.

Study 5 is the only row where the loss is not simulated. Two originators replay recorded
agent sessions, one through an unmodified gateway, and the collectors that relay their
spans and access logs are killed together and restarted 15 to 45 s later. Scope, in the
registration's terms: where a trusted counter exists at the source or the sink, use it;
this is the case of originators that share no counter (two here) and loss from a real
telemetry pipeline, in one rig, at those outage lengths. What goes with the row:

- The caveats the registration puts on the coverage figures: the bounds treat runs as
  independent and they replay shared sessions; one rig; an interval with no upper end
  would have counted as covering (none occurred in the runs left).
- The runs set aside were bunched in time. Post hoc (`artifacts/study5/posthoc3.json`,
  D112; no criterion), against the power log of the laptop that made the runs: of the 8
  runs set aside for the longest delays, 5 sit beside a sleep of the machine and the
  other 3 predate the retained log; and 17 of the 21 set aside for shorter ones are among
  the 58 runs made between the log's last reading on battery and external power being
  plugged in. What slowed the machine was not measured.
- Also post hoc: the registered estimator, scored on the 20 of the 23 runs set aside that
  left a full record, covered 19 of them, which is 288 of 300 with the rest counted as
  not covered (Wilson 0.931 to 0.977). Those runs broke the timing rule, so this is a
  description and not a test, and the two registered figures stand as they are.
- Against the union the comparison shows that the amount added is the right order, not
  that it is accurate; the informative comparison is the sequence-gap one
  (`seq_gap_tail`). That baseline is never shown the end state, so the comparison is of
  information as much as of method.
- The point is the two-list estimate. A plain Chapman interval on the same two lists
  (`chapman_te`, no criterion registered) covers 250 of the 277 runs: what the registered
  estimator adds on these runs is its interval.
- Two descriptions made after the verdicts were read (`artifacts/study5/posthoc.json`,
  D98; no criterion). The lower end of coverage stays at the bar up to a design effect
  of 4.05 from the shared sessions. Estimated later under a model (D112), the design
  effect for misses is 2.92, with an approximate upper limit of 7.51: the runs neither
  show dependence nor exclude enough of it to matter. And
  against the larger of the sequence-gap count and the union, a baseline that does read
  the end state, the point was closer in 219 of the 277 runs (exact lower bound 0.738).
- What the end state hides is by the author's design: the author wrote the task mix and
  its weights, and chose the estimator and the condition after a pilot on the same rig.
- Floors on the runs left: `union_count` covers 2 and `trust_trace` none; `union_bound`
  covers all with no upper end on any; `seq_gap_tail` covers 34. The median absolute error
  is 3.9 actions for `ll_sa_hom` and 22.0 for the union. On average 0.068 of a run's
  actions were in no log.
- Unstressed control: 58 of 60 runs in the test, no interval above the truth. A defect:
  in one of them (`s5-s0-021`), where no log lost anything, the registered `ll_sa_hom`
  gave an infinite point and no upper end. By the registered rule that counts as
  covering; it changes no verdict. The library's `ll_sa_hom` was fixed after the
  verdicts were read (D94, post hoc) and the registered figures were not recomputed.
- A second stressed condition (each relay on its own outage, one originator killed) has no
  criterion and no prediction: 49 of the 51 runs left were covered (Wilson 0.868 to
  0.989), with 9 of the 60 planned set aside.
- No claim is made for or against the estimator by outage length, pool, band or time
  window.

Terms:
- Channel, itemised: a channel is a route an action can take, a direct tool call or
  `run_script`, a batch tool that runs a script of many operations in one call. A log
  itemises a channel if it holds one record per action taken through it.
- Witnesses: W1 `trace` (the agent's own log), W2 `gateway` (calls through an
  interception proxy), W3 `env` (state diff plus a network sink). An LLM auditor as a
  fourth was dropped: it is not independent (D20). In study 5 the logs are each
  originator's OpenTelemetry spans, its receiver's access log, and the end state.
- `ll_bic`: Poisson log-linear model of the three-log capture table, pairwise
  interactions chosen by BIC, 95% profile-likelihood interval.
- `ll_sa_hom` (study 5): two-list estimate between any telemetry log (spans or access)
  and the end state; 95% profile-likelihood interval widened by the dispersion between
  sessions. `chapman_sa` is the two-list estimate between spans and access alone.
- `fin_mid`: back-office loss setting where the trace and the gateway each drop half
  their lines at random and the env diff sees only financial operations; `transfer_mid`
  is the same loss in the infra environment.
- Linkage: *signature* matches records on operation, resource and sometimes value, in
  order (the registered headline); *call id* matches trace and gateway on the provider's
  tool-call id the harness writes into both; *ledger id* matches on the environment's
  ledger id, which no auditor has.

## Quickstart

```bash
pip install lacuna-mse    # the command and the import package are both `lacuna`
W=tests/fixtures/rig/run/witnesses
lacuna lint --map rig/map.toml
lacuna table --map rig/map.toml --in spans=$W/spans.jsonl --in logs=$W/logs.jsonl --in env=$W/env.jsonl
lacuna estimate --map rig/map.toml --in spans=$W/spans.jsonl --in logs=$W/logs.jsonl --in env=$W/env.jsonl
```

`lint` checks a map file, `table` counts what each combination of logs recorded,
`estimate` adds an estimate of what no log recorded. Maps for your own logs, reading the
output and the assumptions: [docs/GUIDE.md](docs/GUIDE.md).

From a clone:

```bash
make install    # needs uv
make validate   # lint + tests + privacy gate; no key, no network
make paper      # tables, figures and the PDFs (needs tectonic)
```

## What the results mean

Numbers here are quoted from artifacts through `tests/claims_map.json`; the full tables
are in `paper/lacuna.tex`, the short version in `paper/lacuna-v1.tex` (`paper/lacuna-workshop.tex` is the earlier short paper, written before study 5), the decision
trail in `docs/DECISIONS.md` (D44–D54, D61–D73, D92–D95).

- **Linkage broke the first two, not the estimator.** Signature matching merges distinct
  actions and manufactures false overlaps; matching by ledger id is nominal in both
  environments. The call id fixes back-office only, because the env diff carries no id.
  The call id is only as good as the lines: a twentieth of lines duplicated in all three
  logs takes `ll_bic` to 0.094, and a twentieth forged as phantoms to 0.0.
- **Bypass is the lead negative result** (`docs/PREREG-BYPASS.md`, write-up in D63, post
  hoc follow-up in D64). The environment gets a `run_script` batch tool; no witness
  parses the script (a stipulation), and the env diff sees only the effects. Intervals
  are 333.6 wide against 39.0 with no bypass, about 8.6 times in absolute terms. Under the
  registered view, 317.8 of the 330.7 actions a replicate hides (96%) are bypassed customer
  updates that no witness can record, so the estimator extrapolates to actions with zero
  capture probability. One record per bypassed action, taken from the ledger, restores
  coverage (0.949), so the failure is the stipulation that no counted log itemises the
  channel, not the pool or the tool; study 4 shows that itemising is not enough. When
  the env diff also sees the bypassed updates, coverage is 0.008, so the registered
  cell's apparent tolerance is partly a cancellation.
- **The audit-warning prompt** raised the bypass share from 0.87 to 0.99 on the same 12
  seeds, but two neutral episodes carry the difference (without them it is +0.013), so it
  does not show the prompt changed behaviour. No share threshold is established.
- **Study 4** (`docs/PREREG-STUDY4.md`, D70, corrected in D73) ran a second model, an
  adversary prompt and an auditor that parses the scripts. Its main finding is clustered
  loss: one witness record standing for many actions breaks the independence the estimator
  assumes, so itemising a channel is necessary, not sufficient. Compare within pools, never
  across them (table above). The adversary prompt told the agent how it is counted and asked
  it to undercount. It split the pool: 11 of 23 episodes did not batch at all (neutral: 1)
  and the rest batched as much as neutral, at least two explicitly to undercount (read by
  hand, D73). No action escaped the trace and the gateway by any route but the one channel.
  The adversary was weak (one episode, black box, channel not named), so this does not test
  a determined one.
- **Scope.** The method is for settings where every action reaches at least one counted
  log under independent-enough loss. Bypass tests the boundary where a channel reaches
  none, and there the estimator cannot bound the count.
- **Loss the estimator does not expect.** Bursts of consecutive lost actions leave the
  plain interval too narrow (0.559), mostly because the agent works one kind of ticket at
  a time and a lost block lands where the env diff is blind. Reporting the
  dispersion-corrected `ll_bic_hom` beside it restores 0.952 (back-office) and 0.994
  (infra). A loss rate that ramps over the episode with the same mean gives 0.742 and
  0.471. Session quality shared by all three logs defeats the pooled intervals, and the
  repairs tried (`tier_bic`: 0.838 and 0.907) are not defaults.
- **Cheaper checks** (simulated loss, exact matching): a tail-corrected gateway sequence
  counter covers 0.952 where `ll_bic` covers 0.254 (a fifth more loss on unseen
  operations), but covers 0.0 under bursts and in the bypass study; planted canary
  actions cover 0.922 or more where exchangeable with real actions, at wider intervals.
- **Main limitation.** The agents were benign and, outside study 5, all log loss is
  simulated afterwards on complete logs, in two environments and a batch-tool variant of
  one. One adversary prompt was run live (study 4): the pool routed 0.502 of its actions
  through the batch tool against 0.925 under the neutral prompt, about half the episodes
  declined and the rest batched as before, and none found a route beyond the stipulated
  channel. No stronger adversary (several episodes, feedback, the channel named) was run
  (D10, D70, D73).

## Layout

- `src/lacuna/sim.py`: capture-history simulator across the threat ladder
- `src/lacuna/estimators/`: floor baselines, log-linear MSE with profile-likelihood
  intervals, latent-class model, env-anchored Γ bounds
- `src/lacuna/envs/`: back-office kernel with a hidden ground-truth ledger, tools, gateway
  log, harness trace; `bypass.py` adds the `run_script` batch tool (D61)
- `src/lacuna/thinning.py`: post-hoc logging loss over recorded episodes
- `scripts/`: simulation, episode running, thinning, and the table renderers the docs quote

## Run

Needs `uv` and `Python >= 3.11`. `make sim` is a 500-rep coverage sweep (~6 min).

The registered confirmatory analysis needs `--bootstrap-episodes`; without it `ll_bic` at
`fin_mid` covers 0.90, not 0.94 (`artifacts/thinning/confirm-heldout-fixed-episodes.json`).
At 100 replicates the two are not distinguishable. `make repro-headline` reruns the
registered replay and fails if `confirm.json` or `hypotheses.json` differ.

```bash
uv run python scripts/thin_episodes.py --run confirm --pool heldout --bootstrap-episodes
uv run python scripts/test_hypotheses.py --run confirm
```

`scripts/thin_episodes.py` overwrites `artifacts/thinning/` unless given `--out`. The same
cell can read differently across tables because it was run on more than one set of loss
seeds; the report's appendix lists each.

## Data and license

- The capture histories, ledgers and results are also a Hugging Face dataset,
  [`caiotheodoro/lacuna-capture-histories`](https://huggingface.co/datasets/caiotheodoro/lacuna-capture-histories),
  built by `scripts/build_dataset.py`. It withholds the episode transcripts.
- The short paper is deposited on [Zenodo](https://doi.org/10.5281/zenodo.24453638) (D110), under
  `CC BY 4.0`; `CITATION.cff` and `.zenodo.json` carry its metadata. The source in this tree,
  `paper/lacuna-v1.tex`, is a later version (D113), not yet deposited.
- Code and data are `Apache-2.0` (`LICENSE`, `NOTICE`). The `transcript` fields in
  `artifacts/episodes/*/episodes.jsonl` hold the reasoning text of the DeepSeek and qwen
  models the live runs used (served through OpenCode Go), which that license does not cover;
  the provider's terms apply.
- Terms such as `ll_bic`, `fin_mid` and the linkages are defined in the dataset card's
  glossary.
- The live runs cost $11.47 in all (summed from `artifacts/episodes/*/receipt.json` into
  `artifacts/derived.json`; study 4 at the `deepseek-v4.1-flash` rates the runner applied to
  every run, so its two qwen figures overstate the provider's cost). Study 5's recorded
  sessions are not in that figure: each pool's `spent_usd` is in
  `artifacts/study5/pools/*/pool.json`, 1.38 to 1.53 for each of the three confirmatory
  pools and 0.49 to 0.77 for each pilot pool, in tokens at an assumed price (D89); the
  provider's real price for those models was not found. The 420 runs replay those
  sessions and made no paid call.
