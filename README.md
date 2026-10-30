# lacuna

How many consequential actions did an AI agent take that no log shows? lacuna
estimates that count with capture-recapture (the method for counting casualties nobody
recorded): each log sees part of what the agent did, and the overlaps between logs
estimate what none of them saw.

## Status: what holds

*Coverage* is the share of simulated replicates whose 95% interval contains the true
count; the bar is 0.90.

| Claim | Evidence | Status |
|---|---|---|
| Counting unlogged actions works in the back-office environment | Pre-registered. Passed at its registered loss seed (coverage 0.94) but covers 0.893 over 3000 replicates, below the 0.90 bar | scoped |
| The registered record matching transfers to a second (infra) environment | Pre-registered, predicted to fail. T1 failed: 0.3 at the registered seed, 0.401 over 20 seeds. Linking the trace and the gateway by a shared call id lifts back-office to 0.946 and infra to 0.859 | fail |
| The count is bounded when a channel is itemised by no counted log | Pre-registered bypass study. The agent ran 0.856 of its actions through a batch tool that the trace and the gateway log as one call (predicted 0.05 to 0.30). Coverage 0.759 against 0.946 with no bypass. Post hoc, parsing the scripts restores 0.949 | fail |
| The estimator detects that the logs are incomplete | Detects a gap in 0.995 of bypass replicates (also 1.0 with no bypass, so it does not say which channel is unseen). Counting the union of the logs covers 0.0 | pass |
| It survives loss that depends on the action, comes in bursts or drifts | Fails silently: 0.647 at a tenth more loss on the operations the env diff cannot see, 0.187 at a fifth more; 0.559 under bursts; 0.742 under a drifting loss rate | fail |

The thesis these rows support: capture-recapture bounds the unlogged actions when every
channel is itemised by at least two logs; when a channel is itemised by none, no
log-based count can see it, and only design review, not the estimator, detects that.

Terms:
- Witnesses: W1 `trace` (the agent's own log), W2 `gateway` (calls through an
  interception proxy), W3 `env` (state diff plus a network sink). An LLM auditor as a
  fourth was dropped: it is not independent (D20).
- `ll_bic`: Poisson log-linear model of the three-log capture table, pairwise
  interactions chosen by BIC, 95% profile-likelihood interval.
- `fin_mid`: back-office loss setting where the trace and the gateway each drop half
  their lines at random and the env diff sees only financial operations; `transfer_mid`
  is the same loss in the infra environment.
- Linkage: *signature* matches records on operation, resource and sometimes value, in
  order (the registered headline); *call id* matches trace and gateway on the provider's
  tool-call id the harness writes into both; *ledger id* matches on the environment's
  ledger id, which no auditor has.

## Quickstart

```bash
make install    # needs uv
make validate   # lint + tests + privacy gate; no key, no network
make paper      # tables, figures and both PDFs (needs tectonic)
```

## What the results mean

Numbers here are quoted from artifacts through `tests/claims_map.json`; the full tables
are in `paper/lacuna.tex`, the short version in `paper/lacuna-workshop.tex`, the decision
trail in `docs/DECISIONS.md` (D44–D54, D61–D64).

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
  capture probability. Parsing the scripts restores coverage (0.949), so the failure is
  the stipulation that no counted log itemises the channel, not the pool or the tool. When
  the env diff also sees the bypassed updates, coverage is 0.008, so the registered
  cell's apparent tolerance is partly a cancellation.
- **The audit-warning prompt** raised the bypass share from 0.87 to 0.99 on the same 12
  seeds, but two neutral episodes carry the difference (without them it is +0.013), so it
  does not show the prompt changed behaviour. No share threshold is established.
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
- **Main limitation.** The agents were benign and all log loss is simulated afterwards on
  complete logs, in two environments and a batch-tool variant of one. Nothing here tests
  an agent that hides on purpose. No adversarial rung has been run live (D10).

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
- Code and data are `Apache-2.0` (`LICENSE`, `NOTICE`). The `transcript` fields in
  `artifacts/episodes/*/episodes.jsonl` hold the reasoning text of the DeepSeek models the
  live runs used (served through OpenCode Go), which that license does not cover; the
  provider's terms apply.
- Terms such as `ll_bic`, `fin_mid` and the linkages are defined in the dataset card's
  glossary.
