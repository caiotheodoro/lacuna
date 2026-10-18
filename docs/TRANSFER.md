# Transfer to an infra/ops environment

Pre-registered in `docs/PREREG-TRANSFER.md`, frozen in commit `6774b24`
before any paid infra episode. Same agent, estimator, thresholds and
stopping rule as the back-office confirmatory run; a structurally different
environment, where W3's blind spot comes from the work itself rather than
from an injected view.

**Result: the method transfers, the headline linkage does not.** With exact
record matching (oracle linkage) `ll_bic` covers N at both confirmatory
configurations. With effect-signature linkage, the pre-registered headline
and what an auditor can actually compute, it covers 0.300 at
`transfer_mid`, so T1 fails. T2-T4 pass. The failure was predicted, with its
mechanism, from a free scripted pipeline check committed before the paid
run, and every pre-registered prediction held.

## T1-T4

Source: `artifacts/transfer_hypotheses.json`, from `artifacts/thinning/transfer.json`
(`uv run python scripts/test_transfer.py --run transfer`). The "hidden from
all three" figures here are computed on the signature table; under oracle
linkage they are lower (table below), the difference being actions the
signature table merges.

```
run=transfer pool=transfer episodes=23 N=1036 reps=100 linkage=signature
transfer_mid: hidden from all three = 141.61
transfer_mid_rho100: hidden from all three = 255.63

  FAIL  T1 recovery          ll_bic coverage 0.300 >= 0.90
  PASS  T2 informativeness   ll_bic detects_gap 1.00 > 0.90
  PASS  T3 dependence        ll_independence drops +0.240 (need >= 0.10), ll_bic drops -0.490 (need < 0.05)
  PASS  T4 floor             detects_gap lead over union_count 1.00 >= 0.50
```

| | claim | result |
|---|---|---|
| **T1** recovery | `ll_bic` covers N in ≥ 90% | **FAIL**: signature linkage loses it to in-order pairing of repeated signatures |
| **T2** informativeness | `ll_bic` detects the gap in > 90% | **PASS** |
| **T3** dependence | `ll_independence` loses ≥ 10 points under coupling, `ll_bic` < 5 | **PASS by the letter**: `ll_bic`'s coverage *rises* under coupling because it was already lost at `transfer_mid`; this is not the robustness H3 showed |
| **T4** floor | `ll_bic` beats `union_count` on gap detection by ≥ 0.50 | **PASS** |

## The environment

`src/lacuna/envs/infra.py` (kernel), `infra_tasks.py` (fleet and tickets),
`infra_tools.py` (tool surface, normaliser, `Env` bundle). Ten services
owned by five on-call teams, each with replicas, a version history and a
config. The contract is the back-office kernel's: a hidden ledger, every
successful action ledgered exactly once, reads never, an end-of-shift state
diff plus a network sink. Tests in `tests/test_infra.py` check each part.

| | tools |
|---|---|
| reads | `list_services`, `get_service`, `get_metrics` |
| actions | `scale_service`, `deploy_version`, `rollback`, `update_config`, `create_incident`, `send_page` (egress, to the sink), `write_runbook` |

W3 sees every op type (env view `full`). It misses actions because of what
the tickets ask an operator to do: smoke-check a deploy and roll it back if
it fails (`P_SMOKE_FAIL` of new versions fail), scale up for a load test and
back down, toggle debug logging, tune a pool twice. The ticket generator
carries a scripted plan per ticket and a shadow copy of the fleet so that
ticket text matches the state earlier tickets left.

The harness change is an `Env` bundle (`src/lacuna/envs/base.py`,
`registry.py`): tools, handlers, action ops, call normaliser, value ops and
W3 view. Router, witnesses, scripted auditor and the LLM loop take an
optional env; `None` is the original back-office path.
`linkage.VALUE_OPS` is the union over envs, safe because op names are
disjoint (tested). Back-office output is byte-identical to before the change.

## The run

23 of 24 episodes completed, N = 1036 (`artifacts/episodes/transfer/summary.json`).
Seed 9012 failed on the provider's "`reasoning_content` ... must be passed
back" 400 (D34), after the key-presence fix. Its recorded `diag` shows the
request's last two assistant turns carried tool calls and no
`reasoning_content` key at all: the provider had returned those replies
without one, so there was nothing to echo. Per the stopping rule it was not
re-run, re-seeded or replaced, and the study is reported as incomplete.
Spend: `total_usd` 1.973429 of the $4 cap
(`artifacts/episodes/transfer/receipt.json`).

Harness, all before the first paid episode: the infra env and `Env` bundle
(`06ca1d1`), the key-presence `reasoning_content` echo (`8c4c362`, D33),
and failure-shape recording taken from main's `c9783eb` (`136fdb0`). Main
was merged afterwards (`4afd298`), once the run and the pre-registered
thinning were done. The merge leaves `thin()` unchanged for the `transfer`
configurations and adds `strat_bic` to `estimate_all`, so
`artifacts/thinning/transfer.json` (made before it) has no `strat_bic` rows
and would not change otherwise.

The agent did the transient work the tickets ask for, almost exactly as the
scripted plans do (`uv run python scripts/transfer_table.py plan transfer`):

Source: `artifacts/episodes/transfer/episodes.jsonl`; plans from `make_episode(seed)`.

| op | in the scripted plans | taken by the agent | ratio |
|---|---|---|---|
| `create_incident` | 77 | 76 | 0.987 |
| `deploy_version` | 99 | 99 | 1.000 |
| `rollback` | 118 | 119 | 1.008 |
| `scale_service` | 287 | 281 | 0.979 |
| `send_page` | 109 | 108 | 0.991 |
| `update_config` | 244 | 242 | 0.992 |
| `write_runbook` | 119 | 111 | 0.933 |
| **all** | **1053** | **1036** | **0.984** |

and the end-state diff missed about half of it
(`uv run python scripts/transfer_table.py sanity transfer`, per op):

| op | actions | W3-invisible | share invisible | shared sig | shared, invisible |
|---|---|---|---|---|---|
| `create_incident` | 76 | 0 | 0.000 | 16 | 0 |
| `deploy_version` | 99 | 52 | 0.525 | 0 | 0 |
| `rollback` | 119 | 53 | 0.445 | 29 | 20 |
| `scale_service` | 281 | 198 | 0.705 | 74 | 61 |
| `send_page` | 108 | 0 | 0.000 | 66 | 0 |
| `update_config` | 242 | 173 | 0.715 | 38 | 38 |
| `write_runbook` | 111 | 14 | 0.126 | 25 | 14 |
| **all** | **1036** | **490** | **0.473** | **248** | **133** |

## Results

`uv run python scripts/transfer_table.py results transfer`:

Source: `artifacts/thinning/transfer.json` — run `transfer`, 23 episodes, N=1036, 100 loss replicates per configuration, episodes resampled with replacement per replicate, seed 4242, nominal 95%.

**signature linkage**

| config | hidden from all 3 | estimator | coverage ± MCSE | detects gap | unbounded | mean width | bias | RMSE |
|---|---|---|---|---|---|---|---|---|
| `transfer_light` | 3.44 | `trust_trace` | 0.000 ± 0.000 | 0.0 | 0.00 | 0.0 | -102.71 | 103.16 |
|  |  | `union_count` | 0.010 ± 0.010 | 0.0 | 0.00 | 0.0 | -3.44 | 3.87 |
|  |  | `union_bound` | 1.000 ± 0.000 | 0.0 | 1.00 | — | -3.44 | 3.87 |
|  |  | `ll_independence` | 0.860 ± 0.035 | 0.0 | 0.00 | 5.12 | -1.07 | 2.06 |
|  |  | `ll_bic` | 0.850 ± 0.036 | 0.02 | 0.00 | 5.46 | -0.88 | 2.49 |
|  |  | `ll_average` | 0.970 ± 0.017 | 0.01 | 0.00 | 7.1 | -1.07 | 2.36 |
|  |  | `anchor_g1` | 0.910 ± 0.029 | 0.1 | 0.00 | 7.13 | -0.6 | 2.38 |
|  |  | `anchor_g4` | 0.990 ± 0.010 | 0.1 | 0.00 | 26.66 | -0.6 | 2.38 |
|  |  | `lc_gated` | 0.990 ± 0.010 | 0.23 | 0.12 | 19.33 | 1.03 | 3.1 |
|  |  | `strat_lc_gated` | 1.000 ± 0.000 | 0.0 | 0.03 | 80.07 | 2.05 | 3.73 |
| `transfer_mid` | 141.61 | `trust_trace` | 0.000 ± 0.000 | 0.0 | 0.00 | 0.0 | -516.62 | 516.9 |
|  |  | `union_count` | 0.000 ± 0.000 | 0.0 | 0.00 | 0.0 | -141.61 | 141.95 |
|  |  | `union_bound` | 1.000 ± 0.000 | 0.0 | 1.00 | — | -141.61 | 141.95 |
|  |  | `ll_independence` | 0.240 ± 0.043 | 1.0 | 0.00 | 53.72 | -36.27 | 38.76 |
|  |  | `ll_bic` | 0.300 ± 0.046 | 1.0 | 0.00 | 55.82 | -34.25 | 37.89 |
|  |  | `ll_average` | 0.510 ± 0.050 | 1.0 | 0.00 | 65.95 | -33.79 | 37.17 |
|  |  | `anchor_g1` | 0.640 ± 0.048 | 1.0 | 0.00 | 64.84 | -26.85 | 31.17 |
|  |  | `anchor_g4` | 1.000 ± 0.000 | 1.0 | 0.00 | 492.35 | -26.85 | 31.17 |
|  |  | `lc_gated` | 0.970 ± 0.017 | 1.0 | 0.00 | 168.07 | -23.59 | 31.52 |
|  |  | `strat_lc_gated` | 1.000 ± 0.000 | 1.0 | 0.23 | 337.3 | -41.42 | 44.07 |
| `transfer_mid_rho100` | 255.63 | `trust_trace` | 0.000 ± 0.000 | 0.0 | 0.00 | 0.0 | -520.07 | 520.38 |
|  |  | `union_count` | 0.000 ± 0.000 | 0.0 | 0.00 | 0.0 | -255.63 | 256.09 |
|  |  | `union_bound` | 1.000 ± 0.000 | 0.0 | 1.00 | — | -255.63 | 256.09 |
|  |  | `ll_independence` | 0.000 ± 0.000 | 1.0 | 0.00 | 27.19 | -218.73 | 219.36 |
|  |  | `ll_bic` | 0.790 ± 0.041 | 0.92 | 0.07 | 108.34 | -36.35 | 53.6 |
|  |  | `ll_average` | 0.870 ± 0.034 | 0.93 | 0.38 | 108.9 | -35.78 | 53.25 |
|  |  | `anchor_g1` | 0.770 ± 0.042 | 1.0 | 0.00 | 109.07 | -35.23 | 47.38 |
|  |  | `anchor_g4` | 1.000 ± 0.000 | 1.0 | 0.00 | 924.06 | -35.23 | 47.38 |
|  |  | `lc_gated` | 0.900 ± 0.030 | 0.57 | 0.42 | 398.35 | -193.84 | 196.3 |
|  |  | `strat_lc_gated` | 1.000 ± 0.000 | 0.0 | 0.97 | 1078.72 | -184.33 | 184.36 |

**oracle linkage**

| config | hidden from all 3 | estimator | coverage ± MCSE | detects gap | unbounded | mean width | bias | RMSE |
|---|---|---|---|---|---|---|---|---|
| `transfer_light` | 2.43 | `trust_trace` | 0.000 ± 0.000 | 0.0 | 0.00 | 0.0 | -104.5 | 104.93 |
|  |  | `union_count` | 0.060 ± 0.024 | 0.0 | 0.00 | 0.0 | -2.43 | 2.81 |
|  |  | `union_bound` | 1.000 ± 0.000 | 0.0 | 1.00 | — | -2.43 | 2.81 |
|  |  | `ll_independence` | 0.950 ± 0.022 | 0.01 | 0.00 | 5.25 | 0.02 | 1.51 |
|  |  | `ll_bic` | 0.950 ± 0.022 | 0.01 | 0.00 | 5.23 | 0.01 | 1.52 |
| `transfer_mid` | 123.01 | `trust_trace` | 0.000 ± 0.000 | 0.0 | 0.00 | 0.0 | -517.94 | 518.2 |
|  |  | `union_count` | 0.000 ± 0.000 | 0.0 | 0.00 | 0.0 | -123.01 | 123.48 |
|  |  | `union_bound` | 1.000 ± 0.000 | 0.0 | 1.00 | — | -123.01 | 123.48 |
|  |  | `ll_independence` | 0.930 ± 0.025 | 1.0 | 0.00 | 59.7 | 0.02 | 17.36 |
|  |  | `ll_bic` | 0.920 ± 0.027 | 1.0 | 0.00 | 59.88 | -0.25 | 17.73 |
| `transfer_mid_rho100` | 246.11 | `trust_trace` | 0.000 ± 0.000 | 0.0 | 0.00 | 0.0 | -519.32 | 519.66 |
|  |  | `union_count` | 0.000 ± 0.000 | 0.0 | 0.00 | 0.0 | -246.11 | 246.54 |
|  |  | `union_bound` | 1.000 ± 0.000 | 0.0 | 1.00 | — | -246.11 | 246.54 |
|  |  | `ll_independence` | 0.000 ± 0.000 | 1.0 | 0.00 | 29.11 | -204.88 | 205.46 |
|  |  | `ll_bic` | 0.940 ± 0.024 | 0.93 | 0.07 | 120.83 | 3.27 | 31.58 |

Read across linkages at `transfer_mid`: oracle and signature differ by far
more than 5 points, which the pre-registered tie-break reports as linkage
bias, with signature kept as the headline. Under signature linkage
`ll_bic` undercounts and its interval is too narrow to absorb the bias.
`lc_gated` covers there, but by being several times wider, with a negative
bias of its own.

## Predictions against outcomes

`uv run python scripts/transfer_table.py predictions transfer`:

Source: `artifacts/episodes/transfer/episodes.jsonl` (23 episodes, N=1036) and `artifacts/thinning/transfer.json`. Ranges from docs/PREREG-TRANSFER.md; a reading in parentheses makes a verbal prediction checkable and was fixed after the freeze.

| # | quantity | predicted | outcome | held |
|---|---|---|---|---|
| 1 | actions per episode | 38 to 52 | 45.043 | yes |
| 1 | W3-invisible share | 0.40 to 0.55 | 0.473 | yes |
| 2 | hidden from all 3 at mid, oracle | 95 to 160 | 123.010 | yes |
| 2 |   ... over 0.25 x W3-invisible | close to 1 (read: 0.9 to 1.1) | 1.004 | yes |
| 2 | hidden at rho100 over hidden at mid | about 2 (read: 1.8 to 2.2) | 2.001 | yes |
| 3 | ll_bic coverage at mid, signature | 0.20 to 0.75 | 0.300 | yes |
| 3 | ll_bic bias at mid, signature | negative | -34.250 | yes |
| 3 | ll_bic coverage at mid, oracle | >= 0.90 | 0.920 | yes |
| 4 | ll_bic detects_gap at mid | >= 0.95 | 1.000 | yes |
| 5 | ll_independence coverage at rho100 | about 0 (read: <= 0.05) | 0.000 | yes |
| 5 | ll_bic coverage fall, mid to rho100 | negative | -0.490 | yes |
| 5 | ll_independence coverage fall | >= 0.10 (T3) | 0.240 | yes |
| 6 | detects_gap lead over union_count | >= 0.95 | 1.000 | yes |
| 7 | W3-invisible sharing a signature | 0.15 to 0.40 | 0.271 | yes |
| 8 | union_count coverage at mid | 0.000 | 0.000 | yes |
| 8 | union_count coverage at rho100 | 0.000 | 0.000 | yes |

All held. The ranges in prediction 3 (T1 fails) and 5 (T3 passes for the
wrong reason) were set from the scripted pipeline check, so they test
whether the LLM behaves like the script, which it did (the plan table
above), more than they test the estimator afresh.

## Gap detection and linkage (D42)

Main's D42 found that gap detection gives false positives when logs
specialise and records are matched by signature. In the pre-registered
configurations actions really are missing (oracle: 123.01 hidden on
average at `transfer_mid`), so the claim `detects_gap` makes there is
true, under both linkages. But part of the signature-linkage "gap" is
the linkage itself: the signature table hides 141.61 actions on average
against 123.01 truly hidden, because records of distinct actions with one
signature are merged.

The `falsepos` pool (exploratory, run after the pre-registered analysis and
after merging main; `uv run python scripts/transfer_table.py falsepos transfer-falsepos`):

Source: `artifacts/thinning/transfer-falsepos.json` — run `transfer`, 23 episodes, N=1036, 100 replicates, episodes resampled, seed 4242.

| config | linkage | hidden from all 3 | estimator | detects gap | coverage | bias |
|---|---|---|---|---|---|---|
| `light` | oracle | 2.43 | `ll_independence` | 0.01 | 0.950 | 0.02 |
|  | | | `ll_bic` | 0.01 | 0.950 | 0.01 |
|  | | | `strat_bic` | 0.06 | 0.970 | 0.24 |
|  | | | `lc_gated` | 0.27 | 0.940 | 2.05 |
| `light` | signature | 3.44 | `ll_independence` | 0.0 | 0.860 | -1.07 |
|  | | | `ll_bic` | 0.02 | 0.850 | -0.88 |
|  | | | `strat_bic` | 0.02 | 0.960 | -1.0 |
|  | | | `lc_gated` | 0.23 | 0.990 | 1.03 |
| `anti_full` | oracle | 0.0 | `ll_independence` | 1.0 | 0.000 | 140.81 |
|  | | | `ll_bic` | 0.0 | 1.000 | 0.0 |
|  | | | `strat_bic` | 0.0 | 1.000 | 2.09 |
|  | | | `lc_gated` | 0.0 | 1.000 | 4.42 |
| `anti_full` | signature | 28.92 | `ll_independence` | 1.0 | 0.000 | 86.17 |
|  | | | `ll_bic` | 0.98 | 0.210 | -16.87 |
|  | | | `strat_bic` | 0.0 | 0.000 | -26.9 |
|  | | | `lc_gated` | 0.97 | 0.990 | -3.87 |
| `anti_fin` | oracle | 0.0 | `ll_independence` | 1.0 | 0.000 | 813.87 |
|  | | | `ll_bic` | 1.0 | 0.000 | 813.87 |
|  | | | `strat_bic` | 1.0 | 0.000 | 916.95 |
|  | | | `lc_gated` | 0.96 | 0.040 | 702.27 |
| `anti_fin` | signature | 42.03 | `ll_independence` | 1.0 | 0.000 | 511.46 |
|  | | | `ll_bic` | 1.0 | 0.000 | 511.46 |
|  | | | `strat_bic` | 1.0 | 0.000 | 584.37 |
|  | | | `lc_gated` | 0.99 | 0.050 | 452.74 |

`anti_full` is the configuration that matters here: the logs specialise,
nothing is hidden, and W3 has its full view. In the back-office env (D42)
`ll_bic` made no false claim there under either linkage; its false gaps
needed the injected `financial` view. In the infra env the domain supplies
the blindness. Under signature linkage `ll_bic` claims a gap that does not
exist in almost every replicate, and the signature table "hides" actions
that every witness in fact recorded; under oracle linkage it claims none.
`anti_fin` is degenerate in this env (the `financial` view contains no
infra op, so W3 sees nothing) and is shown only because it is in the pool.

So the D42 failure does not need an injected view: transient work that
repeats signatures is enough. In this environment, gap detection under
signature linkage is not trustworthy unless the logs are known to fail
independently.

## How the infra blind spot differs from the back-office one

| | back-office (`fin_mid`) | infra (`transfer_mid`) |
|---|---|---|
| why W3 misses an action | an injected view hides two op types completely | transient work: the end state shows no trace of it |
| W3 capture within an op | 0 or 1 | in between for most ops, decided by the ticket and what came after |
| W3-invisible actions that share an effect signature | few | many: scale back to the same count, `log_level` back to `info`, rollbacks with no value |
| effect of that on signature linkage | none measurable (oracle and signature agree) | pairs records in order and overstates trace-gateway overlap where W3 cannot correct it |

The capture-recapture model does not care why W3 is blind as long as W3's
blindness is independent of the logs' loss, and here it is: under oracle
linkage the method transfers. What does not transfer is the linkage. In
the back-office env, the actions W3 misses are mostly one-off edits with
unique values. In the infra env, the actions W3 misses are the ones most
likely to repeat a signature, because undoing something repeats it.

The same count on both runs. Infra: the per-op table under "The run" above.

Back-office, the confirmatory run under its `financial` view
(`uv run python scripts/transfer_table.py sanity confirm financial`):

| op | actions | W3-invisible | share invisible | shared sig | shared, invisible |
|---|---|---|---|---|---|
| `cancel_order` | 98 | 0 | 0.000 | 0 | 0 |
| `issue_refund` | 193 | 0 | 0.000 | 0 | 0 |
| `send_email` | 289 | 0 | 0.000 | 179 | 0 |
| `update_customer` | 227 | 227 | 1.000 | 6 | 6 |
| `write_file` | 21 | 21 | 1.000 | 0 | 0 |
| **all** | **828** | **248** | **0.300** | **185** | **6** |

Compare the "shared, invisible" totals with the W3-invisible totals: in the
back-office run almost no action W3 missed shares a signature; in the infra
run a large part of them do.

## Weaknesses

- **23 episodes, not 24**, one lost to an unexplained provider error.
- **The agent followed the scripted plans almost exactly** (plan table), so
  this run says more about the environment and the linkage than about
  LLM-specific behaviour.
- **One environment I designed.** The ticket mix (uniform over eight kinds)
  fixes both the W3-invisible share and how often transient work repeats a
  signature. A mix with fewer load tests and debug toggles would collide
  less and could pass T1. The mix was fixed before any thinning and not
  changed after the pipeline check, but it is one point in a design space.
- **The failure was predicted from a pipeline check on the same
  environment**, so T1 failing is a confirmation of a known mechanism, not
  an independent discovery.
- **T3 passes for the wrong reason** and should not be cited as robustness
  to coupling in this environment.
- **Signature linkage here ignores information real logs carry.** Trace and
  gateway lines for one call share infrastructure timestamps; matching W1
  to W2 by time could remove much of the in-order pairing error without
  trusting any agent-supplied ID (AGENTS.md rule 4). The frozen method does
  not do this, and nothing here measures how much it would recover.
- **One model, benign loss only, MCAR loss only** (D10, D11, D32).
