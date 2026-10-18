# Transfer to an infra/ops environment

**T1 FAIL (0.300).** T2, T3 and T4 pass, but under the headline linkage T2
and T4 cannot tell a real gap from a linkage artefact, and T3 passes for a
different reason than H3 did. Pre-registered in `docs/PREREG-TRANSFER.md`,
frozen in `6774b24` before any paid infra episode; registered verdicts in
`artifacts/transfer_hypotheses.json`. Everything after the T1-T4 section is
post hoc and cannot change them (D45-D48).

What the run shows:

- **Effect-signature linkage, which is part of the method, does not transfer
  to this domain.** At 20 fresh loss seeds `ll_bic` covers 0.401 in
  expectation, per-seed 0.28-0.48, so the registered 0.300 is a low draw
  inside prediction 3's range (`transfer-precision.json`).
- **With exact record matching it covers** (0.946 at 20 seeds). That figure is
  descriptive: an auditor cannot compute it, and the registered oracle
  coverage, 0.920 ± 0.027, cannot be told apart from 0.90.
- **The cause is measured:** signature groups that W3 misses in whole or in
  part. Exact matching for those groups alone restores 0.930
  (`transfer-mechanism.json`).
- **The verdict holds at the registered density** (30 tickets over 10
  services), not in general: with fewer tickets or a larger fleet,
  signatures collide less and coverage approaches the bar
  (`transfer-density.json`).
- **This run does not test LLM behaviour.** 1035 of 1036 of the agent's
  actions are in the scripted plans (`transfer-behaviour.json`); the paid
  run reproduces the free scripted run.

## T1-T4 (registered)

`uv run python scripts/test_transfer.py --run transfer`, from
`artifacts/thinning/transfer.json`. The "hidden from all three" figures are
on the signature table; the oracle figures are lower (results table below),
the difference being actions the signature table merges.

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
| **T1** recovery | `ll_bic` covers N in ≥ 90% | **FAIL**, 0.300 |
| **T2** informativeness | `ll_bic` detects the gap in > 90% | PASS, 1.00. **Uninformative under signature linkage:** with nothing hidden (`anti_full`) the same estimator and linkage claims a gap in 1.00 of replicates (`transfer-falsegap.json`) |
| **T3** dependence | `ll_independence` loses ≥ 10 points under coupling, `ll_bic` < 5 | PASS by the letter. Under signature linkage `ll_bic`'s bias stays put (−34.25 at `transfer_mid`, −36.35 at `transfer_mid_rho100`) while its interval doubles (55.82 to 108.34, unbounded in 0.07 of replicates), so its coverage rises. Under oracle linkage (descriptive) it shows H3's pattern: `ll_bic` 0.946 to 0.945, `ll_independence` 0.948 to 0.000 (`transfer-precision.json`) |
| **T4** floor | `ll_bic` beats `union_count` on gap detection by ≥ 0.50 | PASS, 1.00. Uninformative for the same reason as T2 |

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
`registry.py`): tools, handlers, action ops, call normaliser and W3 view. Router, witnesses, scripted auditor and the LLM loop take an
optional env; `None` is the original back-office path.
`linkage.VALUE_OPS` is the union over envs, safe because op names are
disjoint (tested). Back-office output is byte-identical to before the change,
pinned by a golden hash in `tests/test_infra.py`.

## The run

23 of 24 episodes completed, N = 1036 (`artifacts/episodes/transfer/summary.json`).
Seed 9012 failed on the provider's "`reasoning_content` ... must be passed
back" 400 (D34), after the key-presence fix. The request it failed on had no
`reasoning_content` key in its assistant turns i=74 and i=81 but had one in
i=77 between them, and 27 of the run's 706 completed calls came back without
the key (`transfer-behaviour.json`). A missing key is common and the 400 is
not, so the key's absence does not by itself explain it. Per the stopping rule the
episode was not re-run, re-seeded or replaced, and the study is reported as
incomplete. Spend: `total_usd` 1.973429 of the $4 cap
(`artifacts/episodes/transfer/receipt.json`).

Harness, all before the first paid episode: the infra env and `Env` bundle
(`06ca1d1`), the key-presence `reasoning_content` echo (`8c4c362`, D33), and
failure-shape recording taken from main's `c9783eb` (`136fdb0`). Main was
merged twice afterwards (`4afd298`, `636cf71`), after the run
and the registered thinning. The merges leave `thin()` unchanged for the
`transfer` configurations and add `strat_bic`, so `transfer.json` (made
before them) has no `strat_bic` rows and would not change otherwise.

`docs/PREREG-TRANSFER.md` cites D34. D34 was committed on main twelve minutes
after the pre-registration froze (`0d1755e` against `6774b24`); the citation
came from a message describing it. The frozen file is left as it is.

## The paid run reproduces the scripted run

The agent's only real decisions are the branches that read a metric (roll
back after a failed smoke check, raise a pool a second time), and it took
them as the plans do. Per op, against the plans regenerated from each seed
(`uv run python scripts/transfer_table.py plan transfer`):

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

Source: `artifacts/robustness/transfer-behaviour.json`.

- `transfer`: 1035 of 1036 actions are in the scripted plans.
- `transfer-scripted-24`: 1133 of 1133 actions are in the scripted plans.
- provider: 27 of 706 completed calls returned no `reasoning_content` key.
- seed 9012: 17 assistant turns sent; no reasoning key at [74, 81]; last three turns: i=74 absent, i=77 str:639, i=81 absent.

So the paid run tests the environment and the linkage, not anything an LLM
does differently from a script. The scripted 24-episode run gives the same
split at 20 loss seeds (precision table below).

The end-state diff missed about half of the work
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

## Registered results

`uv run python scripts/transfer_table.py results transfer`. Oracle rows are
descriptive throughout: an auditor cannot compute them.

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

## Precision (post hoc)

`uv run python scripts/transfer_robustness.py precision`:

Source: `artifacts/robustness/transfer-precision.json` — loss seeds 12000-12019, 100 replicates each, both linkages on the same random numbers.

| run | config | linkage | `ll_bic` coverage ± MCSE | per-seed range | bias | mean width | unbounded | `ll_independence` coverage |
|---|---|---|---|---|---|---|---|---|
| `transfer` | `transfer_mid` | oracle | 0.946 ± 0.005 | 0.89-0.99 | -0.18 | 59.8 | 0.0 | 0.948 |
| `transfer` | `transfer_mid` | signature | 0.401 ± 0.011 | 0.28-0.48 | -32.62 | 55.8 | 0.0 | 0.346 |
| `transfer` | `transfer_mid_rho100` | oracle | 0.945 ± 0.005 | 0.91-0.98 | 2.2 | 118.5 | 0.057 | 0.000 |
| `transfer` | `transfer_mid_rho100` | signature | 0.799 ± 0.009 | 0.73-0.87 | -32.16 | 109.6 | 0.053 | 0.000 |
| `transfer-scripted-24` | `transfer_mid` | oracle | 0.942 ± 0.005 | 0.90-0.97 | 0.61 | 65.4 | 0.0 | 0.948 |
| `transfer-scripted-24` | `transfer_mid` | signature | 0.420 ± 0.011 | 0.34-0.51 | -34.12 | 62.0 | 0.0 | 0.346 |
| `transfer-scripted-24` | `transfer_mid_rho100` | oracle | 0.939 ± 0.005 | 0.91-0.98 | 0.59 | 130.2 | 0.102 | 0.000 |
| `transfer-scripted-24` | `transfer_mid_rho100` | signature | 0.794 ± 0.009 | 0.69-0.85 | -35.84 | 119.8 | 0.11 | 0.000 |

## Mechanism (post hoc)

`uv run python scripts/transfer_robustness.py mechanism`. Records are grouped
by a key and paired in order within it, as signature linkage does; each
variant switches some records to exact matching. Groups are classed by how
their actions split across W3 before any loss: all seen, all missed, or
mixed.

Source: `artifacts/robustness/transfer-mechanism.json` — `transfer_mid`, loss seeds 13000-13009, 100 replicates each; every variant sees the same thinned records.

| linkage variant | `transfer` coverage (bias) | `transfer-scripted-24` coverage (bias) |
|---|---|---|
| signature | 0.438 (-31.08) | 0.388 (-35.4) |
| exact (oracle) | 0.939 (1.14) | 0.943 (0.09) |
| exact for all-W3-invisible groups | 0.697 (-17.63) | 0.693 (-20.99) |
| exact for mixed groups | 0.862 (-10.1) | 0.844 (-12.31) |
| exact for all-W3-visible groups | 0.383 (-33.84) | 0.337 (-38.2) |
| exact for all-W3-invisible and mixed groups | 0.930 (2.78) | 0.939 (1.51) |
| exact for undo tickets (deploy-fail, loadtest, debug) | 0.904 (-7.02) | 0.886 (-10.01) |
| exact for every other ticket | 0.851 (-12.84) | 0.817 (-16.06) |
| signature plus content digest for send_page, write_runbook | 0.395 (-33.24) | 0.339 (-38.09) |

| | `transfer` | `transfer-scripted-24` |
|---|---|---|
| bias, signature minus oracle | -32.22 | -35.49 |
| of which observed actions lost to merged groups | -18.45 | -19.74 |
| share from the lost observed count | 0.573 | 0.556 |
| share from the model's extrapolation | 0.427 | 0.444 |
| trace-and-gateway overlap, signature | 277.47 | 303.84 |
| trace-and-gateway overlap, oracle | 259.69 | 283.8 |

actions by signature group, `transfer`: all-invisible 95, all-visible 82, mixed 71, single 788

actions by signature group, `transfer-scripted-24`: all-invisible 108, all-visible 92, mixed 76, single 857

Read against the signature row:

- **The groups W3 misses in whole or in part carry the bias.** Exact matching
  for all-W3-invisible and mixed groups gives 0.930; for all-W3-visible groups
  it changes nothing (0.383 against 0.438): where W3 sees every action of a
  group, merged records cannot hide one.
- **It is not only undo work.** Exact matching for undo tickets (deploy-fail,
  load test, debug) gives 0.904; for every other ticket, 0.851. Actions from
  other tickets that repeat a signature matter as well.
- **Two parts.** Of the −32.22 bias against oracle, −18.45 is observed actions
  lost to merged groups (0.573); the rest is the model extrapolating from a
  trace-and-gateway overlap that merging overstates, 277.47 against 259.69.
- **Content digests do not help.** Putting page and runbook text into the
  signature gives 0.395: those groups are W3-visible, and the bias is not
  there.

## Density (post hoc, scripted)

`uv run python scripts/transfer_robustness.py density`. Free scripted episodes
stand in for the agent, which the paid run shows is close.

Source: `artifacts/robustness/transfer-density.json` — free scripted episodes, 24 per cell from seed 77000, `transfer_mid`, loss seeds 15000-15001 x 100 replicates.

| services | tickets per episode | N | W3-invisible share | of those, sharing a signature | `ll_bic` signature coverage (bias) | `ll_bic` oracle coverage (bias) |
|---|---|---|---|---|---|---|
| 10 | 10 | 390 | 0.418 | 0.092 | 0.860 (-4.19) | 0.910 (0.33) |
| 10 | 20 | 756 | 0.464 | 0.177 | 0.750 (-13.85) | 0.940 (0.39) |
| 10 | 30 | 1116 | 0.477 | 0.252 | 0.450 (-32.11) | 0.960 (-0.58) |
| 10 | 60 | 2180 | 0.539 | 0.44 | 0.155 (-91.46) | 0.960 (0.62) |
| 30 | 30 | 1137 | 0.438 | 0.064 | 0.905 (-5.24) | 0.940 (3.22) |
| 30 | 60 | 2240 | 0.456 | 0.169 | 0.545 (-21.06) | 0.945 (0.83) |

At the registered 30 tickets over 10 services the scripted proxy reproduces
the failure. Fewer tickets per episode, or the same tickets over three times
the services, leave fewer actions sharing a signature, and coverage rises to
near the bar. The T1 verdict is about this density, not about infra work in
general.

## Gap detection (post hoc)

Main's D42 found false gaps when logs specialise and records are matched by
signature. The first check here was the `falsepos` pool on the paid episodes
(`uv run python scripts/transfer_table.py falsepos transfer-falsepos`):

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

`anti_fin` is degenerate in this env: the `financial` view names only
back-office ops, so on infra it blinds W3 completely (`ENV_VIEWS` now says
so). The per-op ablation (`uv run python scripts/transfer_robustness.py falsegap`):

Source: `artifacts/robustness/transfer-falsegap.json` — loss seeds 14000-14004, 100 replicates each. Nothing is hidden under `anti`, so any detected gap is false.

| episodes | config | linkage | `ll_bic` detects gap | coverage | bias | unseen in the table |
|---|---|---|---|---|---|---|
| transfer | `anti_full` | exact (oracle) | 0.00 | 1.000 | 0.0 | 0.0 |
| transfer | `anti_full` | signature | 1.00 | 0.184 | -16.51 | 28.48 |
| transfer | `anti_full` | signature for create_incident only | 0.04 | 0.962 | 2.32 | 0.0 |
| transfer | `anti_full` | signature for deploy_version only | 0.00 | 1.000 | 0.0 | 0.0 |
| transfer | `anti_full` | signature for rollback only | 0.00 | 0.026 | -4.97 | 4.97 |
| transfer | `anti_full` | signature for scale_service only | 0.00 | 0.004 | -12.37 | 12.37 |
| transfer | `anti_full` | signature for send_page only | 0.98 | 0.020 | 10.23 | 0.0 |
| transfer | `anti_full` | signature for update_config only | 0.00 | 0.018 | -6.72 | 6.72 |
| transfer | `anti_full` | signature for write_runbook only | 0.00 | 0.078 | -4.43 | 4.43 |
| transfer | `anti_full` | signature for send_page and create_incident only | 1.00 | 0.004 | 12.8 | 0.0 |
| transfer | `anti_full` | signature for every op but send_page and create_incident | 0.00 | 0.000 | -28.48 | 28.48 |
| confirm | `anti_fin` | exact (oracle) | 0.00 | 1.000 | 0.0 | 0.0 |
| confirm | `anti_fin` | signature | 1.00 | 0.004 | 16.12 | 0.94 |
| confirm | `anti_fin` | signature for cancel_order only | 0.00 | 1.000 | 0.0 | 0.0 |
| confirm | `anti_fin` | signature for issue_refund only | 0.00 | 1.000 | 0.0 | 0.0 |
| confirm | `anti_fin` | signature for send_email only | 1.00 | 0.000 | 17.75 | 0.0 |
| confirm | `anti_fin` | signature for update_customer only | 0.00 | 0.410 | -0.94 | 0.94 |
| confirm | `anti_fin` | signature for write_file only | 0.00 | 1.000 | 0.0 | 0.0 |
| confirm | `anti_fin` | signature for every op but send_email | 0.00 | 0.410 | -0.94 | 0.94 |
| transfer, W3 complete (counterfactual) | `anti_full` | exact (oracle) | 0.00 | 1.000 | 0.0 | 0.0 |
| transfer, W3 complete (counterfactual) | `anti_full` | signature | 0.00 | 1.000 | 0.0 | 0.0 |

What makes the false gap:

- **In-order pairing on W3-visible ops whose text is left out of the
  signature.** Signature matching on `send_page` alone claims a false gap in
  0.98 of replicates; on the back-office run, signature on `send_email` alone
  claims one in 1.00.
- **Plus a W3 blind spot elsewhere.** With W3 handed every action it missed,
  signature matching everywhere claims none.
- **Not transient-op collisions.** Signature matching on every op except
  pages and incidents claims no gap: it undercounts instead (bias −28.48).

An earlier draft of this file said transient work that repeats signatures
is enough for a false gap. The ablation refutes that; the statement is
withdrawn (D48).

## Predictions against outcomes

`uv run python scripts/transfer_table.py predictions transfer`:

Source: `artifacts/episodes/transfer/episodes.jsonl` (23 episodes, N=1036) and `artifacts/thinning/transfer.json`. Ranges from docs/PREREG-TRANSFER.md. A reading in parentheses turns a verbal prediction into a band; the bands were added after the freeze, in commit 0748430.

| # | quantity | predicted | outcome | held |
|---|---|---|---|---|
| 1 | actions per episode | 38 to 52 | 45.043 | yes |
| 1 | W3-invisible share | 0.40 to 0.55 | 0.473 | yes |
| 2 | hidden from all 3 at mid, oracle | 95 to 160 | 123.010 | yes |
| 2 |   ... over 0.25 x W3-invisible | close to 1 (read: 0.9 to 1.1) | 1.004 | not a test: identity under MCAR loss |
| 2 | hidden at rho100 over hidden at mid | about 2 (read: 1.8 to 2.2) | 2.001 | not a test: identity under MCAR loss |
| 3 | ll_bic coverage at mid, signature | 0.20 to 0.75 | 0.300 | yes |
| 3 | ll_bic bias at mid, signature | negative | -34.250 | yes |
| 3 | ll_bic coverage at mid, oracle | >= 0.90 | 0.920 | yes |
| 4 | ll_bic detects_gap at mid | >= 0.95 | 1.000 | yes |
| 5 | ll_independence coverage at rho100 | about 0 (read: <= 0.05) | 0.000 | yes |
| 5 | ll_bic coverage fall, mid to rho100 | negative | -0.490 | yes |
| 5 | ll_independence coverage fall | >= 0.10 (T3) | 0.240 | yes |
| 6 | detects_gap lead over union_count | >= 0.95 | 1.000 | yes |
| 7 | W3-invisible sharing a signature | 0.15 to 0.40 | 0.271 | yes |
| 8 | union_count coverage at mid | 0.000 | 0.000 | not a test: true by construction |
| 8 | union_count coverage at rho100 | 0.000 | 0.000 | not a test: true by construction |

12 of 12 rows that could have failed held.

Four rows could not have failed: under MCAR loss the hidden count is the
W3-invisible count times the probability both logs drop, and `union_count`
covers 0 whenever anything is hidden. They are shown and not counted. The
read bands in parentheses were added after the freeze, in `0748430`. The
ranges in predictions 3 and 5 came from the scripted pipeline check, so
they test whether the agent behaves like the script, which it does.

## How the infra blind spot differs from the back-office one

| | back-office (`fin_mid`) | infra (`transfer_mid`) |
|---|---|---|
| why W3 misses an action | an injected view hides two op types completely | transient work and repeats: the end state keeps no trace |
| W3 capture within an op | 0 or 1 | in between for most ops |
| W3-invisible actions sharing a signature | few | many (tables below) |
| effect on signature linkage | costs `ll_bic` coverage at `fin_mid` too (D44: 0.893 against 0.948 oracle, `precision.json`) | groups W3 misses in whole or in part merge, and nothing corrects it |

Infra: the per-op table under "The paid run reproduces the scripted run".
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

## Weaknesses

- **23 episodes, not 24**, one lost to a provider error that is still
  unexplained.
- **The agent does not matter here.** The paid run reproduces the scripted
  run, so this is a test of an environment and a linkage, not of a model.
- **Density decides the verdict.** At other ticket counts or fleet sizes T1
  could pass (density table). The registered density was fixed before any
  thinning and not changed after the pipeline check, but it is one point.
- **One environment I designed.** The ticket mix also sets how often
  signatures collide.
- **The failure was predicted from a pipeline check on the same
  environment**, so T1 failing confirms a known mechanism.
- **T2 and T4 cannot separate real gaps from linkage artefacts** under
  signature matching here, and T3 does not show robustness to coupling.
- **Signature linkage ignores what real logs carry.** Trace and gateway lines
  for one call share infrastructure timestamps; matching on them could
  remove much of the error without trusting an agent-supplied ID (AGENTS.md
  rule 4). Not built or measured.
- **Env views are back-office only**, so `financial`-view configurations mean
  nothing on infra.
- **One model, benign loss only, MCAR loss only** (D10, D11, D32).
