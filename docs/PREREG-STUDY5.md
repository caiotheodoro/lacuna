# Pre-registration: several originators and real log loss

Status: **DRAFT. Not frozen, and not yet freezable.** AGENTS.md rule 1 applies from the
commit that freezes this file; until then anything here may change. The freeze needs
four things this draft does not have, listed under "Before the freeze" at the end. The
freeze will be a signed tag `prereg-study5` and a deposit, both recorded in
`docs/DECISIONS.md` before the first confirmatory run (D74).

Fifth pre-registered study, after `docs/PRE-REGISTRATION.md` (back-office),
`docs/PREREG-TRANSFER.md` (infra), `docs/PREREG-BYPASS.md` (bypass) and
`docs/PREREG-STUDY4.md` (a second model and an adversary). The design is
`docs/DESIGN-STUDY5.md`; the rig, its checkpoint and the pilot are
`docs/RIG-FINDINGS.md`; D75 to D80 are the decisions.

## Scope

If a trusted counter exists at the source or at the sink, use it. This study is about
the case where none does: actions come from separate processes that share no counter,
and the logs that record them lose records on their own.

The rig's single kernel is itself a census at the sink. It stands for several
downstream systems whose logs are not collected. That is a stipulation of the study and
not something it shows.

The author works on a commercial AI gateway. Nothing from that product is used: every
component is open source and pinned by digest, and all data is synthetic.

## Questions

1. **Q1. Does the interval cover the true count when records are lost by real telemetry
   components, in blocks, to both planes at once?** (P1.)
2. **Q2. Is the estimate closer to the true count than counting sequence gaps, and than
   counting the union of the logs?** (P2, against each.)
3. **Q3. The same without the end state.** Predicted to fail.

## The rig

`rig/compose.yaml`, driven by `scripts/rig_run.py`. Two originators replay recorded
sessions over MCP: plane A through an unmodified agentgateway, plane B straight to an
MCP front. One kernel service keeps the ledger. Per plane there are three witnesses:

| witness | what it is | how it loses records |
|---|---|---|
| `spans` | the originator's own OpenTelemetry spans, through a relaying collector | the SDK gives a batch up when the relay is away past its retries; a killed originator loses its queue |
| `access` | the receiver's access log, tailed by a second relaying collector | lines written while that relay is away are skipped |
| `env` | the end state and the outbound sink, each row with its last writer | an action overwritten or undone later leaves nothing |

- Every SDK and collector setting is its default. No code in this repository drops a
  record. Only the two relays and the originators are ever killed.
- Both planes' spans pass one relay and both planes' access logs another.
- What makes a run a registered run is written to `artifacts/study5/registered.json` at
  the freeze by `scripts/study5_runs.py --register`: the source hash of the image, the
  hashes of the compose file, the image's Dockerfile and requirements and every file
  under `rig/config/`, the image ids, the driver's parameters, and the hash of each
  pool. A run whose manifest differs is set aside
  (`lacuna.rig.manifest.registered_problems`).
- The task mix is `backoffice_churn` (`lacuna.envs.churn`): later tickets correct,
  revert, delete or rewrite what earlier ones did.

## Sessions (paid, recorded once)

`scripts/study5_record.py`, in-process, ticket by ticket, both planes of a session on
one kernel (`lacuna.rig.record`). The system prompt's sha256 is written with each pool.
Temperature, top-p and seed are the provider's defaults (none sent). Only the model
string is known; a provider-side change behind a name is not detectable.

| pool | orchestrator (plane A) | sub-agent (plane B) | sessions | seeds | tickets | cap |
|---|---|---|---|---|---|---|
| `ds` | `deepseek-v4.1-flash` | `deepseek-v4.1-flash` | 36 | 51000 to 51035 | 10 | $5 |
| `qw` | `qwen3.8-flash` | `qwen3.8-flash` | 36 | 52000 to 52035 | 10 | $5 |
| `mix` | `deepseek-v4.1-flash` | `qwen3.8-flash` | 36 | 53000 to 53035 | 10 | $5 |

- A session whose recording raises is tried once more from the start; one that fails
  twice is not in the pool, and the pool's `pool.json` lists it. A pool with fewer than
  30 sessions is recorded again in full under a new name before the freeze, with the
  reason in `docs/DECISIONS.md`.
- The pools are recorded before the freeze, because their hashes are part of what is
  frozen. Between recording and freeze the only things done with them are
  `scripts/study5_runs.py --check` (which draws sessions and schedules and makes no
  run) and reading `pool.json`'s counts. No table is built from them and no run
  replays them before the tag.
- The pilot's own recorded pool uses seeds 50000 and up and is never replayed in a
  confirmatory run.

## Runs

`scripts/study5_runs.py --list` prints all 420. The plan is a function of nothing.

| condition | what | runs | seeds |
|---|---|---|---|
| `s0` | no stress | 60 | 70000 to 70059 |
| `s1` | both relays killed together at a time drawn from 8 to 90 s and restarted together after 15 to 45 s | 300 | 71000 to 71299 |
| `s2` | each relay its own outage (spans 25 to 45 s, access 10 to 40 s, each starting 8 to 70 s in) and one originator killed for good 15 to 90 s in | 60 | 72000 to 72059 |

- A run replays 24 sessions drawn without replacement from one pool by the run's seed.
  The same seed draws the stress schedule (`lacuna.rig.stress.plan`) and each
  originator's wait before it creates its SDK. Run `i` of a condition uses pool
  `("ds", "qw", "mix")[i % 3]`.
- Sessions start at seeded times over 100 s. The outage lengths of `s1` cross the
  lengths at which the SDK gives up one, two, three or four batches; which it is in a
  given run is partly chance, and that is sampled, not avoided.
- Runs are made one at a time in the plan's order, which spreads every condition over
  the whole sequence. A run is made once. A folder that exists is never remade,
  whatever it holds.
- `scripts/study5_runs.py --check` must print nothing at the freeze: no run of the plan
  is one the driver would refuse.

## Analysis (frozen code)

```
PYTHONPATH=src uv run python scripts/study5_analysis.py --prefix s5- \
    --registered artifacts/study5/registered.json --out artifacts/study5/confirm.json
PYTHONPATH=src uv run python scripts/study5_verdicts.py \
    --analysis artifacts/study5/confirm.json --primary ll_sa_hom --condition s1 \
    --control s0 --no-env chapman_sa --min-runs 270 \
    --out artifacts/robustness/study5-verdicts.json
```

- **The unit is the run.** One run is one realisation of loss over 24 sessions that
  share it, so sessions are not replicates and every rate is over runs.
- **Truth** is the run's own ledger, read to score and to describe loss and by nothing
  that estimates (AGENTS.md rule 3).
- **Tables.** One three-witness table per plane (`lacuna.rig.witnesses`). `spans` and
  `access` are linked by (trace id, JSON-RPC id), both stamped by the client library;
  `env` attaches by what the action was (session, op, resource, value) in call order.
- **Primary estimator: `ll_sa_hom`** (`scripts/study5_lib.py`). The two planes' tables
  are summed. The model has the spans-access interaction and no other: the two
  telemetry logs may depend on each other, the end state on neither. The point is the
  model's closed form, the count seen plus (seen by env alone) x (seen by telemetry
  and not env) / (seen by both). The interval is the profile likelihood with its cutoff
  scaled by the between-session dispersion of the run
  (`loglinear.homogeneity_dispersion`, D51). It is unbounded when no action is in both
  lists or the point passes 20 times the count seen.
- **Reported beside it, with no criterion:** `strat_bic` (the estimator the earlier
  studies registered), `ll_bic_hom`, `ll_bic_boot`, `chapman_te`.
- **Baselines.** `union_count`: the number of distinct actions any witness recorded.
  `seq_gap_tail`: per session and plane, the JSON-RPC ids on the delivered telemetry
  records, read from both logs together, with the first tool call's id known; ids
  missing up to the last one seen are lost requests; requests become actions at the
  share of seen calls that were successful actions, pooled over the run; the expected
  lost tail is added with the earlier studies' formula. It is not shown the end state.
- **Floors (rule 5):** `trust_trace`, `union_count`, `union_bound`, in every table.

## Gates applied before any test

- **Integrity.** A run that fails `lacuna.rig.integrity` (a stress event that did not
  happen, an originator that exited wrongly or left no report, a call the receiver did
  not get, an incomplete access-log copy, a telemetry record with no session or
  matching no ledger entry, a receiver and a ledger that disagree, a call made more
  than 0.25 s late, an originator not ready at the start, clocks more than 0.5 s
  apart), that the driver did not finish, or whose manifest is not the registered one,
  is set aside. It is listed with its reason and is in no figure.
- **Underpowered:** fewer than 270 `s1` runs left. P1 and P2 are then "underpowered".
- **Not exercised:** the mean share of `s1` actions in no log is under 0.02, or both
  telemetry witnesses lose under 0.01 of a plane's actions on average. P1 and P2 are
  then "not exercised".

## Tests and verdict rules (as coded in `scripts/study5_lib.py`)

- **P1.** Over the `s1` runs left: the Wilson 95% lower end of the share whose interval
  holds the true count is at or above 0.90, and under 0.05 of intervals are unbounded.
  An unbounded interval covers and counts toward the 0.05.
- **P2.** Against each of `seq_gap_tail` and `union_count`: over the same runs, the
  number in which `ll_sa_hom`'s point is strictly closer to the true count. A tie is a
  loss, and so is an unbounded estimate. Pass if the exact one-sided 97.5% lower bound
  on that share is above 0.5. P2 passes only if both pass.
- **Q3.** P1 and P2 for `chapman_sa`, Lincoln-Petersen between `spans` and `access`.
- **Control.** On `s0`: how many runs have an `ll_sa_hom` interval wholly above the true
  count. Reported; no criterion. P2 is not defined there, since with no loss every
  method ties.
- `s2` is described with the same tables and has no criterion.
- Results are published whatever they are. If P1 or P2 fails, the paper is the map of
  when log completeness can be estimated and when it cannot.

## Predictions, written before any confirmatory run

| | prediction | why |
|---|---|---|
| gates | `s1` is exercised | scripted pilot: mean share in no log 0.065 over 24 runs of this condition |
| P1 | **uncertain** | it needs 281 of 300; the scripted pilot has 23 of 24, a Wilson interval of 0.798 to 0.993 |
| P2 against `union_count` | pass | 22 of 24 with one tie; the union cannot count what no log holds |
| P2 against `seq_gap_tail` | pass, with less margin | it needs 168 of 300; 19 of 24 in the pilot, but 6 of 9 where the outage was under 25 s, and a third of `s1`'s lengths are under 25 s |
| Q3 | fail on both | `chapman_sa` covered 1 of 24; the two telemetry logs lose overlapping blocks |
| control | no interval above the truth | 6 of 6 unstressed pilot runs returned the count |
| `s2` | `ll_sa_hom` high on average | with an originator killed the pilot's mean error was +18 actions over six runs, standard deviation 19; unexplained |

These rest on scripted sessions. The pilot on recorded sessions is not done; when it
is, this table is revised before the freeze and the revision is dated here.

## What was seen before writing this

- The rig's checkpoint (D78) and the pilot on scripted sessions (D79): 57 runs with no
  model, `artifacts/study5/pilot-summary.json`. `ll_sa_hom` and the condition that
  became `s1` were both chosen after the pilot's first round was seen. The 24 runs the
  predictions cite were made after that.
- No recorded session and no confirmatory run.
- `s1` and `s2` are new names with their own draws: no pilot run used a schedule of
  either.

## Not tested, and caveats fixed now

- **No adversary.** The link id is stamped by the client library in the originator's
  process. An originator whose process is hostile can stamp what it likes.
- **One layout, one laptop.** Two planes, two relays shared by both, one outage shape
  per condition, synthetic tickets, two cheap model families.
- **The estimate assumes the end state hides an action independently of whether the
  telemetry lost it.** Hiding depends on what later happened to the same record; loss
  depends on when the action was made. A session in which late actions are both the
  ones lost and the ones still visible breaks it. The scripted pilot cannot show this:
  its threads do not touch each other's records. The pilot on recorded sessions is
  where it would first show, and no analysis here tests it directly.
- **The dispersion that widens the interval comes from sessions of one run that share
  one outage.** `homogeneity_dispersion` cannot see what every session shares.
- **A run where the end state shows nothing the telemetry missed gives the union as
  its estimate.** One pilot run did, and missed.
- **Killing an originator** changes which actions happen and biased the estimate upward
  in the pilot. It is in `s2` only, which has no criterion.
- **The sequence baseline is one construction.** It is granted the first call id and
  both logs; it is not given the end state and has no floor. A different counter would
  score differently.
- **The gateway's id expression** (`json(request.body).id`) is exercised by runs only.
- **Shares are shares of this run length.** A run is about two minutes with outages of
  15 to 45 s. The share of actions lost says nothing about a deployment's rate.
- **Call order across planes is the recording's.** Replay keeps it to within the
  lateness limit; a model that would have reacted to a lost reply is not modelled.

## Before the freeze

1. Provider access, then the pilot pool (seeds 50000 and up) and a pilot round on it
   under the pilot's `node-restart`; revise the predictions and the caveats.
2. The three pools above; `scripts/study5_runs.py --check` prints nothing.
3. One unstressed run at the frozen commit, and `scripts/study5_runs.py --register` on
   its manifest; `artifacts/study5/registered.json` committed.
4. A review and a refutation of this file. Then the signed tag and the deposit, both
   recorded in `docs/DECISIONS.md` with their times, before the first `s5-` run.
