# Pre-registration: several originators and real log loss

Status: **DRAFT. Not frozen, and not yet freezable.** AGENTS.md rule 1 applies from the
commit that freezes this file; until then anything here may change, and every change
after the pools are recorded is listed in `docs/DECISIONS.md` with its reason. The
freeze needs things this draft does not have ("Before the freeze", at the end). The
freeze will be a
signed tag `prereg-study5` and a deposit, both recorded in `docs/DECISIONS.md` with
their times before the first confirmatory run (D74).

Fifth pre-registered study, after `docs/PRE-REGISTRATION.md` (back-office),
`docs/PREREG-TRANSFER.md` (infra), `docs/PREREG-BYPASS.md` (bypass) and
`docs/PREREG-STUDY4.md` (a second model and an adversary). The design is
`docs/DESIGN-STUDY5.md`; the rig, its checkpoint and the pilot are
`docs/RIG-FINDINGS.md`; D75 to D85 are the decisions. D80 records two reviews of the
code this file names and a refutation of its first draft; D81 the owner's answers to
the four questions that draft left open; D84 the pilot on recorded sessions, which
changed the provider and the second model; D85 a refutation of D84's proposal to split
the primary condition, which is withdrawn.

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
- The task mix is `backoffice_churn` (`lacuna.envs.churn`): later tickets correct,
  revert, delete or rewrite what earlier ones did.

## Sessions (paid, recorded once)

`scripts/study5_record.py`, in-process, ticket by ticket, both planes of a session on
one kernel (`lacuna.rig.record`). Each plane is one conversation with its model. The
design calls plane A's agent the orchestrator and plane B's the sub-agent; neither
delegates to the other. The queue's tickets are dealt to them in turn. The system
prompt's sha256 is written with each pool. Temperature, top-p and seed are the
provider's defaults (none sent). Only the model string is known; a provider-side change
behind a name is not detectable. Both models are served by one provider, at
`api.mistral.ai`; each pool's `pool.json` records that host. Requests carry no project
name.

| pool | plane A | plane B | sessions | seeds | tickets | cap |
|---|---|---|---|---|---|---|
| `ds` | `deepseek-v4-1-flash-1017` | `deepseek-v4-1-flash-1017` | 72 | 51000 to 51071 | 10 | $12 |
| `glm` | `zai-glm-5-3` | `zai-glm-5-3` | 72 | 52000 to 52071 | 10 | $12 |
| `mix` | `deepseek-v4-1-flash-1017` | `zai-glm-5-3` | 72 | 53000 to 53071 | 10 | $12 |

```
PYTHONPATH=src uv run python scripts/study5_record.py --pool ds --seed0 51000 --sessions 72 --tickets 10 --pace 1.0 --model-a deepseek-v4-1-flash-1017 --model-b deepseek-v4-1-flash-1017 --usd-per-m-in 0.30 --usd-per-m-out 1.20 --budget 12
PYTHONPATH=src uv run python scripts/study5_record.py --pool glm --seed0 52000 --sessions 72 --tickets 10 --pace 1.0 --model-a zai-glm-5-3 --model-b zai-glm-5-3 --usd-per-m-in 0.30 --usd-per-m-out 1.20 --budget 12
PYTHONPATH=src uv run python scripts/study5_record.py --pool mix --seed0 53000 --sessions 72 --tickets 10 --pace 1.0 --model-a deepseek-v4-1-flash-1017 --model-b zai-glm-5-3 --usd-per-m-in 0.30 --usd-per-m-out 1.20 --budget 12
```

- The commands read `LACUNA_BASE_URL` and `LACUNA_API_KEY` from the environment. The
  prices on them, $0.30 and $1.20 per million tokens in and out, are an assumption
  carried from the earlier studies' provider and are not this provider's published
  rate, which was not found. Every dollar figure here is tokens at that assumed rate;
  the receipts hold the tokens.
- The pools are the folders `artifacts/study5/pools/{ds,glm,mix}` and are committed.
- **72 sessions a pool, on one number** (D81, corrected by D83). Let `c` be the pilot
  pool's `spent_usd` divided by its sessions. If 216 `c` is at most $30, the pools are
  as in the table. If not, and 108 `c` is at most $30, they are 36 sessions each:
  seeds end at 035, the floor is 30 in place of 60, two runs of a pool share about two
  thirds of their sessions in place of a third, the study rests on 108 sessions in
  place of 216, and `POOL_SESSIONS` and `MIN_POOL` in `scripts/study5_runs.py` change,
  all in one decision entry before any confirmatory pool is recorded. If neither
  holds, no pool is recorded and the owner decides. The cap of $12 a pool is the same
  in both cases and is above what either costs when the rule allows it. Applied: the
  three pilot pools cost $1.75 for 84 sessions at the assumed rate
  (`artifacts/study5/pools/pilot-*/pool.json`), so `c` is $0.021, 216 `c` is $4.50, and
  the pools are 72 sessions each. `c` includes what a retried session's first try cost.
- A session is left out only if its recording raised twice: it is tried once more
  from the start after the first, and `pool.json` lists it after the second. No
  session is left out for what is in it: one with no action, with failed calls or
  with unfinished tickets stays.
- A recording that its budget or an interrupt ended says so in `pool.json`
  (`stopped`), and `--check` names that pool as unfinished.
- A pool with fewer than 60 sessions is not used (`scripts/study5_runs.py --check`
  names it). Its folder is renamed `<pool>-attempt1`, kept, and the pool is recorded
  once more with the same command. Both attempts are named in `docs/DECISIONS.md`.
- The pools are recorded before the freeze, because their hashes are part of what is
  frozen. Between recording and freeze two things are done with them and no others:
  `scripts/study5_runs.py --check`, which draws sessions and schedules and makes no
  run, and reading in `pool.json` what the recorder wrote about the recording itself:
  `sessions_recorded`, `failed_sessions`, `spent_usd`, `stopped`, `commit` and
  `dirty`. No table is built from them, their hidden share is not read, and no
  run replays them before the tag. If `--check` prints anything, the only remedy is
  recording again as above; no range, seed or geometry in this file changes for it.
- The pilot's own recorded pools use seeds 50000 to 50035, 50100 to 50123 and 50200 to
  50223, and are never replayed in a confirmatory run.

## Runs

`scripts/study5_runs.py --list` prints all 420. The plan is a function of nothing.

| condition | what | runs | seeds | runs per pool |
|---|---|---|---|---|
| `s0` | no stress | 60 | 70000 to 70059 | 20 |
| `s1` | the restart: both relays killed together at a time drawn from 8 to 90 s and restarted together 15 to 45 s later | 300 | 71000 to 71299 | 100 |
| `s2` | each relay its own outage (spans 25 to 45 s, access 10 to 40 s, each starting 8 to 70 s in) and one originator killed for good 15 to 90 s in | 60 | 72000 to 72059 | 20 |

- A run replays 24 sessions drawn without replacement from one pool by the run's seed.
  Run `i` of a condition uses pool `("ds", "glm", "mix")[i % 3]`. The same integer
  seeds the stress schedule (`lacuna.rig.stress.plan`) and each originator's wait
  before it creates its SDK; they are separate streams.
- A session's start within the run is a function of the session's name, between 0 and
  100 s, and so is the same in every run that draws it.
- **One restart, and no line through it.** `s1` is the condition the owner accepted
  (D80) and the pilot's `node-restart`. The pilot's three misses were all at outages
  of about 20 s or less, and a draft of this file set the shorter restarts apart. That
  was withdrawn (D85): it chose the primary condition as the part of the pilot where
  nothing failed. `s1` includes the lengths at which the estimator was seen to miss.
- `s1`'s lengths cross the lengths at which the SDK gives up two, three or four
  batches; that is sampled, not avoided. An outage can still be on when the last call
  of a run is due. Each run's manifest says so, and the analysis counts those runs.
- `s1` is the pilot's `node-restart` under its own name, with its own draws. `s2` was
  not piloted.
- Runs are made by `scripts/study5_runs.py --go`, one at a time in the plan's order,
  which spreads every condition over the whole sequence. `--go` starts only from a
  clean tree with the registered driver, lock file and pools, and stops at the first
  run the driver refuses or whose manifest is not the registered one.
- A run is made once. A folder that exists is never remade, whatever it holds. A run
  that is set aside is not replaced. If fewer than 270 `s1` runs are left, P1 and P2
  are "underpowered" and no run is added.
- No interim analysis. `scripts/study5_analysis.py` is run on `s5-` runs once, after
  `--go` has gone through the plan, and after `scripts/study5_runs.py --seal` has
  written one hash per run folder to `artifacts/study5/runs-sha256.json` and that file
  is committed. The raw runs are deposited with the release; they are too large for
  the repository.

### What makes a run a registered run

`scripts/study5_runs.py --sample` makes one unstressed run, `reg-sample`, at the tagged
commit, with the flags `--go` uses; it is not a run of the plan and is in no figure.
`--register` then writes `artifacts/study5/registered.json`: from that run's manifest
the hashes of the image's source (`src/`), of `scripts/rig_run.py`, of `uv.lock`, of the
compose file, the Dockerfile, its requirements and every file in `rig/config/`, the
image ids and the driver's parameters; the hash of each pool; and for each of the 420
runs its condition, seed, pool and the hash of the sessions file it will replay. It
refuses a sample that is not what the plan's runs will be. That file is committed
after the tag, because it holds the manifest of a run made at the tag; the commit
touches nothing under `src`, `rig`, `scripts` or `docs`.

`lacuna.rig.manifest.registered_problems` holds every run to it: made from a clean
tree, by that source, driver, lock file, configs and images, with those parameters, and
being the run of the plan its folder's name says, with that condition, those seeds,
that one pool and those very sessions.

## Analysis (frozen code)

```
PYTHONPATH=src uv run python scripts/study5_runs.py --seal
PYTHONPATH=src uv run python scripts/study5_analysis.py --prefix s5- \
    --registered artifacts/study5/registered.json --out artifacts/study5/confirm.json
PYTHONPATH=src uv run python scripts/study5_verdicts.py \
    --analysis artifacts/study5/confirm.json --primary ll_sa_hom --condition s1 \
    --control s0 --no-env chapman_sa --min-runs 270 \
    --out artifacts/robustness/study5-verdicts.json
```

The analysis refuses `s5-` runs without `--registered`, and the verdict script refuses
an analysis made without it.

- **The unit is the run.** One run is one realisation of loss over 24 sessions that
  share it, so sessions are not replicates and every rate is over runs.
- **Runs are not independent of each other either.** The 100 `s1` runs of a pool draw
  24 of the same 72 sessions, so two of them share about a third of their sessions
  (two fifths if a pool has only the 60 it must), each starting at the same moment in
  both. What differs between such runs is the
  schedule and the phase of each originator's exports. The Wilson and exact bounds
  below treat runs as independent and are therefore too narrow by an amount this study
  does not measure. Every statement is conditional on the sessions recorded, 216 if
  every one records. Coverage per
  pool is reported beside each condition (`by_pool`), with no criterion: a pool is one
  model pairing on its own ticket seeds, so a difference between pools has three
  explanations.
- **Truth** is the run's own ledger, read to score and to describe loss and by nothing
  that estimates (AGENTS.md rule 3).
- **Tables.** One three-witness table per plane (`lacuna.rig.witnesses`). `spans` and
  `access` are linked by (trace id, JSON-RPC id), both stamped by the client library;
  `env` attaches by what the action was (session, op, resource, value) in call order.
- **Primary estimator: `ll_sa_hom`** (`scripts/study5_lib.py`). The two planes' tables
  are summed. Its point is Lincoln-Petersen between two lists, "any telemetry" (spans
  or access) and the end state: the count seen plus (seen by env alone) x (seen by
  telemetry and not env) / (seen by both). That is the maximum-likelihood missing cell
  of the log-linear model whose only interaction is spans-access, which is why it is
  named so; how records split between spans and access does not move the point. The
  interval is the profile likelihood of that model with its cutoff scaled by the
  dispersion between the run's (session, plane) tables
  (`loglinear.homogeneity_dispersion`, D51). The interval is unbounded when no action
  is in both lists, when the point passes 20 times the count seen, when the profile's
  upper end does, or when the profile does not converge.
- **Reported beside it, with no criterion:** `strat_bic` (the estimator the earlier
  studies registered), `ll_bic_hom`, `ll_bic_boot`, `ll_bic`, `ll_average`,
  `chapman_te` (the same two lists with Chapman's correction and a Wald interval), and
  `ll_sa_hom_sig`, the primary estimator on tables built with no id trusted (every
  record attached by what the action was), which shows what trusting the client
  library's ids buys (rule 4).
- **Baselines.** `union_count`: the number of distinct actions in the two planes'
  tables. `seq_gap_tail`: per session and plane with at least one delivered telemetry
  record, the JSON-RPC ids on those records, read from both logs together; the first
  tool call's id is known (it is 2 for this client); ids missing up to the last one
  seen are lost requests; requests become actions at the share of seen calls that were
  successful actions, pooled over the run; the expected lost tail is added with the
  earlier studies' formula. It is not shown the end state, it has no floor, and a
  session with no delivered record does not exist for it. `seq_gap` is reported too.
- **Floors (rule 5):** `trust_trace`, `union_count`, `union_bound`, in every table.

## Gates applied before any test

- **Integrity** (`lacuna.rig.integrity`, as the driver applies it). A run is set aside
  if: a stress event did not happen; an originator exited wrongly, left no report or
  could not replay a session; a witness file is missing or empty; a service that
  should have been up was not; the copy of an access log is incomplete; a plane's
  receiver did not get every planned call; a telemetry record has no session or
  matches no ledger entry; a plane's log relay delivered no line of its access log; a
  receiver and the ledger disagree on the actions; a call was made more than 0.25 s
  late; an originator was not ready at the start; the containers' clock is more than
  0.5 s from the driver's. Also if the driver did not finish, or the run is not a
  registered run (above).
- Two of those can follow from the loss under study: a witness file left empty and a
  relay that delivered no access line. A run lost that way is a run with extreme loss
  left out, which favours passing. Runs set aside are listed by condition, kind and
  reason beside the verdicts, and a rate that differs between conditions is reported
  as a caveat on them.
- **Underpowered:** fewer than 270 `s1` runs left. P1 and P2 are then "underpowered".
  This is said first; nothing else is read.
- **Not exercised:** the mean over the `s1` runs left of each run's share of actions in
  no log is under 0.02, or both telemetry witnesses lose under 0.01 of a plane's
  actions on average over runs and planes. P1 and P2 are then "not exercised". The
  gate is computed on the runs the tests use and only relabels them.

## Tests and verdict rules (as coded in `scripts/study5_lib.py`)

- **P1.** Over the `s1` runs left: the Wilson 95% lower end of the share whose interval
  holds the true count is at or above 0.90, and under 0.05 of intervals are unbounded.
  An unbounded interval covers and counts toward the 0.05, so P1 can pass with up to
  14 of 300 unbounded; coverage over bounded intervals alone is reported beside it.
- **P2.** Against each of `seq_gap_tail` and `union_count`: over the same runs, the
  number in which `ll_sa_hom`'s point is strictly closer to the true count, on the
  unrounded errors. A tie is a loss, and so is an interval with no upper end, whatever
  its point. Pass if the exact one-sided 97.5% lower bound on that share is above 0.5.
  P2 passes only if both pass.
- **How a "fail" is read.** `P1_reading` and `P2_reading` say "missed" when the
  interval excludes meeting the criterion (coverage's Wilson upper end under 0.90, or
  the unbounded share's exact lower bound at 0.05 or more; a win rate whose exact
  upper bound is under 0.5 against a baseline) and "not shown" otherwise. A criterion
  not shown is not evidence that the estimator fails it.
- **What P2 against `union_count` can show.** The union is never above the true count
  and the estimate is the union plus a non-negative amount, so the estimate wins
  whenever that amount is positive and under twice what the union missed, and loses by
  the tie rule in every run where the union is already exact. It shows that the amount
  added is the right order, not that it is accurate. The informative comparison is the
  one with `seq_gap_tail`. The number of runs where the union is exact is reported.
- **Q3.** P1 and P2 for `chapman_sa`: Chapman's form of Lincoln-Petersen between
  `spans` and `access`, with a Wald interval.
- **Control.** On `s0`: how many runs have an `ll_sa_hom` interval wholly above the true
  count. Reported; no criterion. With no loss the end state shows nothing the telemetry
  missed, the estimate is the count seen, and no interval can lie above the truth: the
  control checks that the rig loses nothing unstressed, not the estimator. P2 is not
  defined there, since every method ties.
- **Exploratory, named here so that it is not invented later.** For each condition,
  `breakdown` in `scripts/study5_lib.py`: the number of runs and of runs covered, by
  how long the span relay was down (under 20 s, 20 to 28 s, 28 s or more) and by how
  many actions both telemetry logs lost (under 12, 12 or more); every run a candidate
  did not cover, with how many actions outside its interval the truth fell; and, per
  band, how many of the lost actions and of the others the end state shows. The count
  of actions lost by both logs is read from the receivers' own complete logs, which
  are no witness, to describe a run after it is scored; no estimate reads it, and an
  auditor would not have it. The bands were drawn after the pilot's misses were seen.
  No criterion, no gate, and no reading of P1 or P2 by band.
- `s2` is described with the same tables and has no criterion.
- **No multiplicity correction.** The tests above are fixed here and reported together.
  No estimator reported beside the primary one stands in for it if it fails, and none
  is promoted. Any analysis not named in this file is labelled exploratory.
- Results are published whatever they are. If P1 or P2 is not met, the paper reports
  where the estimate held and where it did not in this rig; one layout and one outage
  shape per condition do not make a general map.

## Predictions, written before any confirmatory run

Figures are from `artifacts/study5/pilot-summary.json` (scripted sessions) and
`artifacts/study5/pilot-recorded-summary.json` (recorded ones), over the pilot's runs of
this restart: 20 on recorded sessions, and the 24 on scripted ones made after
`ll_sa_hom` existed. "Reached" figures are from `thresholds` in those files, for
independent runs at a given true rate; the runs are not independent, so they are a
guide.

| | prediction | basis |
|---|---|---|
| gates | `s1` is exercised: mean share in no log between 0.03 and 0.09 | 0.058 over the recorded runs (6 of the 20 under 0.02 on their own), 0.065 over the scripted |
| P1, coverage | not met, read "not shown": coverage between 0.86 and 0.95 | 18 of 20 recorded (Wilson 0.699 to 0.972), 23 of 24 scripted (0.798 to 0.993). P1 needs 281 of 300, reached 17% of the time at a true 0.92, 65% at 0.94 and 88% at 0.95 |
| P1, unbounded | under 0.05 | no `ll_sa_hom` interval was unbounded in either set |
| P2 against `seq_gap_tail` | pass; win rate 0.70 or more | 16 of 20 recorded, 19 of 24 scripted; it needs 168 of 300, reached 93% of the time at a true 0.60 |
| P2 against `union_count` | pass; win rate 0.65 or more | 15 of 20 with one tie recorded, 22 of 24 with one tie scripted; weak by construction (above) |
| Q3 | "missed" on P1 | `chapman_sa` covered 1 of 20 recorded and 1 of 24 scripted |
| control | no interval above the truth | follows from the table when nothing is lost |
| exploratory | most runs not covered are ones in which both logs lost under 12 actions; where they lost 12 or more, coverage is 0.95 or more | under 12: 1 of 3 covered recorded, 1 of 2 scripted. 12 or more: 17 of 17 and 22 of 22. Five runs under 12, and bands drawn after the misses |
| `s2` | no prediction of size; mean error above zero | with an originator killed under a different schedule the scripted pilot's mean error was +18 actions over six runs, standard deviation 19, two runs near zero; with each relay on its own outage and no kill it was +10. Not piloted on recorded sessions |

**P1 is predicted not to be met.** The interval is built for 95% and the pilot puts
its coverage on recorded sessions near 0.90, which is the bar's value and not above
it. If the prediction holds, the result is an estimator closer than both baselines
whose interval is too narrow when few actions are lost, with the size of that
measured. If P1 passes, the pilot's two misses in 20 were on the unlucky side. Neither
outcome changes what is reported.

The recorded pilot is 20 restart runs on 84 sessions; two of its three pools have 24
sessions, so their six runs each replay the same sessions under six schedules.

## What was seen before writing this

- **Spikes and the rig's checkpoint** (D75 to D78): what the SDK and the collectors
  lose at default settings and at which outage lengths, which is where the range of
  15 to 45 s first came from.
- **The pilot on scripted sessions** (D79, D80): 61 runs with no model,
  `artifacts/study5/pilot-summary.json`.
  - Eight candidates were scored on it. `ll_sa_hom` and a restart of 15 to 45 s were
    both chosen after its first round was seen. On the 24 later runs of that condition
    `ll_sa_hom`, `ll_bic_hom` and `ll_bic_boot` each covered 23 and `strat_bic`, which
    the earlier studies registered, 22 with one unbounded. `ll_sa_hom` is primary on an
    argument (nothing is selected), not on that count.
  - Those 24 runs were made at three commits, between two code reviews; the first 12
    predate the checks on call lateness and clocks. All were scored with the final code.
  - The queue's thread weights were set to hide about two fifths of actions.
  - The sequence baseline was given the first call's id and both logs during it.
- **The pilot on recorded sessions** (D84): 23 runs on three pools of sessions recorded
  from the study's two models, `artifacts/study5/pilot-recorded-summary.json`.
  - `ll_sa_hom` covered 18 of 20 restart runs. Both misses are on the first pool, by
    1 and by 3 actions, at outages of 17.9 and 19.3 s.
  - The cause was looked for with the ledger open on those two runs, then measured
    over all of them: the end state shows 0.42 of the actions in the first fifth of a
    session and 0.82 of those in the last. What it shows of the actions an outage lost
    therefore depends on where they fell, which the estimator assumes it does not.
  - Three other intervals were tried on runs with the truth known and not adopted
    (`docs/RIG-FINDINGS.md`, section 6).
  - A split of the restart at 28 s was proposed (D84) with a prediction for the 12
    runs on the other two pools, committed before they were scored: no miss at 28 s
    or longer. None of the 12 was missed at any length. A refutation then showed the
    line was drawn around the failures and its reason was wrong, and the split was
    withdrawn (D85). The breakdowns registered as exploratory come from that
    episode, and their bands from the misses.
- **Reviews and refutations** of the pilot's code and write-up, of this file's tooling
  and first draft, of the tool, and of the recorded pilot's write-up and the draft that
  split the condition (D79, D80, D83, D84, D85).
- No confirmatory pool and no confirmatory run.

## Not tested, and caveats fixed now

- **No adversary.** The link id is stamped by the client library in the originator's
  process. An originator whose process is hostile can stamp what it likes.
- **One layout, one laptop.** Two planes, two relays shared by both, one outage shape
  per condition, synthetic tickets, two model families from one provider, at most
  216 sessions.
- **The estimate assumes the end state hides an action independently of whether the
  telemetry lost it, and that is false in a way now measured.** Hiding depends on what
  later happened to the same record, so early actions are hidden more than late ones
  (0.42 of a session's first fifth shown against 0.82 of its last, in the recorded
  pilot); loss depends on when the action was made. The actions an outage loses can
  therefore be shown more or less than the rest, and the estimate is then too high or
  too low. In the pilot this put the truth outside the interval in 2 of 20 recorded
  runs and 1 of 24 scripted, by 1 to 3 actions, each a run in which both logs lost
  under 12 actions. `s1` is not drawn to avoid it. No estimator here corrects for it, and sessions of another length or a
  queue built another way would show it differently.
- **The two planes are summed.** Plane A goes through a gateway and plane B does not;
  if their loss or their hiding differ, the pooled two-list estimate is biased and
  nothing here checks for it. `strat_bic`, which keeps the planes apart, is beside it.
- **The dispersion that widens the interval comes from tables of one run that share
  one outage,** and the two planes of a session are two of its rows.
  `homogeneity_dispersion` cannot see what every row shares.
- **A run where the end state shows nothing the telemetry missed gives the union as
  its estimate.** Two stressed pilot runs with actions missing from every log did, one
  recorded and one scripted, and both missed.
- **Killing an originator** changes which actions happen. It is in `s2` only.
- **The sequence baseline is one construction.** A different counter would score
  differently.
- **The gateway's id expression** (`json(request.body).id`) is exercised by runs only.
- **Shares are shares of this run length.** A run is about two minutes with outages of
  15 to 45 s. The share of actions lost says nothing about a deployment's rate.
- **A run is not reproducible from its seeds to the record.** The SDK's retry jitter is
  not seeded. Two pilot schedules made again gave the same batch counts and tables
  within three per cell; that is two schedules.
- **Call order across planes is the recording's.** Replay keeps it to within the
  lateness limit; a model that would have reacted to a lost reply is not modelled.
- **Image ids are held strictly.** A cleared build cache part-way through can change
  them; `--go` then stops at the first such run.

## Settled by the owner

Four questions the first draft left open, answered on 2026-11-15 (D81):

1. **P2 keeps `union_count` as a criterion,** as approved, with what it can show stated
   above. The weight of Q2 is on `seq_gap_tail`.
2. **A fail is read three ways** (pass, not shown, missed). The pass rules are as they
   were.
3. **The restart's start is anywhere from 8 to 90 s,** as piloted. Some outages are
   still on when a run's last call is due; those runs are counted.
4. **72 sessions a pool,** by the rule under "Sessions", now applied.

And on the same day (D84): the provider is the one at `api.mistral.ai`, on the owner's
instruction, and the second model is GLM 5.3, which the owner left to be chosen
between 5.2 and 5.3.

And on the same day, after D85: asked whether to freeze a registration that predicts
its first criterion is not met, or to loosen the bar, redraw the condition or widen the
interval, the owner said to proceed as recommended: this text, with `s1` at 15 to 45 s,
`ll_sa_hom` unchanged, the criteria as they were and P1 predicted not met.

## Proposed and withdrawn

D84's draft of this file split the accepted restart into a primary condition of 28 to
45 s and a set-apart one of 15 to 24 s, 480 runs, and asked the owner to confirm. It
was withdrawn before an answer (D85). Nothing the owner accepted is changed: one
restart of 15 to 45 s, 420 runs, `ll_sa_hom`, the criteria as they were.

## Before the freeze

1. The owner's signing key registered where the tag is published, and the deposit.
2. The three pools, committed; `scripts/study5_runs.py --check` prints nothing.
3. This file lists the sha256 of `scripts/rig_run.py`, `scripts/study5_runs.py`,
   `scripts/study5_lib.py`, `scripts/study5_analysis.py`, `scripts/study5_verdicts.py`
   and `uv.lock` at the commit to be tagged.
4. A review and a refutation of the final text. Then the signed tag and the deposit,
   recorded in `docs/DECISIONS.md` with their times.
5. After the tag and before the first `s5-` run: `scripts/study5_runs.py --sample`,
   then `--register`, and `artifacts/study5/registered.json` committed.
