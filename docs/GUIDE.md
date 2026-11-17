# Using lacuna

For an engineer who has logs from an agent system and wants to know how many actions no log
recorded. This guide is for version 1.0.0. The distribution is `lacuna-mse`; the command and the
import package are `lacuna`.

## What it answers

You give it two or more logs that record the same actions, and a map file that says how to read each
log. It tells you:

- how many distinct actions each combination of logs recorded (`lacuna table`);
- how many actions the logs hold together, which is a count and also a lower bound on what happened;
- with `lacuna estimate`, an estimate of how many actions happened, those in no log included, and an
  interval.

The estimate comes from the overlaps. If two logs that fail separately each hold most of what the
other holds, little is missing from both; if they overlap little, more is. This is
capture-recapture.

## What it cannot answer

- Which actions are missing. It gives a number, not a list.
- Whether its assumptions hold for your logs. It cannot check them (see "What the estimate
  assumes").
- Anything about a route the map does not name. An action that reaches your systems by a path none
  of the mapped logs covers is in no table and in no estimate.
- Whether an agent that hides its actions has succeeded. The study behind this release ran no
  adversary (`docs/PREREG-STUDY5.md`, "Not tested, and caveats fixed now").

## When not to use it

- **A trusted counter exists at the source or at the sink.** Use the counter. Examples: a sequence
  number given by the one process every action passes through, or the ledger of the system of
  record. A counter gives the count. This tool gives an estimate that rests on assumptions.
- **You have one log, or more than five on one plane.** Every plane needs two to five witnesses.
- **All your logs arrive by one pipeline** (one collector, one shipper, one file). What loses one
  loses the others, and the estimate is then too low. `lint` refuses a map that says this. It cannot
  tell when the map says otherwise and is wrong.

## Install

```bash
pip install lacuna-mse
# If PyPI does not have it yet (this guide was written before any upload):
pip install git+https://github.com/caiotheodoro/lacuna
```

It needs Python 3.11 or later and installs one dependency, numpy. The PyPI project named `lacuna` is
a different project by someone else; do not install that one. The extras `agents` and `rig` are for
rerunning the studies. The command needs neither. The examples below use a map and recorded logs
from the repository (`rig/map.toml`, `tests/fixtures/rig`). They are not in the installed package.
Clone the repository and run the commands from its root.

## The three commands

Every command takes `--map FILE` and `--json`. `table` and `estimate` also take `--in NAME=PATH`,
once for each source the map names.

### `lacuna lint`: check a map before you collect anything

`lint` reads the map and no log. It checks that every plane has two witnesses and that they arrive
by two pipelines, as the map names them. The command is `lacuna lint --map rig/map.toml`:

<!-- output of: lacuna lint --map rig/map.toml -->
```
ok: two witnesses on every plane, by two pipelines as the map names them
```

When the map has an error, `lint` prints an `error:` line for each and exits 1. An error is
something no log can fix: a plane with fewer than two witnesses or more than five, a plane whose
witnesses share one pipeline, or a witness with no id that two blocks read from one file. `warning:`
lines are for what weakens a count without preventing it, such as a block with no `session`, or a
plane where every witness is a log and none is a record of the state.

### `lacuna table`: what each combination of logs recorded

```bash
lacuna table --map rig/map.toml \
  --in spans=tests/fixtures/rig/run/witnesses/spans.jsonl \
  --in logs=tests/fixtures/rig/run/witnesses/logs.jsonl \
  --in env=tests/fixtures/rig/run/witnesses/env.jsonl
```

<!-- output of: lacuna table --map rig/map.toml --in spans=tests/fixtures/rig/run/witnesses/spans.jsonl --in logs=tests/fixtures/rig/run/witnesses/logs.jsonl --in env=tests/fixtures/rig/run/witnesses/env.jsonl -->
```
plane a: union_count 10 (union_bound: at least 10), trust spans 10
      10  spans+access+env
plane b: union_count 6 (union_bound: at least 6), trust spans 6
       6  spans+access+env
spans (block 0, spans): {'read': 28, 'not_selected': 4, 'not_an_action': 8, 'kept': 16}
access (block 1, logs): {'read': 46, 'not_selected': 32, 'not_an_action': 4, 'kept': 10}
access (block 2, logs): {'read': 46, 'not_selected': 36, 'not_an_action': 4, 'kept': 6}
env (block 3, env): {'read': 16, 'kept': 16}
```

The fixture is a short run in which no log lost anything, so each plane has one line.

### `lacuna estimate`: the tables, and an estimate

The same command with `estimate` in place of `table`:

<!-- output of: lacuna estimate --map rig/map.toml --in spans=tests/fixtures/rig/run/witnesses/spans.jsonl --in logs=tests/fixtures/rig/run/witnesses/logs.jsonl --in env=tests/fixtures/rig/run/witnesses/env.jsonl -->
```
plane a: union_count 10 (union_bound: at least 10), trust spans 10
      10  spans+access+env
  estimate: 10 actions (interval 10 to 10)
plane b: union_count 6 (union_bound: at least 6), trust spans 6
       6  spans+access+env
  estimate: 6 actions (interval 6 to 6)
spans (block 0, spans): {'read': 28, 'not_selected': 4, 'not_an_action': 8, 'kept': 16}
access (block 1, logs): {'read': 46, 'not_selected': 32, 'not_an_action': 4, 'kept': 10}
access (block 2, logs): {'read': 46, 'not_selected': 36, 'not_an_action': 4, 'kept': 6}
env (block 3, env): {'read': 16, 'kept': 16}
all planes: union_count 16; estimate: 16 actions (interval 16 to 16)
Estimator: ll_sa_hom, the one study 5 tested, on a map with two id-linked logs and one witness without an id per plane.
The tool reads that witness as a state diff and cannot check that it is one.
Study 5 was one rig and scored the total over planes, not a plane's own line.
This assumes the witnesses miss actions independently and every record is a real action.
Witnesses that lose the same records make it too low; it can also be too high.
The interval is the estimator's own and was not calibrated on these logs.
```

Here the logs agree on everything, and the estimate is the count. The estimator adds actions only
when the state diff holds some that neither log has; here it holds none. That is not a finding that
nothing is missing: an action that both logs lost and the state diff does not show leaves no trace
in any of the three. The worked example below shows an estimate above the count, with an interval.

## Writing a map file for your own logs

A map file is TOML. It holds paths and constants only. Nothing in it is evaluated.

Three words first. An **action** is one thing the agent did that changed state or sent data out: a
refund, an email, a deployment. A **plane** is one route by which actions reach your systems; an
agent that calls tools through a gateway and also directly has two. A **witness** is one log, read
one way. Each `[[witness]]` block reads one file for one witness.

Top-level keys:

| key | meaning |
|---|---|
| `version` | Must be `1`. |
| `planes` | The names of the planes. Every plane needs two to five witnesses. |
| `actions` | The operation names that count as actions. A record whose operation is not listed is counted as `not_an_action`. |
| `env` | Optional. One of the study environments in `lacuna.envs.registry`, which turns a tool name and its arguments into an action. For your own logs, leave it out and use `op`, `resource` and `value`. |

Keys of a `[[witness]]` block:

| key | meaning |
|---|---|
| `name` | The witness. Several blocks may share a name: one witness read from several files, or once per plane. |
| `source` | The `NAME` in `--in NAME=PATH`. Defaults to `name`. Several blocks may read one source. |
| `format` | `jsonl`: one JSON object per line. `otlp-spans` or `otlp-logs`: OTLP JSON as the OpenTelemetry Collector's file exporter writes it, one export request per line. |
| `body` | For `otlp-logs` only: `body = "json"` parses each record's body, so paths may start with `body`. |
| `pipeline` | What delivers this log to you, in your words. `lint` counts the distinct names on a plane. If two logs pass through the same collector, give them the same name. |
| `plane` | A plane's name, or `{ path = ..., map = { "value" = "plane" } }` to read it from a field. A record whose value is not in the table is counted as `plane_not_mapped`. |
| `select` | A list of `{ path, equals }` tests. A record that fails one is not this witness's (`not_selected`). |
| `has` | A list of paths that must be present, or the record is `not_selected`. |
| `ok` | A list of `{ path, equals }` tests for success. A record that fails one is counted as `failed` and is not an action. |
| `session` | Path to the session or run id. The same refund in two sessions is two actions; without `session` the two can be merged into one. |
| `op`, `resource`, `value` | Paths to what the action was: the operation, what it acted on, and with what. All three are needed. |
| `tool`, `args` | With `env` only, in place of the three above: the tool name and its arguments. |
| `id` | A list of paths that together identify one action in every log of the plane that has them, such as a trace id and a request id. The id must be unique across the plane, not only within a session: if two sessions reuse request numbers, add the session or a trace id to the list. Leave it out for a log with no such id; its records are then matched by session, operation, resource and value. |
| `seq` | Path to the call's number within its session. It orders the records of witnesses with an id. |

A path is a key, or a list of keys for a nested field: `["meta", "run_id"]`. A key that holds dots
is one key: `["attributes", "gen_ai.tool.name"]`. Three rules to know:

- Equality is by type. `true` does not equal `1`. `200` equals `200.0`.
- A session, an id part, an operation, a resource or a value must be a string or a number. A list or
  an object there makes the record unusable, and it is counted so.
- Use an `id` only if something other than the agent wrote it. An agent can write any id it likes
  into its own log.

### A worked example: two JSON-lines logs that are not OTLP

An agent issues refunds and credits. You have two records of what it did. The first is the harness's
own log, one line per tool call, shipped from stdout:

<!-- file: demo/agent.jsonl -->
```json
{"event":"start","run":"r1"}
{"event":"tool_call","run":"r1","tool":"refund","order":"A-101","amount":"25.00","status":"ok"}
{"event":"tool_call","run":"r1","tool":"refund","order":"A-102","amount":"9.50","status":"ok"}
{"event":"tool_call","run":"r1","tool":"lookup","order":"A-103","amount":"","status":"ok"}
{"event":"tool_call","run":"r1","tool":"credit","order":"A-103","amount":"5.00","status":"ok"}
{"event":"tool_call","run":"r2","tool":"refund","order":"A-105","amount":"12.00","status":"ok"}
{"event":"tool_call","run":"r2","tool":"refund","order":"A-109","amount":"60.00","status":"declined"}
{"event":"tool_call","run":"r2","tool":"credit","order":"A-106","amount":"5.00","status":"ok"}
{"event":"tool_call","run":"r2","tool":"refund","order":"A-107","amount":"18.25","status":"ok"}
```

The second is an export of the payments database after the runs, in another shape:

<!-- file: demo/ledger.jsonl -->
```json
{"meta":{"run_id":"r1"},"kind":"refund","order_id":"A-101","amount":"25.00"}
{"meta":{"run_id":"r1"},"kind":"credit","order_id":"A-103","amount":"5.00"}
{"meta":{"run_id":"r1"},"kind":"refund","order_id":"A-104","amount":"40.00"}
{"meta":{"run_id":"r2"},"kind":"refund","order_id":"A-105","amount":"12.00"}
{"meta":{"run_id":"r2"},"kind":"refund","order_id":"A-107","amount":"18.25"}
{"meta":{"run_id":"r2"},"kind":"refund","order_id":"A-108","amount":"7.00"}
```

The map says where each field is in each log:

<!-- file: demo/orders.toml -->
```toml
version = 1
planes = ["payments"]
actions = ["refund", "credit"]        # a lookup changes nothing, so it is not listed

[[witness]]                           # the harness's own log of tool calls
name = "agent"
format = "jsonl"
pipeline = "harness stdout -> log shipper"
plane = "payments"
select = [{ path = "event", equals = "tool_call" }]
ok = [{ path = "status", equals = "ok" }]
session = "run"
op = "tool"
resource = "order"
value = "amount"

[[witness]]                           # rows exported from the payments database; no id
name = "ledger"
format = "jsonl"
pipeline = "database export"
plane = "payments"
session = ["meta", "run_id"]
op = "kind"
resource = "order_id"
value = "amount"
```

Save the three files in a folder `demo` and run:

```bash
lacuna lint --map demo/orders.toml
lacuna estimate --map demo/orders.toml --in agent=demo/agent.jsonl --in ledger=demo/ledger.jsonl
```

<!-- output of: lacuna lint --map demo/orders.toml -->
```
ok: two witnesses on every plane, by two pipelines as the map names them
```

<!-- output of: lacuna estimate --map demo/orders.toml --in agent=demo/agent.jsonl --in ledger=demo/ledger.jsonl -->
```
plane payments: union_count 8 (union_bound: at least 8), trust agent 6
       2  agent
       2  ledger
       4  agent+ledger
  estimate: 10 actions (interval 8 to 23)
agent (block 0, agent): {'read': 9, 'not_selected': 1, 'failed': 1, 'not_an_action': 1, 'kept': 6}
ledger (block 1, ledger): {'read': 6, 'kept': 6}
all planes: union_count 8; estimate: 10 actions (interval 8 to 23)
Estimator: strat_bic, which is not the one study 5 tested: that one needs two id-linked logs and one witness without an id on every plane, each with a session.
This assumes the witnesses miss actions independently and every record is a real action.
Witnesses that lose the same records make it too low; it can also be too high.
The interval is the estimator's own and was not calibrated on these logs.
```

Neither log has an id, so records are matched when session, operation, resource and value are equal.
`rig/map.toml` shows the other case: two OTLP logs that share a trace id and a request id, and a
state diff with none.

## Reading the output

**1. Read the lines per block first, near the end.** Each says what one `[[witness]]` block did with
its file. `read` is the number of records. Every record is counted once more under one of:
`not_selected`, `plane_not_mapped`, `failed`, `no_session`, `bad_arguments`, `not_an_action`,
`kept`. In the example the agent's log has 9 records: 1 is not a tool call, 1 was declined, 1 is a
lookup, and 6 are kept. Check these against what you know of the file first. A block that kept
nothing, or dropped most records for a reason you did not expect, means the map is wrong, and every
number above it is wrong with it. `unreadable_lines` counts lines that are not JSON objects, such as
a line cut short when a process was killed. `duplicates` counts second copies of a span delivered
twice.

**2. Then the floors.** They need no model.

- `union_count` is the number of distinct actions the logs hold together.
- `union_bound: at least N` is the same number read as a bound: at least that many actions happened,
  if every record is a real action. It has no upper end.
- `trust NAME N` is what the first witness in the map holds on its own. It is the number you would
  report if you trusted that one log. The gap between `trust` and `union_count` is what the other
  logs already show it missed.

**3. Then the table.** One line per combination of witnesses that recorded something. In the
example, 4 before `agent+ledger` means four actions are in both logs, and 2 before `agent` means two
are in the agent's log only. Combinations with no action are not printed. A witness with no record
on a plane is named in a line of its own.

**4. Then the estimate.** `estimate: N actions (interval LOW to HIGH)` is the estimated number of
actions that happened, counting those in no log, and the estimator's own interval (its 2.5% and
97.5% points). The interval was not checked against your logs. Read it as the range the model allows
if its assumptions hold, and read the next section before you repeat the number to anyone. There is
one estimate per plane and one for all planes together.

**5. "No upper bound".** `estimate: at least N (what the logs hold), no upper bound` means the tool
gives no estimate. You have the floor and nothing more. The next line says why. It is one of three
reasons: no witness recorded an action; the witnesses recorded the same actions, or only one
recorded any, so there is no disagreement to estimate from; or too few actions were recorded by more
than one witness to say how many all missed. None of the three means that nothing is missing. In
JSON the same case is `"bounded": false`, with `"actions": null` and the reason under
`"why_unbounded"`. With the tested estimator there is a fourth reason for the total: a plane has
actions and none of them is in both a log and the state diff, so nothing ties that plane's logs to
its state.

**6. Two things the tool does to keep an answer honest.** If the estimate falls outside its own
interval, which can happen because the estimate is a real number and the interval is searched over
whole counts, the interval's end is moved to the estimate and JSON carries
`"interval_extended_to_point": true`. And if link ids repeat across sessions on a plane, `estimate`
stops with exit status 2 and says so, because its per-session counts would no longer add up.

`--json` prints the same content under `planes` and `blocks`, and for `estimate` also `total` and
`assumes`.

## What the estimate assumes, and how each assumption fails

The tool cannot check any of these. `lacuna estimate` ends by repeating the first two.

- **The witnesses miss actions independently.** It fails when two logs lose the same records: both
  pass one collector, one host or one network path, and an outage takes a stretch out of both. The
  estimate is then too low, because the logs agree more than independent logs would. `pipeline`
  names in the map are your statement about this. The header of `rig/map.toml` describes a case
  where the names differ and the two pipelines still lost overlapping stretches.
- **Every record is a real action.** It fails when a failed call is counted (no `ok` test), when a
  retry is logged as a second action, or when a file holds another system's records. The union and
  the estimate are then too high.
- **A state record shows the lost actions as often as the others.** A database export or an
  end-state diff shows only what is still there at the end. An action that was later overwritten or
  undone leaves nothing. If the actions the logs lost are also the ones the state hides, the
  estimate is too low.
- **Each witness sees the whole plane.** If one log covers only part of the traffic on a plane, the
  estimate can be too high. Split the plane in the map.
- **The planes are every route.** An action by a route outside the map is not counted or estimated.
- **Records are matched correctly.** With an id, two logs are joined on it. Without one, records are
  matched in order on session, operation, resource and value, which finds as much overlap as the
  records allow and so leans the estimate low. A forged id, or a missing `session`, joins records
  that are not the same action.

## What has been tested, and what has not

One study tested the setting this tool is for: several logs of one system that lose records through
real telemetry components. It is registered in `docs/PREREG-STUDY5.md` and reported in
`docs/DECISIONS.md`, D92 and D93. Every number below is from
`artifacts/robustness/study5-verdicts.json`; the key is in brackets.

**The setting.** One rig on one laptop. Two planes. On each plane a span log, an access log and an
end-state diff, which is the layout of `rig/map.toml`. Records were lost by killing and restarting
the two collectors that relay the logs. The sessions were recorded once from language models working
synthetic tickets, then replayed. The registered test is of the estimator the artifact calls
`ll_sa_hom` (`primary.primary`).

`lacuna estimate` runs that estimator when every plane in the map has the study's layout: two logs
with an `id` and one witness without, each block giving a `session`. It says so under the estimate.
It reads the witness without an id as a state diff and cannot check that it is one: if yours is a
third log with no id, the estimate's assumption does not hold and the sentence it prints is wrong
about your case. On
any other layout it runs `strat_bic` and says that study 5 did not test it; `strat_bic` was computed
beside the tested one with no prediction registered for it, and its row is
`by_condition.s1.estimators.strat_bic` in `artifacts/study5/confirm.json`. The study scored the total
over planes, not a plane's own line. The library's `ll_sa_hom` differs from the registered code in
one case, fixed after the verdicts were read: where the registered fit gave no answer on a complete
log, it now uses the model's closed form (`docs/DECISIONS.md`, D94).

**What was found.**

- 300 runs of the tested condition were planned. 23 were set aside on integrity before the analysis
  (`runs_set_aside_by.s1.integrity`). Integrity is a check on the run itself, such as a replayed
  call that came late. That leaves 277 (`primary.gates.runs_passing_integrity`).
- In 269 of the 277 the interval held the true count (`primary.coverage`). The 95% Wilson interval
  for that share is 0.9441 to 0.9853. The registered bar was a lower end at or above 0.90, so the
  registered test passed.
- That pass is not to be quoted without the 23. With those counted as not covered the result is 269
  of 300, and its interval is 0.8571 to 0.9262 (`P1_not_in_test_as_uncovered`), which straddles the
  0.90 bar. On that count the bar is neither shown met nor shown missed. Whether the 23 would have
  been covered is not known, and what made those runs fail the check is not established (D93).
- None of the 277 intervals was without an upper end (`primary.coverage.unbounded`). In one run the
  interval lay above the true count (`false_alarms`); the other seven misses ended below it.
- The estimate was closer to the true count than `union_count` in 261 of 277 runs, with 3 ties, and
  closer than a count of gaps in sequence numbers in 223 of 277 (`primary.paired`). The first
  comparison is weak by construction: the union can never be above the true count.
- The floors, in the same 277 runs (`floors`): `union_count` equalled the true count in 2, with a
  median absolute error of 22.0 actions, against 3.949 for the estimate
  (`primary.coverage.median_abs_error`). `union_bound` held in all 277, as a lower bound on real
  records must. The span log alone, `trust_trace`, equalled the true count in none.
- Without the end state, from the two telemetry logs only, the interval held the true count in 10 of
  277 runs (`Q3_without_env.coverage`). This was predicted before the runs. It is the case `lint`
  warns about when every witness on a plane is a log.
- With no loss induced, 60 runs were planned and 58 were in the test (`control`; 2 set aside,
  `runs_set_aside_by.s0`). No interval lay above the true count. In one run the tested estimator
  gave no estimate and no upper end although the logs were complete (`control.coverage.unbounded`).
  The estimator in the tool was changed after the verdicts were read so that it answers there
  (`docs/DECISIONS.md`, D94); the study's figures are the registered code's.

**What was not tested.**

- An agent or a process that hides actions, writes false records or forges an id.
- Any other layout, loss mechanism or run length, and any log from a real deployment.
- The Wilson intervals above treat the runs as independent. The runs share sessions (D92), and the
  intervals do not account for that.

Four earlier studies used loss simulated on complete logs. The README's "Status" section reports
them, the failures included.

## Exit codes

| code | meaning |
|---|---|
| 0 | Done. For `estimate` this includes "no upper bound": read the output. |
| 1 | `lint` found an error in the map. No count can be made with it. |
| 2 | The map or an input cannot be read: a missing file, a file that is not UTF-8 text or not in the format the map says, a map that is not valid, a source with no `--in`, or a plane with fewer than two witnesses given to `table` or `estimate`. One line on stderr says which. |
