# Design: study 5 (several originators, real log loss) and the v1.0 tool

Status: **design, not a registration.** First written 2026-11-15, before any study-5
code; amended the same day after two spikes and after a review and a refutation pass
(D75, D76), again when the rig was built (D78), and again after the first part of the
pilot (D79). A draft of the registration now exists (D80); it is not frozen. The registration will be `docs/PREREG-STUDY5.md`, frozen at a signed tag and deposited
before the first confirmatory run (D74). Anything here may change until then; nothing in
this file is a result.

## Why this study

Four studies left one criticism standing. D43 records it: if every action passes one
channel that numbers its entries, the gaps give N, so the live runs calibrate the
estimator without showing it is needed. D58 measured how far that goes: on unseen
operations at 1.2x loss a tail-corrected sequence count covers 0.952 where `ll_bic`
covers 0.254, and the same counter covers 0.0 under bursts and 0.002 with 5% of actions
bypassing the gateway. Two limits sit beside it. The README's: all log loss was
simulated after the fact on complete logs. And one the README does not state: the
gateway witness is a list the router appends to in the agent's own process
(`src/lacuna/envs/tools.py`), not a proxy on a network path.

Study 5 moves to the setting the method is for:

- **Several originators.** Actions come from separate processes that share no counter.
  Numbers exist only where a receiver assigns them, so loss before the receiver leaves
  no gap.
- **Real loss.** Records are lost by unmodified telemetry components under stress. No
  code in this repository drops a record.
- **Real hops.** Actions travel over the network through an unmodified open-source
  gateway and real MCP servers.

Scope condition, to be stated in the registration: **if a trusted counter exists at the
source or at the sink, use it.** The single kernel in the rig is itself a census at the
sink; it stands for several downstream systems whose logs are not collected, and that
is a stipulation, not a finding.

The author works on a commercial AI gateway. Nothing from that product is used: every
component is open source, pinned by digest, and all data is synthetic.

## Topology

```
ORIGINATOR             PLANE                 WITNESSES
orchestrator  ----->   A: gateway     --->   gateway access log
  agent                (agentgateway)        OTel spans (own SDK -> collector)
                                             env snapshot diff

sub-agent     ----->   B: direct MCP  --->   MCP front access log
  process              (no gateway)          OTel spans (own SDK -> collector)
                                             env snapshot diff

truth: the kernel ledger (unchanged)
```

A third plane (scripts in a sandbox with no SDK, witnessed by an egress proxy and an
exec audit log) is the next study, not this one. It needs id-less linkage and one
record per action from the audit log. D73 records what batch scripts did to coverage:
0.20 to 0.76 when they were not itemised, and 0.34 to 0.61 when a script's lines were
lost together.

### Witnesses

| witness | sees | how it is expected to lose records |
|---|---|---|
| OTel spans | the originator's own tool calls | SDK queue lost on kill; collector outage; export retries exhausted |
| receiver access log | calls that reached the gateway (A) or the MCP front (B) | log shipper outage or kill; rotation during restart |
| env snapshot diff | end state against start state | an action overwritten or reverted later leaves no trace |

- The env witness is a start-to-end snapshot diff. The earlier in-process diff saw
  nearly every action, which made the union count nearly exact and the comparison with
  it empty. The task mix includes updates and rollbacks so that a material share of
  actions is invisible to the snapshot.
- The third column is expectation. Only the first row's first two mechanisms were
  observed (`docs/RIG-FINDINGS.md`, one emitter); log shippers were not tried at all.
- JSON-RPC request ids give planes A and B a per-session number assigned at the source.
  The sequence-gap baseline uses them, which is its best fair case. By construction it
  cannot see a lost tail or a session lost whole; D58 lists other conditions where the
  counter failed in simulation.
- Each plane's two telemetry witnesses are routed through different pipelines, with the
  intent that one collector outage does not remove both. Whether their losses are in
  fact separate is for the pilot to measure.

## Rig

Docker compose, outside `make validate` (`tests/test_ci_hermetic.py` gains the new
targets in its forbidden list).

| service | role |
|---|---|
| `kernel` | hosts one `Kernel` per session behind a lock; not on the plane network (its admin port is published on the host's loopback, and every route but `/health` needs a key); source of truth |
| `mcp-a`, `mcp-b` | MCP fronts, one per plane, each with its own credential |
| `agentgateway` | unmodified, in front of `mcp-a` |
| `agent-a`, `agent-b` | originators; each replays recorded sessions with its own OTel SDK |
| `otelcol-gw` | relays the originators' spans to `backend` |
| `otelcol-node` | tails the access-log files and relays the lines to `backend` |
| `backend` | a collector that writes the two witness files; never stressed, so never restarted |
| `telemetrygen` | background span load (not built yet; the pilot decides if it is needed) |
| stress driver (host) | seeded kills and restarts at registered times |

- Output per run: `artifacts/rig/<run>/witnesses/*.jsonl` and a manifest of image
  digests and config hashes.
- Queue settings stay at their defaults. Stress is kills of originators and collector
  outages past the exporter's retry window, with background load and container limits
  as stressors on the collector. If the pilot shows under 1% loss, a tuned-queue
  condition is added and labelled secondary.
- Per-plane credentials give the kernel service a side map from ledger id to
  principal, which is how an env-only record gets a plane. `LedgerEntry` is unchanged.
- A session is one kernel world worked on from both planes at once: its tickets are
  split between the orchestrator (plane A) and the sub-agent (plane B). The session's
  name travels as a request header set by the replay client.
- An access log is the receiver's stdout, copied to a file by `docker logs --follow`
  and tailed by `otelcol-node` at default settings. The file is complete and is not
  the witness; the witness is what reaches `backend`.
- Agents are recorded once with real models (`deepseek-v4.1-flash`, `qwen3.8-flash`)
  and replayed in every run. A run is one realisation of loss over fixed behaviour.

## Estimation

- One 3-witness table per plane, combined with the existing `strat_bic`. A pooled
  table over all witnesses needs a cell mask the fit does not have, and its model
  space is not enumerable.
- `strat_bic` can be told which witnesses see a plane. It then reports the planes where
  a declared witness recorded nothing, which the counts alone cannot tell from a witness
  that could not look. This is a diagnostic: it changes no estimate.
- Linkage within a plane is by the pair (W3C trace id, JSON-RPC request id), with
  duplicates removed on span id. The trace id alone links a session, not an action
  (`docs/RIG-FINDINGS.md`, section 2). Env records attach by signature in order.
- **These ids and AGENTS.md rule 4.** The rule is that linkage never uses an id the
  agent wrote, because an agent that hides can forge one. The trace id and the JSON-RPC
  id are assigned by the client library in the originator's process; no model output
  reaches them, and that is asserted by a test on the replay driver. This is the same
  standing as the call id of D49. It is not protection against an originator whose
  process is itself hostile: such a process can stamp what it likes, and study 5 has no
  adversary. The registration lists that under "Not tested", and the signature-linked
  table is reported beside the id-linked one so the cost of not trusting the ids is
  visible.
- The primary estimator is fixed after a pilot on seeds disjoint from the confirmatory
  ones. Pilot data is published and excluded from the registered analysis.

## What the registration will contain

- **Questions.** Q1: does the interval cover under real, clustered loss? Q2: is the
  estimate closer than sequence-gap counting and the union count? Q3 (predicted to
  fail): the same without the env witness.
- **Conditions.** S0, no stress (60 runs; a control for false gaps). S1, the
  registered outage schedule (300 runs; primary). S2, heavy stress (60 runs;
  descriptive).
- **Unit.** One rig run of 24 concurrent replayed sessions. A run is one realisation of
  loss, so sessions inside it are not independent replicates and coverage is counted
  over runs.
- **Gates.** Underpowered: fewer than 270 completed S1 runs. Not exercised: mean
  (N − union)/N under 2%, or every telemetry witness under 1% loss. Integrity: the
  ledger differs from the accepted requests, or a manifest hash differs.
- **P1.** Wilson lower end of coverage at or above 0.90 over runs, and fewer than 5%
  of intervals unbounded.
- **P2.** Against each of `seq_gap_tail` and `union_count`: a paired sign test over
  runs on absolute error. Ties and unbounded estimates count against the estimator.
  Pass if the exact one-sided 97.5% lower bound on the win rate is above 0.5. Both
  must pass.
- **Floors, always (AGENTS.md rule 5).** Every table carries `trust_trace` (each
  plane's own spans), `union_count` and `union_bound`. If the estimator loses to a
  floor, the table says so.
- **Known traps, to be written into it.** BIC drops the telemetry-pair term at small
  counts and biases N low. `homogeneity_dispersion` cannot see an outage shared by
  every session in a run. A gateway log that does not mark failed tool calls creates
  phantom records. A gateway log without tool arguments cannot attach to env; the
  fallback is an attempted-action estimand.

Results are published whatever they are. If P1 or P2 fails, the paper is the map of
when log completeness can be estimated and when it cannot.

## Code

Additive only: the frozen study scripts import `lacuna.thinning`, `lacuna.linkage`
and `lacuna.witnesses`, and `make repro-headline` must stay byte-identical.

| new or changed | purpose |
|---|---|
| `src/lacuna/core/records.py` | `Obs` (witness by name, stratum, link id, time, sequence) and `WitnessSpec(name, sees)` |
| `src/lacuna/core/tables.py` | `build` (per-plane tables) and `keyed_table`; `linkage.callid_table` is left as it is, and a test pins the two equal on three witnesses |
| `src/lacuna/estimators/stratified.py` | `strat_bic(tables, live=None)`: declared visibility, reported as `detail["silent"]` |
| `src/lacuna/ingest/{otlp,jsonl}.py` | OTLP JSON and JSON-lines logs to flat records; the map file that turns them into `Obs` comes with the CLI |
| `src/lacuna/envs/service.py` | the kernel behind a network service, through the same `Router` an in-process episode uses |
| `src/lacuna/rig/{sessions,front,replay,stress,manifest,witnesses}.py` | the rig's Python: session files, the MCP front, the replay client, the stress schedule, config hashes, and run files to `Obs` |
| `src/lacuna/envs/churn.py` | the study's queue: later tickets overwrite or undo earlier ones, so the end state hides a share of the actions |
| `src/lacuna/rig/record.py`, `scripts/study5_record.py` | a session recorded from two model conversations over one kernel; a pool of them on disk |
| `src/lacuna/agents/backend.py` | `McpBackend`: not built. Sessions are recorded once and replayed, so recording can use the in-process router (D78) |
| `src/lacuna/mapfile.py`, `src/lacuna/cli.py` | a TOML map from log files to witness records; `lacuna estimate`, `lacuna table`, `lacuna lint` |
| `src/lacuna/envs/prompts.py`, `envs/registry.py` | the system prompt moves out of `agents/llm.py`, so environments and estimators import without `httpx`; only `agents/llm.py` needs it |
| `rig/`, `scripts/rig_run.py`, `scripts/rig_collect.py` | compose file, configs, the image; one run under a stress schedule; runs to a committed artifact |
| `scripts/study5_{lib,analysis,verdicts}.py` | a run's files to estimates, baselines and scores; runs to one artifact; the criteria applied, writing `artifacts/robustness/study5-verdicts.json` |
| `scripts/study5_pilot_summary.py` | every figure the pilot's write-up quotes across rounds, from the round artifacts |
| `scripts/study5_runs.py` | the study's 420 runs: the plan, a check that none would be refused, the registration's record of what makes a run, a launcher that never remakes one, a seal over the runs made |

### The tool

Built (D82): `src/lacuna/mapfile.py`, `src/lacuna/cli.py`, the console script `lacuna`,
and `rig/map.toml`, the rig's own logs written as a map.

- `lacuna lint --map m.toml` reads the map alone. It exits 1 if a plane has fewer than
  two witnesses, or if all its witnesses arrive by one pipeline, and warns about a log
  with no success test, no session, or no pipeline named, a plane with no record of
  state, and a plane where only one witness has an id. This is the first item of the
  workshop paper's deployer checklist ("Inventory the channels first") as a command.
- `lacuna table --map m.toml --in NAME=PATH ...` prints, per plane, how many actions
  each combination of witnesses recorded, the floors (what all witnesses hold, what the
  first one holds alone), and what each block of the map did with the records it read.
- `lacuna estimate` adds `strat_bic` per plane and over all planes, with its interval.
  When the witnesses recorded exactly the same actions it prints a floor and no
  ceiling, and says why.
- The map file is TOML: planes; per witness block a format (`otlp-spans`, `otlp-logs`,
  `jsonl`), the pipeline that delivers it, the plane (a constant, or a path and a
  table), equality tests that select its records and that mark success, and paths for
  the session, the id, the call order, and either tool and arguments (read through a
  registered environment's normaliser) or op, resource and value. A path is a list of
  keys. Nothing in a map is evaluated.
- `rig/map.toml` gives the tables `lacuna.rig.witnesses` gives, on the recorded
  fixture and on the same files with a different part of each log removed (tested).
- The command needs numpy and the standard library; importing it loads no agent, MCP
  or OpenTelemetry package (tested).
- The estimator behind `estimate` is the one the earlier studies registered. It is not
  study 5's primary estimator, which is for a layout with two telemetry logs and a
  state diff; which one the tool should default to is decided after the study.
- Extras: `agents` (httpx), `rig` (MCP and OTel SDKs). The version is still `0.0.0`:
  it changes with the release, when the distribution name is chosen.

## Order of work and the checkpoint

1. Spikes: what agentgateway logs per call (request id, arguments, failure), and what
   a collector kill loses at default settings.
2. Core records and tables; httpx decoupling.
3. Kernel service, MCP fronts, `McpBackend`, spans, replay driver.
4. Compose rig, stress driver, ingest, recorded fixtures.
5. **Checkpoint, on feasibility only and never on an estimate:** the rig smoke run is
   green on both planes; at least 1% loss in a telemetry witness at default settings;
   trace-id link rate at or above 0.98 unstressed; `make validate` and
   `make repro-headline` green. If it fails, plane B is dropped and plane A runs alone
   with the tuned-queue condition.
6. Record the pools; pilot; analysis scripts with tests.
7. Registration, review and refutation, signed tag, deposit, runs.
8. Verdicts, CLI and lint, card and claims map.
9. One paper replacing the workshop draft; release.

## Changed by the spikes (2026-11-15)

Two hand-run spikes (`docs/RIG-FINDINGS.md`) settled some open points. They are one
emitter at one rate on one laptop, so what follows is direction for the rig and the
pilot, not established fact.

- **Stress is kills and long outages.** A killed emitter lost the spans queued since
  its last export (28 to 32 here), and a 25 s collector outage lost two export batches.
  An 8 s outage lost nothing in four runs and one batch in a fifth, on retry jitter
  alone. The outage length at which loss starts was not located, so the rig measures the
  retry window before a schedule is registered, and keeps outages well clear of it.
- **At agent pace a kill loses one or two actions.** The amount lost is what was queued
  in the last 5 s at most. Whether kills and outages together clear the "not exercised"
  gate is the pilot's first question.
- **All observed loss was one contiguous block.** The pilot chooses an estimator for
  block loss, starting from the episode-level dispersion correction of D51. Faults that
  lose a tail (a killed originator) are reported apart from faults that leave an
  interior gap (a long outage).
- **The gateway's access-log settings are part of the registered configuration.** As
  shipped, the log has no arguments and gives a tool that raised the same line as one
  that succeeded. Settings in the gateway's own config add arguments, `isError` and the
  JSON-RPC id. The gateway's session key is set explicitly so its config file, and the
  file's hash, do not change at start.
- **The sequence-gap baseline counts requests.** JSON-RPC ids number every request of
  a session, not only tool calls. The registration says how the baseline turns missing
  requests into a count of actions.
- **A restarted component must not truncate a witness file.** The spike's first
  collector-kill runs reported about 0.30 missing that way, none of it real; a direct
  test then showed the collector's file going from 11,368 bytes to 0 on restart.

Figures in this section are from `artifacts/rig-spikes/collector_kill.json`.

## Changed by the rig (2026-11-15)

The rig for planes A and B was built and met its checkpoint (`docs/RIG-FINDINGS.md`,
sections 3 and 4; D78). Figures below are from `artifacts/rig/checkpoint.json`,
`artifacts/rig/checkpoint-first-round.json` and
`artifacts/rig-spikes/retry_window.json`. They are scripted sessions on one laptop, a
few runs per condition, so they are direction for the pilot.

- **The outage schedule has chance bands.** The SDK attempts a batch four times, the
  last 5.6 to 8.4 s after the first refusal. By the code an outage of the span relay
  under about 4.5 s cannot lose a batch and one over about 14 s must; between them the
  outcome is chance. A second, larger batch is at stake about 11 to 13 s later, so the
  checkpoint's 20 s outage lost one batch for seven originators and two for two. A
  registered outage sits outside these bands or the registration says they are sampled.
  The window was measured from a host process against a refused connection; in the rig
  the failure is a name that does not resolve, and its timing was not measured.
- **Telemetry loss is common-cause across planes, and comes in blocks.** Every loss in
  every checkpoint run is one unbroken block of calls. Both originators' export cycles
  are in step and both planes' access logs pass through one relay, so an outage removes
  one time window from every session on both planes. Per-plane tables do not make the
  planes independent. The pilot has to choose between staggering the originators and
  the relays and registering shared fate as the case studied; the unit stays the run.
- **One schedule does not give one amount of loss** (as measured then; see "Changed
  by the pilot"). Two runs of the same commit,
  configs, seeds, sessions and 20 s outage lost 0.168 and 0.557 of plane B's action
  spans. Across the two rounds one schedule went from 0.401 to 0.007, though the run
  driver changed between the rounds, so that pair is not a clean repeat. The sample
  size is set from the pilot's spread, not from these runs.
- **The "not exercised" gate would fail today.** With scripted tickets the env witness
  misses at most 0.081 of a plane's actions, and an action was in no log in two runs
  only (0.019 of all actions each). The recorded sessions need a task mix that leaves
  more actions invisible in the end state.
- **Linking is reported as two numbers.** Coverage (of the ids either log holds, the
  share both hold) is the checkpoint's "link rate"; it is 1.0 unstressed and falls with
  loss. Agreement (do two logs that share an id name the same action) was added because
  coverage cannot tell a linking fault from loss.
- **A run has an integrity check** (`lacuna.rig.integrity`): the stress events
  happened, originators exited as planned, every planned call reached its receiver, the
  access-log copies are complete, the witness files exist, no telemetry record lacks a
  session or matches no ledger entry, and each receiver confirmed the actions the
  ledger holds for its plane. A run that fails it is kept and not analysed. This is
  most of the registration's integrity gate in code. The part still missing is the
  comparison of a run's manifest (config hashes, image ids, source hash) with the
  registered one, which needs a registration to compare with.
- **Not handled:** a call cut off after the kernel acted can be in the ledger and in no
  log as a success. The integrity check counts it and allows it only on the plane of a
  killed originator; none occurred in these runs. It is listed for the registration.
- **Rule 3 and the env row's plane.** The kernel's diff names each row's last writer by
  ledger id, as it has since the first study. The service turns that id into a
  principal inside itself and hands the env witness the principal only: what a "last
  modified by" column holds. No ledger id leaves the service on that path.

## Changed by the pilot (2026-11-15)

The first part of the pilot ran on scripted sessions (`docs/RIG-FINDINGS.md`,
section 5; D79). Figures below are from `artifacts/study5/pilot-summary.json` and the
round files it is built from: 61 runs on one laptop, no model. The second part, on
recorded sessions, is not done. Where this section and "Changed by the rig" above
disagree, this one is later: the originators' export cycles are no longer in step, and
what one schedule's spread is has to be asked again (below).

- **The queue is `backoffice_churn`.** Same world, tools and kernel; later tickets
  correct, revert, delete or rewrite what earlier ones did. The end state then shows no
  trace of 0.35 to 0.52 of a plane's actions, where it showed all but 0.08 at most. The
  thread weights were set to give about that share, and threads do not touch each
  other's records, so in scripted sessions hiding is independent of the rest of the
  session by construction. Recorded sessions need not have that.
- **A session has one clock, and a run has one start.** Tickets are worked in queue
  order across the planes, and the replay client makes each call when it is due,
  counted from a start time both originators are given. Each originator waits a seeded
  time before it creates its SDK. Two schedules were made again with their seeds
  fixed, one inside a band of the batch table (four runs) and one in an overlap of
  two bands (five runs): each gave the same batch counts every time and tables within
  three per cell. The spread the rig's section reports for a 20 s outage was measured
  with the start left to chance; that pair was not made again.
- **The relays stay shared,** as the approved plan names for the primary case.
- **The key carries the value for every op.** An end state holds a file's content and a
  log holds the call's arguments, so both can hash it. A file written three times is
  three keys.
- **An action is in no log only where the two pipelines' lost blocks overlap** and the
  end state hides it. A restart of both relays for 15 to 24 s left a mean of 0.024 of
  actions in no log over six runs, three of them under 0.02; for 15 to 45 s in a longer
  run, 0.065 over 24 runs, three under 0.02. The first sits at the "not exercised"
  gate and the second clears it.
- **Span batches given up do not follow the outage's length alone.** One at 16 to 22 s,
  two at 21 to 41 s, three at 33 to 44 s, four at 43 to 45 s, and under five schedules
  the two originators gave up different counts for one outage. A schedule that draws its
  length from a range samples this; the registration says so.
- **Candidates for the primary estimator.** `strat_bic` as registered before;
  `ll_bic_hom` and `ll_bic_boot` (the planes pooled, a BIC-selected model, a
  dispersion-scaled or a bootstrap interval); and `ll_sa_hom`, one model named in
  advance (the spans-access interaction only) with the dispersion-scaled interval,
  added after the first round was seen. On the 24 runs of the proposed condition made
  after it existed, `ll_sa_hom`, `ll_bic_hom` and `ll_bic_boot` each cover 23 and
  `strat_bic` 22 with one unbounded. The coverage comes from the scaled interval. The
  case for the named model is an argument, that nothing is selected, and one run of
  the first round where a selected model put the count at 2.5 times the truth.
- **The sequence baseline reads both telemetry logs of a plane together and is told a
  session's first call id,** so a lost head is a gap it sees. Missing requests become
  actions at the share of seen calls that were successful actions. It never sees the
  end state and has no floor. It was exact with no loss; over the 15 runs with both
  relays down under 25 s the estimator was the closer in 10; where an outage took
  session tails it was short by 11 to 100 actions at the median; and it was above the
  truth in 10 of 48 stressed runs.
- **With an originator killed the estimate's mean error was +18 actions in six runs**
  (standard deviation 19, two near zero), unexplained. Kills belong in the heavy,
  descriptive condition, with that stated before the runs.
- **P1 is uncertain and the registration should say so.** At 300 runs the Wilson lower
  end reaches 0.90 at 281 covered. The pilot's interval on the proposed condition is
  0.798 to 0.993 (23 of 24). P2 against the sequence baseline needs 168 of 300; the
  pilot has 19 of 24, whose exact lower bound is 0.579.
- **The integrity check gained four things:** sessions that cannot be one run are
  refused (two pools recorded from one seed), a kill drawn for after the last call is
  refused, a call made later than it was due fails the run, and so does a containers'
  clock that disagrees with the driver's.

## Not in v1.0

The sandbox plane; a reader for Inspect eval logs (its tool and sandbox events are
written by one process to one file, so it can feed `lint` but not `estimate`); the
dataset update (D72's hostname rule refuses rig logs as they are); a frontier-model
pool; an adaptive adversary.
