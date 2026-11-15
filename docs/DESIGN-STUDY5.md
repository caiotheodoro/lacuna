# Design: study 5 (several originators, real log loss) and the v1.0 tool

Status: **design, not a registration.** First written 2026-11-15, before any study-5
code; amended the same day after two spikes and after a review and a refutation pass
(D75, D76). The registration will be `docs/PREREG-STUDY5.md`, frozen at a signed tag and deposited
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
| `kernel` | hosts one `Kernel` per session behind a lock; admin network only; source of truth |
| `mcp-a`, `mcp-b` | MCP fronts, one per plane, each with its own credential |
| `agentgateway` | unmodified, in front of `mcp-a` |
| `agent-a`, `agent-b` | originators; each replays recorded sessions with its own OTel SDK |
| `otelcol-gw` | spans and gateway OTLP, to a throttled backend, to a file |
| `otelcol-node` | tails access logs, to a file |
| `telemetrygen` | background span load |
| stress driver (host) | seeded kills and restarts at registered times |

- Output per run: `artifacts/rig/<run>/witnesses/*.jsonl` and a manifest of image
  digests and config hashes.
- Queue settings stay at their defaults. Stress is kills of originators and collector
  outages past the exporter's retry window, with background load and container limits
  as stressors on the collector. If the pilot shows under 1% loss, a tuned-queue
  condition is added and labelled secondary.
- Per-plane credentials give the kernel service a side map from ledger id to
  principal, which is how an env-only record gets a plane. `LedgerEntry` is unchanged.
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
| `src/lacuna/ingest/{otlp,jsonl}.py` | OTLP JSON and JSON-lines logs to `Obs`, driven by a map file |
| `src/lacuna/envs/service.py` | the kernel behind a network service, reusing `env.handlers(kernel)` |
| `src/lacuna/agents/backend.py` | `McpBackend`, a stand-in for `Router` so `run_agent` is unchanged |
| `src/lacuna/cli.py` | `lacuna estimate`, `lacuna table`, `lacuna lint` |
| `src/lacuna/envs/prompts.py`, `envs/registry.py` | the system prompt moves out of `agents/llm.py`, so environments and estimators import without `httpx`; only `agents/llm.py` needs it |
| `rig/`, `scripts/rig_run.py` | compose files, configs, the stress driver |
| `scripts/study5_{lib,analysis,verdicts}.py` | analysis, writing `artifacts/robustness/study5-verdicts.json` |

### The tool

- `lacuna estimate --map m.toml --otlp spans.json --log gateway=access.jsonl [--json]`
  prints the estimate and interval per plane and in total.
- `lacuna lint --map m.toml` lists every action channel against its recorders and
  exits 1 if any channel has fewer than two independent ones. This is the first item
  of the workshop paper's deployer checklist ("Inventory the channels first") as a
  command.
- The map file is TOML: per witness a format, a selector, field paths for op,
  resource, value, id, time and stratum, a success predicate, and the planes it sees.
- Extras: `agents` (httpx), `rig` (MCP and OTel SDKs). The version stops being
  `0.0.0`.

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
  its last export (31 to 34 here), and a 25 s collector outage lost two export batches;
  an 8 s outage lost nothing. The outage length at which loss starts was not located,
  so the rig measures the retry window before a schedule is registered.
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
  collector-kill runs reported 0.297 to 0.313 missing that way, none of it real; a
  direct test then showed the collector's file going from 11,368 bytes to 0 on restart.

## Not in v1.0

The sandbox plane; a reader for Inspect eval logs (its tool and sandbox events are
written by one process to one file, so it can feed `lint` but not `estimate`); the
dataset update (D72's hostname rule refuses rig logs as they are); a frontier-model
pool; an adaptive adversary.
