# Study 5, milestone 2: the two-plane rig

**Goal:** a docker compose rig that replays sessions over planes A and B, loses records
only in unmodified telemetry components, and writes witness files the `lacuna.core`
tables can be built from. It ends at the feasibility checkpoint of
`docs/DESIGN-STUDY5.md`, which is about the rig and never about an estimate.

**Spec:** `docs/DESIGN-STUDY5.md`, as amended by `docs/RIG-FINDINGS.md`. AGENTS.md
rules 2 to 6 apply to everything here.

**Not in this milestone:** recording LLM sessions, the pilot, the registration, the
tuned-queue condition, `telemetrygen` load, container limits, the CLI and its map file.

## Decisions taken while planning

1. **The retry window is read from the SDK and then checked.** The exporter's loop
   (`opentelemetry/exporter/otlp/common/http.py`, from `opentelemetry-exporter-otlp-common`
   0.66b2, which SDK 1.45.2's exporter pulls in) makes a first attempt and up
   to three retries inside a 10 s deadline, waiting `2**i * U(0.8, 1.2)` seconds before
   retry `i`. The last attempt for a batch is therefore 5.6 to 8.4 s after the first
   refusal. A sweep of outage lengths with clock-stamped SDK messages checks this
   (`rig/spikes/collector_kill`, output in `out-sweep/`).
2. **The session is a request header.** `x-lacuna-session`, set by the replay client,
   not by a model. An ad-hoc check showed agentgateway v1.7.0 forwards it upstream,
   forwards the caller's trace id, keeps JSON-RPC ids, and logs the header through an
   access-log expression. Every run's integrity check fails if a telemetry record
   lacks its session, so the check needs no artifact of its own.
3. **Both planes of a session act on one kernel.** A session's tickets are split
   between the orchestrator (plane A) and the sub-agent (plane B). The kernel service
   records which principal made each ledger entry, which is what gives an env row its
   plane: a "last modified by" column. The service computes it from the kernel's own
   record of each row's last writer (a ledger id) and hands the env witness only the
   principal.
4. **`McpBackend` is not built.** Its purpose was to let `run_agent` drive the rig. With
   sessions recorded once and replayed, recording can use the in-process router, whose
   tool results the MCP front returns unchanged. Whether recording goes through the rig
   is decided with the pools, in the next milestone.
5. **Access logs reach the collector as files.** Each receiver writes its access log to
   stdout; `docker logs --follow` copies stdout to a file in the run folder; the node
   collector tails the files with its `filelog` receiver at default settings. The file
   itself is complete and is not a witness. The witness is what the collector delivers.
6. **Witness files are written by a collector that is never stressed** (`backend`), so
   a restart cannot truncate them (the false finding of the first spike).
7. **Python for the rig lives in `src/lacuna/rig/`** behind the `rig` extra (`mcp<2`,
   OTel SDK and OTLP/HTTP exporter). `rig/` holds compose, configs, the image and pinned
   requirements. `lacuna.core`, `estimators`, `envs` and `ingest` still import with
   numpy alone.

## Files

| file | responsibility | deps |
|---|---|---|
| `src/lacuna/envs/service.py` | `KernelService`: one kernel per session behind a lock, principal per ledger entry, env rows with principal; a stdlib HTTP shell | stdlib |
| `src/lacuna/rig/sessions.py` | session format; `scripted_sessions` splits a scripted plan over planes | numpy |
| `src/lacuna/rig/front.py` | MCP front for an env: forwards calls to the kernel service, writes one access-log line per `tools/call` | mcp |
| `src/lacuna/rig/replay.py` | replays sessions over MCP with one span per tool call; resumes after a kill | mcp, otel |
| `src/lacuna/rig/stress.py` | seeded schedule of kills and restarts | stdlib |
| `src/lacuna/rig/manifest.py` | image digests, config hashes, commit | stdlib |
| `src/lacuna/ingest/otlp.py`, `jsonl.py` | OTLP JSON spans and logs, and JSON lines, to flat dicts | stdlib |
| `src/lacuna/rig/witnesses.py` | flat dicts to `Obs`, per witness; the link id and the key | lacuna.core |
| `rig/compose.yaml`, `rig/config/*`, `rig/Dockerfile`, `rig/requirements.txt` | the rig | docker |
| `scripts/rig_run.py` | one run: up, replay under a schedule, collect, summarise | docker |

## Tasks

Each task is test-first where a test can be hermetic. `make validate` must not need
docker; the rig is exercised by `make rig-smoke`.

1. **Retry window.** Stamp the spike's SDK messages with the clock, record kill and
   ready times, sweep outage lengths, write `artifacts/rig-spikes/retry_window.json`
   through `collect.py`, and state the window in `docs/RIG-FINDINGS.md`.
2. **Kernel service.** `open(session, seed, n_tickets)`, `execute(session, principal,
   tool, args) -> (ok, text)` with the router's result text, `ledger(session)`,
   `observe(session)` giving `(op, resource, value, principal)`. Tests: results equal
   the in-process router's on a scripted plan; two principals on one session get their
   own entries; concurrent calls keep the ledger consistent; unknown session and
   unknown principal are refused; the HTTP shell round-trips on loopback.
3. **Sessions.** `scripted_sessions(seed, n_tickets, planes)`: tickets dealt to planes
   in order, every call kept, the split deterministic. JSON round trip.
4. **MCP front.** Lists the env's tools with their schemas, forwards a call with the
   plane's key and the session header, returns the router's text and `isError`, logs
   `{ts, session, trace_id, rpc_id, tool, args, ok}`. Tested over the SDK's in-memory
   transport against a real `KernelService`.
5. **Replay client.** One span per tool call carrying tool, arguments, `isError`, the
   JSON-RPC id the client library assigned, and the session. Test (AGENTS.md rule 4):
   the id on the span equals the id the server received, for every call, and no value
   from the session file reaches the trace id or the request id.
6. **Ingest and witnesses.** Fixtures trimmed from a real smoke run. Spans are deduped
   on (trace id, span id). Failed tool calls and non-action tools are dropped. The link
   id is (trace id, JSON-RPC id); the key is (session, op, resource, value).
7. **Stress schedule and manifest.** Same seed, same schedule; the manifest hashes every
   config file and names every image by digest.
8. **Compose rig and `scripts/rig_run.py`.** `make rig-smoke` runs one unstressed run
   and prints N, per-witness counts and the link rate; `tests/test_ci_hermetic.py`
   forbids the new targets inside `validate`.
9. **Checkpoint.** Unstressed runs and stressed runs, an artifact per figure, the four
   checkpoint lines answered in `docs/RIG-FINDINGS.md`, a decision entry, then a review
   and a refutation pass.

## Checkpoint (from the design)

- the smoke run is green on both planes;
- at least 1% loss in a telemetry witness at default settings;
- link rate at or above 0.98 unstressed;
- `make validate` and `make repro-headline` green.

If it fails, plane B is dropped and plane A runs alone with the tuned-queue condition.
