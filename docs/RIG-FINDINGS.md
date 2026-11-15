# Rig findings: two spikes, the retry window, the rig's checkpoint, and the two pilots

Measured 2026-11-15 on one laptop (macOS, Docker 29.9.3 as `docker version` printed it),
by hand, with the scripts in `rig/spikes/`. These are observations that shape the rig,
not results of a study: few runs, one machine, one emitter, one schedule, nothing
registered.

Every number in sections 1 and 2 is in `artifacts/rig-spikes/collector_kill.json` or
`artifacts/rig-spikes/gateway_fields.json`, written by `rig/spikes/collect.py` from the
spikes' output folders (AGENTS.md rule 2). Sections 3 and 4 were added when the rig was
built (D78) and name their own artifacts; section 5 is the pilot (D79) and is the
first place an estimator is run on the rig's output. The spikes were run twice. The numbers here
are the second run, made with the final scripts; D75 quotes the first, whose outputs
were not kept. A review and a refutation pass were run on the first version of this
file; D76 and D77 list what changed.

## 1. What a killed emitter and a killed collector lose at default settings

`rig/spikes/collector_kill/`. One Python process emits numbered spans with a 50 ms sleep
between them (about 18 a second in practice), through a `BatchSpanProcessor` and the
OTLP/HTTP exporter with every setting at its default (`opentelemetry-sdk` 1.45.2, Python
3.13.14), to a Collector that writes them to a file
(`otel/opentelemetry-collector-contrib:0.165.0`,
`sha256:e40d61f10c9a617cca1ffd1be4d7e2b3a4b0f0001798ff07e58cabe82e618cea`). The count is
of numbered spans in the file against numbers the emitter printed.

| condition | runs | emitted | missing | where | duplicates |
|---|---|---|---|---|---|
| nothing goes wrong | 5 | 600 | 0 in every run | | 0 |
| emitter killed (SIGKILL) 17 s in | 5 | 304 to 312 | 28 to 32 | one block: the last spans emitted | 0 |
| collector killed 12 s in, back 8 s later | 5 | 600 | 0 in four runs, 94 in one | one block (spans 183 to 276) | 0 |
| collector killed 12 s in, back 25 s later | 3 | 1200 | 313, 308, 322 | one block inside the run (e.g. spans 189 to 501) | 0 |

**Read the missing column as amounts, not rates.** The share of a run that goes missing
is set by the schedule I chose, not by the components:

- A killed emitter lost what it had queued since its last export. The SDK exports every
  5 s; the kill came about 2 s after an export, at about 18 spans a second: 28 to 32
  spans. Against a run cut off at 17 s that is about a tenth of the spans; the same kill
  100 s into a run would be under 0.02. The amount scales with the emission rate and
  with where in the 5 s cycle the kill lands (up to 5 s of spans).
- A 25 s outage lost two export batches, 308 to 322 spans (`sdk_failed_batches` is 2 in
  each run). Spans emitted later in the same outage arrived. As a share of the 1200 I
  chose to emit that is about a quarter.
- All runs of one condition share one phase against the 5 s export timer. They are
  repeats of one case, not samples of a distribution.

What the runs do show:

- **Every loss was one contiguous block.** Nothing was lost span by span, in any run.
- **A killed emitter loses its tail.** A tail leaves no gap in any numbering.
- **Whether an 8 s outage loses anything came down to retry jitter.** In all five runs
  the SDK made three retries after the first refused export. In the four runs that lost
  nothing the three delays summed to 6.98 to 7.65 s and the last attempt found the
  collector back. In the fifth they summed to 5.87 s (0.86, 1.80, 3.21), the last
  attempt was refused, the SDK logged `Failed to export spans batch due to timeout, max
  retries or shutdown`, and one batch of 94 spans was gone. The first version of this
  file said a short outage loses nothing, on five runs that all happened to fall on
  the other side. The length of outage at which loss starts was not located; it depends
  on phase and on the jitter.
- **No duplicates were seen, in a setting where none could occur.** Every failed attempt
  was `Connection refused` (`sdk_connection_refused_lines` equals the retries in every
  run), so no request reached a collector twice. Timeouts and errors after the collector
  has accepted a request were not exercised.
- **A SIGKILL on the collector did not lose what it had already written**, in the one
  case checked: 40 spans delivered two seconds before the kill were all in the file
  after it (11,368 bytes before and after; run `overwrite-test`). Spans delivered less
  than a second before a kill were not probed.

One error on the way, kept because it would have been a false finding. In the first run
of the spike, the five collector-kill runs reported about 0.30 missing, always a block
starting at span 0. Those spans had been delivered before the kill. The file exporter
overwrites its file when the collector starts: in the `overwrite-test` run the file is
11,368 bytes after the kill and 0 bytes after a restart with nothing sent. `run.sh` now
moves the file aside before restarting (`last_span_delivered_before_kill` is 182 to 192
in the 8 s runs). The erroneous outputs were not kept, so their figures in D75 have no
artifact.

### What this changes in the design

- **The rig has to measure the retry window before it registers an outage schedule.**
  The same 8 s outage lost nothing four times and a batch once. A schedule must be
  stated against a measured threshold, with outages well clear of it on either side,
  not against the SDK's documented 10 s timeout.
- **At agent pace a kill loses very little.** An originator making one tool call every
  few seconds loses the calls of its last 5 s at most: one or two actions per kill. The
  loss a run needs (the "not exercised" gate) has to come from how many originators are
  killed and from outages past the retry window, and the pilot has to show it does.
- **Background load is not what produces loss here.** It stays in the design as a
  stressor on the collector; whether the tuned-queue condition is needed is for the
  pilot to say.
- **Independent-loss models are the wrong starting point.** Block loss is the only kind
  observed. The nearest earlier results are the burst conditions (D50 to D54, where an
  episode-level dispersion correction was the repair) and, differently, the script
  lines lost together in D73. D58 also found the sequence counter covering 0.0 under
  simulated bursts, so block loss is hard for the baseline too. The pilot picks the
  estimator, and the unit of analysis is the run.
- **Tail loss and interior loss should be reported apart.** An outage leaves a gap that
  a numbering can show; a killed originator leaves none.
- **A restarted component must not truncate a witness file.**

Not measured: loss under memory pressure or CPU limits; a collector with its own queue
(a network exporter in place of the file exporter); more than one emitter; any rate but
one; log shippers; the outage threshold.

## 2. What agentgateway records for one MCP tool call

`rig/spikes/gateway_fields/`. agentgateway v1.7.0
(`cr.agentgateway.dev/agentgateway@sha256:60216aafb03d37f94e952ff3d51d2f630a65b10ecaebdc0a864882f92610c45b`),
unmodified, in front of a two-tool MCP server (Python `mcp` 1.30.0, streamable HTTP).
The client sends one fixed W3C `traceparent` on every request and makes two `tools/call`
requests: `cancel_order`, which succeeds, and `always_fails`, which raises and comes
back as a tool error (`isError: true`). The access log goes to stdout, one line per
request.

| question | as shipped | with configuration | the setting |
|---|---|---|---|
| tool name | yes (`gen_ai.tool.name`) | | |
| tool arguments | no | yes | `tool_args: 'mcp.tool.arguments'` |
| this failed tool call told from the successful one | no: both are `http.status` 200 | yes: `tool_result.isError` | `tool_result: 'mcp.tool.result'` |
| JSON-RPC request id | no | yes | `rpc_id: 'json(request.body).id'` |
| trace id equal to the caller's | yes (`trace.id`) | | |
| a number assigned by the gateway per request | no | no | none documented |

The settings are entries under `frontendPolicies.accessLog.add` in the gateway's own
config file; the binary is the released one. `config-fields.yaml` has five entries. Three
do the work in the table. Of the other two, `tool_error: 'mcp.tool.error'` added no field
for a tool that raised (the error is inside `tool_result`), and
`caller_traceparent: 'request.headers["traceparent"]'` is a diagnostic used below.

The two lines as shipped (`request_lines` of `default` in the artifact, which drops the
session id and source address; `level` and `scope` are left out here):

```json
{"time": "2026-11-15T07:09:10.324274Z", "gateway": "default/default", "listener": "mcp", "route": "internal/default", "http.method": "POST", "http.host": "localhost", "http.path": "/mcp", "http.version": "HTTP/1.1", "http.status": 200, "trace.id": "0af7651916cd43dd8448eb211c80319c", "span.id": "82bd77fecbbdd88b", "protocol": "mcp", "mcp.method.name": "tools/call", "mcp.target": "spike", "mcp.resource.type": "tool", "gen_ai.tool.name": "cancel_order", "duration": "2ms"}
{"time": "2026-11-15T07:09:10.330852Z", "gateway": "default/default", "listener": "mcp", "route": "internal/default", "http.method": "POST", "http.host": "localhost", "http.path": "/mcp", "http.version": "HTTP/1.1", "http.status": 200, "trace.id": "0af7651916cd43dd8448eb211c80319c", "span.id": "4fb6eb4781103001", "protocol": "mcp", "mcp.method.name": "tools/call", "mcp.target": "spike", "mcp.resource.type": "tool", "gen_ai.tool.name": "always_fails", "duration": "1ms"}
```

With the settings, the same two calls carry these as well (an excerpt: the other fields
are as above, and `caller_traceparent` is left out here):

```json
{"gen_ai.tool.name": "cancel_order", "tool_args": {"order_id": "O-123"}, "tool_result": {"content": [{"type": "text", "text": "cancelled O-123"}], "structuredContent": {"result": "cancelled O-123"}, "isError": false}, "rpc_id": 1}
{"gen_ai.tool.name": "always_fails", "tool_args": {"order_id": "O-456"}, "tool_result": {"content": [{"type": "text", "text": "Error executing tool always_fails: cannot cancel O-456"}], "isError": true}, "rpc_id": 3}
```

### What this changes in the design

- **The precondition for signature linkage on plane A holds.** Arguments can be logged.
  Whether every kernel tool's (op, resource, value) can be rebuilt from its arguments
  and result was not tried: the spike has one toy tool with one argument. The
  attempted-action fallback stays in the design until the rig shows it is not needed.
- **One kind of phantom record can be filtered.** A tool that raises is marked
  `isError` once `mcp.tool.result` is added; as shipped it is the same line as a
  success, so a rig on the default log would count it as an action. JSON-RPC errors,
  upstream failures, timeouts and gateway denials were not tried. The registration must
  name the access-log settings and their hash.
- **The caller's span id is not in the access log.** The log has the caller's trace id
  and the gateway's own `span.id`; the header the gateway's expression sees already
  holds the gateway's span id (`…-9f41356aa33b2324-01` where the client sent parent
  `b7ad6b7169203331`). So where one trace covers several calls, which is the usual shape
  of an agent run, the trace id links the run and not the action, and the link id for
  one action is the pair (trace id, JSON-RPC id), with the caller's span recording the
  JSON-RPC id of the call it made. In this spike every request carried the same trace id
  because the client sent one fixed header; a client that starts a trace per call was
  not tried.
- **The sequence-gap baseline counts requests, not actions.** JSON-RPC ids are assigned
  by the client per session and shared by every request: here `initialize` was 0, the
  first tool call 1, a `tools/list` 2, the second tool call 3. A gap says a line is
  missing, not that a tool call is. The registration has to say how the baseline turns
  missing requests into a count of actions.
- **The gateway writes to its own config file.** With no session key set, v1.7.0 wrote a
  generated `config.session.key` into the mounted file. The rig sets the key explicitly
  so the config's hash is stable and nothing secret-shaped is committed.
- **Pin the MCP SDK.** A first install without a pin brought `mcp` 2.3.0, which failed at
  import with "This is mcp 2.x, where FastMCP was renamed to MCPServer"; the spike runs
  on `mcp<2` (1.30.0). The rig picks one major version on purpose.

Not checked: whether the gateway forwards `traceparent` to the upstream MCP server (the
rewritten header suggests it does); the gateway's log path under load or when its
container is killed; the direct-MCP plane, which has no gateway and whose access log is
whatever the MCP front writes.

## 3. The retry window, read from the SDK and then measured

Section 1 left one thing open: how long a collector outage has to be before a span
batch is lost. The answer is in the exporter's code and a sweep agrees with it.

**From the code** (`opentelemetry/exporter/otlp/common/http.py`, which ships in
`opentelemetry-exporter-otlp-common` 0.66b2, the package the 1.45.2 OTLP/HTTP exporter
pulls in; the defaults are `_DEFAULT_TIMEOUT = 10.0`, `_MAX_RETRIES = 6`,
`_DEFAULT_JITTER = 0.2`). An export gets a 10 s deadline. Each attempt is up to two
requests, because a connection error is retried once at once. After a refused attempt `i` (0, 1, 2, ...) the exporter
waits `2**i` seconds times a uniform draw on [0.8, 1.2], and gives up at once if that
wait is longer than what is left of the deadline. The first three waits sum to between
5.6 and 8.4 s; a fourth (6.4 to 9.6 s) never fits. So a batch is attempted four times,
the last between 5.6 and 8.4 s after the first refusal, and it is delivered if the
collector is back by that last attempt.

Two things set the outcome of one outage, and the outage's length is only part of one:

- how long after the collector dies the first attempt comes, which is anywhere in the
  5 s export cycle;
- the three draws.

**The sweep.** `rig/spikes/collector_kill/run.sh kill-collector` with `OUT_ROOT=out-sweep`,
900 spans, 27 runs; the SDK's messages are stamped with the clock and the script records
when the collector was killed and when it answered an export again. Numbers are in
`artifacts/rig-spikes/retry_window.json`.

| outage asked for | kill at | first refusal after the kill (all runs at that kill time) | runs | runs that lost a batch | spans lost |
|---|---|---|---|---|---|
| 4 s | 12 s | 2.9 to 3.2 s | 3 | 0 | |
| 6 s | 12 s | 2.9 to 3.2 s | 3 | 0 | |
| 8 s | 12 s | 2.9 to 3.2 s | 3 | 0 | |
| 10 s | 10.5 s | 4.5 s | 3 | 0 | |
| 10 s | 12 s | 2.9 to 3.2 s | 3 | 2 | 89, 91 |
| 10 s | 13.5 s | 1.5 s | 3 | 3 | 88 to 90 |
| 12 s | 12 s | 2.9 to 3.2 s | 3 | 3 | 93 to 95 |
| 14 s | 12 s | 2.9 to 3.2 s | 3 | 3 | 92 to 93 |
| 17 s | 12 s | 2.9 to 3.2 s | 3 | 3 | 88 to 90 |

- **The timing holds in every run that can test it.** In the 24 runs where the SDK
  announced three waits, a batch was lost exactly when the first refusal plus the three
  waits (`first_batch_last_attempt_after_kill_s`) came before the collector answered
  again (`ready_after_kill_s`): 14 lost, 10 delivered, no exception. In the other three
  (the 4 s outage) an earlier attempt got through. The three waits summed to 5.95 to
  8.08 s. That an attempt succeeds when the collector is up is no finding; what the
  sweep confirms is the arithmetic of when the attempts come.
- **The same 10 s outage lost nothing or a batch depending on when it started.** Killed
  4.5 s before the next export it lost nothing in three runs; killed 1.5 s before, a
  batch in all three; in between, two of three.
- **The first lost batch is one export cycle of spans**, 88 to 95 here at about 18
  spans a second. The batch after it is bigger. The SDK's worker is blocked while a
  batch retries and then waits a full cycle before the next export
  (`opentelemetry/sdk/_shared_internal/__init__.py`), so the next batch holds the
  retry time plus a cycle. In the 17 s runs the first batch was given up 9.7 to 10.3 s
  after the kill and the second was first refused 14.7 to 15.3 s after it; its retries
  reached the collector, so 17 s lost one batch. Section 1's 25 s outage lost two, 308
  to 322 spans in one block, which makes the second batch about 220 spans.
- The collector answered again 0.2 to 1.1 s after it was started.

### What this changes in the design

- **By the code, an outage under about 4.5 s cannot lose a span batch and one over
  about 14 s must.** The lower figure is the shortest three waits (5.6 s) with a first
  refusal at once, less up to 1.1 s for the collector to answer; the upper is 5 s of
  cycle plus 8.4 s of waits. The sweep tried a short outage at one phase only (4 s,
  first refusal 3 s in), so the lower end is read, not measured. Between the two the
  outcome is chance. A registered schedule uses lengths outside that band, or says the
  band is part of what is being sampled.
- **A second batch is lost only if the outage outlasts the first by a cycle and three
  more waits**, about 11 to 13 s more, and it holds about 12 s of an originator's
  spans, not 5. So the amount lost does not grow evenly with the outage: it jumps.
- **This is the refused-connection case, from a host process.** Inside the rig an
  originator reaches the collector by a container name, and while the container is dead
  the name does not resolve; the SDK retries that too (`sdk_export_messages` in
  `artifacts/rig/checkpoint.json`: all but one of the SDK's transient-error messages in the
  rig were name-resolution errors). Whether a failed lookup takes as long as a refused
  connection was not measured, and the rig records how many batches each originator
  gave up, not when each attempt was made. So the window above is not re-measured in
  the rig; section 4 has the rig's own counts.

## 4. The rig's checkpoint

The rig is `rig/compose.yaml`, driven by `scripts/rig_run.py`: two originators replay
sessions over MCP, plane A through agentgateway and plane B straight to an MCP front,
against one kernel service that keeps the ledger. Spans go from each originator's SDK
through a relaying collector to a backend collector's file; access logs are the
receivers' stdout, tailed by a second relaying collector into the same backend. A run
may kill the two relays and an originator, and nothing else.

The design set a checkpoint for the rig, on feasibility and never on an estimate. This
section answers it. Every number is in `artifacts/rig/checkpoint.json`, written by
`scripts/rig_collect.py` from 15 runs: five conditions, three runs each, 24 scripted
sessions of 7 tickets (no model), a mean of 3 s between a session's calls, each run 41
to 43 s. All 15 were made from one committed tree (`65c91c9`), with one source hash,
one set of config hashes and one set of image ids, and all 15 pass the integrity check
of `lacuna.rig.integrity`, both as the driver ran it and as the collector applies its
file checks again. The summaries are computed by the collector with later code than
made the runs; the artifact names both commits (`summarised_at`, and `commit` per run).
The conditions are in `lacuna/rig/stress.py`. They are not a registered schedule, and
the three runs of a condition differ in sessions and in kill times, so they are three
cases, not three replicates.

"Missed" is the share of a plane's ledger entries a witness has no record for.

| run | actions (A, B) | spans missed (A, B) | access missed (A, B) | env missed (A, B) | in no log, share of all | span batches given up (A, B) |
|---|---|---|---|---|---|---|
| none-0 | 158, 149 | none, none | none, none | 0.032, 0.081 | none | 0, 0 |
| none-1 | 157, 138 | none, none | none, none | 0.064, 0.072 | none | 0, 0 |
| none-2 | 159, 152 | none, none | none, none | 0.031, 0.079 | none | 0, 0 |
| gw-outage-0 | 158, 149 | 0.165, 0.168 | none, none | 0.032, 0.081 | none | 1, 1 |
| gw-outage-1 | 157, 138 | 0.191, 0.181 | none, none | 0.064, 0.072 | none | 1, 1 |
| gw-outage-2 | 159, 152 | 0.013, 0.007 | none, none | 0.031, 0.079 | none | 1, 1 |
| node-outage-0 | 158, 149 | none, none | 0.418, 0.376 | 0.032, 0.081 | none | 0, 0 |
| node-outage-1 | 157, 138 | none, none | 0.338, 0.377 | 0.064, 0.072 | none | 0, 0 |
| node-outage-2 | 159, 152 | none, none | 0.358, 0.375 | 0.031, 0.079 | none | 0, 0 |
| agent-kill-0 | 158, 15 | none, 0.867 | none, none | 0.025, none | none | 0, 0 |
| agent-kill-1 | 157, 35 | none, 0.286 | none, none | 0.057, 0.029 | none | 0, 0 |
| agent-kill-2 | 159, 34 | none, 0.088 | none, none | 0.019, 0.029 | none | 0, 0 |
| mixed-0 | 158, 19 | 0.165, 0.895 | 0.392, 0.579 | 0.025, none | none | 1, 0 |
| mixed-1 | 157, 51 | 0.369, 1.000 | 0.344, 0.941 | 0.057, 0.039 | 0.019 | 2, 1 |
| mixed-2 | 159, 52 | 0.403, 0.981 | 0.333, 0.981 | 0.025, 0.038 | 0.019 | 2, 0 |

`gw-outage` takes the span relay down for 20 s, `node-outage` the log relay for 10 s,
`agent-kill` kills plane B's originator for good, `mixed` does all three. Each starts 4
to 16 s into the run at a seeded time.

### The four lines of the checkpoint

1. **The rig runs on both planes.** The design's wording is "the rig smoke run is
   green"; `make rig-smoke` passes but writes no committed artifact, so the evidence
   here is the three unstressed runs. In them every call a receiver wrote down was
   delivered by both pipelines (255, 255 and 256 calls on plane A; 222, 211 and 224 on
   plane B), and in all 15 runs no witness holds a record that matches no ledger entry,
   no record lacks a session, and each receiver confirmed exactly the actions the
   ledger holds for its plane.
2. **At least 1% loss in a telemetry witness at default settings: yes, in every stressed
   run.** The smallest is gw-outage-2, where plane A's spans miss 0.013 of its actions.
   That is 2 actions of 159: the lost batch held 15 calls, most of them reads. No queue,
   retry or batch setting of the SDK or of a collector is changed anywhere.
3. **Linking.** The design asked for a link rate of at least 0.98 unstressed. The
   number first computed for it is what is now called *coverage*: of the ids either
   spans or the access log hold, the share both hold. It is 1.0 in all six unstressed
   plane-runs, so the line is met as first defined. A review pointed out that coverage
   falls with loss as much as with a linking fault, so a second number was added.
   *Agreement*: of the actions both hold under one id, the share where the two name the
   same action. It is 1.0 in 29 of the 30 plane-runs; in the other (mixed-1, plane B) no
   action was in both. In this rig both logs record the same call, so agreement can only
   fall if the id is wrong, and here it never was. It says little else.
4. **The gates.** `make validate` passes and `make repro-headline` reproduces
   (`all_passed=True`, with the drift lines it has printed since before this work).
   Their counts are in D78.

The checkpoint is met. Plane B stays.

### What the runs show beyond the checkpoint

- **Every loss in every run is one unbroken block of calls.** From the first lost call
  to the last, the receiver wrote no call that was delivered (`written_between` equals
  `calls` in each `lost_calls_written_at_s`).
- **A 20 s outage of the span relay loses one batch per originator or two, and how many
  actions that is depends on what the batches held.** Nine originators sat through a
  whole outage (six in `gw-outage`, plane A's three in `mixed`). Seven gave up one
  batch and two gave up two, as section 3 expects of a length where the second batch is
  chance. One batch was 39 to 43 calls where it was the cycle from 5.4 to 10.4 s into
  the run, and 15 or 19 calls where it was the cycle before (2.7 to 5.4 s); two batches
  were 108 and 116 calls, written over 14 and 15 s.
- **Both originators lose the same window.** They start together and their export
  cycles are in step, so in each `gw-outage` run the two planes' lost blocks start
  within 0.2 s of each other and end within 0.2 s. All 15 runs share that one phase.
- **The same schedule does not lose the same amount twice.** The cleanest case is in
  the first round of these runs, kept as `artifacts/rig/checkpoint-first-round.json`:
  `cp-gw-outage-0` and its repeat `diag-gw-outage-0` have the same commit, config
  hashes, seeds and sessions, and plane B's spans missed 0.557 of its actions in one
  and 0.168 in the other (plane A, 0.165 in both). Between the rounds the spread is
  wider (gw-outage-2 lost 0.403 and 0.401 of the two planes' action spans then, 0.013
  and 0.007 now), but the rounds are not repeats of each other: the run driver changed
  in between, including the moment the schedule counts from, and a kill moved across an
  export tick changes which batch is refused. The first round's cp runs kept no stderr,
  so how many batches they gave up is not known.
- **An outage of the log relay loses the lines written while it was down.** In the
  three `node-outage` runs the lost block (84 to 98 calls per plane) starts at most
  0.3 s before the driver's kill returned and ends at most 0.3 s after its restart
  returned. That fits the `filelog` receiver as documented and as configured here, with
  no checkpoint and reading from the end of a file that exists when it starts. No run
  changed the setting, so the runs fit that cause and do not prove it. Both planes'
  access logs pass through the one relay, so this loss is shared across planes too.
- **A killed originator loses the spans of its last seconds**, here a block that
  starts 0.6 to 2.7 s before the kill and runs up to it (5 to 21 calls). It also stops acting: plane B has 15 to
  35 actions in those runs against 138 to 152 unstressed, so the share lost is large
  (0.088 to 0.867) and the table is small.
- **The env witness is still close to a census.** With scripted tickets it misses at
  most 0.081 of a plane's actions, so an action is in no log only when both telemetry
  witnesses lose it: 4 of 208 and 4 of 211 actions in two `mixed` runs, none anywhere
  else. The registration's "not exercised" gate asks for a mean of 2% in no log. No
  condition here reaches it. The task mix of the recorded sessions has to make the end
  state blinder (overwrites and reverts), as the design already says, and the pilot has
  to show it does.
- **Telemetry loss here is common-cause, across sessions and across planes.** One relay
  outage removes a window of every session's records on both planes. A table per plane
  does not make the planes independent, and sessions inside a run are not replicates.

### Not handled, and limits

- A call cut off after the kernel acted (an originator killed mid-call, or a forward
  that times out) is in the ledger and may be in no log as a success. The integrity
  check holds a plane to its planned calls only when its originator was not killed.
- `docker logs --follow` copies a receiver's stdout to the file the log relay tails.
  It is plumbing of this rig, not a telemetry component; each run checks that the
  copy has as many lines as the container wrote, and fails if not.
- Scripted sessions on one laptop, three runs per condition. The figures say the rig
  works and how it loses records. They say nothing about any estimator.
- The raw runs are not committed (they are large and hold container addresses), so the
  summaries cannot be recomputed from the repository alone. The artifact names the
  commit that made each run and the commit that summarised it.
- Every run has the same export phase, because both originators start together and the
  schedule counts from their start. (True of this section's runs. The driver has since
  given each originator a seeded wait; see section 5.) Which batch an outage hits is then fixed by the
  kill time. A study has to vary the phase or say it is fixed.
- The forged-id test covers the replay client and the MCP front. agentgateway's own
  expression for the JSON-RPC id (`json(request.body).id`) is exercised only by the
  runs, where it agreed with the spans for every linked action.
- On Docker Desktop the schedule's zero is the daemon's clock and the driver sleeps on
  the host's. Events were done 0.09 to 0.27 s after their scheduled time here.
- The plane of an env row comes from the kernel service's own bookkeeping: the kernel's
  diff names each row's last writer by ledger id, and the service turns that id into a
  principal before the row leaves. The env witness is handed the principal, which is
  what a "last modified by" column holds, and never the id.
- Not built: background load, container limits, a tuned-queue condition, more than one
  originator per plane, recorded model sessions.

## 5. The pilot, on scripted sessions

The pilot is allowed by the design to fix an estimator and a sample size on seeds the
confirmatory runs will not use. This section is its first part: sessions scripted from
the tickets' own plans, no model. The second part, on sessions recorded from models, is
not done: the recorder is built and tested (`lacuna.rig.record`,
`scripts/study5_record.py`), and its first call was refused by the provider for want of
an active subscription, so nothing was recorded and nothing was spent. What follows is
direction for the registration. Nothing here is registered and nothing is a result.

Every figure is in `artifacts/study5/`. `pilot-round1.json` to `pilot-round4.json`,
`pilot-repeats.json` and `pilot-repeats-band.json` hold the scored runs
(`scripts/study5_analysis.py`);
`pilot-rig.json` holds the rig's own summary of the same runs
(`scripts/rig_collect.py`); `pilot-summary.json` holds every total, split and band
quoted below that spans more than one of those files
(`scripts/study5_pilot_summary.py`). 61 runs: 54 in four rounds and 7 repeats of two of
them. All pass the integrity check and none is set aside. Each round was made from a
clean tree, at commits `e823c54`, `e37a200`, `d17650f` and `e226faf`; the second set of
repeats at `25b74c0`. Rounds 3 and 4
exist because two code reviews changed the driver; each is six more runs of round 2's
condition on new sessions. Session seeds are 60000 and up, in an environment no earlier
run used.

### The task mix

With the tickets of the earlier studies the end state showed nearly every action, the
union of the logs was nearly exact, and the registration's "not exercised" gate would
have failed (section 4). `lacuna.envs.churn` is a queue over the same world, tools and
kernel in which later tickets overwrite or undo earlier ones: an address corrected, a
trial that ends, a phone number restored, a draft emailed and deleted, a handoff note
rewritten. The kernel's diff is untouched and nothing drops a record. In the 48 pilot
runs with no originator killed, the end state and the sink show no trace of 0.35 to
0.52 of a plane's actions.

Two things about that share are mine and not the world's. The thread weights were set
to put it near two fifths. And no two threads of a queue touch the same field, order or
file, so whether an action is hidden depends on its own thread and on nothing else in
the session. That is close to the independence the estimators below lean on, by
construction. A model's sessions need not have it.

A session now has one clock. Tickets are worked in queue order, each by the plane it
was dealt to, so a correction handled on plane B finds the address plane A wrote. The
replay client makes each call when it is due on that clock, counted from a start time
both originators are given. Each originator waits a seeded 0 to 5 s before it creates
its SDK, so the two export cycles are not in step.

### What each condition loses

24 sessions of 10 tickets a run, a mean of 1 s between a session's calls. In round 1
the sessions start over 40 s and a run lasts 58 to 74 s; in rounds 2 to 4 they start
over 100 s and a run lasts 113 to 130 s. "In no log" is the share of a run's actions
that neither telemetry log delivered and the end state does not show. Shares are
shares of run lengths I chose; the counts are in the artifacts.

| condition (round) | runs | actions per run | in no log, mean (range) | spans missed, mean | access missed, mean |
|---|---|---|---|---|---|
| `none` (1) | 4 | 358 to 372 | none | none | none |
| `node-blip` (1): both relays down 15 to 24 s | 6 | 357 to 376 | 0.024 (0.006 to 0.051) | 0.075 | 0.392 |
| `node-down` (1): both down 28 to 45 s | 6 | 364 to 372 | 0.197 (0.087 to 0.299) | 0.452 | 0.747 |
| `relays` (1): each relay its own outage | 6 | 361 to 384 | 0.186 (none to 0.295) | 0.650 | 0.634 |
| `node-down-kill` (1): `node-down`, one originator killed | 6 | 216 to 297 | 0.138 (0.083 to 0.181) | 0.510 | 0.538 |
| `none` (2) | 2 | 373, 373 | none | none | none |
| `node-restart` (2): both down 15 to 45 s, from 8 to 90 s in | 12 | 359 to 381 | 0.062 (0.010 to 0.132) | 0.173 | 0.249 |
| `node-restart` (3) | 6 | 362 to 378 | 0.062 (0.011 to 0.180) | 0.161 | 0.263 |
| `node-restart` (4) | 6 | 358 to 381 | 0.075 (0.024 to 0.148) | 0.181 | 0.307 |

- An action is in no log only where the two pipelines' lost blocks overlap and the end
  state hides it. Both relays being down at once is not enough: in `relays-4` they
  were, and the spans and the access lines lost were of different calls, so every
  action is in some log.
- The short restart is at the gate, not past it: a mean of 0.024 over six runs with a
  standard deviation of 0.017, three of the six under 0.02. The longer run with a
  restart of 15 to 45 s has a mean of 0.065 over its 24 runs, three of them under 0.02.

### Batches given up, as the rig shows them

Section 3 measured the retry window from a host process against a refused connection.
In the rig the failure is a name that does not resolve: of 884 transient export errors
the SDKs logged in the 61 runs, 882 name a resolution failure and 2 a refused
connection. Span batches given up per originator, against how long the span relay was
down (`pilot-summary.json`, `batches`; 104 originator-runs; runs with no outage and
killed originators are left out; a schedule made several times is counted each time):

| batches given up | span relay down, s | originator-runs | distinct schedules |
|---|---|---|---|
| 1 | 16.2 to 22.2 | 25 | 12 |
| 2 | 21.1 to 41.0 | 40 | 22 |
| 3 | 32.8 to 43.8 | 35 | 16 |
| 4 | 42.7 to 44.9 | 4 | 3 |

The bands overlap widely: two batches were given up in outages from 21 to 41 s and
three in outages from 33 to 44 s. Under five schedules the two originators saw one
outage and gave up different counts. So the count is not a function of the outage's
length; where in each originator's export cycle the outage begins matters as much. Section 3's
"about 11 to 13 s more for a second batch" is neither confirmed nor replaced by this
table. No outage under 15 s was run.

### The same sessions and schedule, made again

Two schedules were made more than once, with the same sessions and warm-up seeds
(`pilot-summary.json`, `repeats`).

| | span relay down, s | where it sits | runs | batches given up (A, B), every run | union counts | largest difference in a table cell |
|---|---|---|---|---|---|---|
| `pt2-node-restart-0` and 3 repeats | 37.0 | inside the three-batch band | 4 | 3, 3 | 344, 345, 344, 345 of 378 | 2 |
| `pt2-node-restart-2` and 4 repeats | 21.1 | in the overlap of the one- and two-batch bands | 5 | 1, 2 | 347, 347, 350, 347, 348 of 375 | 3 |

The second was made after the refutation of this section pointed out that the first
sat where the count could not vary. In it the two originators give up different
counts, and they gave up the same different counts all five times. So the difference
between originators under one outage is their export phase, which the warm-up seed
sets, and not chance, on this schedule.

The lost blocks are not identical between repeats: on the first schedule the last call
whose span was lost was written at 97.1, 98.4, 98.5 and 99.9 s on plane A. The SDK's
retry jitter is not seeded.

Two schedules and nine runs say a run of this rig repeats to within a few records when
its seeds are fixed. They do not say it of every schedule: one that puts an export
attempt within the jitter of a retry deadline could still go either way, and none was
looked for. The spread section 4 reported for one 20 s schedule (0.168 and 0.557 of
plane B's spans) was measured with the originators' start left to chance, which is the
difference from these runs; that exact pair was not made again.

### Estimators and baselines

Per run, every candidate is given the two planes' tables and, where it uses them, the
same tables per session. The sequence baseline is given the JSON-RPC ids on the
delivered telemetry records and nothing from the end state. All are scored against
that run's ledger (`scripts/study5_lib.py`). "Covered" counts runs whose interval holds
the true count. The four `node-restart` columns are one condition in three rounds.

| | `none` | `node-blip` | `node-down` | `relays` | `node-down-kill` | `node-restart` (2) | (3) | (4) |
|---|---|---|---|---|---|---|---|---|
| runs | 6 | 6 | 6 | 6 | 6 | 12 | 6 | 6 |
| `ll_sa_hom` covered | 6 | 6 | 6 | 6 | 5 | 12 | 5 | 6 |
| `ll_bic_hom` covered (unbounded) | 6 | 6 | 6 (1) | 6 | 5 | 12 | 5 | 6 |
| `ll_bic_boot` covered | 6 | 6 | 5 | 5 | 4 | 12 | 5 | 6 |
| `strat_bic` covered (unbounded) | 6 | 5 | 6 (1) | 5 | 5 | 11 (1) | 5 | 6 |
| `chapman_te` covered | 6 | 5 | 5 | 6 | 3 | 11 | 4 | 6 |
| `ll_bic` covered (unbounded) | 6 | 4 | 4 (1) | 4 | 4 | 11 | 4 | 6 |
| `seq_gap_tail` covered | 6 | 1 | 1 | 0 | 0 | 0 | 1 | 1 |
| `union_count` covered | 6 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| `trust_trace` covered | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `union_bound` covered (all unbounded above) | 6 | 6 | 6 | 6 | 6 | 12 | 6 | 6 |
| `chapman_sa` (no end state) covered | 6 | 1 | 0 | 1 | 0 | 0 | 1 | 0 |
| median absolute error: `ll_sa_hom` | 0 | 1.5 | 12.7 | 9.2 | 16.3 | 5.4 | 7.3 | 3.7 |
| median absolute error: `seq_gap_tail` | 0 | 3.5 | 51.0 | 100.5 | 77.0 | 15.9 | 10.9 | 20.1 |
| median absolute error: `union_count` | 0 | 8.0 | 74.5 | 82.5 | 36.5 | 24.0 | 18.5 | 27.0 |
| `ll_sa_hom` closer than `seq_gap_tail`, runs | 0 (6 ties) | 4 | 5 | 6 | 6 | 11 | 3 | 5 |
| `ll_sa_hom` closer than `union_count`, runs | 0 (6 ties) | 6 | 6 | 5 (1 tie) | 4 | 12 | 4 (1 tie) | 6 |

The floors of AGENTS.md rule 5 are the `trust_trace`, `union_count` and `union_bound`
rows. `union_bound` is a lower bound and covers by construction.

The candidates:

- `strat_bic`: the estimator of the earlier studies, one BIC-selected model per plane.
- `ll_bic`, `ll_bic_hom`, `ll_bic_boot`: the planes pooled and one model selected by
  BIC, with a plain profile interval, the same interval scaled by how much sessions
  disagree (`loglinear.homogeneity_dispersion`, D51), and a bootstrap interval.
- `chapman_te`: Lincoln-Petersen between the telemetry taken as one list and the end
  state, with a Wald interval.
- `ll_sa_hom`: one model named in advance, the spans-access interaction and nothing
  else, with the dispersion-scaled profile interval. Its point is `chapman_te`'s
  without the small-sample correction.

What these runs suggest:

- **Unstressed, every estimate is the true count.** In the six `none` runs every
  candidate, floor and baseline returns it. For the candidates this follows from the
  table: with no action seen by the end state alone, the missing cell is zero.
- **Over all 48 stressed runs** `ll_sa_hom` and `ll_bic_hom` each cover 46,
  `strat_bic` and `ll_bic_boot` 43, `chapman_te` 40, `ll_bic` 37; `seq_gap_tail`
  covers 4, `union_count` 1, `chapman_sa` 3. Half of those runs are round 1, which was
  seen before `ll_sa_hom` existed, so its figure there is post hoc.
- **On the 24 runs of the proposed condition made after `ll_sa_hom` existed** (rounds
  2 to 4): `ll_sa_hom`, `ll_bic_hom` and `ll_bic_boot` each cover 23, `strat_bic` 22
  with one unbounded, `chapman_te` and `ll_bic` 21. Twenty-three of 24 is a Wilson
  interval of 0.798 to 0.993. These runs do not separate the first three.
- **The coverage comes from the interval, not from naming the model.** `ll_sa_hom`
  and `chapman_te` have the same point to within the small-sample correction, and
  cover 23 and 21 of those 24; the difference is the dispersion-scaled interval,
  whose median width is 38 actions against 26. `ll_bic_hom`, which selects its model,
  has the same scaling and the same 23.
- **What naming the model buys is seen once, post hoc.** On `node-down-0` in round 1
  the BIC-selected pooled model put the count at 930 where it was 372, with no upper
  end, and `strat_bic` was unbounded there and on `node-restart-9`. `ll_sa_hom` was
  bounded in all 54 runs. One run is not a rate.
- **Its one miss on the 24.** In `pt3-node-restart-0` five actions were lost by both
  telemetry logs and all five are also hidden from the end state, so the end state
  shows nothing the telemetry missed, the model's missing cell is zero and the
  estimate is the union, 367 of 372, with an interval of 367 to 369. With about 0.44
  of actions hidden, five of five has a chance near 0.02 if hiding is independent of
  the outage; one run cannot tell that from a dependence.
- **Two code changes touch "after it existed".** The condition was committed together
  with `ll_sa_hom`, after round 1 was seen. And `ll_sa_hom`'s point was moved from an
  iterative fit to its closed form between rounds 2 and 3, because the fit returned
  no number on a table with no loss; on tables with loss the two agree to ten decimal
  places (the second reviewer checked 15 random tables). All rounds are scored with
  the final code.
- **A killed originator: mean error +18 actions in six runs,** with a standard
  deviation of 19: +22, +11, +34, +45, -3 and 0 on 216 to 297 actions. The one run
  `ll_sa_hom` misses there is high (276 against 242). A kill ends one plane's share
  of every session, so follow-ups that would have overwritten the other plane's
  openers never run, which makes hiding depend on when an action happened; it also
  leaves the two planes very unequal (31 to 131 actions on the killed one) in a
  pooled table. No run separates these, and `relays`, with no kill, has a mean error
  of +10.
- **The sequence baseline.** With no loss it is exact. It is closest to the
  candidates when outages are short: over the 15 runs with both relays down under
  25 s (`node-blip`, and the short `node-restart` runs) `ll_sa_hom` is the closer in
  10 and the baseline in 5. Where an outage takes session tails or whole sessions the
  baseline is short, by 11 to 100 actions at the median across the conditions. It is
  not only ever short: its point is above the true count in 10 of the 48 stressed
  runs, half of them `node-blip`, because missing requests are turned into actions at
  one pooled share. It has no floor of its own and is below the union count, which is
  a lower bound, in 21 of 48. Its interval, built for tails lost one record at a
  time, covers 4 of 48.
- **Without the end state the estimate fails,** as the design predicts for Q3:
  `chapman_sa` covers 3 of 48 stressed runs. The two telemetry logs lose overlapping
  blocks, so their overlap says little about what both missed.

### What this suggests for the registration

These are recommendations. The registration is the owner's to freeze.

- **Primary estimator: `ll_sa_hom`, on an argument and not on these runs.** The runs
  tie it with `ll_bic_hom` and `ll_bic_boot`. The argument is that its model is fixed
  by the rig's layout before any table is seen, so a sparse table cannot tip a
  selection toward a model with an unstable missing cell, which happened once here.
  `strat_bic` and `ll_bic_hom` are reported beside it.
- **Primary condition: a restart of both relays, 15 to 45 s, at a seeded time, in a
  run where sessions start over about 100 s** (the geometry of rounds 2 to 4). Its
  mean share in no log is 0.065 over 24 runs, it includes short outages where the
  sequence baseline does best, and it draws lengths across the bands above instead
  of avoiding them. A kill drawn for after the last call is now refused before a run
  starts; a restart drawn for after it is recorded.
- **Kills of an originator go to the heavy, descriptive condition,** with the +18 of
  six runs stated beforehand as seen and unexplained.
- **P1 is uncertain.** At 300 runs the Wilson lower end reaches 0.90 at 281 covered,
  0.937. The pilot has 23 of 24, 0.958, with an interval of 0.798 to 0.993. More
  runs lower the rate needed (919 of 1000) but do not fix a rate at or under 0.90.
  The registration should state the pilot's interval and predict P1 as uncertain.
- **P2 against the sequence baseline needs 168 wins of 300, 0.56.** The pilot has 19
  of 24, whose exact lower bound is 0.579. Split at 25 s of outage it is 6 of 9 under
  and 13 of 15 over; the split point is not pinned by the data, since any value from
  24.6 to 26.7 s gives the same groups. Over all 15 short-outage runs of the pilot it
  is 10 of 15, lower bound 0.384. So P2 is likely on the proposed condition and open
  on a schedule of short outages alone.
- **P2 against the union count:** 22 of 24 with one tie, lower bound 0.73.

### Review

Two reviews of the code and one refutation of this section, each given the files and
not the conclusions.

The first code review:

- Two pools recorded from the same seed would have replayed as one session with four
  shares, and the run would have passed. The driver now refuses sessions that cannot
  be one run, and the manifest lists every pool by path.
- P1 was tested on a Wilson bound rounded to four places, so 260 covered of 278 (lower
  end 0.89998) passed. It is tested on the bound itself.
- Nothing checked that calls were made when they were due, or that the containers'
  clock agrees with the driver's, which times the stress events. Each originator now
  reports how late its latest call was and whether it was ready before the start, and
  a late run fails the integrity check.
- The exact lower bound overflowed past about a thousand runs; a coverage block's two
  error figures were over different sets of runs; four docstrings said more than the
  code does. Tests that passed with a broken implementation were tightened.
- It also checked the sequence baseline for a hidden handicap or advantage in its code
  and found neither.

The second, on those fixes:

- A stress event drawn for after the run's end was only noted. A kill after the last
  call is now refused before the run starts.
- The clock difference was measured through `docker exec` and so included the exec's
  start-up. It is now asked of the kernel service, the quickest of five asks. Round
  3's figure (0.047 to 0.104 s) is the biased one; round 4's is -0.001 to 0.000 s.
- `ll_sa_hom` had lost the cap every other log-linear estimate here has, and gave a
  point where no action is in both lists. Both are unbounded now. Neither case occurs
  in the pilot.
- Three more tests passed with a broken implementation and were tightened.

The refutation changed this section in these ways: the repeats are no longer said to
show a run is a function of its seeds (and a schedule in an overlap of bands was then
repeated, above); the one miss is no longer called the smallest
loss (two runs with fewer actions in no log were covered); the coverage is attributed
to the interval and `ll_bic_hom` and `ll_bic_boot` are named as ties; "short outages
alone would probably fail P2" is withdrawn (the short-only schedule was 4 of 6); the
batch table no longer claims a rate; the kill bias carries its spread and a second
explanation; the sequence baseline's over-counts and its lack of a floor are stated;
"more runs do not fix a rate under 0.937" was wrong and is corrected; totals are over
all stressed runs with the post hoc half named; the floors have rows; and every
cross-round figure now comes from `pilot-summary.json`, where before it came from
joins made by hand.

### Limits

- Scripted sessions. A model makes lookups, failed calls and repeated writes that a
  plan does not; its hidden share may differ, and above all its hiding need not be
  independent of everything else in the session, which the scripted queue makes it by
  construction. The pilot on recorded sessions has to be run before anything is
  frozen.
- One laptop, 61 runs, conditions and run lengths I chose. Shares are shares of those
  run lengths.
- `ll_sa_hom` and the proposed condition were both chosen after seeing round 1. The 24
  later runs are one condition from one session generator.
- The estimate rests on the end state missing actions independently of when the
  telemetry was dark. No analysis here tests that, for instance by an action's
  position in its session.
- In 10 runs of rounds 1 and 2 the last restart came within 3 s of the run's end
  (`pilot-summary.json`, `restart_within_3_s_of_end`), where what the originators
  flush as they shut down could change what is lost. Rounds 3 and 4 have none.
- Rounds 1 and 2 were made before call lateness was recorded. In the 12 runs of
  rounds 3 and 4 the latest call of any originator was 0.007 to 0.013 s after it was
  due.
- The session-level dispersion that widens the interval is computed from sessions in
  one run that share one outage (the trap the design names). It widened enough here;
  that is an observation on these runs, not a property.
- Section 4's limit that every run has one export phase no longer holds for runs made
  with this driver; its runs are as they were.

## 6. The pilot, on recorded sessions

The second part of the pilot: sessions recorded from the study's two models and
replayed through the rig under the scripted pilot's `node-restart` (both relays down 15
to 45 s, from 8 to 90 s in, sessions starting over 100 s). Direction for the
registration, not a result. Figures are from `artifacts/study5/pilot-recorded-ds.json`,
`pilot-recorded-glm.json`, `pilot-recorded-mix.json`, `pilot-recorded-rig.json` and
`pilot-recorded-summary.json`, and from each pool's `pool.json` under
`artifacts/study5/pools/`.

### The pools

Recorded in-process, ticket by ticket, through `api.mistral.ai`. The provider of the
earlier studies refused for want of a subscription; the owner supplied this one and
left the second model to be chosen between GLM 5.2 and 5.3. In a probe of two sessions
each, 5.3 finished both and 5.2 lost one to a rate limit while another recording was
running, so 5.3 was taken; that is a reason, not a comparison.

| pool | plane A | plane B | sessions | failed | actions | calls a session | failed calls | hidden by the end state | spend |
|---|---|---|---|---|---|---|---|---|---|
| `pilot-ds` | `deepseek-v4-1-flash-1017` | the same | 36 | 0 | 495 | 23 to 33 | 30 | 0.410 | $0.77 |
| `pilot-glm` | `zai-glm-5-3` | the same | 24 | 0 | 333 | 22 to 33 | 15 | 0.396 | $0.50 |
| `pilot-mix` | `deepseek-v4-1-flash-1017` | `zai-glm-5-3` | 24 | 0 | 329 | 21 to 33 | 19 | 0.416 | $0.49 |

- Spend is tokens at $0.30 and $1.20 per million in and out, the earlier provider's
  rate carried over. This provider's rate for these models was not found, so the
  dollar figures are an assumption; the receipts hold the tokens.
- The models hide about what the scripted queue hides (0.40 there): they do what the
  tickets ask, with lookups and a few failed calls added.
- A run draws 24 sessions. `pilot-ds` has 36; the other two have 24, so each of their
  runs replays the same sessions under a different schedule.

### What the runs show

23 runs, all passing integrity: 20 under `node-restart` and one unstressed per pool.
Calls were at most 0.072 s late.

| | unstressed | `pilot-ds` | `pilot-glm` | `pilot-mix` | restarts under 25 s | restarts of 25 s or more |
|---|---|---|---|---|---|---|
| runs | 3 | 8 | 6 | 6 | 7 | 13 |
| mean share of actions in no log | none | 0.064 | 0.061 | 0.046 | 0.014 | 0.082 |
| `ll_sa_hom` covered | 3 | 6 | 6 | 6 | 5 | 13 |
| `ll_bic_hom` covered | 3 | | | | 5 | 12 |
| `ll_bic_boot` covered | 3 | | | | 6 | 12 |
| `strat_bic` covered | 3 | 6 | 6 | 5 | 5 | 12 |
| `chapman_te` covered | 3 | | | | 5 | 11 |
| `seq_gap_tail` covered | 3 | | | | 0 | 1 |
| `union_count` covered | 3 | | | | 2 | 0 |
| `trust_trace` covered | 3 | | | | 0 | 0 |
| `chapman_sa` (no end state) covered | 3 | | | | 1 | 0 |
| `ll_sa_hom` closer than `seq_gap_tail` | ties | 5 | 6 | 5 | 4 | 12 |
| `ll_sa_hom` closer than `union_count` | ties | | | | 4 (1 tie) | 11 |
| median absolute error, `ll_sa_hom` | 0 | | | | 1.9 | 3.8 |
| median absolute error, `seq_gap_tail` | 0 | | | | 4.4 | 26.9 |

`union_bound` covers every run and is unbounded above in every run. No candidate was
unbounded. Empty cells are in the pool files and not in the summary.

- **Unstressed, every estimate is the count,** on recorded sessions as on scripted.
- **`ll_sa_hom` covers 18 of the 20 restart runs,** a Wilson interval of 0.699 to
  0.972. On scripted sessions it covered 23 of 24.
- **The two misses are restarts of 18 and 19 s on the first pool, and the cause was
  found.** With the ledger open: in `ptr-ds-restart-7` the kill came 10.5 s into the
  run, when every session under way had just begun, and the four actions both
  telemetry logs lost were openers (two trials started, two drafts written) that
  later tickets undid. The end state shows none, so the estimate is the union, 326
  with an interval of 326 to 327, against 330. In `ptr-ds-restart-6` the eleven lost
  actions were four closing summaries, two handoff notes, two corrections and a
  refund with its email, each the last word on its record. The end state shows all
  eleven, and the estimate adds eight actions that were not made: 333, interval 326
  to 342, against 325.
- **So hiding depends on where in its session an action falls, and a short outage
  loses one part of each session.** The estimator assumes hiding is independent of
  loss. Section 5 listed that as untested; here it fails, in both directions, when the
  outage is short against a session (21 to 33 calls about a second apart).
- **The split was predicted before the other two pools were scored.** After the first
  pool, `docs/superpowers/plans/2026-11-15-study5-recorded-pilot.md` was committed
  with the prediction that on `pilot-glm` and `pilot-mix` no restart of 28 s or longer
  would be missed and any miss would be under 25 s. None of those 12 runs was missed
  at any length, so the prediction was not broken and was not much tested: four of
  the twelve were short.
- **By length, over recorded and scripted runs of this condition:** 25 s or longer,
  28 of 28 covered (Wilson 0.879 to 1); under 25 s, 13 of 16 (0.570 to 0.934).
- **Against the sequence baseline** the same split holds: `ll_sa_hom` is the closer in
  12 of 13 long recorded runs and 4 of 7 short ones.
- **Linking by id and by what the action was gave the same estimate** in all 23 runs.

### Two intervals tried and not adopted

Both were tried on the first pool's runs and the scripted ones, with those runs' truth
known, and are not in the code.

- A negative-binomial interval for the hidden part of what both logs lost, with
  between-session dispersion. It covers the run with nothing visible among the lost
  actions and cannot cover eleven of eleven visible.
- An estimate that uses the end state only for operations it cannot lose (emails,
  refunds, cancels). It covers all 8 runs of the first pool and 45 of the 48 scripted
  stressed runs, against 46 for `ll_sa_hom`, with wider intervals and failures of its
  own in the other direction: which operations an outage catches also depends on
  where in a session it falls.

Neither removes the dependence. Each moves it.

### What this suggests for the registration

Recommendations; the registration is the owner's to freeze.

- **Split the restart.** A long one, 28 to 45 s, as the primary condition: an outage
  as long as a session loses every part of it alike, and nothing failed there in 28
  pilot runs. A short one, 15 to 24 s, as a registered condition of its own with a
  predicted shortfall and no criterion, so that where the estimator fails is measured
  and not left out. This changes what the owner accepted (one restart of 15 to 45 s),
  and it selects the primary condition after seeing where the estimator failed. The
  registration draft says both.
- **Keep `ll_sa_hom`.** Changing the interval after two misses, on the runs that
  showed them, would be fitting to them.
- **The other course** is one condition of 15 to 45 s as accepted, with P1 predicted
  "not shown".

### Limits

- 20 restart runs, 84 sessions, one laptop. Two pools have as many sessions as a run
  draws.
- The prediction made before the second and third pools were scored could have failed
  on eight long runs and did not. It is not strong evidence for the line at 28 s.
- The line depends on how long a session is. These are 21 to 33 calls at a second
  apart. A deployment's sessions are not.
- The heavy condition, with an originator killed, was not piloted on recorded
  sessions.
- The cause was found by reading two runs with the ledger. No analysis measures the
  dependence between hiding and position across all runs.
- Dollar figures rest on an assumed rate.

