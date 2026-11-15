# Rig findings: two spikes, the retry window, and the rig's checkpoint

Measured 2026-11-15 on one laptop (macOS, Docker 29.9.3 as `docker version` printed it),
by hand, with the scripts in `rig/spikes/`. These are observations that shape the rig,
not results of a study: few runs, one machine, one emitter, one schedule, nothing
registered.

Every number in sections 1 and 2 is in `artifacts/rig-spikes/collector_kill.json` or
`artifacts/rig-spikes/gateway_fields.json`, written by `rig/spikes/collect.py` from the
spikes' output folders (AGENTS.md rule 2). Sections 3 and 4 were added when the rig was
built (D78) and name their own artifacts. The spikes were run twice. The numbers here
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
  schedule counts from their start. Which batch an outage hits is then fixed by the
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
