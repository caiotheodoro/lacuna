# Rig findings: two spikes before the study-5 rig

Measured 2026-11-15 on one laptop (macOS, Docker 29.9.3 as `docker version` printed it),
by hand, with the scripts in `rig/spikes/`. These are observations that shape the rig,
not results of a study: few runs, one machine, one emitter, one schedule, nothing
registered.

Every number below is in `artifacts/rig-spikes/collector_kill.json` or
`artifacts/rig-spikes/gateway_fields.json`, written by `rig/spikes/collect.py` from the
spikes' output folders (AGENTS.md rule 2). The spikes were run twice. The numbers here
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
