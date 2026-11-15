# Rig findings: two spikes before the study-5 rig

Measured 2026-11-15 on one laptop (macOS, Docker 29.9.3), by hand, with the scripts in
`rig/spikes/`. These are observations that shape the rig, not results of a study: small
runs, one machine, nothing registered. Raw outputs are under `rig/spikes/*/out/`, which
is not committed; rerun the scripts to regenerate them.

## 1. What a killed emitter and a killed collector lose at default settings

`rig/spikes/collector_kill/`. One Python process emits numbered spans, one every 50 ms,
through a `BatchSpanProcessor` and the OTLP/HTTP exporter with every setting at its
default (`opentelemetry-sdk` 1.45.2, Python 3.13.14), to a Collector that writes them
to a file (`otel/opentelemetry-collector-contrib:0.165.0`,
`sha256:e40d61f10c9a617cca1ffd1be4d7e2b3a4b0f0001798ff07e58cabe82e618cea`). The count is
of numbered spans in the file against numbers the emitter printed.

| condition | runs | emitted | missing | share | where | duplicates |
|---|---|---|---|---|---|---|
| nothing goes wrong | 5 | 600 | 0 in every run | 0 | | 0 |
| emitter killed (SIGKILL) at 17 s | 5 | 297 to 313 | 31 to 34 | 0.104 to 0.109 | one block, always the last spans emitted | 0 |
| collector killed at 12 s, back after 8 s | 5 | 600 | 0 in every run | 0 | | 0 |
| collector killed at 12 s, back after 25 s | 3 | 1200 | 305, 320, 305 | 0.254 to 0.267 | one block inside the run (e.g. spans 179 to 483) | 0 |

- **Loss at default settings is well above 1%, and none of it is independent.** Every
  loss was one contiguous block. Nothing was lost span by span.
- **A killed emitter loses its tail.** What sat in the SDK's queue since the last export
  (up to 5 s of spans) goes with the process. A tail leaves no gap in any numbering,
  which is the case a sequence count cannot see.
- **A short collector outage loses nothing.** Eight seconds down plus start-up was
  covered in 5 of 5 runs. The SDK's documented default is a 10 s export timeout that
  includes retries, which fits; the timeout itself was not measured here. Loss needs an outage longer than the retry
  budget: at 25 s the SDK logged `Failed to export spans batch due to timeout, max
  retries or shutdown` twice and the two batches (about 310 spans) were gone, while
  spans emitted later in the same outage arrived.
- **No duplicates** in 18 runs, including the ones with retries.

One error on the way, kept here because it would have been a false finding: the first
five collector-kill runs reported 0.297 to 0.313 missing, always spans 0 to about 180.
Those spans had been delivered before the kill. The file exporter overwrites its file
when the collector starts, so the restart erased them. `run.sh` now moves the file aside
before restarting, and the five runs were redone (the row above). The rig must not let
a restarted component truncate a witness file, and must not count that as loss if it
does.

### What this changes in the design

- **The outage schedule needs outages longer than 10 s, or kills of the originators.**
  Brief collector restarts are absorbed. The registered schedule should be stated in
  units of the exporter's retry budget.
- **Background load is not needed to get loss.** Kills and long outages give 10 to 27%
  at default queue sizes. `telemetrygen` stays in the design as a stressor on the
  collector, but the tuned-queue condition is unlikely to be needed.
- **Independent-loss models will be wrong here, by construction.** Block loss is the
  only kind observed, which is the case where study 4's coverage fell (D73). The pilot
  has to pick an estimator for block loss, and the unit of analysis has to be the run.
- **Tail loss and interior loss behave differently against the baseline.** A long
  outage leaves an interior gap that a sequence count sees; a killed originator leaves
  none. The registration should report the two kinds of fault separately.

Not measured: loss under memory pressure or CPU limits; a killed collector with spans
in its own queue (the file exporter has none to speak of); more than one emitter; loss
in log shippers.

## 2. What agentgateway records for one MCP tool call

`rig/spikes/gateway_fields/`. agentgateway v1.7.0
(`cr.agentgateway.dev/agentgateway@sha256:60216aafb03d37f94e952ff3d51d2f630a65b10ecaebdc0a864882f92610c45b`),
unmodified, in front of a two-tool MCP server (Python `mcp` 1.30.0, streamable HTTP).
The client sends a fixed W3C `traceparent` and makes two `tools/call` requests:
`cancel_order`, which succeeds, and `always_fails`, which returns a tool error
(`isError: true`). The access log goes to stdout, one line per request.

| question | as shipped | with configuration | the setting |
|---|---|---|---|
| tool name | yes (`gen_ai.tool.name`) | | |
| tool arguments | no | yes | `tool_args: 'mcp.tool.arguments'` |
| a failed tool call told from a successful one | no: both are `http.status` 200 | yes: `tool_result.isError` | `tool_result: 'mcp.tool.result'` |
| JSON-RPC request id | no | yes | `rpc_id: 'json(request.body).id'` |
| trace id equal to the caller's | yes (`trace.id`) | | |
| a number assigned by the gateway per request | no | no | none documented |

The settings are entries under `frontendPolicies.accessLog.add` in the gateway's own
config file (`rig/spikes/gateway_fields/config-fields.yaml`); the binary is the released
one. `tool_error: 'mcp.tool.error'` added nothing for a tool that raised: the error is
inside `tool_result`.

The two lines as shipped (session id and source address removed):

```json
{"time": "2026-11-15T06:24:58.153036Z", "gateway": "default/default", "listener": "mcp", "route": "internal/default", "http.method": "POST", "http.host": "localhost", "http.path": "/mcp", "http.version": "HTTP/1.1", "http.status": 200, "trace.id": "0af7651916cd43dd8448eb211c80319c", "span.id": "89849abd5997f22e", "protocol": "mcp", "mcp.method.name": "tools/call", "mcp.target": "spike", "mcp.resource.type": "tool", "gen_ai.tool.name": "cancel_order", "duration": "4ms"}
{"time": "2026-11-15T06:24:58.161126Z", "gateway": "default/default", "listener": "mcp", "route": "internal/default", "http.method": "POST", "http.host": "localhost", "http.path": "/mcp", "http.version": "HTTP/1.1", "http.status": 200, "trace.id": "0af7651916cd43dd8448eb211c80319c", "span.id": "a32f485cc48bba47", "protocol": "mcp", "mcp.method.name": "tools/call", "mcp.target": "spike", "mcp.resource.type": "tool", "gen_ai.tool.name": "always_fails", "duration": "2ms"}
```

With the settings, the same two calls add (other fields as above):

```json
{"gen_ai.tool.name": "cancel_order", "tool_args": {"order_id": "O-123"}, "tool_result": {"content": [{"type": "text", "text": "cancelled O-123"}], "structuredContent": {"result": "cancelled O-123"}, "isError": false}, "rpc_id": 1}
{"gen_ai.tool.name": "always_fails", "tool_args": {"order_id": "O-456"}, "tool_result": {"content": [{"type": "text", "text": "Error executing tool always_fails: cannot cancel O-456"}], "isError": true}, "rpc_id": 3}
```

### What this changes in the design

- **Signature linkage to env is possible on plane A.** Arguments can be logged, so a
  gateway record can be normalised to (op, resource, value). The attempted-action
  fallback is not needed.
- **Phantom records can be filtered.** `isError` is in the log once `mcp.tool.result` is
  added. As shipped, a failed tool call and a successful one are the same line, so a
  rig that used the default log would count failures as actions. The registration must
  name the access-log settings and their hash.
- **The trace id links a session, not an action.** Every request of one session carried
  the caller's trace id, and the gateway's `span.id` is its own. The header the gateway's
  expression sees already holds the gateway's span id
  (`00-0af7…319c-21b59141d79cd4c8-01` where the client sent parent `b7ad6b7169203331`),
  so the caller's span id is not in the access log. The link id for one action is
  therefore the pair (trace id, JSON-RPC id), and the caller's span must record the
  JSON-RPC id of the call it made.
- **The sequence-gap baseline counts requests, not actions.** JSON-RPC ids are assigned
  by the client per session and shared by every request: in this run `initialize` was 0,
  the first tool call 1, a `tools/list` 2, the second tool call 3. A gap says a line was
  lost, not that a tool call was. The baseline must either count every request or be
  told which ids were tool calls by another log, and the registration has to say which.
- **The gateway writes to its own config file.** With no session key set, v1.7.0 wrote a
  generated `config.session.key` into the mounted file. The rig must set the key
  explicitly (`SESSION_KEY`) so the config's hash is stable and nothing secret-shaped is
  committed.
- **Pin the MCP SDK.** `mcp` 2.3.0 renamed `FastMCP` and the client helpers; the spike
  runs on `mcp<2`. The rig picks one major version on purpose.

Not checked here: whether the gateway forwards `traceparent` to the upstream MCP server
(the rewritten header suggests it does); how the gateway's log path behaves under load
or when the container is killed.
