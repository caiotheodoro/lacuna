#!/usr/bin/env bash
# Usage: [N=600] [OUTAGE=8] run.sh <condition> [tag]
#   condition: clean | kill-emitter | kill-collector | overwrite-test
# Runs one condition and prints its JSON result. Needs docker and uv.
# Hand-run only: nothing under rig/ is part of `make validate`.
# `rig/spikes/collect.py` turns the outputs into artifacts/rig-spikes/collector_kill.json.
set -euo pipefail
cond="${1:?condition}"
tag="${2:-1}"
here="$(cd "$(dirname "$0")" && pwd)"
out="$here/out/$cond-$tag"
rm -rf "$out" && mkdir -p "$out"
image="otel/opentelemetry-collector-contrib:0.165.0"
sdk="1.45.2"
N="${N:-600}"            # spans to emit, with a 50 ms sleep between them
OUTAGE="${OUTAGE:-8}"    # seconds the collector stays down in kill-collector
KILL_EMITTER_AT=17       # seconds after the emitter starts
KILL_COLLECTOR_AT=12
# A venv of its own, so the emitter is a plain python process that SIGKILL reaches
# directly (`uv run` is a parent process and would leave the child running).
venv="$here/.venv"
if [ ! -x "$venv/bin/python" ]; then
  uv venv -q "$venv"
  uv pip install -q -p "$venv/bin/python" \
    "opentelemetry-sdk==$sdk" "opentelemetry-exporter-otlp-proto-http==$sdk"
fi

start_collector() {
  docker run -d --rm --name spike-otelcol -p 4318:4318 \
    -v "$here/collector.yaml:/etc/otelcol-contrib/config.yaml:ro" \
    -v "$out:/out" "$image" >/dev/null
  sleep 3
}
size() { if [ -f "$1" ]; then wc -c <"$1" | tr -d ' '; else echo 0; fi; }

docker rm -f spike-otelcol >/dev/null 2>&1 || true
start_collector
digest="$(docker inspect --format '{{index .RepoDigests 0}}' "$image")"
py="$("$venv/bin/python" -c 'import sys; print(sys.version.split()[0])')"
printf '{"condition": "%s", "tag": "%s", "n": %s, "outage_s": %s, "kill_emitter_at_s": %s, "kill_collector_at_s": %s, "sleep_between_spans_s": 0.05, "image": "%s", "image_digest": "%s", "opentelemetry_sdk": "%s", "python": "%s"}\n' \
  "$cond" "$tag" "$N" "$OUTAGE" "$KILL_EMITTER_AT" "$KILL_COLLECTOR_AT" "$image" "$digest" "$sdk" "$py" \
  >"$out/params.json"

"$venv/bin/python" "$here/emit.py" --n "$N" >"$out/emitted.txt" 2>"$out/emitter.err" &
pid=$!
case "$cond" in
  clean) wait "$pid" ;;
  kill-emitter)
    sleep "$KILL_EMITTER_AT"; kill -9 "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true ;;
  kill-collector)
    # The file exporter overwrites its file when the collector starts, so what arrived
    # before the kill is moved aside first; otherwise the restart erases it and the
    # count blames the kill for spans that were delivered.
    sleep "$KILL_COLLECTOR_AT"; docker kill spike-otelcol >/dev/null 2>&1 || true
    mv "$out/spans.jsonl" "$out/spans-before-kill.jsonl" 2>/dev/null || true
    sleep "$OUTAGE"; start_collector; wait "$pid" ;;
  overwrite-test)
    # Shows the overwrite directly: send a few spans, kill, restart with nothing sent.
    wait "$pid"; sleep 2
    before="$(size "$out/spans.jsonl")"
    arrived="$("$venv/bin/python" "$here/count.py" "$(wc -l <"$out/emitted.txt" | tr -d ' ')" "$out/spans.jsonl")"
    docker kill spike-otelcol >/dev/null 2>&1 || true; sleep 2
    after_kill="$(size "$out/spans.jsonl")"
    start_collector; sleep 1
    after_restart="$(size "$out/spans.jsonl")"
    docker stop spike-otelcol >/dev/null 2>&1 || true
    printf '{"bytes_before_kill": %s, "bytes_after_kill": %s, "bytes_after_restart_nothing_sent": %s, "before_kill": %s}\n' \
      "$before" "$after_kill" "$after_restart" "$arrived" | tee "$out/result.json"
    exit 0 ;;
  *) echo "unknown condition $cond" >&2; exit 2 ;;
esac
sleep 8
docker stop spike-otelcol >/dev/null 2>&1 || true
emitted="$(wc -l <"$out/emitted.txt" | tr -d ' ')"
"$venv/bin/python" "$here/count.py" "$emitted" "$out"/spans*.jsonl | tee "$out/result.json"
