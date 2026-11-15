#!/usr/bin/env bash
# Usage: [N=600] [OUTAGE=8] run.sh <condition> [tag]
#   condition: clean | kill-emitter | kill-collector
# Runs one condition and prints the counter's JSON line. Needs docker and uv.
# Hand-run only: nothing under rig/ is part of `make validate`.
set -euo pipefail
cond="${1:?condition}"
tag="${2:-1}"
here="$(cd "$(dirname "$0")" && pwd)"
out="$here/out/$cond-$tag"
rm -rf "$out" && mkdir -p "$out"
image="otel/opentelemetry-collector-contrib:0.165.0"
N="${N:-600}"            # spans to emit, one every 50 ms
OUTAGE="${OUTAGE:-8}"    # seconds the collector stays down in kill-collector
# A venv of its own, so the emitter is a plain python process that SIGKILL reaches
# directly (`uv run` is a parent process and would leave the child running).
venv="$here/.venv"
if [ ! -x "$venv/bin/python" ]; then
  uv venv -q "$venv"
  uv pip install -q -p "$venv/bin/python" opentelemetry-sdk opentelemetry-exporter-otlp-proto-http
fi

start_collector() {
  docker run -d --rm --name spike-otelcol -p 4318:4318 \
    -v "$here/collector.yaml:/etc/otelcol-contrib/config.yaml:ro" \
    -v "$out:/out" "$image" >/dev/null
  sleep 3
}

docker rm -f spike-otelcol >/dev/null 2>&1 || true
start_collector
"$venv/bin/python" "$here/emit.py" --n "$N" >"$out/emitted.txt" 2>"$out/emitter.err" &
pid=$!
case "$cond" in
  clean) wait "$pid" ;;
  kill-emitter) sleep 17; kill -9 "$pid"; wait "$pid" 2>/dev/null || true ;;
  kill-collector)
    # The file exporter overwrites its file when the collector starts, so what arrived
    # before the kill is moved aside first; otherwise the restart erases it and the
    # count blames the kill for spans that were delivered.
    sleep 12; docker kill spike-otelcol >/dev/null
    mv "$out/spans.jsonl" "$out/spans-before-kill.jsonl"
    sleep "$OUTAGE"; start_collector; wait "$pid" ;;
  *) echo "unknown condition $cond" >&2; exit 2 ;;
esac
sleep 8
docker stop spike-otelcol >/dev/null 2>&1 || true
emitted="$(wc -l <"$out/emitted.txt" | tr -d ' ')"
"$venv/bin/python" "$here/count.py" "$emitted" "$out"/spans*.jsonl | tee "$out/result.json"
