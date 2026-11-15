#!/usr/bin/env bash
# Usage: run.sh <default|fields>
# Starts the MCP server and the gateway, makes two tool calls (one succeeds, one is a
# tool error), and saves the gateway's log. Hand-run only; needs docker and uv.
set -euo pipefail
which="${1:?default or fields}"
here="$(cd "$(dirname "$0")" && pwd)"
out="$here/out/$which"
rm -rf "$out" && mkdir -p "$out"
image="cr.agentgateway.dev/agentgateway:v1.7.0"

venv="$here/.venv"
if [ ! -x "$venv/bin/python" ]; then
  uv venv -q "$venv"
  # mcp 2.x renamed FastMCP and the client helpers; the spike is written against 1.x.
  uv pip install -q -p "$venv/bin/python" "mcp<2"
fi

"$venv/bin/python" "$here/server.py" >"$out/server.log" 2>&1 &
server=$!
trap 'kill $server 2>/dev/null || true; docker rm -f spike-agw >/dev/null 2>&1 || true' EXIT
sleep 3
docker rm -f spike-agw >/dev/null 2>&1 || true
# A copy, because the gateway may write a generated session key into its config file.
cp "$here/config-$which.yaml" "$out/config.yaml"
# --add-host makes host.docker.internal resolve on Linux; Docker Desktop has it already.
docker run -d --name spike-agw -p 3000:3000 --add-host host.docker.internal:host-gateway \
  -v "$out/config.yaml:/config.yaml" "$image" -f /config.yaml >/dev/null
sleep 4
"$venv/bin/python" "$here/call.py" http://localhost:3000/mcp 2>&1 | tee "$out/client.txt" || true
sleep 2
docker logs spike-agw >"$out/gateway.log" 2>&1
docker inspect --format '{{index .RepoDigests 0}}' "$image" | tee "$out/image-digest.txt"
echo "gateway log lines: $(grep -c "" "$out/gateway.log")"
