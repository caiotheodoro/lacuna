"""Turn the spikes' output folders into committed artifacts.

AGENTS.md rule 2: a number in a document comes from an artifact. The spikes
write to `rig/spikes/*/out/`, which is not committed; this script reads those
folders and writes

    artifacts/rig-spikes/collector_kill.json
    artifacts/rig-spikes/gateway_fields.json

so `docs/RIG-FINDINGS.md` can name them. Run it after the spikes, by hand:

    python rig/spikes/collect.py

Nothing is computed here that the spike did not measure: counts are copied
from each run's `result.json`, retry delays and failures are read from the
SDK's own stderr, and gateway lines are the gateway's, with four fields
removed (session id, source address, and the generated session key never
enters: the config copy is not read).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ART = HERE.parents[1] / "artifacts" / "rig-spikes"
RETRY = re.compile(r"retrying in ([0-9.]+)s")
FAILED = "Failed to export spans batch"
DROP = ("mcp.session.id", "src.addr")


def _last_seq(path: Path) -> int | None:
    """The highest numbered span in a collector file, or None if there is none."""
    top = None
    if not path.exists():
        return None
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        for rs in json.loads(line).get("resourceSpans", []):
            for ss in rs.get("scopeSpans", []):
                for sp in ss.get("spans", []):
                    for attr in sp.get("attributes", []):
                        if attr["key"] == "spike.seq":
                            v = int(attr["value"]["intValue"])
                            top = v if top is None else max(top, v)
    return top


def collector_kill() -> dict:
    runs = []
    for d in sorted((HERE / "collector_kill" / "out").iterdir()):
        params, result = d / "params.json", d / "result.json"
        if not (params.exists() and result.exists()):
            continue
        err = (d / "emitter.err").read_text() if (d / "emitter.err").exists() else ""
        run = {
            **json.loads(params.read_text()),
            "result": json.loads(result.read_text()),
            "sdk_retry_delays_s": [float(x) for x in RETRY.findall(err)],
            "sdk_failed_batches": err.count(FAILED),
            "sdk_connection_refused_lines": err.count("Connection refused"),
            "sdk_error_lines": len([ln for ln in err.splitlines() if ln.strip()]),
        }
        before = d / "spans-before-kill.jsonl"
        if before.exists():
            run["last_span_delivered_before_kill"] = _last_seq(before)
        runs.append(run)
    return {
        "what": "rig/spikes/collector_kill: numbered spans through a default "
        "BatchSpanProcessor to a Collector with a file exporter; hand-run, one laptop",
        "generated_by": "rig/spikes/collect.py from rig/spikes/collector_kill/out/",
        "runs": runs,
    }


def gateway_fields() -> dict:
    out = {
        "what": "rig/spikes/gateway_fields: agentgateway's access log for two MCP "
        "tools/call requests, one that succeeds and one that raises; hand-run",
        "generated_by": "rig/spikes/collect.py from rig/spikes/gateway_fields/out/",
        "removed_fields": list(DROP),
        "configs": {},
    }
    base = HERE / "gateway_fields"
    for which in ("default", "fields"):
        d = base / "out" / which
        if not (d / "gateway.log").exists():
            continue
        requests = []
        for line in (d / "gateway.log").read_text().splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("scope") != "request":
                continue
            requests.append({k: v for k, v in rec.items() if k not in DROP})
        out["configs"][which] = {
            "config_file": f"rig/spikes/gateway_fields/config-{which}.yaml",
            "access_log_add": _added(base / f"config-{which}.yaml"),
            "image_digest": (d / "image-digest.txt").read_text().strip(),
            "client_output": (d / "client.txt").read_text().splitlines(),
            "request_lines": requests,
        }
    return out


def _added(config: Path) -> dict[str, str]:
    """The `frontendPolicies.accessLog.add` entries of a spike config, read as text."""
    added, inside = {}, False
    for line in config.read_text().splitlines():
        if line.strip() == "add:":
            inside = True
            continue
        if inside:
            m = re.match(r"\s+([a-z_]+): '(.*)'\s*$", line)
            if not m:
                break
            added[m.group(1)] = m.group(2)
    return added


def main() -> None:
    ART.mkdir(parents=True, exist_ok=True)
    for name, data in (
        ("collector_kill", collector_kill()),
        ("gateway_fields", gateway_fields()),
    ):
        path = ART / f"{name}.json"
        path.write_text(json.dumps(data, indent=1) + "\n")
        print(f"wrote {path.relative_to(HERE.parents[1])}")


if __name__ == "__main__":
    main()
