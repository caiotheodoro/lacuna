"""One run of the study-5 rig: start it, replay sessions under a stress schedule, collect.

    PYTHONPATH=src uv run python scripts/rig_run.py --run smoke --sessions 4

Needs docker. Not part of `make validate`. A run writes, under `--out`/<run>/:

    sessions.json            what the originators replay
    manifest.json            image digests, config hashes, commit, parameters, schedule
    logs/*.jsonl             each receiver's access log, complete (docker's copy of stdout)
    witnesses/spans.jsonl    spans as the backend collector received them
    witnesses/logs.jsonl     access-log lines as the backend collector received them
    witnesses/env.jsonl      the env witness: end state and sink, with the last writer
    truth/ledger.jsonl       the kernels' ledgers. For scoring only; no witness reads it.
    events.json              what the stress driver did and when, and the integrity check
    summary.json             counts, tables and loss per witness (lacuna.rig.witnesses)

A run whose plumbing failed (`lacuna.rig.integrity`) is kept, marked, and exits 1.
One run at a time: the script refuses to start while the rig's containers exist.

Keys are drawn per run and passed in the environment; none is written to a file.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import secrets
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from lacuna.rig.integrity import problems
from lacuna.rig.manifest import config_hashes, sha256_file, tree_hash
from lacuna.rig.sessions import dump, scripted_sessions
from lacuna.rig.stress import plan
from lacuna.rig.witnesses import LOG_PLANE, LOG_SERVICE, SERVICE_PLANE, summarise

ROOT = Path(__file__).resolve().parents[1]
RIG = ROOT / "rig"
COMPOSE = RIG / "compose.yaml"
SERVICES = ["backend", "otelcol-gw", "otelcol-node", "kernel", "mcp-a", "mcp-b", "agentgateway"]
AGENTS = ["agent-a", "agent-b"]
PROBE = """
import socket, sys, time
targets = [("localhost", 8000), ("mcp-b", 8000), ("agentgateway", 3000),
           ("otelcol-gw", 4318), ("backend", 4317)]
deadline = time.time() + 60
while targets and time.time() < deadline:
    host, port = targets[0]
    try:
        socket.create_connection((host, port), timeout=1).close()
        targets.pop(0)
    except OSError:
        time.sleep(0.2)
sys.exit(1 if targets else 0)
"""


def sh(*cmd: str, env: dict | None = None, check: bool = True) -> str:
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if check and r.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)}\n{r.stderr.strip()}")
    return r.stdout.strip()


def started_at(container: str) -> float:
    """When docker started a container, as a unix time."""
    stamp = sh("docker", "inspect", "--format", "{{.State.StartedAt}}", container)
    whole, _, frac = stamp.rstrip("Z").partition(".")
    t = datetime.datetime.fromisoformat(f"{whole}.{frac[:6] or '0'}+00:00")
    return t.timestamp()


def running(container: str) -> bool:
    return sh("docker", "inspect", "--format", "{{.State.Running}}", container) == "true"


def count_lines(text: str) -> int:
    return sum(1 for line in text.splitlines() if line.strip())


def http(url: str, key: str, body: dict | None = None) -> dict:
    req = urllib.request.Request(
        url,
        data=None if body is None else json.dumps(body).encode(),
        headers={"content-type": "application/json", "x-lacuna-key": key},
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run", required=True, help="name of the run folder")
    p.add_argument("--out", type=Path, default=ROOT / "artifacts" / "rig" / "runs")
    p.add_argument("--sessions", type=int, default=24)
    p.add_argument("--seed0", type=int, default=0, help="sessions are seeds seed0..seed0+n-1")
    p.add_argument("--tickets", type=int, default=7)
    p.add_argument("--pace", type=float, default=1.5, help="mean seconds between a session's calls")
    p.add_argument("--stagger", type=float, default=3.0)
    p.add_argument("--condition", default="none")
    p.add_argument("--stress-seed", type=int, default=0)
    p.add_argument("--settle", type=float, default=8.0)
    p.add_argument("--kernel-port", type=int, default=18080)
    a = p.parse_args()

    if shutil.which("docker") is None:
        sys.exit("rig_run needs docker")
    run_dir = a.out / a.run
    if run_dir.exists():
        sys.exit(f"{run_dir} exists; a run is never overwritten")
    events = plan(a.condition, a.stress_seed)

    admin = secrets.token_hex(8)
    env = {
        **os.environ,
        "RUN_DIR": str(run_dir.resolve()),
        "KEY_A": secrets.token_hex(8),
        "KEY_B": secrets.token_hex(8),
        "ADMIN_KEY": admin,
        "SESSION_KEY": secrets.token_hex(32),
        "STAGGER": str(a.stagger),
        "KERNEL_PORT": str(a.kernel_port),
    }
    compose = ["docker", "compose", "-f", str(COMPOSE)]
    kernel = f"http://127.0.0.1:{a.kernel_port}"
    # The project name and the kernel port are fixed, so a second run would tear down
    # the first. Refuse instead; leftovers of a crashed run are removed by hand.
    if sh(*compose, "--profile", "agents", "ps", "-aq", env={**env, "RUN_DIR": "/"}):
        sys.exit(
            "the rig's containers exist (another run, or a crashed one). When no run is "
            f"going: RUN_DIR=/ KEY_A=x KEY_B=x ADMIN_KEY=x SESSION_KEY=x docker compose "
            f"-f {COMPOSE.relative_to(ROOT)} --profile agents down -v"
        )

    for sub in ("witnesses", "logs", "truth", "agents"):
        (run_dir / sub).mkdir(parents=True)
    (run_dir / "witnesses").chmod(0o777)  # the collector image runs as its own user
    sessions = [
        s
        for seed in range(a.seed0, a.seed0 + a.sessions)
        for s in scripted_sessions(seed, a.tickets, ("a", "b"), pace_s=a.pace)
    ]
    (run_dir / "sessions.json").write_text(dump(sessions))

    def cid(service: str) -> str:
        return sh(*compose, "ps", "-aq", service, env=env)

    followers: list[subprocess.Popen] = []
    done: list[dict] = []
    not_running: list[str] = []
    try:
        sh("docker", "build", "-q", "--provenance=false", "-f", str(RIG / "Dockerfile"),
           "-t", "lacuna-rig:dev", str(ROOT))  # fmt: skip
        sh(*compose, "up", "-d", *SERVICES, env=env)
        deadline = time.time() + 60
        while True:
            try:
                http(f"{kernel}/health", admin)
                break
            except (urllib.error.URLError, OSError):
                if time.time() > deadline:
                    raise RuntimeError("the kernel service did not come up") from None
                time.sleep(0.2)
        sh(*compose, "exec", "-T", "mcp-a", "python", "-c", PROBE, env=env)
        ids = {s: cid(s) for s in SERVICES}

        # The node collector has no port to probe, so prove the whole log path instead:
        # a line written to a file in the tailed folder must come out of the backend.
        canary, delivered = run_dir / "logs" / "canary.jsonl", run_dir / "witnesses" / "logs.jsonl"
        deadline = time.time() + 60
        while not (delivered.exists() and "lacuna-canary" in delivered.read_text()):
            if time.time() > deadline:
                raise RuntimeError("no log line came through the node collector")
            with canary.open("a") as f:
                f.write(json.dumps({"lacuna-canary": time.time()}) + "\n")
            time.sleep(0.5)

        images = {s: sh("docker", "inspect", "--format", "{{.Image}}", ids[s]) for s in SERVICES}
        manifest = {
            "run": a.run,
            "commit": sh("git", "-C", str(ROOT), "rev-parse", "HEAD"),
            "dirty": bool(sh("git", "-C", str(ROOT), "status", "--porcelain")),
            # The source the image was built from, whether or not it is committed.
            "source_sha256": tree_hash(ROOT, (ROOT / "src").rglob("*.py")),
            "configs": config_hashes(
                RIG, [COMPOSE, RIG / "Dockerfile", RIG / "requirements.txt", *RIG.glob("config/*")]
            ),
            "images": images,
            "sessions_sha256": sha256_file(run_dir / "sessions.json"),
            "params": {
                k: getattr(a, k)
                for k in (
                    "sessions",
                    "seed0",
                    "tickets",
                    "pace",
                    "stagger",
                    "condition",
                    "stress_seed",
                    "settle",
                )  # fmt: skip
            },
            "schedule": [vars(e) for e in events],
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")

        for name, service in LOG_SERVICE.items():
            followers.append(
                subprocess.Popen(
                    ["docker", "logs", "-f", ids[service]],
                    stdout=(run_dir / "logs" / name).open("wb"),
                    stderr=subprocess.DEVNULL,
                )
            )
        for seed in range(a.seed0, a.seed0 + a.sessions):
            opened = {"session": f"s{seed}", "seed": seed, "n_tickets": a.tickets}
            http(f"{kernel}/open", admin, opened)
        not_running += [s for s in SERVICES if not running(ids[s])]

        sh(*compose, "--profile", "agents", "up", "-d", *AGENTS, env=env)
        ids.update({s: cid(s) for s in AGENTS})
        # The schedule counts from the moment docker started the first originator.
        t0 = min(started_at(ids[s]) for s in AGENTS)

        def stress() -> None:
            for e in events:
                time.sleep(max(0.0, t0 + e.at_s - time.time()))
                try:
                    ok = subprocess.run(
                        ["docker", e.action, ids[e.service]], capture_output=True
                    ).returncode == 0  # fmt: skip
                except (OSError, KeyError):
                    ok = False
                done.append({**vars(e), "done_at_s": round(time.time() - t0, 3), "ok": ok})

        driver = threading.Thread(target=stress, daemon=True)
        driver.start()
        codes = [int(c) for c in sh("docker", "wait", *(ids[s] for s in AGENTS)).split()]
        driver.join()
        ended = time.time()
        time.sleep(a.settle)

        reports: dict[str, dict | None] = {}
        for s in AGENTS:
            logs = subprocess.run(["docker", "logs", ids[s]], capture_output=True, text=True)
            (run_dir / "agents" / f"{s}.txt").write_text(logs.stdout)
            # What the SDK said about refused exports. Diagnosis only; not a witness.
            (run_dir / "agents" / f"{s}.err").write_text(logs.stderr)
            try:
                reports[s] = json.loads(logs.stdout.strip().splitlines()[-1])["failed"]
            except (IndexError, ValueError, KeyError):
                reports[s] = None
        ledger, rows = [], []
        for name in http(f"{kernel}/sessions", admin)["sessions"]:
            for e in http(f"{kernel}/ledger?session={name}", admin)["ledger"]:
                ledger.append({"session": name, **e})
            for r in http(f"{kernel}/observe?session={name}", admin)["rows"]:
                rows.append({"session": name, **r})
        (run_dir / "truth" / "ledger.jsonl").write_text(
            "".join(json.dumps(e) + "\n" for e in ledger)
        )
        (run_dir / "witnesses" / "env.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows)
        )
        # `docker logs --follow` is plumbing of ours, not a telemetry component: the file
        # it wrote must hold every line the container wrote.
        full: dict[str, int] = {}
        followed: dict[str, int] = {}
        for _ in range(6):  # the follower can be a moment behind the container
            for name, service in LOG_SERVICE.items():
                full[name] = count_lines(sh("docker", "logs", ids[service]))
                followed[name] = count_lines((run_dir / "logs" / name).read_text())
            if followed == full:
                break
            time.sleep(0.5)
        not_running += [s for s in SERVICES if not running(ids[s])]
        sh(*compose, "stop", "-t", "10", "backend", env=env)  # lets the file exporter flush
    finally:
        for f in followers:
            f.terminate()
        sh(*compose, "--profile", "agents", "down", "-v", "--remove-orphans", env=env, check=False)

    summary = summarise(run_dir)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    planes = sorted(set(LOG_PLANE.values()))
    wrong = problems(
        schedule=events,
        events_done=done,
        exit_codes=dict(zip(AGENTS, codes, strict=True)),
        reports=reports,
        file_sizes={
            name: (run_dir / "witnesses" / name).stat().st_size
            if (run_dir / "witnesses" / name).exists()
            else 0
            for name in ("spans.jsonl", "logs.jsonl", "env.jsonl")
        },
        not_running=sorted(set(not_running)),
        follower_lines=followed,
        full_lines=full,
        planned_calls={p: sum(len(s.steps) for s in sessions if s.plane == p) for p in planes},
        written_calls={p: summary["planes"][p]["calls_written_by_receiver"] for p in planes},
        agent_plane=SERVICE_PLANE,
    )
    (run_dir / "events.json").write_text(
        json.dumps(
            {
                "started_unix": round(t0, 3),
                "duration_s": round(ended - t0, 3),
                "agent_exit_codes": dict(zip(AGENTS, codes, strict=True)),
                "events": done,
                "integrity": {"ok": not wrong, "problems": wrong},
            },
            indent=1,
        )
        + "\n"
    )
    print(json.dumps({**summary["headline"], "integrity": wrong or "ok"}, indent=1))
    if wrong:
        sys.exit(1)


if __name__ == "__main__":
    main()
