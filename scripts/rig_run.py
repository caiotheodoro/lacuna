"""One run of the study-5 rig: start it, replay sessions under a stress schedule, collect.

    PYTHONPATH=src uv run python scripts/rig_run.py --run smoke --sessions 4

Needs docker. Not part of `make validate`. A run writes, under `--out`/<run>/:

    sessions.json            what the originators replay (scripted, or drawn from a pool)
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
import random
import secrets
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from lacuna.envs.registry import ENVS
from lacuna.rig.integrity import file_problems, killed_for_good, problems
from lacuna.rig.manifest import config_hashes, sha256_file, tree_hash
from lacuna.rig.sessions import (
    clashes,
    draw,
    dump,
    load,
    offsets,
    scripted_sessions,
    session_delay,
)
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


def source_files() -> list[Path]:
    """The files `rig/Dockerfile` copies from `src`, as `.dockerignore` leaves them."""
    skip = {"__pycache__", ".DS_Store"}
    return [
        f
        for f in (ROOT / "src").rglob("*")
        if f.is_file() and f.suffix != ".pyc" and not skip & set(f.parts)
    ]


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
    p.add_argument("--env", default="backoffice", choices=sorted(ENVS))
    p.add_argument("--sessions", type=int, default=24)
    p.add_argument("--seed0", type=int, default=0, help="sessions are seeds seed0..seed0+n-1")
    p.add_argument("--tickets", type=int, default=7)
    p.add_argument("--pace", type=float, default=1.5, help="mean seconds between a session's calls")
    p.add_argument(
        "--timeline", action="store_true", help="scripted sessions on one clock, in queue order"
    )
    p.add_argument(
        "--sessions-file", type=Path, action="append", default=[],
        help="recorded sessions to replay instead of scripted ones; --sessions of them are "
        "drawn by --draw-seed. Repeatable.",
    )  # fmt: skip
    p.add_argument("--draw-seed", type=int, default=0)
    p.add_argument("--stagger", type=float, default=3.0)
    p.add_argument(
        "--lead", type=float, default=6.0,
        help="seconds an originator has to come up; the start is --lead + --warmup-max "
        "after launch",
    )  # fmt: skip
    p.add_argument("--warmup-max", type=float, default=5.0)
    p.add_argument("--max-lag", type=float, default=0.25, help="seconds a call may be late")
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
        "LACUNA_ENV": a.env,
        "T0": "0",  # set when the originators are launched
    }
    # Each originator waits its own seeded time before it creates its SDK.
    warm = random.Random(f"warmup/{a.stress_seed}")
    warmup = {s: round(warm.uniform(0.0, a.warmup_max), 3) for s in AGENTS}
    env |= {"WARMUP_A": str(warmup["agent-a"]), "WARMUP_B": str(warmup["agent-b"])}
    compose = ["docker", "compose", "-f", str(COMPOSE)]
    kernel = f"http://127.0.0.1:{a.kernel_port}"
    # The project name and the kernel port are fixed, so a second run would tear down
    # the first. Refuse instead; leftovers of a crashed run are removed by hand.
    if sh(*compose, "--profile", "agents", "ps", "-aq", env={**env, "RUN_DIR": "/"}):
        sys.exit(
            "the rig's containers exist (another run, or a crashed one). When no run is "
            f"going: RUN_DIR=/ KEY_A=x KEY_B=x ADMIN_KEY=x SESSION_KEY=x T0=0 docker compose "
            f"-f {COMPOSE.relative_to(ROOT)} --profile agents down -v"
        )

    for sub in ("witnesses", "logs", "truth", "agents"):
        (run_dir / sub).mkdir(parents=True)
    (run_dir / "witnesses").chmod(0o777)  # the collector image runs as its own user
    if a.sessions_file:
        pool = [s for f in a.sessions_file for s in load(f.read_text())]
        sessions = draw(pool, a.sessions, a.draw_seed)
    else:
        sessions = [
            s
            for seed in range(a.seed0, a.seed0 + a.sessions)
            for s in scripted_sessions(
                seed, a.tickets, ("a", "b"), pace_s=a.pace, env=ENVS[a.env], timeline=a.timeline
            )
        ]
    wrong_sessions = clashes(sessions)
    if wrong_sessions:
        shutil.rmtree(run_dir)
        sys.exit("these sessions cannot be one run: " + "; ".join(wrong_sessions))
    (run_dir / "sessions.json").write_text(dump(sessions))
    opened = sorted({(s.session, s.seed, s.n_tickets) for s in sessions})
    # When the last call of the run is due, if every call is made on time.
    planned_end = max(
        (session_delay(s.session, a.stagger) + offsets(s)[-1] for s in sessions if s.steps),
        default=0.0,
    )

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
            # Whether what a run executes differs from that commit. Documents may differ.
            "dirty": bool(
                sh("git", "-C", str(ROOT), "status", "--porcelain", "--", "src", "rig", "scripts")
            ),
            # The source the image was built from, whether or not it is committed.
            "source_sha256": tree_hash(ROOT, source_files()),
            "configs": config_hashes(
                RIG, [COMPOSE, RIG / "Dockerfile", RIG / "requirements.txt", *RIG.glob("config/*")]
            ),
            "images": images,
            "sessions_sha256": sha256_file(run_dir / "sessions.json"),
            "params": {
                k: getattr(a, k)
                for k in (
                    "env",
                    "sessions",
                    "seed0",
                    "tickets",
                    "pace",
                    "timeline",
                    "draw_seed",
                    "stagger",
                    "lead",
                    "warmup_max",
                    "max_lag",
                    "condition",
                    "stress_seed",
                    "settle",
                )  # fmt: skip
            },
            "pools": [{"path": str(f), "sha256": sha256_file(f)} for f in a.sessions_file],
            "warmup_s": warmup,
            "planned_end_s": round(planned_end, 3),
            # Stress events drawn for after the last call: they cost nothing, and a
            # kill of an originator that has already exited cannot happen.
            "events_after_planned_end": [vars(e) for e in events if e.at_s > planned_end],
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
        for name, seed, n_tickets in opened:
            http(f"{kernel}/open", admin, {"session": name, "seed": seed, "n_tickets": n_tickets})
        not_running += [s for s in SERVICES if not running(ids[s])]

        # The run's start: both originators count every session from it, and so does
        # the stress schedule. It is far enough ahead for both to be up and warm.
        t0 = round(time.time() + a.lead + a.warmup_max, 3)
        env["T0"] = str(t0)
        sh(*compose, "--profile", "agents", "up", "-d", *AGENTS, env=env)
        ids.update({s: cid(s) for s in AGENTS})
        late = [s for s in AGENTS if started_at(ids[s]) + warmup[s] > t0 - 1.0]
        if late:
            raise RuntimeError(f"originators were not up before the start: {late}")
        # The originators count from t0 on their clock; the stress events are timed
        # on this one. Docker Desktop's VM can drift from the host.
        before = time.time()
        inside = float(
            sh(
                *compose,
                "exec",
                "-T",
                "kernel",
                "python",
                "-c",
                "import time; print(time.time())",
                env=env,
            )  # fmt: skip
        )
        clock_offset = round(inside - (before + time.time()) / 2, 3)

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
        timing: dict[str, dict | None] = {}
        for s in AGENTS:
            logs = subprocess.run(["docker", "logs", ids[s]], capture_output=True, text=True)
            (run_dir / "agents" / f"{s}.txt").write_text(logs.stdout)
            # What the SDK said about refused exports. Diagnosis only; not a witness.
            (run_dir / "agents" / f"{s}.err").write_text(logs.stderr)
            try:
                closing = json.loads(logs.stdout.strip().splitlines()[-1])
                reports[s] = closing["failed"]
                timing[s] = {k: closing[k] for k in ("start_lag_s", "max_lag_s")}
            except (IndexError, ValueError, KeyError):
                reports[s] = timing[s] = None
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
    except BaseException as error:
        crashed(run_dir, error)
        raise
    finally:
        for f in followers:
            f.terminate()
        sh(*compose, "--profile", "agents", "down", "-v", "--remove-orphans", env=env, check=False)

    try:
        codes_by_agent = dict(zip(AGENTS, codes, strict=True))
        # Written before the summary, which counts times from the run's start.
        record = {
            "started_unix": round(t0, 3),
            "duration_s": round(ended - t0, 3),
            "agent_exit_codes": codes_by_agent,
            "events": done,
            "timing": timing,
            "containers_clock_minus_driver_s": clock_offset,
        }
        (run_dir / "events.json").write_text(json.dumps(record, indent=1) + "\n")
        summary = summarise(run_dir, ENVS[a.env])
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
        planes = sorted(set(LOG_PLANE.values()))
        dead = killed_for_good(events)
        wrong = problems(
            schedule=events,
            events_done=done,
            exit_codes=codes_by_agent,
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
            timing=timing,
            clock_offset_s=clock_offset,
            max_lag_s=a.max_lag,
        ) + file_problems(summary, {SERVICE_PLANE[s] for s in dead if s in SERVICE_PLANE})
        record["integrity"] = {"ok": not wrong, "problems": wrong}
        (run_dir / "events.json").write_text(json.dumps(record, indent=1) + "\n")
        print(json.dumps({**summary["headline"], "integrity": wrong or "ok"}, indent=1))
        if wrong:
            sys.exit(1)
    except BaseException as error:
        crashed(run_dir, error)  # no-op once the integrity result is written
        raise


def crashed(run_dir: Path, error: BaseException) -> None:
    """Mark a run the driver did not finish, so it is kept and never mistaken for a run."""
    if not run_dir.is_dir():
        return
    path = run_dir / "events.json"
    record = json.loads(path.read_text()) if path.exists() else {}
    if "integrity" in record:
        return
    what = f"the run driver stopped: {type(error).__name__}: {error}"
    record["integrity"] = {"ok": False, "problems": [what]}
    path.write_text(json.dumps(record, indent=1) + "\n")


if __name__ == "__main__":
    main()
