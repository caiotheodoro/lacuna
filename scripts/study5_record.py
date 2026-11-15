#!/usr/bin/env python3
"""Record a pool of study-5 sessions from real models. Spends money; needs a key.

    PYTHONPATH=src uv run python scripts/study5_record.py --pool pilot-ds --seed0 50000 \\
        --sessions 8 --model-a deepseek-v4.1-flash --model-b deepseek-v4.1-flash \\
        --usd-per-m-in 0.30 --usd-per-m-out 1.20 --budget 2

The base URL and the key come from the environment or the gitignored `.env`
(LACUNA_BASE_URL, LACUNA_API_KEY). `--model-a` is the orchestrator's model (plane A),
`--model-b` the sub-agent's (plane B).

Writes artifacts/study5/pools/<pool>/:

    sessions.json      what the rig replays (`lacuna.rig.sessions`)
    transcripts.jsonl  one line per session: both conversations as recorded
    pool.json          seeds, models, the prompt's hash, per-session counts, what failed
    receipt.json       every model call's tokens and price

`--budget` is checked before each session and inside each model's own client, so a
pool can pass it by what one session costs.

A pool is never overwritten. A session whose recording raised is tried once more from
the start; one that fails twice is listed in pool.json and is not in the pool.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlparse

from lacuna.envs.base import Env
from lacuna.envs.registry import ENVS
from lacuna.rig.record import RECORD_SYSTEM, Recorded, record_session
from lacuna.rig.sessions import dump

ROOT = Path(__file__).resolve().parents[1]
PLANES = ("a", "b")


def load_dotenv(path: Path) -> None:
    """KEY=VALUE lines from the repo's gitignored .env; the real environment wins."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        key, sep, value = line.strip().partition("=")
        if sep and key and not key.startswith("#"):
            os.environ.setdefault(key, value.strip().strip('"'))


def describe(rec: Recorded, env: Env) -> dict:
    """One session's counts as recorded. A run's own ledger is what a run is scored on."""
    actions = len(rec.ledger)
    return {
        "session": rec.session,
        "seed": rec.seed,
        "calls": {p: sum(1 for q, *_ in rec.calls if q == p) for p in rec.planes},
        "failed_calls": rec.outcomes.count(False),
        "reads": sum(
            1
            for (_, _, tool, _), ok in zip(rec.calls, rec.outcomes, strict=True)
            if ok and tool not in env.action_tools
        ),
        "actions": {
            p: sum(1 for e in rec.ledger if e["principal"] == f"plane-{p}") for p in rec.planes
        },
        "rows_in_end_state": len(rec.rows),
        "hidden_share": round(1 - len(rec.rows) / actions, 4) if actions else None,
        "unfinished_tickets": rec.unfinished,
    }


def record_pool(
    out: Path,
    seeds: list[int],
    n_tickets: int,
    env: Env,
    models_for: Callable[[int], dict],
    pace_s: float,
    meta: dict,
    attempts: int = 2,
    stop: Callable[[], str] = lambda: "",
) -> dict:
    """Record `seeds` into `out`. `models_for(seed)` gives the per-plane `complete` callables.

    `stop()` returns a reason to stop before the next session (a spent budget), or "".
    Returns what is written to pool.json.
    """
    if out.exists():
        raise FileExistsError(f"{out} exists; a pool is never overwritten")
    out.mkdir(parents=True)
    sessions, per_session, failed = [], [], []
    stopped = ""
    with (out / "transcripts.jsonl").open("w") as fh:
        for seed in seeds:
            stopped = stop()
            if stopped:
                break
            rec, errors = None, []
            for _ in range(attempts):
                try:
                    rec = record_session(seed, n_tickets, env, models_for(seed), PLANES)
                    break
                except Exception as e:  # one session failing must not end the pool
                    errors.append(f"{type(e).__name__}: {e}"[:300])
            if rec is None:
                failed.append({"seed": seed, "errors": errors})
                print(f"session {seed}: FAILED {errors[-1][:80]}", flush=True)
                continue
            sessions += rec.sessions(pace_s)
            row = describe(rec, env) | {"attempts": len(errors) + 1}
            per_session.append(row)
            line = {"session": rec.session, "seed": seed, "transcripts": rec.transcripts}
            fh.write(json.dumps(line) + "\n")
            print(f"session {seed}: actions={row['actions']} calls={row['calls']}", flush=True)
    (out / "sessions.json").write_text(dump(sessions))
    actions = sum(sum(r["actions"].values()) for r in per_session)
    rows = sum(r["rows_in_end_state"] for r in per_session)
    pool = {
        **meta,
        "env": env.name,
        "n_tickets": n_tickets,
        "pace_s": pace_s,
        "seeds_asked": seeds,
        "sessions_recorded": len(per_session),
        "stopped": stopped,
        "failed_sessions": failed,
        "prompt_sha256": hashlib.sha256(RECORD_SYSTEM.encode()).hexdigest(),
        "sessions_sha256": hashlib.sha256((out / "sessions.json").read_bytes()).hexdigest(),
        "transcripts_sha256": hashlib.sha256((out / "transcripts.jsonl").read_bytes()).hexdigest(),
        "actions_recorded": actions,
        "hidden_share": round(1 - rows / actions, 4) if actions else None,
        "sessions": per_session,
    }
    (out / "pool.json").write_text(json.dumps(pool, indent=1) + "\n")
    return pool


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--pool", required=True, help="name of the pool's folder")
    p.add_argument("--out", type=Path, default=ROOT / "artifacts" / "study5" / "pools")
    p.add_argument("--env", default="backoffice_churn", choices=sorted(ENVS))
    p.add_argument("--seed0", type=int, required=True)
    p.add_argument("--sessions", type=int, required=True)
    p.add_argument("--tickets", type=int, default=10)
    p.add_argument("--pace", type=float, default=1.0, help="mean seconds between a session's calls")
    p.add_argument("--model-a", required=True)
    p.add_argument("--model-b", required=True)
    p.add_argument("--usd-per-m-in", type=float, required=True)
    p.add_argument("--usd-per-m-out", type=float, required=True)
    p.add_argument("--budget", type=float, required=True, help="USD, for the whole pool")
    a = p.parse_args()

    from lacuna.agents.llm import ChatClient

    load_dotenv(ROOT / ".env")
    base_url = os.environ.get("LACUNA_BASE_URL")
    key = os.environ.get("LACUNA_API_KEY")
    if not base_url or not key:
        sys.exit("recording needs LACUNA_BASE_URL and LACUNA_API_KEY")
    if (a.out / a.pool).exists():
        sys.exit(f"{a.out / a.pool} exists; a pool is never overwritten")

    # One client per model, so a pool with one model has one receipt. Each client's own
    # cap is the pool's; the pool stops when the clients together have spent it.
    clients = {
        m: ChatClient(base_url, key, m, a.usd_per_m_in, a.usd_per_m_out, a.budget)
        for m in {a.model_a, a.model_b}
    }
    model = {"a": a.model_a, "b": a.model_b}

    def spent() -> float:
        return sum(c.spent for c in clients.values())

    def models_for(seed: int) -> dict:
        for c in clients.values():
            c.headers = {
                "User-Agent": "lacuna/0.0.0",
                "x-opencode-session": f"lacuna-{a.pool}-{seed}",
            }
        return {plane: clients[model[plane]].complete for plane in PLANES}

    def stop() -> str:
        return f"budget: spent ${spent():.4f} of ${a.budget:.2f}" if spent() >= a.budget else ""

    meta = {
        "pool": a.pool,
        "models": model,
        "usd_per_m_in": a.usd_per_m_in,
        "usd_per_m_out": a.usd_per_m_out,
        # The host and nothing else: no scheme, port, path or key.
        "base_url_host": urlparse(base_url if "//" in base_url else "//" + base_url).hostname,
        "commit": subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip(),
    }
    seeds = list(range(a.seed0, a.seed0 + a.sessions))
    pool = record_pool(
        a.out / a.pool, seeds, a.tickets, ENVS[a.env], models_for, a.pace, meta, stop=stop
    )
    receipt = {m: c.receipt for m, c in clients.items()}
    (a.out / a.pool / "receipt.json").write_text(json.dumps(receipt) + "\n")
    path = a.out / a.pool / "pool.json"
    pool["spent_usd"] = round(spent(), 4)
    path.write_text(json.dumps(pool, indent=1) + "\n")
    print(
        f"{pool['sessions_recorded']} sessions, {pool['actions_recorded']} actions, "
        f"hidden share {pool['hidden_share']}, ${pool['spent_usd']}"
    )


if __name__ == "__main__":
    main()
