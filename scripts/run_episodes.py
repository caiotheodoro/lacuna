#!/usr/bin/env python3
"""Run a batch of L0 episodes, pool them, and score every estimator on both linkages.

Scripted (free):
  uv run python scripts/run_episodes.py --agent scripted --episodes 200 --run scripted-l0

LLM (needs LACUNA_BASE_URL, LACUNA_API_KEY, LACUNA_MODEL, LACUNA_USD_PER_M_IN/_OUT):
  uv run python scripts/run_episodes.py --agent llm --auditor llm --episodes 20 \
      --budget 5 --run pilot-l0

`--env infra` runs the infra/ops environment (lacuna.envs.infra) instead of
the back-office one; the scripted auditor follows the env, the LLM auditor
exists for the back-office env only.

Writes artifacts/episodes/<run>/{episodes.jsonl,summary.json,receipt.json}.
The pooled truth N is the ledger total; estimators see only the witness tables.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np

from lacuna.agents.llm import BudgetExceeded
from lacuna.agents.scripted import ScriptedAgent, ScriptedAuditor
from lacuna.coverage import ANCHOR_GAMMAS, PER_SET
from lacuna.envs.registry import ENVS
from lacuna.episode import run_episode
from lacuna.estimators import baselines, bounds
from lacuna.histories import keep_witnesses
from lacuna.linkage import oracle_table, signature_table

ROOT = Path(__file__).resolve().parents[1]


def load_dotenv(path: Path) -> None:
    """KEY=VALUE lines from the repo's gitignored .env; the real environment wins."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        key, sep, value = line.strip().partition("=")
        if sep and key and not key.startswith("#"):
            os.environ.setdefault(key, value.strip().strip('"'))


def _client(budget: float):
    from lacuna.agents.llm import ChatClient

    return ChatClient(
        base_url=os.environ["LACUNA_BASE_URL"],
        api_key=os.environ["LACUNA_API_KEY"],
        model=os.environ["LACUNA_MODEL"],
        usd_per_m_in=float(os.environ["LACUNA_USD_PER_M_IN"]),
        usd_per_m_out=float(os.environ["LACUNA_USD_PER_M_OUT"]),
        budget_usd=budget,
    )


def score(table: np.ndarray, n_true: int) -> dict:
    out = {}
    for keep, label in (((0, 1, 2), "trace+proxy+env"), ((0, 1, 2, 3), "all4")):
        t = keep_witnesses(table, keep)
        for name, fn in PER_SET.items():
            e = fn(t)
            out[f"{name}@{label}"] = _row(e, n_true)
        if keep == (0, 1, 2):
            for g in ANCHOR_GAMMAS:
                e = bounds.anchor_bounds(t, gamma=g, anchor=2)
                out[f"{e.name}@{label}"] = _row(e, n_true)
    out["trust_trace"] = _row(baselines.trust_trace(table), n_true)
    out["auditor_count"] = _row(baselines.auditor_count(table), n_true)
    return out


def _row(e, n_true: int) -> dict:
    f = lambda v: v if math.isfinite(v) else "inf"  # noqa: E731
    return {"point": f(e.point), "lo": f(e.lo), "hi": f(e.hi), "covers": e.covers(n_true)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", choices=sorted(ENVS), default="backoffice")
    ap.add_argument("--agent", choices=["scripted", "llm"], default="scripted")
    ap.add_argument("--auditor", choices=["scripted", "llm"], default="scripted")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed0", type=int, default=1000)
    # Explicit, non-contiguous seeds; overrides --seed0/--episodes. Used to rerun
    # specific lost episodes as a labelled supplement (docs/DECISIONS.md D33).
    ap.add_argument("--seeds", default="")
    ap.add_argument("--budget", type=float, default=5.0)
    ap.add_argument("--run", required=True)
    ap.add_argument("--tickets", type=int, default=7)
    # "full" lets W4 read every call verbatim, which makes it degenerate
    # (docs/DECISIONS.md D14). "narration" gives it only what the agent said.
    ap.add_argument("--auditor-view", choices=["full", "narration"], default="full")
    ap.add_argument("--max-turns", type=int, default=40)
    # 0/0 records complete logs; scripts/thin_episodes.py applies loss afterwards.
    ap.add_argument("--p-gateway-drop", type=float, default=0.05)
    ap.add_argument("--p-trace-drop", type=float, default=0.10)
    args = ap.parse_args()
    if args.env != "backoffice" and args.auditor == "llm":
        # W4 is dropped from the live arm (docs/DECISIONS.md D20) and its
        # prompt names the back-office ops only.
        ap.error("--auditor llm is back-office only")
    env = ENVS[args.env]
    load_dotenv(ROOT / ".env")

    out_dir = ROOT / "artifacts" / "episodes" / args.run
    out_dir.mkdir(parents=True, exist_ok=True)
    client = _client(args.budget) if "llm" in (args.agent, args.auditor) else None

    if args.auditor == "llm":
        from lacuna.agents.llm import LLMAuditor

        auditor = LLMAuditor(client, view=args.auditor_view)
    else:
        auditor = ScriptedAuditor(np.random.default_rng([args.seed0, 3]), env=env)

    oracle = np.zeros(16, dtype=np.int64)
    sig = np.zeros(16, dtype=np.int64)
    n_true, done, fp, stopped = 0, 0, 0, ""
    failed: list[dict] = []
    t0 = time.perf_counter()
    with (out_dir / "episodes.jsonl").open("w") as fh:
        seeds = (
            [int(x) for x in args.seeds.split(",")]
            if args.seeds
            else [args.seed0 + i for i in range(args.episodes)]
        )
        for seed in seeds:
            agent = client if args.agent == "llm" else ScriptedAgent(np.random.default_rng(seed))
            if client:
                client.headers = {
                    "User-Agent": "lacuna/0.0.0",
                    "x-opencode-session": f"lacuna-{args.run}-{seed}",
                }
            try:
                res = run_episode(
                    seed,
                    agent,
                    auditor,
                    args.p_gateway_drop,
                    args.p_trace_drop,
                    args.tickets,
                    args.max_turns,
                    env,
                )
            except BudgetExceeded as e:
                stopped = f"{type(e).__name__}: {e}"
                break
            except Exception as e:
                # One episode failing must not end the study. Record it and
                # carry on; the summary reports how many were lost (D30).
                failed.append(
                    {
                        "seed": seed,
                        "error": f"{type(e).__name__}: {e}"[:300],
                        "diag": getattr(client, "last_error_diag", None) if client else None,
                    }
                )
                print(f"episode {seed}: FAILED {type(e).__name__}", flush=True)
                continue
            fh.write(json.dumps(res.to_json()) + "\n")
            o, stats = oracle_table(res.ledger, res.records, 4)
            oracle += o
            sig += signature_table(res.records, 4)
            fp += stats["auditor_false_positives"]
            n_true += len(res.ledger)
            done += 1
            spent = f" ${client.spent:.4f}" if client else ""
            print(f"episode {seed}: N={len(res.ledger)} calls={res.n_calls}{spent}", flush=True)

    oracle_obs = oracle.copy()
    oracle_obs[0] = 0
    summary = {
        "env": args.env,
        "agent": args.agent,
        "auditor": args.auditor,
        "model": os.environ.get("LACUNA_MODEL") if client else None,
        "episodes": done,
        "seed0": args.seed0,
        "seeds": args.seeds or None,
        "stopped": stopped,
        "failed_episodes": failed,
        "n_true": n_true,
        "oracle_table": oracle.tolist(),
        "signature_table": sig.tolist(),
        "auditor_false_positives": fp,
        "fault_rates": {"p_gateway_drop": args.p_gateway_drop, "p_trace_drop": args.p_trace_drop},
        "tickets": args.tickets,
        "auditor_view": args.auditor_view,
        "oracle": score(oracle_obs, n_true),
        "signature": score(sig, n_true),
        "seconds": round(time.perf_counter() - t0, 1),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if client:
        receipt = {"total_usd": round(client.spent, 6), "calls": client.receipt}
        (out_dir / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")

    print(f"\nN={n_true} over {done} episodes; true missing cell={int(oracle[0])}")
    print(f"oracle table    {oracle.tolist()}")
    print(f"signature table {sig.tolist()}")
    for mode in ("oracle", "signature"):
        print(f"\n[{mode}]")
        for name, r in summary[mode].items():
            print(f"  {name:38} {r['point']!s:>9} [{r['lo']!s:>7}, {r['hi']!s:>7}] {r['covers']}")
    if stopped:
        print(f"\nstopped early: {stopped}")


if __name__ == "__main__":
    main()
