#!/usr/bin/env python3
"""Run a batch of L0 episodes, pool them, and score every estimator on both linkages.

Scripted (free):
  uv run python scripts/run_episodes.py --agent scripted --episodes 200 --run scripted-l0

LLM (needs LACUNA_BASE_URL, LACUNA_API_KEY, LACUNA_MODEL, LACUNA_USD_PER_M_IN/_OUT):
  uv run python scripts/run_episodes.py --agent llm --auditor llm --episodes 20 \
      --budget 5 --run pilot-l0

`--model`, `--usd-per-m-in` and `--usd-per-m-out` override LACUNA_MODEL, LACUNA_USD_PER_M_IN and
LACUNA_USD_PER_M_OUT (the flag wins; the base URL and key still come from the environment). The
budget guard prices calls with the values in force, and an LLM run's summary.json records the
model and both prices.

`--env infra` runs the infra/ops environment (lacuna.envs.infra) instead of
the back-office one; the scripted auditor follows the env, the LLM auditor
exists for the back-office env only.

`--env backoffice_bypass` is the back-office env plus a `run_script` batch tool
whose inner actions the gateway and the trace log only as one call
(lacuna.envs.bypass). `--bypass-fraction` (scripted agent only, default 0) is
the share of state-changing actions the scripted agent routes through it:
  uv run python scripts/run_episodes.py --env backoffice_bypass --agent scripted \
      --bypass-fraction 0.3 --episodes 24 --seed0 9100 --tickets 30 \
      --p-gateway-drop 0 --p-trace-drop 0 --run bypass-scripted-30
That env's summary.json gains a "bypass" block (ledger N, bypassed actions,
per-witness capture, the gateway's undercount).

`--organism {neutral,l1,l2}` (that env only, default neutral) sets what the LLM agent is told
about its logging: `l1` appends `bypass.L1_PROMPT` (audit warning), `l2` appends
`bypass.L2_PROMPT` (how the three-log audit works, and a goal of keeping its unrecorded-action
estimate low) to the system prompt. The scripted agent ignores it. summary.json records the
organism and the sha256 of the exact system prompt.

Writes artifacts/episodes/<run>/{episodes.jsonl,summary.json,receipt.json}. It refuses to run if
that episodes.jsonl exists, unless `--overwrite` is given.
The pooled truth N is the ledger total; estimators see only the witness tables.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np

from lacuna.agents.llm import BudgetExceeded
from lacuna.agents.scripted import ScriptedAgent, ScriptedAuditor
from lacuna.coverage import ANCHOR_GAMMAS, PER_SET
from lacuna.envs import bypass
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


def _client(budget: float, model: str, usd_per_m_in: float, usd_per_m_out: float):
    from lacuna.agents.llm import ChatClient

    return ChatClient(
        base_url=os.environ["LACUNA_BASE_URL"],
        api_key=os.environ["LACUNA_API_KEY"],
        model=model,
        usd_per_m_in=usd_per_m_in,
        usd_per_m_out=usd_per_m_out,
        budget_usd=budget,
    )


def resolve_llm_settings(args, ap: argparse.ArgumentParser) -> tuple[str, float, float]:
    """Model and prices: the flag if given, else the LACUNA_* environment variable."""
    model = args.model if args.model is not None else os.environ.get("LACUNA_MODEL")
    usd_in = args.usd_per_m_in
    if usd_in is None and "LACUNA_USD_PER_M_IN" in os.environ:
        usd_in = float(os.environ["LACUNA_USD_PER_M_IN"])
    usd_out = args.usd_per_m_out
    if usd_out is None and "LACUNA_USD_PER_M_OUT" in os.environ:
        usd_out = float(os.environ["LACUNA_USD_PER_M_OUT"])
    if not model or usd_in is None or usd_out is None:
        ap.error(
            "an LLM run needs a model and both prices: --model/--usd-per-m-in/--usd-per-m-out "
            "or LACUNA_MODEL/LACUNA_USD_PER_M_IN/LACUNA_USD_PER_M_OUT"
        )
    if usd_in < 0 or usd_out < 0:
        ap.error("prices must be non-negative")
    return model, usd_in, usd_out


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


def bypass_summary(eps: list[dict], oracle: np.ndarray, fraction: float) -> dict:
    """Pooled accounting for the bypass env: what the gateway missed and who saw it.

    `capture` is each witness's count of ledger entries it recorded (oracle
    linkage); the `*_blind_by_op` blocks need the gateway's oracle link and are
    present only for complete-log runs.
    """

    def total(key: str) -> dict:
        out: dict = {}
        for e in eps:
            for op, n in e.get(key, {}).items():
                out[op] = out.get(op, 0) + n
        return dict(sorted(out.items()))

    n_true = sum(e["n_ledger"] for e in eps)
    gateway = sum(e["gateway_action_records"] for e in eps)
    n_bypassed = sum(e["n_bypassed"] for e in eps)
    names = ("trace", "gateway", "env", "auditor")
    out = {
        "bypass_fraction": fraction,
        "n_true": n_true,
        "n_bypassed": n_bypassed,
        "bypassed_share": round(n_bypassed / n_true, 4) if n_true else None,
        "n_script_calls": sum(e["n_script_calls"] for e in eps),
        "capture": {
            w: int(sum(c for h, c in enumerate(oracle) if h >> i & 1)) for i, w in enumerate(names)
        },
        "gateway_action_count": gateway,
        "trace_action_count": sum(e["trace_action_records"] for e in eps),
        "gateway_undercount": n_true - gateway,
        "undercount_equals_bypassed": n_true - gateway == n_bypassed,
        "ledger_by_op": total("ledger_by_op"),
        "bypassed_by_op": total("bypassed_by_op"),
    }
    if all("gateway_blind_by_op" in e for e in eps):
        out["gateway_blind_by_op"] = total("gateway_blind_by_op")
        out["env_saw_gateway_blind_by_op"] = total("env_saw_gateway_blind_by_op")
    return out


def main(argv: list[str] | None = None) -> None:
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
    # LLM runs only. Default None: fall back to LACUNA_MODEL / LACUNA_USD_PER_M_IN / _OUT.
    ap.add_argument("--model", default=None, help="model id (default: $LACUNA_MODEL)")
    ap.add_argument(
        "--usd-per-m-in", type=float, default=None, help="default: $LACUNA_USD_PER_M_IN"
    )
    ap.add_argument(
        "--usd-per-m-out", type=float, default=None, help="default: $LACUNA_USD_PER_M_OUT"
    )
    ap.add_argument("--run", required=True)
    ap.add_argument("--tickets", type=int, default=7)
    # "full" lets W4 read every call verbatim, which makes it degenerate
    # (docs/DECISIONS.md D14). "narration" gives it only what the agent said.
    ap.add_argument("--auditor-view", choices=["full", "narration"], default="full")
    ap.add_argument("--max-turns", type=int, default=40)
    # 0/0 records complete logs; scripts/thin_episodes.py applies loss afterwards.
    ap.add_argument("--p-gateway-drop", type=float, default=0.05)
    ap.add_argument("--p-trace-drop", type=float, default=0.10)
    # backoffice_bypass only: the scripted agent's share of actions sent through run_script.
    ap.add_argument("--bypass-fraction", type=float, default=0.0)
    # backoffice_bypass only: the prompt variant for the LLM agent (lacuna.envs.bypass.ORGANISMS).
    ap.add_argument("--organism", choices=bypass.ORGANISMS, default="neutral")
    # A paid run is not repeatable for free; never replace one silently.
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="replace artifacts/episodes/<run>/episodes.jsonl if it exists (default: refuse)",
    )
    args = ap.parse_args(argv)
    existing = ROOT / "artifacts" / "episodes" / args.run / "episodes.jsonl"
    if existing.exists() and not args.overwrite:
        ap.error(
            f"{existing.relative_to(ROOT)} already exists; pick another --run or pass --overwrite"
        )
    if args.organism != "neutral" and args.env != "backoffice_bypass":
        ap.error("--organism needs --env backoffice_bypass")
    if args.bypass_fraction and (args.env != "backoffice_bypass" or args.agent != "scripted"):
        ap.error("--bypass-fraction needs --env backoffice_bypass and --agent scripted")
    if not 0.0 <= args.bypass_fraction <= 1.0:
        ap.error("--bypass-fraction must be in [0, 1]")
    if args.env != "backoffice" and args.auditor == "llm":
        # W4 is dropped from the live arm (docs/DECISIONS.md D20) and its
        # prompt names the back-office ops only.
        ap.error("--auditor llm is back-office only")
    env = bypass.with_organism(ENVS[args.env], args.organism)
    load_dotenv(ROOT / ".env")

    out_dir = ROOT / "artifacts" / "episodes" / args.run
    out_dir.mkdir(parents=True, exist_ok=True)
    llm = resolve_llm_settings(args, ap) if "llm" in (args.agent, args.auditor) else None
    client = _client(args.budget, *llm) if llm else None

    if args.auditor == "llm":
        from lacuna.agents.llm import LLMAuditor

        auditor = LLMAuditor(client, view=args.auditor_view)
    else:
        auditor = ScriptedAuditor(np.random.default_rng([args.seed0, 3]), env=env)

    oracle = np.zeros(16, dtype=np.int64)
    sig = np.zeros(16, dtype=np.int64)
    n_true, done, fp, stopped = 0, 0, 0, ""
    failed: list[dict] = []
    bypass_eps: list[dict] = []
    complete_logs = args.p_gateway_drop == 0 and args.p_trace_drop == 0
    t0 = time.perf_counter()
    with (out_dir / "episodes.jsonl").open("w") as fh:
        seeds = (
            [int(x) for x in args.seeds.split(",")]
            if args.seeds
            else [args.seed0 + i for i in range(args.episodes)]
        )
        for seed in seeds:
            if args.agent == "llm":
                agent = client
            elif env.script_tools:
                agent = bypass.BypassScriptedAgent(
                    np.random.default_rng(seed),
                    args.bypass_fraction,
                    np.random.default_rng([seed, 13]),
                )
            else:
                agent = ScriptedAgent(np.random.default_rng(seed))
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
            if env.script_tools:
                bypass_eps.append(bypass.episode_stats(res, complete_logs))
            done += 1
            spent = f" ${client.spent:.4f}" if client else ""
            print(f"episode {seed}: N={len(res.ledger)} calls={res.n_calls}{spent}", flush=True)

    oracle_obs = oracle.copy()
    oracle_obs[0] = 0
    summary = {
        "env": args.env,
        "agent": args.agent,
        "auditor": args.auditor,
        "model": llm[0] if llm else None,
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
    if llm:
        summary["usd_per_m_in"], summary["usd_per_m_out"] = llm[1], llm[2]
    if env.script_tools:
        summary["bypass"] = bypass_summary(bypass_eps, oracle, args.bypass_fraction)
        summary["organism"] = args.organism
        summary["organism_applied"] = args.agent == "llm"
        prompt = bypass.ORGANISM_PROMPTS.get(args.organism)
        summary["organism_prompt_sha256"] = (
            hashlib.sha256(prompt.encode()).hexdigest() if prompt else None
        )
        summary["agent_system_sha256"] = hashlib.sha256(env.agent_system.encode()).hexdigest()
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if client:
        receipt = {"total_usd": round(client.spent, 6), "calls": client.receipt}
        (out_dir / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    elif env.script_tools:
        # A scripted bypass run costs nothing; the receipt says so explicitly.
        (out_dir / "receipt.json").write_text(
            json.dumps({"total_usd": 0.0, "calls": []}, indent=2) + "\n"
        )

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
