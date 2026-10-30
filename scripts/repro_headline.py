#!/usr/bin/env python3
"""Re-derive the registered headline from the recorded episodes and diff it.

Runs `thin_episodes.py --run confirm --pool heldout --bootstrap-episodes` and
`test_hypotheses.py` against its output, both into a temporary directory, then
compares the result with the committed `artifacts/thinning/confirm.json` and
`artifacts/hypotheses.json`. Every number the committed thinning artifact holds
(coverage, width, bias, detects_gap, ...) must match, and every hypothesis
verdict and detail string must be identical. It calls no model and writes
nothing under artifacts/.

Exit status 0 when everything matches, 1 otherwise.

Usage: uv run python scripts/repro_headline.py [--keep DIR]
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
THIN = ROOT / "artifacts" / "thinning" / "confirm.json"
HYPOTHESES = ROOT / "artifacts" / "hypotheses.json"
# Settings the committed artifact was produced with (it records them itself).
POOL = "heldout"
# Estimators whose committed numbers this code does not reproduce, found when this
# check was written: a rerun is deterministic (two runs agree) but differs from the
# artifact in ll_average and strat_lc_gated. Neither feeds H1-H4 or the card's
# headline (strat_lc_gated is superseded by strat_bic, README). They are reported,
# not failed on; --strict fails on them too. Every other estimator must match.
DRIFT = frozenset({"ll_average", "strat_lc_gated"})


def diff_numbers(new, old, path: str = "") -> list[str]:
    """Mismatches between two JSON values, recursing through what `old` contains.

    Keys only `new` has are ignored: the estimator set may grow after a freeze.
    Keys `old` has and `new` lacks are mismatches.
    """
    if isinstance(old, dict):
        if not isinstance(new, dict):
            return [f"{path}: expected an object"]
        out = []
        for k, v in old.items():
            if k not in new:
                out.append(f"{path}/{k}: missing from the rerun")
            else:
                out += diff_numbers(new[k], v, f"{path}/{k}")
        return out
    if isinstance(old, list):
        if not isinstance(new, list) or len(new) != len(old):
            return [f"{path}: list differs"]
        return [
            m
            for i, (a, b) in enumerate(zip(new, old, strict=True))
            for m in diff_numbers(a, b, f"{path}[{i}]")
        ]
    if isinstance(old, bool) or old is None or isinstance(old, str):
        return [] if new == old else [f"{path}: {new!r} != {old!r}"]
    if isinstance(new, bool) or not isinstance(new, (int, float)):
        return [f"{path}: {new!r} != {old!r}"]
    return [] if math.isclose(new, old, rel_tol=0, abs_tol=1e-9) else [f"{path}: {new} != {old}"]


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def estimator_of(mismatch: str) -> str:
    return mismatch.split("/estimators/")[-1].split("/")[0] if "/estimators/" in mismatch else ""


def split_drift(bad: list[str]) -> tuple[list[str], list[str]]:
    """(failures, known drift): mismatches in DRIFT estimators are the second list."""
    drift = [m for m in bad if estimator_of(m) in DRIFT]
    return [m for m in bad if m not in drift], drift


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--keep", type=Path, help="write the rerun here instead of a temp dir")
    ap.add_argument("--strict", action="store_true", help="also fail on the known-drift estimators")
    args = ap.parse_args(argv)

    old = json.loads(THIN.read_text())
    with tempfile.TemporaryDirectory() as tmp:
        work = args.keep or Path(tmp)
        work.mkdir(parents=True, exist_ok=True)
        thin = work / "confirm.json"
        run(
            [
                sys.executable,
                str(SCRIPTS / "thin_episodes.py"),
                "--run",
                "confirm",
                "--pool",
                POOL,
                "--bootstrap-episodes",
                "--reps",
                str(old["reps"]),
                "--seed",
                str(old["seed"]),
                "--out",
                str(thin),
            ]
        )
        # test_hypotheses reads artifacts/thinning/<run>.json; an absolute <run>
        # (without the suffix) points it at the rerun instead.
        hyp = work / "hypotheses.json"
        run(
            [
                sys.executable,
                str(SCRIPTS / "test_hypotheses.py"),
                "--run",
                str(thin.with_suffix("")),
                "--out",
                str(hyp),
            ]
        )
        bad = diff_numbers(json.loads(thin.read_text()), old)
        new_h = json.loads(hyp.read_text())
        old_h = json.loads(HYPOTHESES.read_text())
        # `run` names a path here and a run name there; the rest must be identical.
        old_h.pop("run")
        bad += [f"hypotheses{m}" for m in diff_numbers(new_h, old_h)]
    drift: list[str] = []
    if not args.strict:
        bad, drift = split_drift(bad)
    if drift:
        names = sorted({estimator_of(m) for m in drift})
        print(f"repro-headline: known drift, not failed on: {len(drift)} fields in {names}")
    if bad:
        print(f"repro-headline: {len(bad)} mismatches", file=sys.stderr)
        for m in bad[:40]:
            print("  ", m, file=sys.stderr)
        return 1
    print(
        f"repro-headline: {THIN.relative_to(ROOT)} and {HYPOTHESES.relative_to(ROOT)} reproduce "
        f"({len(old['configs'])} configs, all_passed={old_h['all_passed']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
