"""What a run used, by hash: the part of a run's manifest that needs no docker.

`scripts/rig_run.py` adds the image digests, the commit and the run's parameters.
A run whose hashes differ from the registered ones fails the integrity gate.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def config_hashes(root: Path, paths: Iterable[Path]) -> dict[str, str]:
    """sha256 of each file, keyed by its path relative to `root`, in path order."""
    return {p.relative_to(root).as_posix(): sha256_file(p) for p in sorted(paths)}


def tree_hash(root: Path, paths: Iterable[Path]) -> str:
    """One sha256 over a set of files: their relative paths and contents, in path order.

    The run driver takes it over the source the rig image is built from, so a run made
    from an uncommitted tree still says exactly which source it ran.
    """
    lines = [f"{rel} {digest}" for rel, digest in config_hashes(root, paths).items()]
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


# What a registration pins about every run of a condition, besides the hashes.
REGISTERED_PARAMS = ("env", "sessions", "stagger", "lead", "warmup_max", "max_lag", "settle")


def registered(manifest: dict, pools: dict[str, str]) -> dict:
    """What a registration records from the manifest of a run made at the frozen commit.

    `pools` maps each pool's name to the sha256 of its sessions file.
    """
    return {
        "commit": manifest["commit"],
        "source_sha256": manifest["source_sha256"],
        "configs": dict(manifest["configs"]),
        "images": dict(manifest["images"]),
        "params": {k: manifest["params"][k] for k in REGISTERED_PARAMS},
        "pools": dict(pools),
    }


def registered_problems(manifest: dict, reg: dict) -> list[str]:
    """How a run's manifest differs from the registered one. Empty means it does not.

    A run that differs was not made by what was registered: it is kept and not analysed.
    """
    out: list[str] = []
    if manifest.get("dirty"):
        out.append("the run was made from a tree with uncommitted changes")
    if manifest.get("source_sha256") != reg["source_sha256"]:
        out.append("the source the image was built from is not the registered source")
    for what in ("configs", "images"):
        mine, want = manifest.get(what) or {}, reg[what]
        for name in sorted(set(mine) | set(want)):
            if mine.get(name) != want.get(name):
                out.append(f"{what}: {name} is not the registered one")
    params = manifest.get("params") or {}
    for k, v in reg["params"].items():
        if params.get(k) != v:
            out.append(f"parameter {k} is {params.get(k)!r}, registered {v!r}")
    pools = manifest.get("pools") or []
    if not pools:
        out.append("the run replayed no registered pool")
    known = set(reg["pools"].values())
    for pool in pools:
        if pool.get("sha256") not in known:
            out.append(f"pool {pool.get('path')} is not a registered pool")
    return out
