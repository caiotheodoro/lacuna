"""The distribution: what `pyproject.toml` promises, and what a built wheel does.

The import package and the command are `lacuna`; the distribution is `lacuna-mse`,
because the name `lacuna` on PyPI is someone else's. The first tests read files and
run in `make validate`. The rest build (`uv build` fetches the build backend, and the
install fetches numpy), so they run only under `make dist-check`, which sets
`LACUNA_DIST_CHECK=1`. Nothing here uploads anything.

The map and the logs the installed command is run on are this repository's
(`rig/map.toml`, `tests/fixtures/rig`). They are not in the wheel.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tarfile
import tomllib
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text())
NAME, VERSION = "lacuna-mse", "1.0.0"
STEM = f"lacuna_mse-{VERSION}"
MAP = ROOT / "rig" / "map.toml"
W = ROOT / "tests" / "fixtures" / "rig" / "run" / "witnesses"

# A plane only one witness can see: what `lint` exists to refuse.
ONE_RECORDER = """
version = 1
planes = ["p"]
actions = ["pay"]
[[witness]]
name = "log"
format = "jsonl"
pipeline = "shipper"
plane = "p"
session = "sid"
op = "op"
resource = "res"
value = "val"
id = ["n"]
"""

building = pytest.mark.skipif(
    os.environ.get("LACUNA_DIST_CHECK") != "1" or shutil.which("uv") is None,
    reason="builds and installs: run `make dist-check`",
)


# What the files say ------------------------------------------------------------------


def test_the_distribution_is_named_apart_from_the_import_package_and_the_command():
    p = PROJECT["project"]
    assert (p["name"], p["version"]) == (NAME, VERSION)
    assert p["scripts"] == {"lacuna": "lacuna.cli:main"}
    assert PROJECT["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"] == ["src/lacuna"]
    # The estimator needs numpy and nothing else; the rest are extras.
    assert p["dependencies"] == ["numpy>=1.26"] and p["requires-python"] == ">=3.11"
    assert set(p["optional-dependencies"]) == {"agents", "rig"}


def test_the_metadata_a_release_needs_is_there():
    p = PROJECT["project"]
    assert p["readme"] == "README.md" and len(p["description"]) > 40
    assert p["license"] == "Apache-2.0" and p["license-files"] == ["LICENSE", "NOTICE"]
    assert all((ROOT / f).is_file() for f in p["license-files"])
    assert p["authors"] == [{"name": "Caio Theodoro"}]  # the paper's author line; no address
    assert len(p["keywords"]) >= 4
    assert "Programming Language :: Python :: 3.11" in p["classifiers"]
    # A licence classifier beside an SPDX `license` is refused by the build backend.
    assert not [c for c in p["classifiers"] if c.startswith("License ::")]
    assert p["urls"]["Repository"] == "https://github.com/caiotheodoro/lacuna"
    assert all(u.startswith("https://") for u in p["urls"].values())


def test_the_lock_file_is_for_this_name_and_version():
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    (own,) = [k for k in lock["package"] if k.get("source") == {"editable": "."}]
    assert (own["name"], own["version"]) == (NAME, VERSION)


# What a build gives ------------------------------------------------------------------


def _run(argv: list, cwd: Path, timeout: int = 600) -> subprocess.CompletedProcess:
    """Run outside this checkout's environment: no PYTHONPATH (the Makefile exports one
    that points at `src`) and none of the virtualenv's variables."""
    drop = ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT")
    env = {k: v for k, v in os.environ.items() if k not in drop}
    return subprocess.run(
        [str(a) for a in argv], cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout
    )


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> dict[str, Path]:
    out = tmp_path_factory.mktemp("dist")
    r = _run(["uv", "build", "--out-dir", out], ROOT)
    assert r.returncode == 0, r.stderr
    assert sorted(p.name for p in out.iterdir() if p.name != ".gitignore") == [
        f"{STEM}-py3-none-any.whl",
        f"{STEM}.tar.gz",
    ]
    return {"wheel": out / f"{STEM}-py3-none-any.whl", "sdist": out / f"{STEM}.tar.gz"}


@pytest.fixture(scope="module")
def venv(built, tmp_path_factory) -> Path:
    """A fresh environment holding the wheel and what it requires, no extra."""
    home = tmp_path_factory.mktemp("venv")
    r = _run(["uv", "venv", "--python", sys.executable, home / "v"], home)
    assert r.returncode == 0, r.stderr
    bin_ = home / "v" / ("Scripts" if os.name == "nt" else "bin")
    r = _run(["uv", "pip", "install", "--python", bin_ / "python", built["wheel"]], home)
    assert r.returncode == 0, r.stderr
    return bin_


@building
def test_the_wheel_holds_the_package_and_nothing_else(built):
    with zipfile.ZipFile(built["wheel"]) as z:
        names = z.namelist()
        meta = z.read(f"{STEM}.dist-info/METADATA").decode()
        entry = z.read(f"{STEM}.dist-info/entry_points.txt").decode()
    assert {n.split("/")[0] for n in names} == {"lacuna", f"{STEM}.dist-info"}
    assert "lacuna/cli.py" in names and "lacuna/estimators/stratified.py" in names
    assert not [n for n in names if n.endswith(".pyc") or "__pycache__" in n]
    source = {str(p.relative_to(ROOT / "src")) for p in (ROOT / "src" / "lacuna").rglob("*.py")}
    assert {n for n in names if n.startswith("lacuna/")} == source
    assert meta.startswith("Metadata-Version: 2.4\n")
    assert f"Name: {NAME}\n" in meta and f"Version: {VERSION}\n" in meta
    assert "License-Expression: Apache-2.0\n" in meta and "Requires-Python: >=3.11\n" in meta
    required = [x for x in meta.splitlines() if x.startswith("Requires-Dist:")]
    assert [x for x in required if "extra ==" not in x] == ["Requires-Dist: numpy>=1.26"]
    assert "lacuna = lacuna.cli:main" in entry
    licences = {n for n in names if "/licenses/" in n}
    assert licences == {f"{STEM}.dist-info/licenses/{f}" for f in ("LICENSE", "NOTICE")}


@building
def test_the_sdist_holds_the_source_and_the_guide_and_no_study(built):
    with tarfile.open(built["sdist"]) as t:
        names = {m.name.removeprefix(f"{STEM}/") for m in t.getmembers() if m.isfile()}
        info = t.extractfile(f"{STEM}/PKG-INFO").read().decode()
    assert info.startswith("Metadata-Version: 2.4\n") and f"Name: {NAME}\n" in info
    top = {"pyproject.toml", "README.md", "LICENSE", "NOTICE", "PKG-INFO", ".gitignore"}
    rest = {n for n in names - top if not n.startswith("src/lacuna/")}
    assert rest == {"docs/GUIDE.md"}, sorted(rest)
    assert top <= names and "src/lacuna/cli.py" in names
    assert built["sdist"].stat().st_size < 400_000  # no artifacts, no paper, no fixtures


@building
def test_the_installed_package_imports_without_httpx_or_any_extra(venv):
    code = (
        "import sys, importlib.metadata as md, importlib.util as u\n"
        "import lacuna, lacuna.cli\n"
        "print(lacuna.__file__)\n"
        "print(sorted(m for m in sys.modules if m.split('.')[0] in "
        "('httpx', 'mcp', 'opentelemetry')))\n"
        "print(sorted(n for n in ('httpx', 'mcp', 'opentelemetry') if u.find_spec(n)))\n"
        "print(sorted(d.metadata['Name'] for d in md.distributions()))\n"
        f"print(md.version({NAME!r}))\n"
    )
    r = _run([venv / "python", "-c", code], venv)
    assert r.returncode == 0, r.stderr
    where, loaded, installable, dists, version = r.stdout.splitlines()
    # The copy imported is the wheel's, not this checkout's `src`.
    assert Path(where).resolve().is_relative_to(venv.parent.resolve()), where
    assert loaded == "[]" and installable == "[]"
    assert dists == str(sorted([NAME, "numpy"])) and version == VERSION


@building
def test_the_installed_command_lints_and_estimates_and_refuses_one_recorder(venv, tmp_path):
    lacuna = venv / "lacuna"
    r = _run([lacuna, "lint", "--map", MAP], tmp_path)
    assert (r.returncode, r.stderr) == (0, ""), r.stderr
    assert r.stdout.startswith("ok: two witnesses on every plane")

    inputs = [f"--in={n}={W / (n + '.jsonl')}" for n in ("spans", "logs", "env")]
    r = _run([lacuna, "estimate", "--map", MAP, *inputs], tmp_path)
    assert (r.returncode, r.stderr) == (0, ""), r.stderr
    assert "all planes: union_count 16" in r.stdout

    bad = tmp_path / "one.toml"
    bad.write_text(ONE_RECORDER)
    r = _run([lacuna, "lint", "--map", bad], tmp_path)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "plane p: 1 of the map's witnesses can see it" in r.stdout
