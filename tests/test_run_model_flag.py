"""`--model`, `--usd-per-m-in`, `--usd-per-m-out` for scripts/run_episodes.py.

The flags override LACUNA_MODEL / LACUNA_USD_PER_M_IN / LACUNA_USD_PER_M_OUT, the budget guard
uses the values in force, and the summary records them. No network: the real ChatClient is wired
to an httpx mock transport that answers every request with an empty reply and fixed token usage.
"""

from __future__ import annotations

import functools
import json
import sys
from pathlib import Path

import httpx
import pytest

import lacuna.agents.llm as llm
from lacuna.agents.llm import ChatClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_episodes  # noqa: E402

TOKENS = 1_000  # prompt and completion tokens of every mock reply
ENV = {
    "LACUNA_BASE_URL": "http://mock.invalid/v1",
    "LACUNA_API_KEY": "test-key",
    "LACUNA_MODEL": "env-model",
    "LACUNA_USD_PER_M_IN": "1",
    "LACUNA_USD_PER_M_OUT": "2",
}


@pytest.fixture
def seen(monkeypatch):
    """Patch ChatClient to use a mock transport; return the list of request bodies."""
    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": "done"}}],
                "usage": {"prompt_tokens": TOKENS, "completion_tokens": TOKENS},
            },
        )

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(llm, "ChatClient", functools.partial(ChatClient, transport=transport))
    for k in ENV:
        monkeypatch.delenv(k, raising=False)
    return bodies


def _main(monkeypatch, tmp_path, env: dict, *argv: str) -> Path:
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(run_episodes, "ROOT", tmp_path)
    full = ["run_episodes.py", "--agent", "llm", "--episodes", "1", "--run", "r", *argv]
    monkeypatch.setattr(sys, "argv", full)
    run_episodes.main()
    return tmp_path / "artifacts" / "episodes" / "r"


def test_environment_variables_are_the_default(monkeypatch, tmp_path, seen):
    out = _main(monkeypatch, tmp_path, ENV)
    s = json.loads((out / "summary.json").read_text())
    assert s["model"] == "env-model"
    assert (s["usd_per_m_in"], s["usd_per_m_out"]) == (1.0, 2.0)
    assert {b["model"] for b in seen} == {"env-model"}
    assert json.loads((out / "receipt.json").read_text())["total_usd"] == pytest.approx(0.003)


def test_flags_override_the_environment(monkeypatch, tmp_path, seen):
    out = _main(
        monkeypatch, tmp_path, ENV,
        "--model", "flag-model", "--usd-per-m-in", "10", "--usd-per-m-out", "20",
    )  # fmt: skip
    s = json.loads((out / "summary.json").read_text())
    assert s["model"] == "flag-model"
    assert (s["usd_per_m_in"], s["usd_per_m_out"]) == (10.0, 20.0)
    assert {b["model"] for b in seen} == {"flag-model"}
    # 1000 tokens each way at $10 / $20 per million
    assert json.loads((out / "receipt.json").read_text())["total_usd"] == pytest.approx(0.03)


def test_each_flag_overrides_on_its_own(monkeypatch, tmp_path, seen):
    out = _main(monkeypatch, tmp_path, ENV, "--usd-per-m-out", "6")
    s = json.loads((out / "summary.json").read_text())
    assert (s["model"], s["usd_per_m_in"], s["usd_per_m_out"]) == ("env-model", 1.0, 6.0)


def test_flags_alone_suffice_without_the_model_and_price_variables(monkeypatch, tmp_path, seen):
    only_conn = {k: ENV[k] for k in ("LACUNA_BASE_URL", "LACUNA_API_KEY")}
    out = _main(
        monkeypatch, tmp_path, only_conn,
        "--model", "m", "--usd-per-m-in", "0", "--usd-per-m-out", "0",
    )  # fmt: skip
    s = json.loads((out / "summary.json").read_text())
    assert s["model"] == "m" and s["usd_per_m_in"] == 0.0 and s["usd_per_m_out"] == 0.0


def test_missing_model_or_price_is_a_usage_error(monkeypatch, tmp_path, seen):
    only_conn = {k: ENV[k] for k in ("LACUNA_BASE_URL", "LACUNA_API_KEY")}
    with pytest.raises(SystemExit):
        _main(monkeypatch, tmp_path, only_conn, "--model", "m")


def test_budget_guard_uses_the_flag_prices(monkeypatch, tmp_path, seen):
    # Env prices make one reply cost $0.003, far under the budget; the flag prices make it $3.
    out = _main(
        monkeypatch, tmp_path, ENV,
        "--episodes", "3", "--budget", "1", "--usd-per-m-in", "1000", "--usd-per-m-out", "2000",
    )  # fmt: skip
    s = json.loads((out / "summary.json").read_text())
    assert s["episodes"] == 1 and s["stopped"].startswith("BudgetExceeded")
    assert len(seen) == 1


def test_budget_guard_does_not_trip_at_the_environment_prices(monkeypatch, tmp_path, seen):
    out = _main(monkeypatch, tmp_path, ENV, "--episodes", "3", "--budget", "1")
    s = json.loads((out / "summary.json").read_text())
    assert s["episodes"] == 3 and s["stopped"] == ""


def test_scripted_runs_keep_their_summary_shape(monkeypatch, tmp_path):
    monkeypatch.setattr(run_episodes, "ROOT", tmp_path)
    argv = ["run_episodes.py", "--agent", "scripted", "--episodes", "1", "--run", "s"]
    monkeypatch.setattr(sys, "argv", argv)
    run_episodes.main()
    s = json.loads((tmp_path / "artifacts/episodes/s/summary.json").read_text())
    assert s["model"] is None and "usd_per_m_in" not in s and "usd_per_m_out" not in s
