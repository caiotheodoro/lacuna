"""Every figure in paper/lacuna.tex and the two short papers must trace to an
artifact, a frozen pre-registration, or a code constant checked here.

Tables are generated (scripts/paper_tables.py) and excluded. Prose is not, and
prose is where a hand-typed number slips in: D26 caught four, the README gate
caught six more.

`make paper-check` additionally sets PAPER_FINAL=1, which fails while any
\\pending{} box or MISSING ARTIFACT table remains.

paper/lacuna-v1.tex also carries study 5, whose registration fixes how its
verdict may be worded; the last tests here hold the paper to that wording.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import claims_lib
import pytest

ROOT = Path(__file__).resolve().parents[1]
TEX = ROOT / "paper" / "lacuna.tex"
# The workshop version condenses the report; its numbers face the same gate.
WORKSHOP = ROOT / "paper" / "lacuna-workshop.tex"
# The v1 paper is the short paper from study 5 on.
V1 = ROOT / "paper" / "lacuna-v1.tex"
TEXES = [TEX, WORKSHOP, V1]
TABLES = ROOT / "paper" / "tables"

# Cited from another paper, with the citation next to it in the text.
CITED = {"4.1"}
# Names, not measurements: ISO/IEC 42001, SB 53, the 2^3 cells of a 3-witness table.
NAMES = {"42001", "53", "2", "3"}
# Structural integers: Gamma values, section-sized counts, rungs.
STRUCTURAL = {str(i) for i in range(0, 11)}


def _code_facts() -> set[str]:
    import inspect

    from lacuna.envs.tasks import make_world
    from lacuna.envs.tools import TOOLS
    from lacuna.estimators.diagnostics import DELTA_BIC
    from lacuna.estimators.latent_class import sample_posterior
    from lacuna.estimators.loglinear import CAP_RATIO

    sig = inspect.signature(make_world).parameters
    lc = inspect.signature(sample_posterior).parameters
    facts = {
        str(len(TOOLS)),  # "eleven tools" is spelled out, but keep the check honest
        str(sig["n_customers"].default),
        str(sig["n_orders"].default),
        str(int(DELTA_BIC)),
        # The methods appendix: search cap and the latent-class sampler.
        str(CAP_RATIO),
        *(str(lc[p].default) for p in ("k_max", "burn", "draws", "a_alpha", "b_alpha")),
    }
    return facts


def _study5_gate_constants() -> set[str]:
    """The limits study 5's registration sets before any test (lateness, the clocks, the
    two gates): constants of a frozen text, which the v1 paper states in its design."""
    text = (ROOT / "docs" / "PREREG-STUDY5.md").read_text()
    gates = text.split("## Gates applied before any test", 1)[1].split("\n## ", 1)[0]
    return set(re.findall(r"(?<![\w.])\d+(?:\.\d+)?(?![\w])", gates))


def test_the_integrity_limits_in_v1_are_the_registration_s():
    """0.25 s and 0.5 s are accepted because the registration sets them, not because some
    artifact happens to hold the same digits."""
    para = V1.read_text().split("\\paragraph{Integrity rules.}", 1)[1].split("\n\n", 1)[0]
    limits = set(re.findall(r"(\d+\.\d+) s\b", para))
    assert limits == {"0.25", "0.5"}
    assert limits <= _study5_gate_constants()


def _numbers_in(obj, out: set[str]) -> set[str]:
    if isinstance(obj, dict):
        for v in obj.values():
            _numbers_in(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _numbers_in(v, out)
    elif isinstance(obj, bool):
        pass
    elif isinstance(obj, (int, float)):
        for x in (obj, abs(obj)):
            out.add(str(x))
            for fmt in ("{:.0f}", "{:.1f}", "{:.2f}", "{:.3f}", "{:.4f}"):
                out.add(fmt.format(float(x)))
            out.add(f"{float(x) * 100:.0f}")
    return out


def _allowed() -> set[str]:
    allowed: set[str] = set()
    for p in list((ROOT / "artifacts").rglob("*.json")) + [ROOT / "MEASUREMENT_CARD.json"]:
        try:
            _numbers_in(json.loads(p.read_text()), allowed)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
    prereg = (ROOT / "docs" / "PRE-REGISTRATION.md").read_text()
    allowed |= set(re.findall(r"\d+(?:\.\d+)?", prereg))
    allowed |= _study5_gate_constants()
    return allowed | CITED | NAMES | STRUCTURAL | _code_facts()


def _prose(tex: str) -> str:
    body = tex.split("\\begin{document}", 1)[1]
    # A diagram's coordinates and lengths are not results; its caption is outside it.
    body = re.sub(r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}", " ", body, flags=re.S)
    body = re.sub(r"(?m)%.*$", "", body)  # comments
    body = re.sub(r"\\pending\{(?:[^{}]|\{[^{}]*\})*\}", " ", body)  # unfinished boxes
    body = re.sub(r"``.*?''", " ", body, flags=re.S)  # quoted model output
    body = re.sub(r"\\(?:input|label|ref|citep?|bibliography\w*)\{[^}]*\}", " ", body)
    body = re.sub(r"\\texttt\{[^}]*\}", " ", body)  # identifiers like fin_mid_rho100
    return body


@pytest.mark.parametrize("tex", TEXES, ids=lambda p: p.name)
def test_every_prose_number_has_a_source(tex):
    if not tex.exists():
        pytest.skip(f"{tex.name} not written")
    allowed = _allowed()
    found = set(re.findall(r"(?<![\w.])\d+(?:\.\d+)?(?![\w])", _prose(tex.read_text())))
    missing = sorted(found - allowed, key=lambda x: float(x))
    assert not missing, f"{tex.name} states figures with no artifact behind them: {missing}"


def _inputs(tex: Path) -> list[str]:
    return re.findall(r"\\input\{tables/([^}]+)\}", tex.read_text())


@pytest.mark.skipif(os.environ.get("PAPER_FINAL") != "1", reason="draft: pending sections allowed")
@pytest.mark.parametrize("tex", TEXES, ids=lambda p: p.name)
def test_paper_is_finished(tex):
    body = tex.read_text().split("\\begin{document}", 1)[1]
    assert "\\pending{" not in body, "unfinished \\pending box"
    bad = [p.name for p in TABLES.glob("*.tex") if "MISSING ARTIFACT" in p.read_text()]
    assert not bad, f"tables without an artifact: {bad}"
    unwritten = [t for t in _inputs(tex) if not (TABLES / f"{t}.tex").exists()]
    assert not unwritten, f"{tex.name} inputs tables nothing generated: {unwritten}"


def test_the_tables_are_what_the_generator_writes(tmp_path, monkeypatch):
    """Generated, not typed: the v1 paper's tables equal a fresh run over the artifacts,
    and paper/tables holds no file the generator has stopped writing."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import paper_tables

    monkeypatch.setattr(paper_tables, "OUT", tmp_path)
    paper_tables.main([])
    stale = [
        name
        for name in _inputs(V1)
        if (tmp_path / f"{name}.tex").read_text() != (TABLES / f"{name}.tex").read_text()
    ]
    assert not stale, f"tables that differ from what the artifacts give: {stale}"
    orphans = sorted(q.name for q in TABLES.glob("*.tex") if not (tmp_path / q.name).exists())
    assert not orphans, (
        f"paper/tables holds files scripts/paper_tables.py does not write: {orphans}"
    )


def _art(rel: str) -> dict:
    return json.loads((ROOT / "artifacts" / rel).read_text())


def _table_rows(tex: str) -> list[list[str]]:
    """The body rows of a generated table, cell by cell."""
    body = tex.split("\\midrule", 1)[1].split("\\bottomrule")[0]
    rows = [line.removesuffix(" \\\\").split(" & ") for line in body.splitlines()]
    return [[c.strip() for c in r] for r in rows if len(r) > 1]


# Why a run was set aside, by the words of the integrity check, in the table's column order.
ASIDE_WORDS = (
    ("late call", "made a call"),
    ("not ready", "was ready"),
    ("clock off", "containers' clock"),
    ("driver stopped", "run driver stopped"),
    ("no access line", "log relay delivered no line"),
)
# The first run of each stretch after the first, as D93 names them.
STRETCH_STARTS = ("s5-s1-219", "s5-s1-230", "s5-s1-231", "s5-s1-270")


def test_the_set_aside_table_is_generated_and_counts_what_the_artifact_holds(tmp_path, monkeypatch):
    """The v1 paper's table of runs set aside: in the paper, equal to a fresh generation,
    and every cell recounted here from the artifact and the plan's order, by this test's
    own code and its own boundaries."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import paper_tables
    import study5_runs

    assert "study5_set_aside" in _inputs(V1)
    monkeypatch.setattr(paper_tables, "OUT", tmp_path)
    paper_tables.study5_set_aside()
    fresh = (tmp_path / "study5_set_aside.tex").read_text()
    assert fresh == (TABLES / "study5_set_aside.tex").read_text()

    c = _art("study5/confirm.json")
    aside = {r["run"]: r["why"] for r in c["runs_set_aside"]}
    plan = study5_runs.runs()
    order = [r["run"] for r in plan]
    cond = {r["run"]: r["condition"] for r in plan}
    assert len(order) == c["planned_runs"]

    def counts(names: list[str], of_s1: bool) -> list[str]:
        hit = [n for n in names if n in aside]
        reasons = [sum(any(words in w for w in aside[n]) for n in hit) for _, words in ASIDE_WORDS]
        s1 = str(sum(cond[n] == "s1" for n in hit)) if of_s1 else "--"
        return [str(len(names)), str(len(hit)), s1, *map(str, reasons)]

    cuts = [0, *(order.index(name) for name in STRETCH_STARTS), len(order)]
    assert cuts == sorted(cuts)
    want = [counts([n for n in order if cond[n] == k], False) for k in ("s0", "s1", "s2")]
    want.append(counts(order, True))
    want += [counts(order[i:j], True) for i, j in zip(cuts, cuts[1:], strict=False)]

    header = fresh.split("\\toprule", 1)[1].split("\\midrule", 1)[0]
    assert [h.strip().removesuffix(" \\\\") for h in header.split(" & ")][4:] == [
        head for head, _ in ASIDE_WORDS
    ]
    rows = _table_rows(fresh)
    assert [r[1:] for r in rows] == want
    assert [r[0] for r in rows[:4]] == ["\\texttt{s0}", "\\texttt{s1}", "\\texttt{s2}", "all"]
    assert [r[0].split(",")[0] for r in rows[4:]] == [
        f"runs {i + 1} to {j}" if j - i > 1 else f"run {j}"
        for i, j in zip(cuts, cuts[1:], strict=False)
    ]
    assert [int(n) for n in want[3][:3]] == [420, 34, 23]
    # Counts of runs and nothing else: no coverage, estimate or error by stretch.
    assert not re.search(r"\d\.\d{3}", fresh), "the set-aside table holds a share or a coverage"


def test_the_table_beside_the_power_log_is_generated_and_counts_what_the_artifact_holds(
    tmp_path, monkeypatch
):
    """Post hoc (D112). In the paper, equal to a fresh generation, every cell a count of
    runs from artifacts/study5/posthoc3.json, and no estimate in it."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import paper_tables

    assert "study5_host" in _inputs(V1)
    monkeypatch.setattr(paper_tables, "OUT", tmp_path)
    paper_tables.study5_host()
    fresh = (tmp_path / "study5_host.tex").read_text()
    assert fresh == (TABLES / "study5_host.tex").read_text()
    b = _art("study5/posthoc3.json")["runs_set_aside"]["beside_the_power_log"]
    rows = _table_rows(fresh)
    want = []
    for key in (
        "before_the_last_battery_reading",
        "on_battery_until_external_power",
        "after_external_power",
    ):
        w = b["windows"][key]
        by = w["by_condition"]
        want.append(
            [str(w["runs"]), str(w["set_aside_long"]), str(w["set_aside_short"])]
            + [f"{by[c]['set_aside_short']} of {by[c]['runs']}" for c in ("s0", "s1", "s2")]
        )
    assert [r[1:] for r in rows] == want
    assert fresh.count("Post hoc") == 1 and "no estimate is read" in fresh
    assert not re.search(r"\d\.\d{3}", fresh), "the table holds a share or a coverage"
    # The sleeps and the long runs, as the note states them.
    lg, sl = b["long"], b["sleeps_while_runs_were_made"]
    assert lg["beside_a_sleep"] == lg["in_the_log"], "the note says each is beside a sleep"
    assert sl["beside_a_run_set_aside"] == sl["sleeps"], "the note says each sleep is beside one"
    assert f"each of the {sl['sleeps']} sleeps" in fresh


def _earlier_cells() -> list[dict]:
    """The `ll_bic` cell of each row of the v1 paper's first table, by its artifact path."""
    base = _art("robustness/baselines.json")["cells"]

    def callid(name: str) -> dict:
        return _art(f"robustness/{name}.json")["cells"]["fin_mid/callid"]["estimators"]

    def post(key: str) -> dict:
        return base[key]["linkage"]["oracle"]["estimators"]

    cells = [
        _art("thinning/confirm.json")["configs"]["fin_mid/signature"]["estimators"],
        _art("thinning/transfer.json")["configs"]["transfer_mid/signature"]["estimators"],
        callid("bypass-analysis-bypass-llm"),
        callid("bypass-analysis-confirm"),
        callid("study4-adversary-s4-qwen-bypass"),
        post("per_record"),
        post("mnar_op_x1.2"),
        post("burst_w12"),
        post("shared_quality_fin"),
        post("bypass_b0.05/fin_mid"),
    ]
    return [e["ll_bic"] for e in cells]


def test_the_failure_map_is_the_artifact_cells():
    """Generated, not typed, and with no plotting library: each row's value, both ends of
    its bar and its style class, against the artifact cells read here by their paths."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import paper_tables

    earlier = _earlier_cells()
    v = _art("robustness/study5-verdicts.json")
    cv, u = v["primary"]["coverage"], v["P1_not_in_test_as_uncovered"]
    post = _art("study5/posthoc3.json")["runs_set_aside"]["estimator"]
    scored = post["with_the_runs_set_aside_as_scored"]["no_record_as_uncovered"]
    want = [
        (
            m["coverage"],
            m["coverage"] - 1.96 * m["coverage_mcse"],
            m["coverage"] + 1.96 * m["coverage_mcse"],
        )
        for m in earlier
    ] + [
        (cv["covered"] / cv["runs"], cv["wilson_lo"], cv["wilson_hi"]),
        (u["covered"] / u["runs"], *u["wilson"]),
        # Post hoc (D112): the runs set aside that left a full record, scored as they were.
        (scored["covered"] / scored["runs"], *scored["wilson"]),
    ]
    assert (cv["covered"], cv["runs"], u["covered"], u["runs"]) == (269, 277, 269, 300)
    assert scored["covered"] == cv["covered"] + post["covered"] and scored["runs"] == u["runs"]
    rows = paper_tables.failure_map_rows()
    assert len(rows) == len(want)
    for row, (value, lo, hi) in zip(rows, want, strict=True):
        assert (row["value"], row["lo"], row["hi"]) == pytest.approx((value, lo, hi), abs=1e-12)
        assert row["lo"] < row["value"] < row["hi"]
    # Filled circle, open circle, filled square, open square: what the caption says each means.
    assert [r["style"] for r in rows] == (
        ["registered"] * 5 + ["post_hoc"] * 5 + ["study5"] * 2 + ["study5_post_hoc"]
    )
    assert all(r["label"].startswith("post hoc: ") == r["style"].endswith("post_hoc") for r in rows)
    # The row of the pass names the runs left, and the row under it the runs set aside.
    assert "277 runs left" in rows[-3]["label"] and "23 set aside" in rows[-2]["label"]
    assert rows[-2]["lo"] < 0.90 < rows[-2]["hi"], "the caption says this one straddles the bar"
    # The post hoc row says what it counts, and the caption says its bar is above the bar.
    assert (
        f"the {post['with_a_full_record']} set aside with a record as scored" in (rows[-1]["label"])
    )
    assert rows[-1]["lo"] > 0.90

    # The first table shows the same cells, row for row.
    table = _table_rows((TABLES / "v1_earlier.tex").read_text())
    assert [r[2].split(" ")[0] for r in table] == [f"{m['coverage']:.3f}" for m in earlier]
    assert "figures/fig_failure_map" in V1.read_text()
    assert (ROOT / "paper" / "figures" / "fig_failure_map.pdf").exists()


def test_the_failure_map_draws_its_rows():
    """Optional (matplotlib is in the `paper` group only): the drawing puts the rows the
    test above checks on the axes, with their bars and marker styles, and no others."""
    pytest.importorskip("matplotlib")
    sys.path.insert(0, str(ROOT / "scripts"))
    import paper_figures
    import paper_tables

    fig, ax = paper_figures.plt.subplots()
    paper_figures.failure_map(ax)
    rows = paper_tables.failure_map_rows()
    marks = [line for line in ax.get_lines() if line.get_marker() in ("o", "s")]
    marks.sort(key=lambda line: float(line.get_ydata()[0]))
    assert [(float(m.get_ydata()[0]), float(m.get_xdata()[0])) for m in marks] == [
        (float(i), r["value"]) for i, r in enumerate(rows)
    ]
    hollow = paper_figures.matplotlib.colors.to_rgba("white")
    styles = [
        {
            ("s", False): "study5",
            ("s", True): "study5_post_hoc",
            ("o", True): "post_hoc",
            ("o", False): "registered",
        }[
            m.get_marker(),
            paper_figures.matplotlib.colors.to_rgba(m.get_markerfacecolor()) == hollow,
        ]
        for m in marks
    ]
    assert styles == [r["style"] for r in rows]
    bars = sorted(
        (float(seg[0][1]), float(seg[0][0]), float(seg[1][0]))
        for coll in ax.collections
        for seg in coll.get_segments()
        if len(seg) == 2 and seg[0][1] == seg[1][1] and seg[0][0] != seg[1][0]
    )
    assert bars == pytest.approx([(float(i), r["lo"], r["hi"]) for i, r in enumerate(rows)])
    assert [t.get_text() for t in ax.get_yticklabels()] == [r["label"] for r in rows]
    paper_figures.plt.close(fig)


def test_v1_states_the_cap_and_the_cutoff_the_code_uses():
    """The prose gate does not read a number glued to a letter. These are the two in the
    estimator's equations: the cap, a multiple of n, and the level of the chi-squared point
    with the one half in front of it."""
    import inspect
    import math

    from lacuna.estimators import loglinear

    tex = V1.read_text()
    caps = re.findall(r"(?<![\w.])(\d+)n(?!\w)", tex)
    assert caps and set(caps) == {str(loglinear.CAP_RATIO)}, caps
    level = re.search(r"is the (\d+)\\% point of the chi-squared distribution on one degree", tex)
    assert level, "the paper no longer says which point of the chi-squared the cutoff is"
    # One degree of freedom: P(X <= x) = erf(sqrt(x / 2)).
    assert math.erf(math.sqrt(loglinear.CHI2_1_95 / 2)) == pytest.approx(
        int(level.group(1)) / 100, abs=1e-9
    )
    assert "-\\tfrac12\\,\\hat\\phi\\,\\chi^2_{1}" in tex
    assert "cut = mle_ll - chi2 / 2" in inspect.getsource(loglinear.Profile.interval)
    registered = (ROOT / "scripts" / "study5_lib.py").read_text()
    assert "prof.interval(mle, mle_ll, chi2=loglinear.CHI2_1_95 * phi)" in registered
    assert "if n + z0 > loglinear.CAP_RATIO * n:" in registered


# --- key-aware gate for the abstract ----------------------------------------
# The test above accepts a number found anywhere. The abstract is where a reader
# takes figures from, so each of its numbers is bound to one artifact cell in
# tests/claims_map.json. Only the abstracts of the two short papers are mapped, not
# the bodies: a body has hundreds of figures, and its tables are generated from
# artifacts.

ABSTRACTS = {"paper_abstract": WORKSHOP, "paper_v1_abstract": V1}


def _abstract_numbers(tex: Path) -> tuple[str, dict[int, str]]:
    text = tex.read_text()
    body = claims_lib.squash(text.split("\\begin{abstract}", 1)[1].split("\\end{abstract}", 1)[0])
    pattern = r"(?<![\w.])\d+(?:\.\d+)?(?![\w])"
    return body, {m.start(): m.group() for m in re.finditer(pattern, body)}


def _abstract_claims(key: str) -> list[dict]:
    block = json.loads(claims_lib.MAP.read_text())[key]
    assert ROOT / block["file"] == ABSTRACTS[key], f"{key} maps {block['file']}"
    return block["claims"]


@pytest.mark.parametrize("key", ABSTRACTS)
def test_every_abstract_number_is_mapped_to_an_artifact_value(key):
    tex = ABSTRACTS[key]
    if not tex.exists():
        pytest.skip(f"{tex.name} not written")
    body, numbers = _abstract_numbers(tex)
    problems = claims_lib.unmapped_and_failing(body, numbers, _abstract_claims(key))
    assert not problems, f"{tex.name} abstract numbers not backed by their mapped cell:\n" + (
        "\n".join(problems)
    )


# --- study 5: the wording its registration fixes -----------------------------
# docs/PREREG-STUDY5.md says what the paper may say in each outcome, and D93 what
# goes beside the pass. These hold paper/lacuna-v1.tex and its tables to it.

FLOORS = ("trust_trace", "union_count", "union_bound")
# Words the registration bars where a reading is "not shown"; the paper uses none anywhere.
LEANING = (r"\bnear\b", r"\bnearly\b", "consistent with", "close to nominal", "calibrated")
# The pass: 269 of the 277 runs left. What must follow it, in that sentence or the next.
PASS = re.compile(r"269 of (?:the )?277")
BESIDE_THE_PASS = (
    r"23 (?:of the 300 planned|set aside)",  # the planned runs not in the test
    r"269 of 300",  # with those counted as not covered
    r"``not shown''",  # how the registration reads that figure
)


def _v1_text() -> str:
    """The v1 paper as a reader meets it: the body with its generated tables inlined."""
    body = V1.read_text().split("\\begin{document}", 1)[1]
    body = re.sub(r"(?m)%.*$", "", body)
    return claims_lib.squash(
        re.sub(
            r"\\input\{tables/([^}]+)\}",
            lambda m: (TABLES / f"{m.group(1)}.tex").read_text(),
            body,
        )
    )


def _two_sentences(text: str, start: int) -> str:
    """From `start` to the end of the sentence after the one it is in."""
    ends = [m.end() for m in re.finditer(r"\.(?:'')?\)? (?=[A-Z0-9\\`(])", text[start:])]
    return text[start : start + ends[1]] if len(ends) > 1 else text[start:]


@pytest.mark.skipif(not V1.exists(), reason="lacuna-v1.tex not written")
def test_v1_never_quotes_the_pass_without_the_runs_set_aside():
    text = _v1_text()
    quoted = [m.start() for m in PASS.finditer(text)]
    assert quoted, "the v1 paper does not quote P1's count"
    bare = []
    for at in quoted:
        window = _two_sentences(text, at)
        if not all(re.search(need, window) for need in BESIDE_THE_PASS):
            bare.append(window[:160])
    assert not bare, "P1's pass quoted without the runs set aside beside it:\n" + "\n".join(bare)


# A statement that P1 passed, in any words: the clause or the criterion "passed", "passes",
# "is met", "was met", or "the pass". Not the rule that defines a pass ("if the clause passes").
STATES_THE_PASS = re.compile(
    r"\b(?:clause|criterion)\b[^.;:]{0,60}?\b(?:passed|passes|(?:is|was) met)\b|\b[Tt]he pass\b"
)
SENTENCE_END = re.compile(r"\.(?:'')?\)? (?=[A-Z0-9\\`(])")


def _around(text: str, at: int) -> str:
    """What must hold the count for a statement at `at`: its table row if it is in a
    table, otherwise its own sentence and the next."""
    top, bottom = text.rfind("\\toprule", 0, at), text.rfind("\\bottomrule", 0, at)
    if top > bottom:
        start = max(text.rfind("\\\\ ", 0, at), text.rfind("\\midrule", 0, at)) + 1
        end = text.find("\\\\", at)
        return text[start:end]
    before = [m.end() for m in SENTENCE_END.finditer(text, 0, at)]
    start = before[-1] if before else 0
    return text[start:at] + _two_sentences(text, at)


def _bare_passes(text: str) -> list[str]:
    bare = []
    for m in STATES_THE_PASS.finditer(text):
        if re.search(r"\bif\b[^.]{0,40}$", text[: m.start()]):
            continue  # the rule, not a result
        window = _around(text, m.start())
        if not (re.search(r"\b23\b", window) and re.search(r"\b300\b", window)):
            bare.append(window.strip()[:200])
    return bare


@pytest.mark.skipif(not V1.exists(), reason="lacuna-v1.tex not written")
def test_v1_never_states_the_pass_without_the_runs_not_in_the_test():
    """The registration: "The number of planned s1 runs not in the test is given wherever
    a verdict is quoted." The test above fires on the count of the pass; this one fires on
    the pass stated in words, in the prose and in a table row."""
    text = _v1_text()
    assert len(STATES_THE_PASS.findall(text)) >= 5, "the gate no longer finds the verdict"
    bare = _bare_passes(text)
    assert not bare, "P1's pass stated without the 23 of 300 beside it:\n" + "\n".join(bare)


def test_the_gate_on_the_pass_in_words_fires_and_holds():
    fires = (
        "The clause passed on the runs left. Its gloss was another. The mark stands. ",
        "The pass is on the runs left and does not survive. It does not show more. ",
        "\\toprule a & b \\\\ \\midrule P1 & the clause passes on the runs left & wrong \\\\ "
        "Q3 & 23 of 300 \\\\ \\bottomrule ",
    )
    for text in fires:
        assert _bare_passes(text), text
    holds = (
        "The clause passed on 277 of the 300 planned runs, 23 set aside. Next one. ",
        "The criterion is met on the runs left. 23 of the 300 planned runs were set aside. ",
        "The mark is wrong if the coverage clause passes. Nothing else. ",
        "\\toprule a \\\\ \\midrule P1 & the clause passes on 277 of 300, 23 set aside \\\\ "
        "\\bottomrule ",
    )
    for text in holds:
        assert not _bare_passes(text), text


def test_the_window_is_two_sentences_and_no_more():
    text = "It covered 269 of 277 runs. That is all. 23 of the 300 planned; 269 of 300."
    assert _two_sentences(text, text.index("269")) == "269 of 277 runs. That is all. "


@pytest.mark.skipif(not V1.exists(), reason="lacuna-v1.tex not written")
def test_v1_quotes_the_counts_the_verdict_artifact_holds():
    v = json.loads((ROOT / "artifacts" / "robustness" / "study5-verdicts.json").read_text())
    cov, beside = v["primary"]["coverage"], v["P1_not_in_test_as_uncovered"]
    assert (cov["covered"], cov["runs"]) == (269, 277), "PASS above quotes another count"
    assert (beside["not_in_test"], beside["covered"], beside["runs"]) == (23, 269, 300)
    lo, hi = beside["wilson"]
    assert lo < 0.90 < hi, "the paper says this interval straddles the bar"
    assert v["primary"]["P1_clauses"] == {"coverage": "pass", "unbounded": "pass"}


POST_HOC_2 = "\\paragraph{Two descriptions made after the verdicts were read.}"
# What the owner's decision (D107) has beside the two descriptions wherever they are quoted.
FENCE_2 = (
    "post hoc",
    "no criterion",
    "no verdict depends on either",
    "the registration labels exploratory",
    "for no claim for or against the estimate by band, pool or outage length",
    "pooled over runs that share sessions",
    "say nothing about any one run",
    "not a test of the assumption",
    "none of those papers' methods",
    "nothing is computed from them",
)


@pytest.mark.skipif(not V1.exists(), reason="lacuna-v1.tex not written")
def test_v1_quotes_the_two_later_descriptions_from_their_artifact_with_the_fence():
    """The pooled shares and the implied odds ratio: each figure is the cell of
    artifacts/study5/posthoc2.json it is said to be, the fence is beside them, and none of
    them is in the abstract."""
    tex = V1.read_text()
    assert tex.count(POST_HOC_2) == 1
    para = claims_lib.squash(tex.split(POST_HOC_2, 1)[1].split("\n\n", 1)[0])
    # After P1's material and before P2.
    assert tex.index("\\paragraph{P1.}") < tex.index(POST_HOC_2) < tex.index("\\paragraph{P2.}")
    p = _art("study5/posthoc2.json")
    assert p["note"].startswith("Post hoc") and "D107" in p["note"]
    shown, need = (
        p["end_state_shows_what_both_telemetry_logs_lost"],
        p["dependence_each_run_would_need"],
    )
    lost, rest = shown["lost_by_both_telemetry_logs"], shown["the_rest"]
    q = need["implied_odds_ratio"]
    by_factor = {w["factor"]: w["runs"] for w in need["within_a_factor_of_1"]}
    want = (
        f"lost {lost['actions']} actions, of which the end state shows {lost['shown']}, "
        f"a share of {lost['share']:.3f}",
        f"of the other {rest['actions']} actions it shows {rest['shown']}, "
        f"a share of {rest['share']:.3f}",
        f"defined in {need['defined']} of the {need['runs']} runs; in the other "
        f"{need['not_defined']} ",
        f"median is {q['median']:.3f} and its quartiles are {q['q1']:.3f} and {q['q3']:.3f}",
        f"{by_factor[1.25]} runs are within a factor of 1.25 of 1 and {by_factor[1.5]} "
        "within a factor of 1.5",
    )
    missing = [w for w in want if w not in para]
    assert not missing, f"the paragraph does not quote the artifact: {missing}"
    unfenced = [w for w in FENCE_2 if w not in para]
    assert not unfenced, f"the two descriptions are quoted without: {unfenced}"
    assert (need["runs"], shown["runs"], shown["runs_not_described"]) == (277, 277, 0)
    # The sentence that carries them carries the runs not in the test.
    assert "277 \\texttt{s1} runs in the test, 23 of the 300 planned set aside" in para
    abstract = tex.split("\\begin{abstract}", 1)[1].split("\\end{abstract}", 1)[0]
    figures = (lost["actions"], lost["shown"], rest["actions"], rest["shown"])
    shares = [f"{x:.3f}" for x in (lost["share"], rest["share"], q["q1"], q["median"], q["q3"])]
    in_abstract = [str(x) for x in (*figures, *shares) if str(x) in abstract]
    assert not in_abstract, f"post hoc figures in the abstract: {in_abstract}"
    assert "odds ratio" not in abstract


@pytest.mark.skipif(not V1.exists(), reason="lacuna-v1.tex not written")
def test_v1_uses_no_leaning_word():
    text = _v1_text()
    hits = [w for w in LEANING if re.search(w, text, flags=re.I)]
    assert not hits, f"words the study-5 registration bars: {hits}"


@pytest.mark.skipif(not V1.exists(), reason="lacuna-v1.tex not written")
def test_v1_tables_of_estimates_carry_the_floors():
    """AGENTS.md rule 5. A table of estimates is one whose header names an estimator."""
    names = re.compile(r"\\texttt\{(?:ll\\_\w+|chapman\\_\w+|seq\\_gap\w*|strat\\_bic)")
    judged, lacking = [], []
    for name in _inputs(V1):
        rows = (TABLES / f"{name}.tex").read_text().split("\\bottomrule", 1)[0]
        header = rows.split("\\toprule", 1)[1].split("\\midrule", 1)[0]
        if not names.search(header):
            continue
        judged.append(name)
        if not all(f.replace("_", "\\_") in rows for f in FLOORS):
            lacking.append(name)
    assert len(judged) >= 3, f"the check found only these tables of estimates: {judged}"
    assert not lacking, f"tables of estimates without all three floors: {lacking}"
