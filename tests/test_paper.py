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
    ]
    assert (cv["covered"], cv["runs"], u["covered"], u["runs"]) == (269, 277, 269, 300)
    rows = paper_tables.failure_map_rows()
    assert len(rows) == len(want)
    for row, (value, lo, hi) in zip(rows, want, strict=True):
        assert (row["value"], row["lo"], row["hi"]) == pytest.approx((value, lo, hi), abs=1e-12)
        assert row["lo"] < row["value"] < row["hi"]
    # Filled circle, open circle, filled square: what the caption says each means.
    assert [r["style"] for r in rows] == ["registered"] * 5 + ["post_hoc"] * 5 + ["study5"] * 2
    assert all(r["label"].startswith("post hoc: ") == (r["style"] == "post_hoc") for r in rows)
    # The row of the pass names the runs left, and the row under it the runs set aside.
    assert "277 runs left" in rows[-2]["label"] and "23 set aside" in rows[-1]["label"]
    assert rows[-1]["lo"] < 0.90 < rows[-1]["hi"], "the caption says this one straddles the bar"

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
        "study5"
        if m.get_marker() == "s"
        else "post_hoc"
        if paper_figures.matplotlib.colors.to_rgba(m.get_markerfacecolor()) == hollow
        else "registered"
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
