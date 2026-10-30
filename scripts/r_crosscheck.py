#!/usr/bin/env python3
# ruff: noqa: E501  (the embedded R source has long lines)
"""Cross-check our log-linear fits against R's Rcapture (closes docs/DECISIONS.md D1).

Usage:
  uv run python scripts/r_crosscheck.py            # build image if needed, run, write artifact
  uv run python scripts/r_crosscheck.py --out /tmp/x.json

R is not installed on the build machine, so R runs in Docker: a pinned
rocker/r-ver image (whose CRAN snapshot date is fixed by the tag, so package
versions are reproducible) with Rcapture and dga installed on top. Datasets are
embedded in the R script on stdin; nothing is mounted.

For every 3-witness table and each of the 8 identifiable models (independence,
each single pair, each two pairs, all three pairs) we compare, against
`closedpCI.t(mX=<hierarchical name>)` and `closedpMS.t`:

  point   n + exp(intercept)                 ours fit_conditional  vs  results$abundance
  prof    argmax of the multinomial profile  ours Profile (continuous)  vs  CI$abundance
  lo, hi  95% profile-likelihood endpoints   ours Profile (continuous)  vs  CI$infCL, CI$supCL
  bic     BIC minus the best BIC in the table (the constant cancels)

and, for independence, `closedp.t`'s Mt row. The shipped intervals are integer
valued (`Profile.interval` returns integers), so they are reported next to the
continuous ones: the rounding is 1 unit by construction and is not an
estimation difference.

`dga` (Bayesian model averaging, hyper-Dirichlet prior) is a different method
from `ll_average` (BIC-weighted mixture of profile likelihoods). It is run and
recorded, but it is not held to the 1e-3 tolerance; see `dga` in the artifact.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from lacuna.estimators import loglinear
from lacuna.estimators import loglinear as ll
from lacuna.histories import keep_witnesses, observed_total
from lacuna.sim import SCENARIOS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "robustness" / "r-crosscheck.json"

R_BASE = (
    "rocker/r-ver:4.4.2@sha256:df26749182af64d5263bf64149d51a427b476ed28c4e046997143be3f97fdd7c"
)
IMAGE = "lacuna-rcheck:r4.4.2"
DOCKERFILE = f"""FROM {R_BASE}
RUN Rscript -e 'install.packages(c("Rcapture", "dga"))'
"""
TOL_REL = 1e-3  # docs/PLAN.md: agreement with R to 1e-3
FMAX = 20  # Rcapture's search limit as a multiple of N-hat; ours is CAP_RATIO * n


def have_docker() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=30).returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


def ensure_image() -> None:
    have = subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True)
    if have.returncode != 0:
        subprocess.run(["docker", "build", "-t", IMAGE, "-"], input=DOCKERFILE.encode(), check=True)


# --- models -----------------------------------------------------------------


def rcapture_name(pairs: ll.Pairs, k: int = 3) -> str:
    """Hierarchical model name, witness j written as j+1: ((0,1),) -> "[12,3]"."""
    terms = [f"{a + 1}{b + 1}" for a, b in pairs]
    covered = {w for p in pairs for w in p}
    terms += [str(w + 1) for w in range(k) if w not in covered]
    return "[" + ",".join(terms) + "]"


# --- datasets ---------------------------------------------------------------


def sim_datasets() -> list[dict]:
    from dataclasses import replace

    out = []
    for si, (name, sc) in enumerate(SCENARIOS.items()):
        variants = [(sc, 200, s) for s in (0, 1)]
        if name in ("l0_independent", "l0_heterogeneous", "l1_hides_from_trace"):
            variants.append((replace(sc, n_actions=2000), 2000, 2))
        for scn, n_actions, s in variants:
            rng = np.random.default_rng([20261029, si, s])
            full = scn.sample(rng)
            full[0] = 0
            t3 = keep_witnesses(full, (0, 1, 2))
            out.append(
                {
                    "name": f"sim/{name}/N{n_actions}/s{s}",
                    "source": "sim",
                    "counts": [int(c) for c in t3],
                }
            )
    return out


def confirm_datasets() -> list[dict]:
    sys.path.insert(0, str(ROOT / "scripts"))
    from thin_episodes import load

    from lacuna.thinning import HELDOUT, tables, thin

    episodes = load("confirm")
    cfgs = {c.name: c for c in HELDOUT}

    def pooled(cfg, rng):
        parts = []
        for led, recs in episodes:
            kept = recs if cfg is None else thin(recs, cfg, rng)
            parts.append(tables(led, kept, "oracle")[0])
        t4 = np.sum(parts, axis=0)
        return [int(c) for c in keep_witnesses(t4, (0, 1, 2))]

    out = [{"name": "confirm/unthinned/oracle", "source": "confirm", "counts": pooled(None, None)}]
    for cname, draws in (("fin_mid", 4), ("fin_mid_rho100", 2)):
        for i in range(draws):
            rng = np.random.default_rng([20261029, 7, hash_name(cname), i])
            out.append(
                {
                    "name": f"confirm/{cname}/oracle/d{i}",
                    "source": "confirm",
                    "counts": pooled(cfgs[cname], rng),
                }
            )
    return out


def hash_name(s: str) -> int:
    return sum(ord(c) * 31**i for i, c in enumerate(s)) % (2**31)


def boundary_datasets() -> list[dict]:
    """Hand-built tables where the MLE is on the boundary or the pair terms vanish."""
    return [
        # Two witnesses never overlap with the third: z0 diverges for independence.
        {"name": "edge/zero-overlap", "source": "edge", "counts": [0, 30, 25, 0, 20, 0, 0, 0]},
        # One empty cell, interactions identifiable.
        {"name": "edge/one-zero-cell", "source": "edge", "counts": [0, 31, 22, 40, 17, 28, 19, 0]},
        # Tiny counts.
        {"name": "edge/small", "source": "edge", "counts": [0, 3, 2, 1, 2, 1, 1, 2]},
        # The closed-form example used in tests/test_loglinear.py.
        {"name": "edge/closed-form", "source": "edge", "counts": [0, 31, 22, 40, 17, 28, 19, 55]},
    ]


# --- R side -----------------------------------------------------------------

R_PRELUDE = r"""
suppressMessages(library(Rcapture))
H <- t(sapply(1:7, function(h) c(bitwAnd(h,1)>0, bitwAnd(h,2)>0, bitwAnd(h,4)>0)*1))
fmt <- function(v) if (is.character(v)) v else sprintf("%.17g", v)
emit <- function(...) {
  kv <- list(...)
  cat("@@", paste(names(kv), sapply(kv, function(v) gsub("[\t\n]", " ", paste(fmt(v), collapse=";"))),
      sep="=", collapse="\t"), "\n", sep="\t")
}
num <- function(x) suppressWarnings(as.numeric(gsub(">", "", x)))
"""

R_MODEL = r"""
for (nm in names(MODELPAIRS)) {
  warns <- character(0)
  ci <- withCallingHandlers(
    tryCatch(closedpCI.t(X, dfreq=TRUE, mX=nm, fmaxSupCL=FMAX, mname=nm), error=function(e) e),
    warning=function(w) { warns <<- c(warns, conditionMessage(w)); invokeRestart("muffleWarning") })
  if (inherits(ci, "error")) {
    emit(kind="model", ds=ds, model=nm, error=conditionMessage(ci))
  } else {
    sup <- as.character(ci$CI[1, "supCL"])
    emit(kind="model", ds=ds, model=nm,
         point=ci$results[1, "abundance"], prof=ci$CI[1, "abundance"],
         lo=num(ci$CI[1, "infCL"]), hi=num(sup), hi_open=as.integer(grepl(">", sup)),
         bic=ci$results[1, "BIC"], aic=ci$results[1, "AIC"], dev=ci$results[1, "deviance"],
         df=ci$results[1, "df"], info_fit=ci$results[1, "infoFit"], info_ci=ci$CI[1, "infoCI"],
         warn=c(warns, ci$CI.warn), ncurve=ci$N.CI, lcurve=ci$loglik.CI)
  }
}
"""

# An independent reference for the profile: stats::glm.fit on the full table with
# the unseen cell set to N - n, the exact multinomial profile (lgamma, no Stirling),
# optimize and uniroot at 1e-10. Not Rcapture; it isolates Rcapture's numerics.
R_EXACT = r"""
design_for <- function(pairs) {
  H8 <- rbind(c(0,0,0), H)
  cols <- list(rep(1, 8), H8[,1], H8[,2], H8[,3])
  for (p in pairs) cols[[length(cols)+1]] <- H8[,p[1]] * H8[,p[2]]
  do.call(cbind, cols)
}
exact_one <- function(ds, nm, pairs, cnt) {
  x <- design_for(pairs); n <- sum(cnt)
  cond <- tryCatch(suppressWarnings(glm.fit(x[-1, , drop=FALSE], cnt, family=poisson(),
                   control=glm.control(epsilon=1e-12, maxit=200))), error=function(e) NULL)
  if (is.null(cond)) { emit(kind="exact", ds=ds, model=nm, status="glm_error"); return(invisible()) }
  nhat <- n + exp(cond$coefficients[1])
  lN <- function(N) {
    y <- c(N - n, cnt)
    f <- suppressWarnings(glm.fit(x, y, family=poisson(), control=glm.control(epsilon=1e-12, maxit=200)))
    mu <- f$fitted.values
    lgamma(N + 1) - lgamma(N - n + 1) + sum(ifelse(y > 0, y * log(mu / N), 0))
  }
  cap <- FMAX * n
  if (!is.finite(nhat) || nhat > cap) { emit(kind="exact", ds=ds, model=nm, status="unbounded", point=nhat); return(invisible()) }
  op <- optimize(lN, c(n, min(cap, 2 * nhat + 10)), maximum=TRUE, tol=1e-10)
  nmax <- op$maximum; lmax <- op$objective; cut <- lmax - qchisq(0.95, 1) / 2
  f <- function(N) lN(N) - cut
  lo <- if (f(n) >= 0) n else uniroot(f, c(n, nmax), tol=1e-10)$root
  up <- tryCatch(uniroot(f, c(nmax, cap), tol=1e-10)$root, error=function(e) NA)
  emit(kind="exact", ds=ds, model=nm, status=if (is.na(up)) "open" else "ok", point=nhat,
       prof=nmax, lo=lo, hi=up)
}
"""

R_MS_MT = r"""
ms <- tryCatch(closedpMS.t(X, dfreq=TRUE, maxorder=2), error=function(e) NULL)
if (!is.null(ms)) {
  r <- ms$results
  for (i in seq_len(nrow(r)))
    emit(kind="ms", ds=ds, model=rownames(r)[i], point=r[i, "abundance"],
         bic=r[i, "BIC"], info_fit=r[i, "infoFit"])
}
cp <- tryCatch(suppressWarnings(closedp.t(X, dfreq=TRUE)), error=function(e) NULL)
if (!is.null(cp) && "Mt" %in% rownames(cp$results))
  emit(kind="closedp.t", ds=ds, model="Mt", point=cp$results["Mt", "abundance"],
       bic=cp$results["Mt", "BIC"])
"""

# dga: Bayesian model averaging over decomposable graphical models with a
# hyper-Dirichlet prior. The table is a 2x2x2 array with the unseen cell NA.
R_DGA = r"""
suppressMessages(library(dga))
data(graphs3)
dga_one <- function(ds, cnt) {
  # array() fills the first index fastest, so the vector in history order
  # c(x000, x100, x010, ...) lands witness 1 on index 1, as in ours.
  Y <- array(c(0, cnt), dim=c(2,2,2))
  n <- sum(cnt)
  Nmissing <- 1:(FMAX * n)
  res <- tryCatch(bma.cr(Y, Nmissing, delta=1/8, graphs=graphs3, logprior=rep(0, length(Nmissing))),
                  error=function(e) e)
  if (inherits(res, "error")) { emit(kind="dga", ds=ds, error=conditionMessage(res)); return(invisible()) }
  post <- colSums(res); post <- post / sum(post)
  N <- n + Nmissing
  cdf <- cumsum(post)
  q <- function(p) N[which(cdf >= p)[1]]
  emit(kind="dga", ds=ds, mean=sum(N * post), median=q(0.5), lo=q(0.025), hi=q(0.975),
       mass_at_cap=post[length(post)], model_post=paste(sprintf("%.4f", rowSums(res) / sum(res)), collapse=";"))
}
"""


def r_script(datasets: list[dict], with_dga: bool) -> str:
    parts = [R_PRELUDE, R_EXACT]
    mp = ", ".join(
        f'"{rcapture_name(m)}"=list(' + ", ".join(f"c({a + 1},{b + 1})" for a, b in m) + ")"
        for m in ll.model_space(3)
    )
    parts.append(f"MODELPAIRS <- list({mp})\nFMAX <- {FMAX}\n")
    parts.append(
        'emit(kind="versions", R=R.version.string, Rcapture=as.character(packageVersion("Rcapture")), '
        'dga=if (requireNamespace("dga", quietly=TRUE)) as.character(packageVersion("dga")) else "absent")\n'
    )
    for d in datasets:
        cnt = ", ".join(str(c) for c in d["counts"][1:])
        parts.append(f'ds <- "{d["name"]}"\nX <- cbind(H, c({cnt}))\n')
        parts.append(R_MODEL)
        parts.append(R_MS_MT)
        parts.append("for (nm in names(MODELPAIRS)) exact_one(ds, nm, MODELPAIRS[[nm]], X[, 4])\n")
        parts.append("for (nm in names(MODELPAIRS)) exact_one(ds, nm, MODELPAIRS[[nm]], X[, 4])\n")
    if with_dga:
        parts.append(R_DGA)
        for d in datasets:
            cnt = ", ".join(str(c) for c in d["counts"][1:])
            parts.append(f'dga_one("{d["name"]}", c({cnt}))\n')
    return "\n".join(parts)


def run_r(datasets: list[dict], with_dga: bool = False) -> list[dict]:
    script = r_script(datasets, with_dga)
    proc = subprocess.run(
        ["docker", "run", "-i", "--rm", IMAGE, "R", "--slave", "--no-save"],
        input=script,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"R failed:\n{proc.stderr[-2000:]}")
    rows = []
    for line in proc.stdout.splitlines():
        if not line.startswith("@@"):
            continue
        rec = {}
        for cell in line.split("\t")[1:]:
            if "=" in cell:
                k, v = cell.split("=", 1)
                rec[k] = v
        rows.append(rec)
    return rows


# --- our side ---------------------------------------------------------------


def _bisect(f, a: float, b: float, tol: float = 1e-9) -> float:
    """Root of a monotone-sign-change f on [a, b], f(a) < 0 <= f(b) or the reverse."""
    fa = f(a)
    for _ in range(200):
        m = 0.5 * (a + b)
        fm = f(m)
        if (fm < 0) == (fa < 0):
            a, fa = m, fm
        else:
            b = m
        if b - a < tol:
            break
    return 0.5 * (a + b)


def continuous_profile(counts: np.ndarray, pairs: ll.Pairs) -> dict:
    """The same profile likelihood as `Profile`, maximised and cut at real-valued N.

    `Profile.maximise` and `interval` search integers because the shipped
    estimator reports integer counts. Rcapture searches reals with uniroot, so
    the like-for-like comparison is the real-valued root of the same function.
    """
    n = observed_total(counts)
    cap = ll.CAP_RATIO * n
    cond = ll.fit_conditional(counts, pairs)
    if not cond.converged or not math.isfinite(cond.z0) or n + cond.z0 > cap:
        return {"status": "unbounded", "prof": None, "lo": None, "hi": None}
    prof = ll.Profile(counts, pairs)
    try:
        m, _ = prof.maximise(n + cond.z0)
        # golden-section polish of the integer argmax to a real argmax
        a, b = max(n, m - 1.0), min(cap, m + 1.0)
        phi = (math.sqrt(5) - 1) / 2
        c, d = b - phi * (b - a), a + phi * (b - a)
        for _ in range(80):
            if prof(c) > prof(d):
                b, d = d, c
                c = b - phi * (b - a)
            else:
                a, c = c, d
                d = a + phi * (b - a)
        nhat = 0.5 * (a + b)
        top = prof(nhat)
        cut = top - ll.CHI2_1_95 / 2
        lo = float(n) if prof(n) >= cut else _bisect(lambda x: prof(x) - cut, n, nhat)
        step = max(1.0, nhat - n)
        hi_b = nhat + step
        while prof(hi_b) >= cut:
            if hi_b > cap:
                return {"status": "open", "prof": nhat, "lo": lo, "hi": None}
            hi_b = nhat + 2 * (hi_b - nhat)
        hi = _bisect(lambda x: prof(x) - cut, nhat, hi_b)
    except (ll.ConvergenceError, FloatingPointError):
        return {"status": "nonconverged", "prof": None, "lo": None, "hi": None}
    return {"status": "ok", "prof": nhat, "lo": lo, "hi": hi}


def ours(counts: list[int]) -> dict:
    c = np.asarray(counts, dtype=np.int64)
    n = observed_total(c)
    out = {}
    for pairs in ll.model_space(3):
        fit = ll.fit_conditional(c, pairs)
        cont = continuous_profile(c, pairs)
        est = ll._single(c, pairs, "x")
        out[rcapture_name(pairs)] = {
            "n": n,
            "z0": fit.z0,
            "point": n + fit.z0 if fit.converged else None,
            "bic": fit.bic,
            "converged": fit.converged,
            "profile": cont,
            "shipped": {
                "point": None if not math.isfinite(est.point) else est.point,
                "lo": est.lo,
                "hi": None if not math.isfinite(est.hi) else est.hi,
            },
        }
    return out


# --- comparison -------------------------------------------------------------


def fnum(s: str | None) -> float | None:
    if s is None or s == "" or s == "NA":
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    return v if math.isfinite(v) else None


def diff(a: float | None, b: float | None) -> dict | None:
    if a is None or b is None or not (math.isfinite(a) and math.isfinite(b)):
        return None
    return {"ours": a, "r": b, "abs": abs(a - b), "rel": abs(a - b) / max(abs(b), 1e-12)}


def ct_exact(big_n: float, n: int) -> float:
    n0 = big_n - n
    return (
        (n0 * math.log(n0) if n0 > 0 else 0.0)
        + math.lgamma(big_n + 1)
        - big_n * math.log(big_n)
        - math.lgamma(n0 + 1)
    )


def ct_rcapture(big_n: float, n: int) -> float:
    """The correction term in Rcapture's closedpCI.internal (loglikemult), as written there.

    With N above 100 it replaces the log-factorials by Stirling's series
    without the 1/(12 x) terms, in n0 = N - n as well as in N. That is exact
    in the limit and wrong for small n0; for non-integer 0 < n0 < 1 (where
    `optimize` evaluates) it is not close.
    """
    n0 = big_n - n
    if big_n > 100:
        if n0 == 0 or n0 == 1:
            return -big_n + 0.5 * math.log(2 * math.pi * big_n)
        return n0 - big_n - 0.5 * math.log(n0 / big_n)
    return ct_exact(big_n, n)


def stirling_residual(counts: list[int], pairs: ll.Pairs, ncurve, lcurve) -> dict | None:
    """Does Rcapture's own (N, loglik) curve equal our profile plus its Stirling term?

    Rcapture's value is  -deviance/2 + ct.  With the exact ct that equals our
    profile minus sum(y log y) over the observed cells, a constant. So
    residual_exact(N) = R(N) - ours(N) + sum(y log y) is 0 iff R used the exact
    term, and residual_stirling(N) = residual_exact(N) - (ct_R - ct_exact) is 0
    iff the only difference is the Stirling approximation.
    """
    if not ncurve or not lcurve or len(ncurve) != len(lcurve):
        return None
    c = np.asarray(counts, dtype=np.int64)
    n = observed_total(c)
    prof = ll.Profile(c, pairs)
    const = float(sum(v * math.log(v) for v in c[1:] if v > 0))
    exact, stir = [], []
    for big_n, r_ll in zip(ncurve, lcurve, strict=True):
        if big_n <= n:
            continue
        try:
            res = r_ll - prof(float(big_n)) + const
        except (ll.ConvergenceError, FloatingPointError):
            continue
        exact.append(abs(res))
        stir.append(abs(res - (ct_rcapture(big_n, n) - ct_exact(big_n, n))))
    if not stir:
        return None
    return {
        "points": len(stir),
        "max_abs_vs_exact_ct": max(exact),
        "max_abs_vs_rcapture_stirling_ct": max(stir),
    }


def compare(datasets: list[dict], r_rows: list[dict]) -> dict:
    def index(kind):
        return {(r["ds"], r["model"]): r for r in r_rows if r.get("kind") == kind}

    by_model, by_ms, by_exact = index("model"), index("ms"), index("exact")
    by_cp = {r["ds"]: r for r in r_rows if r.get("kind") == "closedp.t"}
    pairs_of = {rcapture_name(p): p for p in ll.model_space(3)}
    out = []
    for d in datasets:
        mine = ours(d["counts"])

        def key(m, d=d):
            return (d["name"], m)

        r_bics = {
            m: fnum(by_model[key(m)].get("bic"))
            for m in mine
            if key(m) in by_model and fnum(by_model[key(m)].get("bic")) is not None
        }
        r_bic_min = min(r_bics.values(), default=None)
        my_bic_min = min(v["bic"] for v in mine.values() if math.isfinite(v["bic"]))
        models = {}
        for m, o in mine.items():
            r, ex, ms = by_model.get(key(m), {}), by_exact.get(key(m), {}), by_ms.get(key(m), {})
            if not r or "error" in r:
                models[m] = {"class": "r_error", "r_error": r.get("error", "missing"), "ours": o}
                continue
            prof = o["profile"]
            r_point, r_prof = fnum(r.get("point")), fnum(r.get("prof"))
            r_lo, r_hi = fnum(r.get("lo")), fnum(r.get("hi"))
            hi_open = r.get("hi_open") == "1"
            info_fit, info_ci = int(float(r["info_fit"])), int(float(r["info_ci"]))
            ex_ok = ex.get("status") == "ok"
            regular = prof["status"] == "ok" and ex_ok and info_fit == 0 and o["point"] is not None
            curve = stirling_residual(
                d["counts"],
                pairs_of[m],
                [float(v) for v in r["ncurve"].split(";")] if r.get("ncurve") else [],
                [float(v) for v in r["lcurve"].split(";")] if r.get("lcurve") else [],
            )
            models[m] = {
                "class": "regular" if regular else "boundary",
                "ours": {
                    "point": o["point"],
                    "profile_status": prof["status"],
                    "prof": prof["prof"],
                    "lo": prof["lo"],
                    "hi": prof["hi"],
                    "shipped": o["shipped"],
                },
                "rcapture": {
                    "point": r_point,
                    "prof": r_prof,
                    "lo": r_lo,
                    "hi": r_hi,
                    "hi_open": hi_open,
                    "info_fit": info_fit,
                    "info_ci": info_ci,
                    "warn": r.get("warn", ""),
                },
                "r_exact_profile": {
                    "status": ex.get("status"),
                    "point": fnum(ex.get("point")),
                    "prof": fnum(ex.get("prof")),
                    "lo": fnum(ex.get("lo")),
                    "hi": fnum(ex.get("hi")),
                },
                "stirling_check": curve,
                # point, dbic: ours vs Rcapture. x_*: ours vs the exact R reference
                # (glm.fit + 1e-10 root finding). rc_*: ours vs Rcapture's closedpCI.t.
                "diff": {
                    "point": diff(o["point"], r_point),
                    "point_vs_closedpMS": diff(o["point"], fnum(ms.get("point"))) if ms else None,
                    "dbic": diff(
                        o["bic"] - my_bic_min,
                        None if r_bic_min is None else r_bics.get(m, math.nan) - r_bic_min,
                    ),
                    "x_point": diff(o["point"], fnum(ex.get("point"))),
                    "x_prof": diff(prof["prof"], fnum(ex.get("prof"))),
                    "x_lo": diff(prof["lo"], fnum(ex.get("lo"))),
                    "x_hi": diff(prof["hi"], fnum(ex.get("hi"))),
                    "rc_prof": diff(prof["prof"], r_prof),
                    "rc_lo": diff(prof["lo"], r_lo),
                    "rc_hi": diff(prof["hi"], r_hi),
                    "shipped_lo": diff(o["shipped"]["lo"], fnum(ex.get("lo"))),
                    "shipped_hi": diff(o["shipped"]["hi"], fnum(ex.get("hi"))),
                },
            }
        mine_pick = min(mine, key=lambda m: mine[m]["bic"])
        r_pick = min(r_bics, key=r_bics.get) if r_bics else None
        cp = by_cp.get(d["name"])
        out.append(
            {
                "name": d["name"],
                "source": d["source"],
                "counts": d["counts"],
                "n_observed": observed_total(np.asarray(d["counts"])),
                "bic_pick": {"ours": mine_pick, "rcapture": r_pick, "agree": mine_pick == r_pick},
                "closedp_t_Mt": None
                if cp is None
                else diff(mine["[1,2,3]"]["point"], fnum(cp.get("point"))),
                "models": models,
            }
        )
    return {"datasets": out}


GATED = ("point", "point_vs_closedpMS", "dbic", "x_point", "x_prof", "x_lo", "x_hi")
INFORMATIONAL = ("rc_prof", "rc_lo", "rc_hi", "shipped_lo", "shipped_hi")


def summarise(result: dict) -> dict:
    """Max |diff| per model and quantity over regular fits; boundary fits listed apart."""
    models = [rcapture_name(m) for m in ll.model_space(3)]
    table = {}
    for m in models:
        row = {}
        for f in GATED + INFORMATIONAL:
            ds = [
                (x["models"][m]["diff"][f], x["name"])
                for x in result["datasets"]
                if m in x["models"]
                and x["models"][m].get("class") == "regular"
                and x["models"][m]["diff"][f] is not None
            ]
            row[f] = (
                None
                if not ds
                else {
                    "max_abs": max(d["abs"] for d, _ in ds),
                    "max_rel": max(d["rel"] for d, _ in ds),
                    "argmax_abs": max(ds, key=lambda t: t[0]["abs"])[1],
                    "n": len(ds),
                }
            )
        table[m] = row

    def worst(fields, key):
        vals = [row[f][key] for row in table.values() for f in fields if row[f]]
        return max(vals, default=None)

    boundary = [
        {
            "dataset": x["name"],
            "model": m,
            "ours": v["ours"],
            "rcapture": v.get("rcapture"),
            "r_exact_profile": v.get("r_exact_profile"),
        }
        for x in result["datasets"]
        for m, v in x["models"].items()
        if v.get("class") != "regular"
    ]
    stir = [
        v["stirling_check"]
        for x in result["datasets"]
        for v in x["models"].values()
        if v.get("stirling_check") and v.get("class") == "regular"
    ]
    # closedp.t only where independence is a regular fit: a diverging N-hat
    # (zero-overlap table) has no meaningful relative difference.
    cp = [
        x["closedp_t_Mt"]
        for x in result["datasets"]
        if x["closedp_t_Mt"] and x["models"]["[1,2,3]"].get("class") == "regular"
    ]
    return {
        "tolerance_rel": TOL_REL,
        "max_diff_by_model": table,
        "gated": {
            "max_rel_point_vs_rcapture": worst(("point", "point_vs_closedpMS"), "max_rel"),
            "max_rel_point_vs_r_glm": worst(("x_point",), "max_rel"),
            "max_rel_profile_vs_r_glm": worst(("x_prof", "x_lo", "x_hi"), "max_rel"),
            "max_abs_delta_bic": worst(("dbic",), "max_abs"),
            "closedp_t_Mt_max_rel": max((d["rel"] for d in cp), default=None),
        },
        "rcapture_profile_ci": {
            "note": "closedpCI.t endpoints vs our exact profile; see stirling_check",
            "max_abs_prof": worst(("rc_prof",), "max_abs"),
            "max_abs_lo": worst(("rc_lo",), "max_abs"),
            "max_abs_hi": worst(("rc_hi",), "max_abs"),
            "max_rel_hi": worst(("rc_hi",), "max_rel"),
            "stirling_max_abs_residual_vs_exact_ct": max(
                (s["max_abs_vs_exact_ct"] for s in stir), default=None
            ),
            "stirling_max_abs_residual_vs_rcapture_ct": max(
                (s["max_abs_vs_rcapture_stirling_ct"] for s in stir), default=None
            ),
        },
        "shipped_integer_interval_max_abs": worst(("shipped_lo", "shipped_hi"), "max_abs"),
        "bic_pick_agreement": sum(x["bic_pick"]["agree"] for x in result["datasets"]),
        "datasets": len(result["datasets"]),
        "regular_fits": sum(
            v.get("class") == "regular" for x in result["datasets"] for v in x["models"].values()
        ),
        "boundary_fits": boundary,
    }


def datasets_all() -> list[dict]:
    return sim_datasets() + confirm_datasets() + boundary_datasets()


def dga_vs_average(datasets: list[dict], dga_rows: list[dict]) -> list[dict]:
    """dga's posterior on N next to `ll_average` and `ll_bic`. Informational, not gated.

    The two do not estimate the same object: dga averages decomposable
    graphical models (saturated 3-way model included, weighted by a
    hyper-Dirichlet prior with delta = 1/8), `ll_average` mixes the 8
    no-3-way profile likelihoods with BIC weights. Both put a flat prior on N.
    """
    counts = {d["name"]: np.asarray(d["counts"], dtype=np.int64) for d in datasets}
    out = []
    for r in dga_rows:
        if "error" in r:
            out.append({"dataset": r["ds"], "error": r["error"]})
            continue
        t = counts[r["ds"]]
        avg, bic = loglinear.averaged(t), loglinear.best_bic(t)

        def fin(x):
            return x if math.isfinite(x) else None

        out.append(
            {
                "dataset": r["ds"],
                "dga": {k: fnum(r[k]) for k in ("mean", "median", "lo", "hi", "mass_at_cap")},
                "ll_average": {"point": fin(avg.point), "lo": avg.lo, "hi": fin(avg.hi)},
                "ll_bic": {"point": fin(bic.point), "lo": bic.lo, "hi": fin(bic.hi)},
            }
        )
    return out


def run(datasets: list[dict], with_dga: bool = False) -> dict:
    ensure_image()
    rows = run_r(datasets, with_dga)
    ver = next(r for r in rows if r.get("kind") == "versions")
    result = compare(datasets, rows)
    result["versions"] = {k: v for k, v in ver.items() if k != "kind"}
    result["dga"] = dga_vs_average(datasets, [r for r in rows if r.get("kind") == "dga"])
    result["image"] = {"tag": IMAGE, "base": R_BASE, "dockerfile": DOCKERFILE}
    result["summary"] = summarise(result)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--no-dga", action="store_true")
    args = ap.parse_args()
    if not have_docker():
        raise SystemExit("docker is not available")
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; refusing to overwrite an artifact")
    t0 = time.perf_counter()
    result = run(datasets_all(), with_dga=not args.no_dga)
    result["runtime_s"] = round(time.perf_counter() - t0, 1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    s = result["summary"]
    print(json.dumps({k: v for k, v in s.items() if k != "boundary_fits"}, indent=2)[:6000])
    print(f"{len(s['boundary_fits'])} boundary fits; wrote {args.out}")


if __name__ == "__main__":
    main()
