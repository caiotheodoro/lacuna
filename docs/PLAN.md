# lacuna: estimating what an agent did that its trace doesn't show

> The original plan, written before any code (Oct 2026). Kept for the record;
> several parts were never built (OpenRouter, the τ²-bench env, `PREREG.md`, a pip
> library). What was built and decided is in `README.md` and `docs/DECISIONS.md`.

## Context
Goal: an unstudied measurement problem in agent oversight, scoped against the literature as of Oct 2026.

The gap: agents can tamper with their own traces (Qin et al., arXiv 2609.30266, Sep 2026). Every audit regime (EU AI Act Art. 12, ISO 42001, SB 53) assumes logs are *complete*, but current work only checks *integrity*. LLM auditors relay agents' claims instead of checking them (2609.07680). Incident registries can't give rates because they have no denominator (2609.11030).

The idea: treat "how many consequential actions did this agent take?" as a **population-size problem**. Several imperfect, possibly correlated witnesses each catch part of the actions. Multiple-systems estimation, the method HRDAG uses for uncounted casualties, gives N̂ with an interval. For adaptive adversaries, report bounds instead.

Constraints: ~3 months part-time, <$500. Public/synthetic data only. Audience: safety/eval orgs (AISI/CAISI, METR, Apollo, lab eval teams).

## Locked design
| Axis | Decision |
|---|---|
| Action unit | Tool calls that change state or send data out, weighted by cost. Reads out of scope for v1 |
| Threat ladder | L0 benign loss (sampling, async drops, logger bugs) → L1 agent prompted to hide → L2 adaptive agent (told which witnesses exist, tries to correlate its omissions) |
| Witnesses | W1 agent's own trace (agent-controlled). W2 MCP/HTTP interception proxy, bypassable through a code-exec side channel in the ops env. W3 env state diff + network sink (blind to transient write-then-revert). W4 LLM auditor rebuilding actions from the transcript |
| Ground truth | Env kernel ledger the agent can't see. A program, not labels |
| Linkage | Oracle (ground-truth IDs, the upper bound) vs effect-signature (hash of resource, op and payload digest within a time window). Never trust IDs the agent supplies |
| Estimators | L0/L1: log-linear MSE with pairwise interactions, model-averaged, plus a Bayesian latent-class model for robustness. L2: partial-identification bounds, assuming W3 is independent of the agent's policy |
| Baselines (floor) | `trust_trace` (\|W1\|), `union_count` (\|W1∪W2∪W3\|), `auditor_count` (W4) |
| Subjects | Open-weight Qwen3/GLM/DeepSeek via OpenRouter, plus a check on one frontier model. Prompted model organisms only, no fine-tuning in v1 |
| Envs | τ²-bench (DB state) + one custom back-office env (files, egress, payments, code-exec side channel) |
| Release | arXiv preprint + HF dataset (capture histories + ground truth) + `lacuna` pip library. Next SaTML/safety-workshop deadline after week 12 |

## Pre-registered hypotheses (`PREREG.md`, frozen before any agent run)
- **H1**: at L0/L1, 95% MSE intervals cover the true N in ≥90% of runs, and RMSE beats `union_count`.
- **H2**: at L2 the MSE point estimate undercounts, but the bounds stay valid as long as W3 is in the witness set.
- **H3**: W4 is positively correlated with W1. Treating W4 as independent gives tighter intervals that miss the true N.

## Repo layout (as planned; uv + pyproject)
```
src/lacuna/
  sim/          capture-history simulator: known N, dependence graph, heterogeneity by action type
  estimators/   baselines.py, mse_loglinear.py (statsmodels Poisson GLM), latent_class.py (numpyro), bounds.py
  linkage/      oracle.py, effect_signature.py
  witnesses/    w1_selftrace.py, w2_proxy.py, w3_envdiff.py (+ netsink), w4_auditor.py
  envs/         tau2/ (vendored adapter), ops_env/
  organisms/    L0 loss injectors, L1/L2 prompt organisms
  intervals.py  (vendored)
PREREG.md  MEASUREMENT_CARD.md  tests/test_published_claims.py
```

## Reuse (patterns from the author's earlier eval projects, vendored with provenance)
- interval code: paired/cluster bootstrap, design effect
- a τ² adapter + Docker sandbox → `envs/tau2`, ops_env sandbox
- seeded generator, deterministic oracle, signature-disjoint splits, leak probe: planted-omission generator
- MEASUREMENT_CARD + `test_published_claims.py` + `make validate` claim gates
- HF publish scripts + gates → dataset release

## Milestones
0. **Week 0**
   - Prior-art sweep. Capture-recapture is already used for estimating software-inspection defects (Eick et al. 1992 onward) and vulnerability counts, so cite it and make clear the novelty is agent audit + adversarial witnesses + bounds.
   - Confirm nobody has applied MSE to agent or trace completeness.
1. **Weeks 1–2, simulator only ($0).** Synthetic capture histories over a grid of dependence/heterogeneity settings. Check estimator coverage and cross-check against R `Rcapture`/`dga`/`LCMCR`.
2. **Weeks 3–6.** Env + witness plumbing, ground-truth ledger, L0/L1 runs, both linkage modes. Freeze PREREG before the first agent run.
   - **Gate (week 4):** if L0 coverage fails in sim + L0 → pivot to benchmark-only (the floor).
3. **Weeks 7–9.** L2 adaptive organisms, bounds, W4 dependence study (H3), frontier-model check.
4. **Weeks 10–12.** MEASUREMENT_CARD, paper, HF dataset, `lacuna` release.

## Budget (<$500)
OpenRouter open-weight runs ~$250 · frontier check capped at $100 · GPU $0 (no fine-tune) · buffer $150. Track spend in a `receipt.json`.

## Verification
- **Estimator correctness:** Python estimates match R `Rcapture`/`dga` to 1e-3 on shared histories (pytest). Coverage on ≥1,000 sim replicates per config is reported with a Monte Carlo SE.
- **Ground truth:** the env kernel ledger is invisible to the agent. A leak probe asserts no witness reads it.
- **Claims:** every number in README/paper is gated by `tests/test_published_claims.py` against MEASUREMENT_CARD. Unchecked numbers are marked `NOT_VERIFIED`.
- **Independent review:** a fresh subagent red-teams PREREG (given the spec only) before the first agent run, and a second one checks the paper's claims against the card before release.
- **Floor check:** each estimator must beat `union_count` on RMSE and `trust_trace` on coverage. If not, report it as a negative result.
