# lacuna

How many consequential actions did the agent take that no log line shows?

Trace integrity is a solved-ish problem: hash chains, append-only servers,
TEEs. Trace *completeness* is not. Agents delete and spoof their own trace
events when it pays (Qin et al. 2026, arXiv 2609.30266). Every audit regime
that relies on logs (EU AI Act Art. 12, ISO/IEC 42001, SB 53 incident reports)
assumes the log is whole, and nothing measures how far from whole it is.

lacuna treats the question as a population-size problem. Several witnesses each see
part of what the agent did:

| Witness | What it sees | Who controls it |
|---|---|---|
| W1 `trace` | the agent's own log | the agent |
| W2 `proxy` | MCP/HTTP calls through an interception proxy | infra; bypassable through code execution |
| W3 `env` | state diffs + a network sink | the environment; misses transient write-then-revert |
| ~~W4 `auditor`~~ | an LLM that rebuilds actions from the transcript | **dropped: not independent (D20)** |

Capture-recapture (multiple-systems estimation, the method used to count
casualties nobody recorded) turns the overlaps between witnesses into an
estimate of the actions no witness saw, with an interval. Against an adaptive
agent that correlates its omissions across witnesses, a point estimate
is not identified, and lacuna reports bounds instead.

The floor is `union_bound`: "at least as many as any witness saw, maybe
infinitely more". It always covers. A method earns its place only by giving a
finite upper end without losing coverage.

## Status

The claim now has live evidence. In an environment where 49 of 589 agent
actions (8.4%) reached no witness, counting the union of all three logs
undercounts by exactly that and never covers the truth; capture-recapture
over the same three logs recovers N with 95.0% coverage and an interval about
6% of N wide (`docs/ENV-FINDINGS.md`).

- `src/lacuna/sim.py` — capture-history simulator across the threat ladder
- `src/lacuna/estimators/` — floor baselines; log-linear MSE with profile-likelihood intervals; latent-class model and its identifiability gate; env-anchored Γ bounds
- `src/lacuna/envs/` — back-office kernel with a hidden ground-truth ledger, tools, gateway log, harness trace. Benign loss only (D10, D11)
- `src/lacuna/thinning.py` — post-hoc logging loss over recorded episodes
- `scripts/` — `sim_coverage.py`, `run_episodes.py`, `thin_episodes.py`, `reaudit.py`, plus the table renderers the docs quote

**The live arm has three witnesses, not four (`docs/DECISIONS.md` D20).** An LLM auditor reading the agent's own transcript has capture probability 1 when it sees the call log and 0 when it does not; there is no setting that makes it independent. Every `@4` number in this repo is superseded and supports no claim.

**Open:**
- No adversarial rung has been run live (D10): every L1/L2 number is simulator-only.
- `strat_lc_gated` is dominated by the plain fit and needs redesign (D17).
- Pre-registration has not frozen; everything so far is exploratory.

## Run

```bash
make install
make validate   # lint + tests + privacy gate; no key, no network
make sim        # 500-rep coverage sweep, ~6 min on a laptop
```
