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

**Confirmatory result, pre-registered and frozen before the run**
(`docs/PRE-REGISTRATION.md`, on loss configurations held out of every
exploratory run and of the estimator choice). The registered analysis of 21
episodes passed all four hypotheses. Its coverage hypothesis passed only
marginally:

| | coverage at four loss seeds | mean |
|---|---|---|
| registered, 21 episodes | 0.94, 0.92, 0.91, 0.89 | 0.915 |
| three lost episodes restored, 24 | 0.86, 0.88, 0.91, 0.86 | 0.877 |

With the lost episodes back, H1 (coverage at least 0.90) fails. They were
lost to a harness fault that depends on the agent's trajectory
(`docs/DECISIONS.md` D33 to D36), so the attrition was not ignorable.

What holds on every seed and every pool:
- counting the union of all three logs never covers the truth (coverage 0.0);
- `ll_bic` asserts the logs are incomplete in every replicate;
- when the gateway and the trace fail together, the independence model
  collapses to 0.0 while `ll_bic` holds.

**The limit.** When log loss concentrates on the actions no witness sees, the
method still detects incompleteness but cannot measure it.
- Stratifying on the operation restores validity when the loss depends on
  something the auditor can observe. It refuses rather than answering wrongly
  (D38).
- When the loss depends on something no record carries, both estimators are
  confidently wrong: `ll_bic` covers 0.6 and the stratified one 0.51.

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
