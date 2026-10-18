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
episodes passed all four hypotheses, the coverage one (H1) at 0.94 on one loss
seed. At 3000 replicates it sits at the bar, and the shortfall is record
linkage, not the estimator (`docs/DECISIONS.md` D44):

| `ll_bic` coverage at `fin_mid` | exact matching | signature matching |
|---|---|---|
| registered, 21 episodes | 0.948 | 0.893 |
| three lost episodes restored, 24 | 0.945 | 0.862 |

Signature matching merges actions that share an effect signature (mostly two
emails to one customer), which fakes trace–gateway overlap; BIC then fits that
interaction in 0.358 of replicates, against 0.01 under exact matching. With the
lost episodes back, H1 fails by the rule fixed in advance (D35, D36).

What holds on every seed and every pool:
- counting the union of all three logs never covers the truth (coverage 0.0);
- `ll_bic` asserts the logs are incomplete in every replicate;
- when the gateway and the trace fail together, the independence model
  collapses to 0.0 while `ll_bic` holds.

Before thinning, the trace and the gateway each held every action, so all
hidden mass is simulated loss plus an env view cut by configuration.

**Transfer** (`docs/TRANSFER.md`, separately pre-registered). On an infra/ops
environment whose env diff misses transient work, T1 fails: `ll_bic` covers 0.3
at the registered seed, 0.401 over 20 seeds with signature matching, and 0.946
with exact matching. The paid run reproduces a free scripted run, so this tests
the environment and the linkage, not the agent.

**The limits.**
- When log loss concentrates on the actions the env diff cannot see, the pooled
  estimator fails silently from 1.1x extra loss (covers 0.657, then 0.197 at
  1.2x), under either linkage. Stratifying on the operation stays calibrated,
  refusing more often as the loss grows.
- When the env diff is blind to actions chosen by a key no record carries, both
  fail.
- Signature matching gives false gaps whenever logs lose mutually exclusive
  actions and the env view is partial, in every pool and drop rate tested.

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
