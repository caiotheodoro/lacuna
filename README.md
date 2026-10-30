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
| W2 `gateway` (`proxy` in result labels) | MCP/HTTP calls through an interception proxy | infra; bypassable through code execution |
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

## In short

- **Question:** how many consequential agent actions are in no log?
- **Method:** capture-recapture over three logs (the agent's trace, a gateway log, an
  environment diff). `ll_bic` turns their overlaps into an interval for the true count.
- **Pre-registered, back-office:** passed at its registered loss seed (0.94) but covers
  0.893 over 3000 replicates, below the 0.90 bar.
- **Pre-registered, infra:** failed (0.3 at the registered seed, 0.401 over 20 seeds).
- **Why:** both used effect-signature record matching; matching by the ledger id, which
  no auditor has, is nominal in both.
- **Main limitation:** the agents were benign and all log loss is simulated afterwards on
  complete logs. Nothing here tests an agent that hides on purpose.

Terms used below:
- `ll_bic`: Poisson log-linear model of the three-log capture table, pairwise
  interactions chosen by BIC, 95% profile-likelihood interval.
- `fin_mid`: back-office loss setting where the trace and the gateway each drop half
  their lines at random and the env diff sees only financial operations; `transfer_mid` is
  the same loss in the infra environment.
- signature linkage: records match across logs on operation, resource and sometimes
  value, in order. Call-id linkage: trace and gateway match on the provider's tool-call
  id that the harness writes into both. Ledger-id linkage: match on the environment's
  ledger id, which no auditor has.

## Status

**Two pre-registered tests: back-office passed at its seed and misses in expectation, infra failed** (`docs/DECISIONS.md` D44–D54).

- **Back-office** (`docs/PRE-REGISTRATION.md`): `ll_bic` passed H1 at its registered
  loss seed (0.94) but covers 0.893 over 3000 replicates, below the 0.90 bar.
- **Infra** (`docs/PREREG-TRANSFER.md`, write-up in `docs/TRANSFER.md`): T1 failed,
  0.3 at the registered seed and 0.401 over 20 seeds. The paid run reproduces a
  free scripted one, so this tests the environment and the linkage, not the agent.

Both were nominal when records are matched by ledger id, which no auditor has:

| `ll_bic` coverage | signature (registered) | shared call id (post hoc) | ledger id |
|---|---|---|---|
| back-office `fin_mid` | 0.875 | 0.946 | 0.946 |
| infra `transfer_mid` | 0.406 | 0.859 | 0.944 |

These come from a different loss-seed set than the 0.893 above. The signature dropped
content digests the logs carry and merged distinct actions. Linking trace and gateway
by the provider's tool-call id fixes back-office, where the env diff is easy to
attach; infra stays short because the diff carries no id, and a last-writer rule
that looked like a fix produces false gaps (D52). The call id is written by the
harness; an agent writing its own trace could forge it (untested).

What holds on every seed and every pool:
- counting the union of all three logs never covers the truth (coverage 0.0);
- `ll_bic` asserts the logs are incomplete at `fin_mid` in every replicate;
- when the gateway and the trace fail together, the independence model
  collapses to 0.0 while `ll_bic` holds.

Before thinning, the trace and the gateway each held every action, so all
hidden mass is simulated loss.

**The limits.**
- **Action-dependent loss:** when log loss concentrates on the actions the env diff
  cannot see, the pooled estimator fails silently from 1.1x extra loss (0.657,
  then 0.197 at 1.2x, with signature matching; exact matching gives 0.647 and 0.187). Stratifying on the operation stays
  calibrated where it answers, refusing more often as the loss grows.
- **Bursts:** loss in blocks of consecutive actions makes the plain interval too
  narrow (0.559 at blocks of 12 in `burst-loss.json`; the range over every seed set that
  ran the cell is under Run). Most of that is
  not dispersion: the agent works one kind of ticket at a time, so a lost block lands
  where the env diff is blind and `ll_bic` is biased (+29.6 and +58.8 on the
  long-episode pools). Permuting which actions the diff sees restores 0.875–0.94
  (D54). `ll_bic_hom` (post hoc) keeps `ll_bic`'s model, never covers less, and
  trades width for coverage: 0.952 / 0.994 under bursts in
  `interval-comparison.json`, 0.894 on long-60 and 0.846 on long-120. Report it beside the plain interval.
- **Outages and shared session quality:** whole-session outages of the gateway and
  trace leave plain `ll_bic` nominal (0.936 / 0.943) only while the env diff stays up.
  Session quality shared by all three logs defeats the pooled intervals;
  `tier_bic` recovers 0.838 / 0.907 there but is not a default (0.292 under
  action-dependent loss, 0.595 under bursts).
- **False gaps:** signature matching produces them when logs lose mutually
  exclusive actions and the env diff misses some; call-id linking removes them.

## Layout

- `src/lacuna/sim.py` — capture-history simulator across the threat ladder
- `src/lacuna/estimators/` — floor baselines; log-linear MSE with profile-likelihood intervals; latent-class model and its identifiability gate; env-anchored Γ bounds
- `src/lacuna/envs/` — back-office kernel with a hidden ground-truth ledger, tools, gateway log, harness trace. Benign loss only (D10, D11)
- `src/lacuna/thinning.py` — post-hoc logging loss over recorded episodes
- `scripts/` — `sim_coverage.py`, `run_episodes.py`, `thin_episodes.py`, `reaudit.py`, plus the table renderers the docs quote

**The live arm has three witnesses, not four (`docs/DECISIONS.md` D20).** An LLM auditor reading the agent's own transcript has capture probability 1 when it sees the call log and 0 when it does not; there is no setting that makes it independent. Every `@4` number in this repo is superseded and supports no claim.

**Open:**
- No adversarial rung has been run live (D10): every L1/L2 number is simulator-only.
- Canary actions and sequence-number gaps, the cheaper checks, were not compared head to head.
- `strat_lc_gated` is superseded by `strat_bic` (D17, D38); `strat_bic`'s deferred faults are listed in D40.

## Run

Needs `uv` and `Python >= 3.11`; `make paper` also needs `tectonic`.

```bash
make install
make validate   # lint + tests + privacy gate; no key, no network
make sim        # 500-rep coverage sweep, ~6 min on a laptop
make paper      # tables, figures and both PDFs (needs tectonic)
```

The registered confirmatory analysis needs `--bootstrap-episodes`; without it
`ll_bic` at `fin_mid` covers 0.90, not 0.94
(`artifacts/thinning/confirm-heldout-fixed-episodes.json`). It rewrites
`artifacts/thinning/confirm.json` with the same numbers unless given `--out`:

```bash
uv run python scripts/thin_episodes.py --run confirm --pool heldout --bootstrap-episodes
uv run python scripts/test_hypotheses.py --run confirm
```

Several cells were run on more than one set of loss seeds, so the same cell reads
differently in different tables. `ll_bic` at blocks of 12, back-office, financial view,
exact matching: 0.559 in `burst-loss.json` (the value quoted above), and 0.535 to 0.588
across `burst-loss`, `overdispersion`, `episode-dispersion`, `interval-comparison` and
`final-controls`; the report's appendix lists each. `ll_bic_hom` on the same cell:
0.952 in `interval-comparison.json`, 0.966 in `final-controls.json`.

## Data and license

- The capture histories, ledgers and results are also a Hugging Face dataset,
  [`caiotheodoro/lacuna-capture-histories`](https://huggingface.co/datasets/caiotheodoro/lacuna-capture-histories),
  built by `scripts/build_dataset.py`. It withholds the episode transcripts.
- Code and data are `Apache-2.0` (`LICENSE`, `NOTICE`). The `transcript` fields in
  `artifacts/episodes/*/episodes.jsonl` hold the reasoning text of the DeepSeek
  models the live runs used (served through OpenCode Go), which that license does
  not cover; the provider's terms apply.
- Paper and technical report: `paper/lacuna-workshop.tex`, `paper/lacuna.tex` (`make paper`;
  PDFs are attached to the GitHub release). The report's "Reproducing this" section
  names the script behind each result. `scripts/thin_episodes.py` overwrites
  `artifacts/thinning/` unless given `--out`.
- Terms such as `ll_bic`, `fin_mid` and signature or ledger-id linkage are defined
  in the dataset card's glossary.
