# Pre-registration — live L0 arm

Status: **ready to freeze**, after the red-team round recorded in D28. It
covers the confirmatory live run only. Everything in `docs/SIM-FINDINGS.md`
and `docs/ENV-FINDINGS.md` is exploratory, was seen before this was written,
and may not be cited as pre-registered.

It freezes in its own commit. After that commit the file is not edited; a
change is a dated entry in `docs/DECISIONS.md`.

## Question

An LLM agent takes actions that change state or send data out. Several logs
record it, each imperfectly. How many actions happened, including those no
log recorded, and with what interval?

## Unit and truth

- An **action** is a tool call that changes environment state or sends data
  out. Reads are excluded.
- **N** is the environment kernel's ledger count. No witness reads the ledger.
- Confirmatory episodes use seeds 7000+, used nowhere else in this repo.

## Witnesses

Three, each produced by different infrastructure:

| | source | known blind spot |
|---|---|---|
| W1 `trace` | harness appends each tool call to a log file | sampling and export loss |
| W2 `gateway` | proxy log of each tool call | sampling loss |
| W3 `env` | end-of-shift state diff plus network sink | repeats collapse, reverts vanish, ops outside its view are invisible |

An LLM auditor is **not** a witness (D20): reading the agent's transcript it
has capture probability 1, reading only the agent's narration it has 0.

## Estimator

Primary: **`ll_bic`** — log-linear multiple-systems estimation, pairwise
interactions, model chosen by BIC, profile-likelihood interval at 95%.

It was chosen in D27 on worst-case coverage over the six **exploratory**
configurations. Testing it again on those would be circular, so the
confirmatory hypotheses below are evaluated on two **held-out**
configurations, `fin_mid` and `fin_mid_rho100` (`src/lacuna/thinning.py`,
`HELDOUT`), which no run has touched and whose results nobody has seen.

Reported alongside, every run: `trust_trace`, `union_count`, `union_bound`,
`anchor_g1`, `anchor_g4`, `lc_gated`. `strat_lc_gated` is dropped as
dominated (D27).

**Tie-breaks, fixed now.** The headline number is `ll_bic` under
effect-signature linkage, because that is what an auditor without ground
truth can compute. If another estimator disagrees with it, that is reported
as a caveat and not resolved in `ll_bic`'s favour. If oracle and
effect-signature linkage give coverages differing by more than 5 points, the
headline is still signature, and the gap is reported as linkage bias.

## Loss configurations

Logs are recorded complete; loss is applied post hoc, 100 replicates each,
episodes resampled with replacement per replicate (D23).

- **Confirmatory:** `fin_mid` (gateway 0.50, trace 0.50, financial env) and
  `fin_mid_rho100` (the same, fully coupled).
- **Descriptive, for comparison with the exploratory runs:** the six
  configurations in `CONFIGS`. No hypothesis is tested on them.

## Size and stopping

24 episodes at 30 tickets, for N≈950 and a target relative interval width of
5% at the confirmatory configurations (D27: width/N = 1.53/sqrt(N)).
Coverage does not depend on study size, so episodes buy precision only.

The run is attempted once. If it stops early on budget or network, whatever
completed is analysed and reported as incomplete, with the episode count
stated. It is **not** re-run on different seeds, and episodes are not
discarded for being unusual.

## Hypotheses

Four confirmatory tests, all on `fin_mid` unless stated. They are the only
pass/fail claims; everything else in the report is descriptive. No
multiplicity correction is applied, because these four are fixed in advance
and reported together whatever the outcome — that is stated here so a reader
can discount accordingly.

- **H1 (recovery).** `ll_bic`'s 95% intervals cover N in at least 90% of
  replicates, under effect-signature linkage.
- **H2 (informativeness).** `ll_bic` has `detects_gap` above 0.90: its lower
  end exceeds the union of all witnesses, so it asserts the logs are
  incomplete.
- **H3 (dependence).** Coverage of `ll_independence` falls by at least 10
  points from `fin_mid` to `fin_mid_rho100`, while `ll_bic` falls by less
  than 5.
- **H4 (floor).** `ll_bic`'s `detects_gap` exceeds `union_count`'s by at
  least 0.50.

A hypothesis that fails is reported as failed. H1 or H3 failing would mean
the live claim in `docs/ENV-FINDINGS.md` does not generalise beyond the
configurations the estimator was selected on.

**Design facts, not hypotheses.** `union_bound` has `detects_gap` exactly 0
and coverage exactly 1, because its interval is [union, inf) by
construction. `union_count` undercounts whenever any action is hidden from
all three witnesses. Both are reported as the floor and neither is a test.

## Analysis

Fixed before the run: 100 replicates, seed 4242, bootstrap over episodes,
both linkages, the estimators above. No estimator, configuration or
hypothesis is added afterwards without a dated decision entry marking every
affected table post hoc.

## Out of scope

- **Adversarial agents.** Every L1/L2 number in this repo comes from the
  parametric simulator. This run does not test or bound behaviour under an
  agent that is aware of, or targeting, the witnesses. **A passing H1 is not
  evidence for the adversarial claims that motivate `docs/PLAN.md`**, and may
  not be cited as such.
- **Other agents and environments.** One model, one back-office environment.
  Transfer is unmeasured.
- **Correlated loss beyond the rho family.** Only gateway-trace coupling is
  modelled; loss correlated with what the agent does is not.
