# Design: study 6 (sessions no two runs share, and the conditions study 5 left untested)

Status: **design, not a registration.** Written 2026-11-23 on the owner's instruction to
take the study as far as it goes (D115). Nothing here is frozen, nothing has been run, no
paid call has been made, and every choice marked **owner** is open. The registration would
be `docs/PREREG-STUDY6.md`, frozen at a signed tag that is pushed before the first paid
call and the first run (D74), with the owner's own predictions in it.

## Why

Study 5 met its criterion on the runs left and left five things without an answer. Each is
a limit the paper states, and each needs new runs.

1. **Runs share sessions.** A run replays 24 of its pool's 72 sessions, so no two runs of
   a pool are independent. The dependence was estimated after the fact and too loosely to
   settle: a design effect of 2.92 for misses with an upper limit of 7.51, where the bound
   bears 4.05 (`artifacts/study5/posthoc3.json`).
2. **Runs set aside.** 23 of the 300 planned `s1` runs failed integrity, bunched in time.
   The machine's power log puts the long delays beside its sleeps and most of the short
   ones in a stretch on battery (same artifact). That is a description after the fact.
3. **The relays fail together.** `s1` kills both by design, so the result says nothing
   about collectors that fail independently of each other.
4. **The assumption is not tested within a run.** The estimate takes what both telemetry
   logs lose to be as visible in the end state as the rest. No condition makes that false
   on purpose.
5. **One rig, one layout, no adversary, and simulated loss in studies 1 to 4.**

## What stays

The rig, the kernel and its ledger, the three witnesses, the task mix, the two models and
the provider of study 5; `ll_sa_hom` as the primary estimator, unchanged; the floors and
`seq_gap_tail`; the bar of 0.90 on the Wilson lower end; the integrity rules; the readings
"pass", "not shown" and "missed"; one analysis, run once, after the folders are sealed.

## What changes

**A. Sessions.** Every run replays sessions that no other run replays. With 24 sessions a
run, as in study 5, a condition of 300 runs needs 7,200 sessions. A pool of 72 cost 1.38
to 1.53 in tokens at the price study 5 assumed (`spent_usd` in
`artifacts/study5/pools/*/pool.json`), about 0.02 a session, so 7,200 sessions is about
150 at that price and the provider's real price is still not known (D89). **Owner:** 24
sessions a run at that cost, or fewer sessions a run, which is cheaper and changes what a
run is.

**B. The machine.** Runs are made on external power with sleep disabled, on a machine
doing nothing else, and the driver records the power source, the load and the wall clock
at the start and end of each run. A run made on battery, or across a sleep, is set aside
by a rule written before the runs. This turns D112's reading into a control. **Owner:**
this laptop under those rules, or a rented machine.

**C. Conditions.** One primary and three beside it, each with its own criterion or none.

| condition | what it is | role |
|---|---|---|
| `r1` | study 5's `s1` again: both relays killed together, 15 to 45 s | primary; the same P1 and P2 |
| `r2` | each relay on its own outage, drawn independently, no originator killed | does the result need the relays to fail together? |
| `r3` | the outage is timed to the part of a session the end state hides most | the assumption, made false on purpose; predicted to fail |
| `r0` | no stress | control |

`r3` is the one that can break the estimate, and the registration says so in advance. Its
timing uses what study 5's pilot saw: the end state shows less of a session's early
actions than of its late ones (`docs/PREREG-STUDY5.md`, "Not tested, and caveats fixed
now").

**D. A second layout.** A second compose file: each plane with its own pair of relays, or
a third plane. The registered estimator sums planes, so this is where a difference between
planes would show. **Owner:** in this study, or the next.

**E. An adversary.** Study 4's was one episode, black box, the channel not named. Here: an
originator told which witnesses exist and how the count is made, over several sessions.
It needs live agents and not replay, so it is a separate arm with its own sessions.
**Owner:** in or out.

**F. Loss from a real pipeline on the earlier episodes.** Replay the back-office episodes
of studies 1 to 4 through the rig under `r1`, so that at least one of their cells is
scored under loss that no code simulated.

**G. A registered head-to-head.** `chapman_te` covered 250 of the 277 runs of study 5 on a
narrower interval, with no criterion. Here it gets one, so that what `ll_sa_hom` adds over
a plain Chapman interval is a test and not a description.

## What the analysis can then say

With A, runs are independent by construction and the Wilson bound means what it says. With
B, a run set aside has a recorded reason on the machine's side. `r2` and `r3` are read by
their own criteria; `r3` failing as predicted is a result about the fifth condition, and
passing would be one too.

## What the owner decides before a freeze

1. The spend in A, and the size of a run.
2. The machine in B.
3. Whether D and E are in this study.
4. The predictions, each in the owner's words: for P1 on `r1`, for `r2`, for `r3`.
5. Where the registration is deposited outside GitHub. D74 asked for that and study 5 did
   not do it (D90).

## Order of work once those are decided

Tooling for A and B with tests; a pilot on scripted sessions, which costs nothing; the
registration drafted, reviewed and refuted as study 5's was; the freeze; the paid
sessions; the runs; the seal; the analysis, once.

## Not here

No estimator is changed and none is added to rescue a result. No band, pool or outage
length becomes a condition after the fact.
