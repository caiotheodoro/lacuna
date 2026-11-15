# Study 5: the pilot on recorded sessions, and a prediction made before its second half

Written 2026-11-15, after the runs on the first recorded pool were scored and before
any run on the second and third was. It is committed before those are scored so that
what it predicts can fail.

## What the first pool showed

`artifacts/study5/pilot-recorded-ds.json`: pool `pilot-ds` (36 sessions recorded from
`deepseek-v4-1-flash-1017`, both planes), 8 runs under the pilot's `node-restart` and
1 unstressed, 24 sessions drawn a run.

- Unstressed: every estimate is the count.
- `ll_sa_hom` covers 6 of the 8. On scripted sessions it covered 23 of 24.
- The five runs whose outage was 32 s or longer are covered (errors of -7 to +3 on 326
  to 334 actions). Of the three whose outage was 18 to 19 s, one is covered and two are
  missed: `ptr-ds-restart-6` (333, interval 326 to 342, truth 325) and
  `ptr-ds-restart-7` (326, interval 326 to 327, truth 330).

## Why

In the two missed runs the actions both telemetry logs lost were few, 11 and 4, and
all were made within about four seconds of the kill. In `restart-7` the kill came 10 s
into the run, when every session under way had just begun: all four lost actions were
openers (two trials started, two drafts written), which later tickets undo, so the end
state shows none of them and the estimate is the union. In `restart-6` the lost actions
were four closing summaries, two handoff notes, two corrections and a refund with its
email: the last word on each record, so the end state shows all eleven and the
estimate adds eight that did not exist.

So whether the end state hides an action depends on where in its session the action
falls, and a short outage takes a slice of each session at one point. The estimator
assumes hiding is independent of loss. The registration draft lists that as untested;
it is now seen to fail when the outage is short against a session's length (a recorded
session is about 23 to 33 calls a second apart).

Two other intervals were tried on these 8 runs and on the scripted ones and are not
adopted: a negative-binomial interval with between-session dispersion (covers 7 of 8;
it cannot cover eleven of eleven visible), and an estimate that uses the end state only
for operations it cannot lose (emails, refunds, cancels), which covers 8 of 8 here and
45 of the 48 scripted stressed runs against 46 for `ll_sa_hom`, with wider intervals
and its own failures in the other direction. Neither removes the dependence; each
moves it.

## Prediction for the runs not yet scored

Pools `pilot-glm` (GLM 5.3 on both planes) and `pilot-mix` (DeepSeek on plane A, GLM
5.3 on plane B), six `node-restart` runs each, already made or being made, not scored.

1. Every run whose span relay was down 28 s or longer is covered by `ll_sa_hom`.
2. Any miss is in a run whose outage was under 25 s.

## What follows

- If both hold: the primary condition becomes the long restart (28 to 45 s), where an
  outage is as long as a session and takes every part of it, and the short restart
  (15 to 24 s) becomes a registered condition of its own with a predicted shortfall in
  coverage. That is a change to what the owner accepted in D80 and D81 (a restart of
  15 to 45 s) and is put to the owner as such.
- If a long-outage run is missed: no split. The condition stays, and P1 is predicted
  to be missed.

## Afterword (2026-11-15, after the runs were scored and the proposal refuted)

The text above is as committed at `59fe444` and is not edited.

- **What happened.** None of the 12 runs on `pilot-glm` and `pilot-mix` was missed, at
  any length. Both parts of the prediction held.
- **What it was worth.** Little. It could fail only on the long runs of two pools
  whose 24 sessions each are replayed in every run, and it tested a line that had
  been drawn around the first pool's two misses.
- **The split under "What follows" was proposed (D84) and withdrawn (D85).** A
  refutation showed the line was drawn around the first pool's misses. "Why" above is
  also too sure: over the runs together the end state shows the lost actions about as
  often as the rest, and what it shows varies by kind of action as much as by place
  in the session (`exploratory` in `artifacts/study5/pilot-recorded-summary.json`;
  D86). The registration keeps one restart of 15 to 45 s and predicts that P1 is not
  met.
- **Figures above with no artifact.** "covers 7 of 8", "covers 8 of 8" and "45 of the
  48 ... against 46" came from throwaway scripts that are not in the tree. They should
  not have been written here and are not repeated anywhere else. The description of
  what the lost actions were (openers; closing summaries, notes, corrections, a
  refund) is a reading of the ledger of two runs and is in no artifact either; the
  counts that are (4 lost and none shown; 11 lost and all shown) are in each run's
  `lost_by_both`.
