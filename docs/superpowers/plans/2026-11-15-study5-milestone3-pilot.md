# Study 5, milestone 3: task mix, recorded sessions, pilot

**Goal:** everything the registration needs and nothing it forbids: a task mix whose
end state hides a material share of actions, sessions recorded once from real models,
a pilot on seeds disjoint from the confirmatory ones, and the analysis code the
registration will name. It ends with a pilot report and a decision entry. It does not
write the registration.

**Spec:** `docs/DESIGN-STUDY5.md`, sections "Changed by the rig" and "What the
registration will contain". AGENTS.md rules 2 to 6 apply.

**Not in this milestone:** `docs/PREREG-STUDY5.md`, the confirmatory pools and runs,
the CLI, the paper.

## Decisions taken while planning

1. **The task mix is a new queue, not a new kernel.** `lacuna.envs.churn` builds
   threads of tickets over the back-office world and tools: an address corrected, a
   trial ended, a draft sent and deleted, a handoff note rewritten. The kernel and its
   diff are untouched. The env is registered as `backoffice_churn`.
2. **A session has one clock.** Tickets are handled in queue order, each by the plane
   it was dealt to, so a follow-up finds what its opener left. A session file's gaps
   are differences on that one clock, and the replay client sleeps to a schedule
   counted from a start time both originators are given, not after each reply.
3. **Originators do not start in step.** Each waits a seeded time before it creates its
   SDK, so the two export cycles have their own phase and the phase differs by run.
   The two relays stay shared by both planes, which the approved plan names as the
   primary case.
4. **The rig's key carries the value for every op.** The end state holds a file's
   content and a sent email's text, and the telemetry holds the arguments, so both can
   compute the digest. Rewrites of one file then stop colliding.
5. **Recording is in-process, ticket by ticket.** Each plane is one conversation with
   its model; a ticket is one user turn. Calls go through `KernelService`, the same
   code the rig's kernel runs. The file keeps requests, not outcomes.
6. **Pools.** `ds`: both planes `deepseek-v4.1-flash`. `qw`: both `qwen3.8-flash`.
   `mix`: orchestrator deepseek, sub-agent qwen. The pilot pool is recorded first on
   its own seeds; the confirmatory pools are recorded with the same frozen recorder
   before the registration, and their content is not analysed before the freeze.
7. **The sequence-gap baseline is built before the pilot** and scored in it, so the
   registration's predictions about it are made with its pilot numbers in view.

## Files

| file | responsibility |
|---|---|
| `src/lacuna/envs/churn.py` | the queue of threads; `make_episode` |
| `src/lacuna/rig/sessions.py` | `timed`: calls in session order to per-plane files on one clock |
| `src/lacuna/rig/replay.py` | schedule-based sleeps; `--t0`, `--warmup` |
| `src/lacuna/rig/record.py` | one session recorded from two conversations over one kernel |
| `src/lacuna/rig/stress.py` | the pilot's seeded schedules |
| `scripts/study5_record.py` | a pool: sessions file, transcripts, receipt, provenance |
| `scripts/rig_run.py` | `--env`, `--sessions-file`, the shared start time |
| `scripts/study5_lib.py` | a run's files to tables, estimates, baselines and scores |
| `scripts/study5_analysis.py` | runs to an artifact |
| `scripts/study5_verdicts.py` | the registered criteria as code (gates, P1, P2) |

## Tasks

1. Churn queue, with tests for order, determinism and the hidden share.
2. One clock per session; replay to a schedule; originator warm-up.
3. The rig's key; `--env` through the kernel service, the fronts and the driver.
4. Recorder and its tests against a scripted stand-in for the model; record the pilot
   pool; receipt and provenance.
5. `study5_lib`: per-plane tables from a run, the candidate estimators, `union_count`,
   the JSON-RPC sequence-gap baseline with and without a tail term, scores per run.
6. Pilot schedules; pilot runs on scripted sessions first (free), then on the pilot
   pool; the retry window as the rig shows it.
7. Pilot report (`docs/RIG-FINDINGS.md`), artifact, decision entry, then a review and a
   refutation pass in parallel.

## What the pilot must answer

- Does the churn mix clear the "not exercised" gate, and under which schedule?
- How much does one schedule's loss vary between runs?
- Which estimator and interval hold up under block loss shared by both planes?
- How do `union_count` and the sequence-gap baseline score on the same runs?
- How many runs does P1 need at the coverage the pilot shows?
