# lacuna: operating rules

The count no witness saw has a price. `union_bound` is the floor. `README.md` is the argument. `docs/PLAN.md` is the plan. This file is how work is done.

## Rules

**1. The protocol is binding.** `docs/PRE-REGISTRATION.md` is a draft until the commit that freezes it, which comes before the first agent run. After that commit the file is not edited. A change is a dated `docs/DECISIONS.md` entry.

**2. No transcribed numbers.** Numbers come from an artifact (`artifacts/*.json`) with the seed and rep count that produced them. A doc that quotes a number names its artifact. Once `MEASUREMENT_CARD.json` exists, `tests/test_published_claims.py` gates the README against it.

**3. The truth is a program.** In simulation, `full[0]` is set to zero in exactly one place (`coverage.run_once`) before any estimator sees the table. In agent runs, the ground-truth ledger is the environment kernel's, and no witness reads it.

**4. Never trust an agent-supplied ID.** Linkage across witnesses uses effect signatures or the oracle, never IDs the agent wrote. An L2 agent can forge them.

**5. The floor is always reported.** Every table includes `trust_trace`, `union_count`, `union_bound`, and `auditor_count` when W4 exists. A method that loses to the floor is reported as losing.

**6. Keys stay in the environment.** `.env` is gitignored. `make privacy-gate` scans for key-shaped strings.

## What not to do

- Do not put a customer trace, tenant name, or any data from work in the tree. Public and synthetic data only.
- Do not fit the K-way interaction. It is not identifiable; the reason is in the `estimators/loglinear.py` docstring.
- Do not add a scenario, estimator, or witness after the freeze without a decision entry that marks every affected table as post hoc.
- Do not report effort as elapsed time.
