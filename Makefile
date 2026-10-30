# Targets are the table of contents. validate needs no key, no GPU, no network.
# PYTHONPATH: uv's editable .pth is flagged hidden on macOS and Python 3.13 skips it.
export PYTHONPATH := $(CURDIR)/src
.PHONY: help install lint test sim sim-smoke privacy-gate validate repro-headline card paper paper-check dataset

help: ## List targets.
	@grep -E '^[a-z0-9-]+:.*?## .*$$' $(MAKEFILE_LIST) | sed 's/:.*## /\t/'

install: ## uv sync with the dev group.
	uv sync --group dev

lint: ## ruff check + format check.
	uv run ruff check src tests scripts
	uv run ruff format --check src tests scripts

test: ## pytest.
	uv run pytest -q

sim: ## Full simulator coverage sweep (~6 min). Writes artifacts/sim_coverage.json.
	uv run python scripts/sim_coverage.py --reps 500 --seed 2026

sim-smoke: ## 40-rep sweep to a scratch file, for iterating.
	uv run python scripts/sim_coverage.py --reps 40 --out artifacts/smoke.json

privacy-gate: ## Fail if a key-shaped string is in a tracked or untracked non-ignored file.
	@files="$$(git ls-files -co --exclude-standard)"; \
	if [ -z "$$files" ]; then echo "privacy-gate: no files"; exit 0; fi; \
	if echo "$$files" | xargs grep -nE '(sk-[A-Za-z0-9]{20,}|oc_sk_[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{30,}|gh[pousr]_[A-Za-z0-9]{30,}|Bearer [A-Za-z0-9._-]{20,}|AKIA[0-9A-Z]{16})' -- 2>/dev/null; then \
	  echo "privacy-gate: secret-looking string found" >&2; exit 1; fi; \
	echo "privacy-gate: clean"

card: ## Rewrite MEASUREMENT_CARD.json from the artifacts on disk.
	uv run python scripts/make_card.py

paper: ## Regenerate the paper's tables and figures from artifacts and build both PDFs (tectonic).
	uv run python scripts/paper_tables.py
	uv run --group paper python scripts/paper_figures.py
	cd paper && tectonic -X compile lacuna.tex && tectonic -X compile lacuna-workshop.tex

paper-check: ## Fail on any unsourced number, pending section or missing-artifact table.
	PAPER_FINAL=1 uv run pytest -q tests/test_paper.py

dataset: ## Build dataset/ and dry-run the HF publish, which runs the tests and validator itself. Uploads nothing.
	uv run --group dataset python scripts/build_dataset.py
	uv run --group dataset python scripts/publish_dataset.py --dry-run

repro-headline: ## Rerun the registered held-out replay and H1-H4 into a temp dir; fail if confirm.json or hypotheses.json differ. No key, no network.
	uv run python scripts/repro_headline.py

validate: ## Exit condition.
	$(MAKE) lint
	$(MAKE) test
	$(MAKE) privacy-gate
