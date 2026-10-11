# Core-only engineering entrypoints (no adapter/builder assumptions).
# Override with absolute interpreter paths, e.g.:
#   make check PYTHON=/opt/envs/dpa4/bin/python PIPELINE_PYTHON=/opt/envs/dpa4/bin/python
PYTHON ?= python
PIPELINE_PYTHON ?= $(PYTHON)
LINT_PYTHON ?= $(PYTHON)

.PHONY: check lint test-gate install-smoke

check: lint test-gate install-smoke

lint:
	"$(LINT_PYTHON)" -m ruff check handlers ccqn utils run_dp_ts.py slabsite.py scripts

test-gate:
	"$(PYTHON)" -m pytest scripts/tests -q

install-smoke:
	"$(PYTHON)" scripts/check_core_install.py --pipeline-python "$(PIPELINE_PYTHON)"
