# Core and explicit model-input engineering entrypoints.
# Override with absolute interpreter paths, e.g.:
#   make check PYTHON=/opt/envs/dpa4/bin/python PIPELINE_PYTHON=/opt/envs/dpa4/bin/python
PYTHON ?= python
PIPELINE_PYTHON ?= $(PYTHON)
LINT_PYTHON ?= $(PYTHON)

.PHONY: check lint model-tests test-gate install-smoke

check: lint model-tests test-gate install-smoke

lint:
	"$(LINT_PYTHON)" -m ruff check handlers ccqn utils recnet_backend recnet_adapter run_dp_ts.py slabsite.py scripts

model-tests:
	"$(PIPELINE_PYTHON)" -m pytest recnet_backend/tests/test_boundary.py recnet_adapter/tests/test_model_selection.py -q

test-gate:
	"$(PYTHON)" -m pytest scripts/tests -q

install-smoke:
	"$(PYTHON)" scripts/check_core_install.py --pipeline-python "$(PIPELINE_PYTHON)"
