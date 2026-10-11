# Select independent scientific interpreters without changing the user's environment.
PYTHON ?= python
ADAPTER_PYTHON ?= $(PYTHON)
PIPELINE_PYTHON ?= $(ADAPTER_PYTHON)
BUILDER_PYTHON ?= $(PYTHON)
LINT_PYTHON ?= $(PYTHON)

.PHONY: check lint model-tests test test-adapter test-builder packaging-tests help-smoke install-smoke package-smoke deployment-tests
check: lint test packaging-tests deployment-tests help-smoke install-smoke
lint:
	"$(LINT_PYTHON)" -m ruff check handlers ccqn utils recnet_backend recnet_adapter network_builder run_dp_ts.py slabsite.py scripts
model-tests test-adapter:
	"$(PIPELINE_PYTHON)" -m pytest recnet_backend/tests recnet_adapter/tests -q
test: test-adapter test-builder
test-builder:
	"$(BUILDER_PYTHON)" -m pytest network_builder/tests -q
packaging-tests:
	"$(PYTHON)" -m pytest scripts/tests -q
help-smoke:
	"$(PIPELINE_PYTHON)" scripts/check_help.py adapter
	"$(BUILDER_PYTHON)" scripts/check_help.py builder
install-smoke:
	"$(PYTHON)" scripts/check_core_install.py --pipeline-python "$(PIPELINE_PYTHON)" --builder-python "$(BUILDER_PYTHON)"
package-smoke: install-smoke

deployment-tests:
	"$(PYTHON)" -m pytest recnet_adapter/sai/tests -q
