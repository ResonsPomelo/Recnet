"""The packaging gate must reject missing core files and fork payloads."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

GATE = Path(__file__).resolve().parents[1] / "check_core_install.py"
SPEC = importlib.util.spec_from_file_location("check_core_install", GATE)
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


@pytest.mark.parametrize("missing", ["handlers/context.py", "recnet_backend/calculator.py",
                                      "recnet_adapter/model_config.json", "network_builder/build.py"])
def test_wheel_members_reject_missing_core_file(missing):
    names = gate.expected_members() | {"recnet-0.0.1.dist-info/METADATA"}
    gate.validate_wheel_members(names)
    names.remove(missing)
    with pytest.raises(RuntimeError, match="Missing wheel members"):
        gate.validate_wheel_members(names)


@pytest.mark.parametrize("payload", ["tools/prepare.py",
                                       "recnet_adapter/tools/campaign.py",
                                       "recnet_adapter/runtime_config.py",
                                       "network_builder/tests/leak.py", "graph_generator.py",
                                       "evil.dist-info/METADATA"])
def test_wheel_members_reject_noncore_payload(payload):
    names = gate.expected_members() | {"recnet-0.0.1.dist-info/METADATA", payload}
    with pytest.raises(RuntimeError, match="Unexpected wheel members"):
        gate.validate_wheel_members(names)
