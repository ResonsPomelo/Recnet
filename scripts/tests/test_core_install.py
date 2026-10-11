"""The packaging gate must reject missing core files and fork payloads."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

GATE = Path(__file__).resolve().parents[1] / "check_core_install.py"
SPEC = importlib.util.spec_from_file_location("check_core_install", GATE)
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def test_wheel_members_reject_missing_core_file():
    names = gate.expected_members() | {"recnet-0.0.1.dist-info/METADATA"}
    gate.validate_wheel_members(names)
    names.remove("handlers/context.py")
    with pytest.raises(RuntimeError, match="Missing wheel members"):
        gate.validate_wheel_members(names)


@pytest.mark.parametrize("payload", ["recnet_adapter/__init__.py",
                                       "recnet_backend/__init__.py",
                                       "network_builder/__init__.py",
                                       "tools/prepare.py"])
def test_wheel_members_reject_noncore_payload(payload):
    names = gate.expected_members() | {"recnet-0.0.1.dist-info/METADATA", payload}
    with pytest.raises(RuntimeError, match="Unexpected top-level wheel entries"):
        gate.validate_wheel_members(names)
