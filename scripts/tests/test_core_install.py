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


def test_source_stage_excludes_prior_build_and_keeps_current_source(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "pyproject.toml").write_text("current metadata", encoding="utf-8")
    (root / "recnet_adapter").mkdir()
    (root / "recnet_adapter" / "current.py").write_text("current source", encoding="utf-8")
    (root / "recnet_adapter" / "tools").mkdir()
    (root / "recnet_adapter" / "tools" / "campaign.py").write_text("fork payload", encoding="utf-8")
    (root / "tools").mkdir()
    (root / "tools" / "prepare.py").write_text("fork payload", encoding="utf-8")
    (root / "graph_generator.py").write_text("legacy payload", encoding="utf-8")
    (root / "build" / "lib").mkdir(parents=True)
    (root / "build" / "lib" / "stale.py").write_text("stale build", encoding="utf-8")
    (root / "recnet.egg-info").mkdir()
    (root / "recnet.egg-info" / "SOURCES.txt").write_text("stale manifest", encoding="utf-8")
    monkeypatch.setattr(gate, "ROOT", root)

    source = gate.stage_source(tmp_path / "scratch")

    assert (source / "pyproject.toml").read_text(encoding="utf-8") == "current metadata"
    assert (source / "recnet_adapter" / "current.py").read_text(encoding="utf-8") == "current source"
    assert (source / "recnet_adapter" / "tools" / "campaign.py").is_file()
    assert (source / "tools" / "prepare.py").is_file()
    assert (source / "graph_generator.py").is_file()
    assert not (source / "build").exists()
    assert not (source / "recnet.egg-info").exists()
