import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location("install_gate", Path(__file__).parents[1] / "check_core_install.py")
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)

@pytest.mark.parametrize("payload", ["tests/test_leak.py", "graph_generator.py", "prepare_rmg_data.py", "submit.sh", "recnet_adapter.py", "recnet_adapter/tests/leak.py", "recnet_adapter/tools/leak.py", "recnet_adapter/sai/run.sh", "evil.dist-info/METADATA"])
def test_rejects_unexpected_wheel_payload(payload):
    with pytest.raises(RuntimeError, match="Unexpected wheel members"):
        gate.validate_members(gate.expected_members() | {payload})

@pytest.mark.parametrize("member", sorted(gate.expected_members()))
def test_rejects_each_missing_source_member(member):
    with pytest.raises(RuntimeError, match="Missing wheel members"):
        gate.validate_members(gate.expected_members() - {member})

def test_accepts_current_runtime_tree_and_metadata():
    gate.validate_members(gate.expected_members() | {"recnet-0.0.1.dist-info/METADATA"})
    assert "recnet_adapter/runtime_config.py" in gate.expected_members()
    assert not any("/tests/" in path for path in gate.expected_members())

def test_new_runtime_source_requires_wheel_member(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "ROOT", tmp_path)
    (tmp_path / "handlers").mkdir()
    (tmp_path / "handlers" / "new_runtime.py").write_text("pass")
    assert "handlers/new_runtime.py" in gate.expected_members()
    with pytest.raises(RuntimeError, match="new_runtime"):
        gate.validate_members(set(gate.PY_MODULES) | {"recnet_adapter/model_config.json", "recnet_adapter/tools/finalize_network.sh"})


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
