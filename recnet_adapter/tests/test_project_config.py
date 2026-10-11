"""Explicit user project input and per-model selection contracts."""
import json
import os
from pathlib import Path

import pytest

from recnet_adapter.runtime_config import resolve_runtime
from recnet_adapter.tools.campaign import build_campaign
import synth_case


@pytest.fixture(autouse=True)
def clear_runtime_environment(monkeypatch):
    for name in list(os.environ):
        if name.startswith("RECNET_") or name == "DPEVA_PYTHON":
            monkeypatch.delenv(name)


def test_no_project_defaults_and_explicit_profile(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name in ("RECNET_PROJECT_CONFIG", "RECNET_DP_HEAD", "RECNET_MODEL_S", "RECNET_MODEL_M",
                 "RECNET_WORKSPACE_ROOT", "RECNET_PYTHON"):
        monkeypatch.delenv(name, raising=False)
    assert resolve_runtime().root == tmp_path
    assert resolve_runtime().models == {}
    profile = tmp_path / "user/project.json"
    profile.parent.mkdir()
    profile.write_text(json.dumps({
        "root": "../project-workspace", "python": "/project/venv/bin/python",
        "head": "shared", "models": {
            "alpha": {"path": "models/a.pt", "head": "A"},
            "beta": {"path": "models/b.pt"},
        },
    }))
    cfg = resolve_runtime(project_config=profile)
    assert cfg.root == tmp_path / "project-workspace"
    assert cfg.models == {"alpha": cfg.root / "models/a.pt", "beta": cfg.root / "models/b.pt"}
    assert cfg.heads == {"alpha": "A", "beta": "shared"}
    assert cfg.python == "/project/venv/bin/python"
    monkeypatch.setenv("RECNET_PROJECT_CONFIG", str(profile))
    assert resolve_runtime().models == cfg.models


def test_cli_overrides_profile_and_supports_arbitrary_labels(tmp_path, monkeypatch):
    profile = tmp_path / "project.json"
    profile.write_text(json.dumps({"root": str(tmp_path), "head": "config", "models": {
        "S": {"path": "s.pt", "head": "s-old"}, "X": {"path": "x.pt", "head": "x-old"}}}))
    monkeypatch.setenv("RECNET_DP_HEAD", "env")
    monkeypatch.setenv("RECNET_MODEL_HEAD_X", "env-x")
    cfg = resolve_runtime(project_config=profile, models=["X=override.pt", "Y=new.pt"],
                          model_heads=["S=Default", "Y=none"], head="cli")
    assert cfg.models == {"S": tmp_path / "s.pt", "X": tmp_path / "override.pt",
                          "Y": tmp_path / "new.pt"}
    assert cfg.heads == {"S": "Default", "X": "cli", "Y": None}


@pytest.mark.parametrize("body", ["{", "[]", '{"models":[]}', '{"models":{"A":{}}}',
                                      '{"unknown":1}'])
def test_bad_explicit_project_config_is_error(tmp_path, body):
    profile = tmp_path / "bad.json"
    profile.write_text(body)
    with pytest.raises(ValueError):
        resolve_runtime(project_config=profile)


def test_missing_models_fail_before_campaign_output(tmp_path, capsys):
    template = tmp_path / "template"
    template.mkdir()
    with pytest.raises(SystemExit) as error:
        build_campaign.main(["--out", str(tmp_path / "output"), "--template-case", str(template)])
    assert error.value.code == 2
    assert "select at least one model" in capsys.readouterr().err
    assert not (tmp_path / "output").exists()


def test_bad_or_unmatched_cli_assignment_fails(tmp_path):
    with pytest.raises(ValueError, match="LABEL=VALUE"):
        resolve_runtime(root=tmp_path, models=["invalid"])
    with pytest.raises(ValueError, match="no matching model"):
        resolve_runtime(root=tmp_path, models=["A=a.pt"], model_heads=["B=b"])
    with pytest.raises(ValueError, match="safe directory"):
        resolve_runtime(root=tmp_path, models=["../escape=a.pt"])


def test_project_with_single_and_named_models_uses_named_selection(tmp_path):
    profile = tmp_path / "settings" / "project.json"
    profile.parent.mkdir()
    profile.write_text(json.dumps({
        "model": "pipeline.pt", "head": "pipeline-head",
        "models": {"alpha": {"path": "comparison.pt", "head": "comparison-head"}},
    }))
    cfg = resolve_runtime(project_config=profile)
    assert cfg.root == profile.parent
    assert cfg.models == {"alpha": profile.parent / "comparison.pt"}
    assert cfg.heads == {"alpha": "comparison-head"}


def test_single_model_project_is_optional(tmp_path):
    profile = tmp_path / "project.json"
    profile.write_text('{"model": null}')
    assert resolve_runtime(project_config=profile).models == {}
    profile.write_text('{"model": "model.pt", "head": "Default"}')
    cfg = resolve_runtime(project_config=profile)
    assert cfg.models == {"default": tmp_path / "model.pt"}
    assert cfg.heads == {"default": "Default"}


def test_campaign_writes_each_selected_model_head(tmp_path):
    case = synth_case.create_case(tmp_path / "template")
    (case.dir / "c_vacancy_structures").mkdir()
    out = tmp_path / "campaign"
    assert build_campaign.main([
        "--root", str(tmp_path), "--out", str(out), "--template-case", str(case.dir),
        "--prepared", str(case.dir / "prepared_data/prepared_rmg_data.yaml"),
        "--model", "alpha=a.pt", "--model", "beta=b.pt",
        "--model-head", "alpha=Default", "--model-head", "beta=branch2",
    ]) == 0
    rows = [line.split("\t") for line in (out / "cases.tsv").read_text().splitlines()
            if not line.startswith("#")]
    assert [(row[0], row[2], row[3]) for row in rows] == [
        ("alpha-c0", str(tmp_path / "a.pt"), "Default"),
        ("beta-c0", str(tmp_path / "b.pt"), "branch2"),
    ]
