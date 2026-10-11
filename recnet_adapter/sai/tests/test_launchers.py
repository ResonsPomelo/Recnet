"""Local launcher contract tests: no scheduler, GPU, DP or scientific run."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

SAI = Path(__file__).resolve().parents[1]
ROOT = SAI.parents[1]


@pytest.fixture
def fake_site(tmp_path):
    code = tmp_path / "code"
    adapter = code / "recnet_adapter"
    (adapter / "sai").mkdir(parents=True)
    (adapter / "tools").mkdir()
    shutil.copy2(SAI / "runtime.sh", adapter / "sai/runtime.sh")
    for name in ("run_dp_ts_ft2dp.py", "tools/farm_runner.py", "tools/make_seeds_scan.py",
                 "tools/gpu_concurrency_probe.py"):
        (adapter / name).touch()
    source = Path(os.environ.get("RECNET_REAL_SOURCE", ROOT))
    if not (source / "recnet_adapter/runtime_config.py").is_file():
        pytest.skip("runtime_config.py is supplied by the integration branch")
    shutil.copy2(source / "recnet_adapter/runtime_config.py", adapter / "runtime_config.py")
    shutil.copy2(source / "recnet_adapter/__init__.py", adapter / "__init__.py")
    backend = code / "recnet_backend"
    backend.mkdir()
    shutil.copy2(source / "recnet_backend/__init__.py", backend / "__init__.py")
    shutil.copy2(source / "recnet_backend/model_selection.py", backend / "model_selection.py")
    if (SAI / "runtime_bridge.py").is_file():
        shutil.copy2(SAI / "runtime_bridge.py", adapter / "sai/runtime_bridge.py")
    fake_python = tmp_path / "fake-python"
    fake_python.write_text(
        f"#!{sys.executable}\n"
        "import json, os, pathlib, subprocess, sys\n"
        "with open(os.environ['FAKE_ARGV_LOG'], 'a') as stream:\n"
        " stream.write(json.dumps({'argv': sys.argv[1:], 'head': os.environ.get('RECNET_DP_HEAD'), "
        "'model': os.environ.get('RECNET_DP_MODEL'), 'site': os.environ.get('PYTHONNOUSERSITE'), "
        "'device': os.environ.get('CUDA_VISIBLE_DEVICES'), "
        "'project': os.environ.get('RECNET_PROJECT_CONFIG'), "
        "'model_config': os.environ.get('RECNET_MODEL_CONFIG')}) + '\\n')\n"
        "if os.environ.get('FAKE_RESOLVE_MODEL') == '1' and sys.argv[1:3] == ['-m','recnet_adapter']:\n"
        " from recnet_backend import resolve\n"
        " selected=resolve()\n"
        " with open(os.environ['FAKE_ARGV_LOG'], 'a') as stream:\n"
        "  stream.write(json.dumps({'resolved_model': str(selected.model), 'resolved_head': selected.head})+'\\n')\n"
        "if os.environ.get('FAKE_RUN_VALIDATOR') == '1' and sys.argv[1:2] == ['-']:\n"
        " script=sys.stdin.read()\n"
        " if 'validate_model_head' in script:\n"
        "  sys.exit(subprocess.run([sys.executable, *sys.argv[1:]], input=script, text=True).returncode)\n"
        "if os.environ.get('FAKE_FREEZE') == '1' and sys.argv[1:3] == ['-m','deepmd']:\n"
        " args=sys.argv[1:]; pathlib.Path(args[args.index('-o')+1]).touch()\n"
        "if os.environ.get('FAKE_SCAN_RC') and any(a.endswith('make_seeds_scan.py') for a in sys.argv[1:]):\n"
        " sys.exit(int(os.environ['FAKE_SCAN_RC']))\n"
    )
    fake_python.chmod(0o755)
    log = tmp_path / "argv.jsonl"
    root = tmp_path / "workspace"
    root.mkdir()
    inherited = {key: value for key, value in os.environ.items()
                 if not key.startswith("RECNET_") and key != "DPEVA_PYTHON"}
    env = {**inherited, "RECNET_DIR": str(code), "RECNET_WORKSPACE_ROOT": str(root),
           "RECNET_CONFIG_PYTHON": sys.executable, "RECNET_PYTHON": str(fake_python),
           "DPEVA_PYTHON": str(fake_python), "FAKE_ARGV_LOG": str(log),
           "SLURM_JOB_ID": "101", "SLURM_JOB_PARTITION": "fake",
           "CUDA_VISIBLE_DEVICES": "0", "PYTHONDONTWRITEBYTECODE": "1",
           "PYTHONPATH": str(code)}
    return code, root, log, env


def invoke(name, env, *args):
    return subprocess.run(["bash", str(SAI / name), *map(str, args)], env=env,
                          capture_output=True, text=True, timeout=15)


def calls(log):
    return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []


@pytest.mark.parametrize("name", sorted(p.name for p in SAI.glob("*.sbatch")))
def test_help_has_no_runtime_or_output(name, tmp_path):
    env = {"PATH": os.environ["PATH"], "HOME": str(tmp_path)}
    result = invoke(name, env, "--help")
    assert result.returncode == 0, result.stderr
    assert "Usage:" in result.stdout
    assert not list(tmp_path.iterdir())


def test_backend_requires_model_before_output(fake_site):
    _, root, log, env = fake_site
    env["RECNET_RUN_ROOT"] = str(root / "runs")
    result = invoke("00_backend_check.sbatch", env)
    assert result.returncode != 0
    assert not (root / "runs").exists()
    assert not calls(log)


def test_repo_model_config_is_not_implicit_deployment_selection(fake_site):
    code, root, log, env = fake_site
    model = root / "local-only.pt"
    model.touch()
    (code / "recnet_adapter/model_config.json").write_text(
        json.dumps({"model": str(model), "head": "stale-branch"}))
    env["RECNET_RUN_ROOT"] = str(root / "runs")
    for key in ("RECNET_DP_MODEL", "RECNET_DP_HEAD", "RECNET_PROJECT_CONFIG", "RECNET_MODEL_CONFIG"):
        env.pop(key, None)
    result = invoke("00_backend_check.sbatch", env)
    assert result.returncode != 0
    assert not (root / "runs").exists()
    assert not calls(log)


def test_explicit_model_does_not_inherit_legacy_config_head(fake_site):
    code, root, log, env = fake_site
    model = root / "single.pt"
    model.touch()
    (code / "recnet_adapter/model_config.json").write_text(
        json.dumps({"model": str(root / "other.pt"), "head": "stale-branch"}))
    env.update(RECNET_RUN_ROOT=str(root / "runs"), RECNET_DP_MODEL=str(model))
    env.pop("RECNET_DP_HEAD", None)
    result = invoke("00_backend_check.sbatch", env)
    assert result.returncode == 0, result.stderr
    assert calls(log)[-1]["model"] == str(model)
    assert calls(log)[-1]["head"] == "none"


def test_backend_forwards_selected_runtime_and_environment(fake_site):
    _, root, log, env = fake_site
    model = root / "model with spaces.pt"
    model.touch()
    env.update(RECNET_RUN_ROOT=str(root / "runs"), RECNET_DP_MODEL=str(model), RECNET_DP_HEAD="chosen")
    result = invoke("00_backend_check.sbatch", env)
    assert result.returncode == 0, result.stderr
    executed = calls(log)
    assert executed[-1]["argv"] == ["-m", "recnet_adapter", "check", "--repeats", "5"]
    assert executed[-1]["model"] == str(model)
    assert executed[-1]["head"] == "chosen"
    assert executed[-1]["site"] == "1"


def test_pipeline_forwards_case_and_quoted_arguments(fake_site):
    code, root, log, env = fake_site
    model = root / "single.pt"
    model.touch()
    case = root / "case with spaces"
    prepared = case / "prepared_data/prepared_rmg_data.yaml"
    prepared.parent.mkdir(parents=True)
    prepared.touch()
    env.update(RECNET_CASE=str(case), RECNET_DP_MODEL=str(model), RECNET_DP_HEAD="none")
    result = invoke("01_run_pipeline.sbatch", env, "--top-x", "3", "--custom", "value with spaces")
    assert result.returncode == 0, result.stderr
    argv = calls(log)[-1]["argv"]
    assert argv[0] == str(code / "recnet_adapter/run_dp_ts_ft2dp.py")
    assert argv[-4:] == ["--top-x", "3", "--custom", "value with spaces"]
    assert argv[argv.index("--path") + 1] == str(case)
    assert argv[argv.index("--prepared") + 1] == str(prepared)


def test_farm_forwards_manifest_runtime_and_explicit_head(fake_site):
    code, root, log, env = fake_site
    manifest = root / "cases.tsv"
    manifest.write_text("case\t/path\t/model.pt\tnone\t\t\n")
    env.update(RECNET_RUN_ROOT=str(root / "runs"), RECNET_CONCURRENCY="1",
               RECNET_THREADS="2", RECNET_DP_HEAD="none")
    result = invoke("03_case_farm.sbatch", env, manifest, "trial")
    assert result.returncode == 0, result.stderr
    argv = calls(log)[-1]["argv"]
    assert argv[0] == str(code / "recnet_adapter/tools/farm_runner.py")
    assert argv[argv.index("--cases") + 1] == str(manifest)
    assert argv[argv.index("--head") + 1] == "none"
    assert argv[argv.index("--summary") + 1] == str(root / "runs/trial/farm_summary.tsv")


def test_farm_rejects_unselected_model_and_head_before_launch(fake_site):
    _, root, log, env = fake_site
    manifest = root / "cases.tsv"
    manifest.write_text("case\t/path\t\t\t\t\n")
    env.update(RECNET_RUN_ROOT=str(root / "runs"), RECNET_CONCURRENCY="1",
               RECNET_THREADS="2")
    result = invoke("03_case_farm.sbatch", env, manifest, "trial")
    assert result.returncode != 0
    assert "model must be explicit" in result.stderr
    assert not calls(log)


def test_farm_accepts_explicit_project_default_model(fake_site):
    _, root, log, env = fake_site
    manifest = root / "cases.tsv"
    manifest.write_text("case\t/path\t\t\t\t\n")
    project = root / "project.json"
    project.write_text('{"model": "selected.pt", "head": "none"}')
    env.update(RECNET_PROJECT_CONFIG=str(project), RECNET_RUN_ROOT=str(root / "runs"),
               RECNET_CONCURRENCY="1", RECNET_THREADS="2")
    result = invoke("03_case_farm.sbatch", env, manifest, "trial")
    assert result.returncode == 0, result.stderr
    assert calls(log)[-1]["argv"][0].endswith("tools/farm_runner.py")


def test_seed_scan_selects_arm_model_head_and_threshold(fake_site):
    code, root, log, env = fake_site
    plan = root / "scan.tsv"
    plan.write_text("S\t1\t0\nM\t2\t1\n")
    models = [root / "S.pt", root / "M.pt"]
    for model in models: model.touch()
    env.update(RECNET_RUN_ROOT=str(root / "runs"), RECNET_SCAN_CASE_ROOT=str(root / "cases"),
               RECNET_SCAN_SPECIES="CO", RECNET_RXN="reaction-id",
               RECNET_SCAN_CONCURRENCY="1", RECNET_SCAN_THREADS="2",
               RECNET_SCAN_TIMEOUT_S="10", RECNET_SCAN_FREEZE_THRESHOLD="12.5",
               RECNET_MODEL_S=str(models[0]), RECNET_MODEL_M=str(models[1]),
               RECNET_SCAN_HEAD_S="none", RECNET_SCAN_HEAD_M="multi",
               RECNET_SCAN_DRY_RUN="1")
    result = invoke("04_seed_scan.sbatch", env, plan, "trial")
    assert result.returncode == 0, result.stderr
    scans = [record["argv"] for record in calls(log)
             if record["argv"] and record["argv"][0] == str(code / "recnet_adapter/tools/make_seeds_scan.py")]
    assert len(scans) == 2
    assert [x[x.index("--model") + 1] for x in scans] == list(map(str, models))
    assert [x[x.index("--head") + 1] for x in scans] == ["none", "multi"]
    assert all(x[x.index("--bottom-freeze-threshold") + 1] == "12.5" for x in scans)
    assert all("--dry-run" in x for x in scans)


def test_seed_scan_failure_propagates_after_summary(fake_site):
    _, root, _, env = fake_site
    plan = root / "scan.tsv"
    plan.write_text("S\t1\t0\n")
    for name in ("S.pt", "M.pt"):
        (root / name).touch()
    env.update(RECNET_RUN_ROOT=str(root / "runs"), RECNET_SCAN_CASE_ROOT=str(root / "cases"),
               RECNET_SCAN_SPECIES="CO", RECNET_RXN="reaction",
               RECNET_SCAN_CONCURRENCY="1", RECNET_SCAN_THREADS="2",
               RECNET_SCAN_TIMEOUT_S="10", RECNET_SCAN_FREEZE_THRESHOLD="12.5",
               RECNET_MODEL_S=str(root / "S.pt"), RECNET_MODEL_M=str(root / "M.pt"),
               FAKE_SCAN_RC="9")
    result = invoke("04_seed_scan.sbatch", env, plan, "trial")
    assert result.returncode != 0
    assert "\t9\t" in (root / "runs/trial/seed_scan_summary.tsv").read_text()


def test_seed_scan_invalid_arm_fails_with_summary_evidence(fake_site):
    _, root, log, env = fake_site
    plan = root / "scan.tsv"
    plan.write_text("X\t1\t0\n")
    for name in ("S.pt", "M.pt"):
        (root / name).touch()
    env.update(RECNET_RUN_ROOT=str(root / "runs"), RECNET_SCAN_CASE_ROOT=str(root / "cases"),
               RECNET_SCAN_SPECIES="CO", RECNET_RXN="reaction",
               RECNET_SCAN_CONCURRENCY="1", RECNET_SCAN_THREADS="2",
               RECNET_SCAN_TIMEOUT_S="10", RECNET_SCAN_FREEZE_THRESHOLD="12.5",
               RECNET_MODEL_S=str(root / "S.pt"), RECNET_MODEL_M=str(root / "M.pt"))
    result = invoke("04_seed_scan.sbatch", env, plan, "invalid-arm")
    assert result.returncode != 0
    assert "bad arm X" in result.stderr
    summary = (root / "runs/invalid-arm/seed_scan_summary.tsv").read_text()
    assert "invalid-arm:X\t2\t" in summary
    assert not any(record["argv"] and record["argv"][0].endswith("make_seeds_scan.py")
                   for record in calls(log))


def test_d2s_passes_each_row_head_and_inherited_device(fake_site):
    _, root, log, env = fake_site
    reaction = root / "reaction"
    reaction.mkdir()
    (reaction / "d2s_run.py").touch()
    model = root / "model.pt"
    model.touch()
    cases = root / "d2s.tsv"
    cases.write_text(f"case_a\t{root / 'case a'}\t{model}\tnone\t{root / 'out a'}\n")
    env.update(RECNET_REACTION_DIR=str(reaction), RECNET_D2S_OUT_ROOT=str(root / "logs"),
               RECNET_D2S_CONCURRENCY="1")
    result = invoke("06_co_d2s.sbatch", env, cases)
    assert result.returncode == 0, result.stderr
    executed = calls(log)
    assert executed[-1]["argv"] == ["-u", str(reaction / "d2s_run.py"), str(root / "case a"),
                                     str(model), "none", str(root / "out a")]
    assert executed[-1]["device"] == "0"


def test_probe_uses_selected_frames_and_head(fake_site):
    code, root, log, env = fake_site
    model, ts, md = (root / name for name in ("model.pt", "ts.xyz", "md.xyz"))
    for path in (model, ts, md): path.touch()
    env.update(RECNET_RUN_ROOT=str(root / "runs"), RECNET_DP_MODEL=str(model),
               RECNET_DP_HEAD="selected", RECNET_TS_FRAME=str(ts), RECNET_MD_FRAME=str(md),
               RECNET_PROBE_WORKERS="1,2", RECNET_PROBE_REPEATS="3")
    result = invoke("07_gpu_concurrency_probe.sbatch", env)
    assert result.returncode == 0, result.stderr
    probes = [record["argv"] for record in calls(log)
              if record["argv"] and record["argv"][0] == str(code / "recnet_adapter/tools/gpu_concurrency_probe.py")]
    assert len(probes) == 2
    assert [args[args.index("--structure") + 1] for args in probes] == [str(ts), str(md)]
    assert all(args[args.index("--head") + 1] == "selected" for args in probes)


def fake_deepmd_metadata(code):
    deepmd = code / "deepmd"
    (deepmd / "infer").mkdir(parents=True)
    (deepmd / "utils").mkdir()
    (deepmd / "__init__.py").write_text('__version__ = "fake"\n')
    (deepmd / "infer/__init__.py").write_text(
        "class DeepEval:\n"
        " def __init__(self, path, head): self.path=path\n"
        " def get_model_def_script(self):\n"
        "  return {'model_dict': {'selected': {}}} if 'multi' in self.path else {}\n"
    )
    (deepmd / "utils/__init__.py").touch()
    (deepmd / "utils/model_branch_dict.py").write_text(
        "def get_model_dict(value): return ({'selected': {}}, {})\n"
    )


@pytest.mark.parametrize("name,head", [("single with spaces.pt", None), ("multi.pt", "selected")])
def test_freeze_uses_selected_python_and_validated_head(fake_site, name, head):
    code, root, log, env = fake_site
    fake_deepmd_metadata(code)
    checkpoint = root / name
    checkpoint.touch()
    output = root / "frozen output"
    env.update(OUT_DIR=str(output), FAKE_RUN_VALIDATOR="1", FAKE_FREEZE="1")
    if head is not None: env["RECNET_DP_HEAD"] = head
    result = invoke("02_freeze_export.sbatch", env, checkpoint)
    assert result.returncode == 0, result.stderr
    args = next(record["argv"] for record in calls(log)
                if record["argv"][:2] == ["-m", "deepmd"])
    assert args[:6] == ["-m", "deepmd", "--pt", "freeze", "-c", str(checkpoint)]
    assert ("--head" in args) == (head is not None)
    assert Path(args[args.index("-o") + 1]).is_file()
    assert calls(log)[-1]["argv"] == ["-", args[args.index("-o") + 1]]


def test_freeze_rejects_multihead_without_selection_before_output(fake_site):
    code, root, log, env = fake_site
    fake_deepmd_metadata(code)
    checkpoint = root / "multi.pt"
    checkpoint.touch()
    output = root / "frozen"
    env.update(OUT_DIR=str(output), FAKE_RUN_VALIDATOR="1", FAKE_FREEZE="1")
    env.pop("RECNET_DP_HEAD", None)
    result = invoke("02_freeze_export.sbatch", env, checkpoint)
    assert result.returncode != 0
    assert "requires an explicit head" in result.stderr
    assert not output.exists()
    assert all(record["argv"][:2] != ["-m", "deepmd"] for record in calls(log))


def test_round2_gate_pass_skips_farm_and_output(fake_site):
    _, root, log, env = fake_site
    gate = root / "c2_gate.json"
    gate.write_text('{"gate_pass": true}')
    manifest = root / "cases.tsv"
    manifest.write_text("case\t/path\t/model.pt\tnone\t\t\n")
    env.update(RECNET_RUN_ROOT=str(root / "runs"), RECNET_CONCURRENCY="1",
               RECNET_THREADS="2", RECNET_PYTHON=sys.executable)
    result = invoke("06_round2_if_needed.sbatch", env, "--gate", gate,
                    "--cases", manifest, "--run-id", "trial")
    assert result.returncode == 0, result.stderr
    assert "gate PASS" in result.stdout
    assert not (root / "runs").exists()
    assert not calls(log)


def test_round2_refuses_empty_gate_result_from_broken_interpreter(fake_site):
    _, root, _, env = fake_site
    gate = root / "c2_gate.json"
    gate.write_text('{"gate_pass": false}')
    manifest = root / "cases.tsv"
    manifest.write_text("case\t/path\t/model.pt\tnone\t\t\n")
    env.update(RECNET_RUN_ROOT=str(root / "runs"), RECNET_CONCURRENCY="1",
               RECNET_THREADS="2")
    result = invoke("06_round2_if_needed.sbatch", env, "--gate", gate,
                    "--cases", manifest, "--run-id", "trial")
    assert result.returncode != 0
    assert not (root / "runs").exists()


def test_round2_failed_gate_forwards_selected_manifest(fake_site):
    code, root, log, env = fake_site
    gate = root / "c2_gate.json"
    gate.write_text('{"gate_pass": false}')
    manifest = root / "cases.tsv"
    manifest.write_text("case\t/path\t/model.pt\tnone\t\t\n")
    runner = code / "recnet_adapter/tools/farm_runner.py"
    runner.write_text(
        "import json,os,sys\n"
        "with open(os.environ['FAKE_ARGV_LOG'],'a') as f:\n"
        " f.write(json.dumps({'argv':sys.argv[1:], 'device':os.environ.get('CUDA_VISIBLE_DEVICES')})+'\\n')\n"
    )
    env.update(RECNET_RUN_ROOT=str(root / "runs"), RECNET_CONCURRENCY="1",
               RECNET_THREADS="2", RECNET_PYTHON=sys.executable)
    result = invoke("06_round2_if_needed.sbatch", env, "--gate", gate,
                    "--cases", manifest, "--run-id", "round2")
    assert result.returncode == 0, result.stderr
    argv = calls(log)[-1]["argv"]
    assert argv[argv.index("--cases") + 1] == str(manifest)
    assert argv[argv.index("--run-id") + 1] == "round2"
    assert argv[argv.index("--summary") + 1] == str(root / "runs/round2/farm_summary.tsv")


def test_c2_finalizer_receives_only_selected_inputs(fake_site):
    code, root, _, env = fake_site
    tools = code / "recnet_adapter/tools"
    for relative in ("c2_gate.py", "network/c2_gate.py", "network/aggregate_network.py",
                     "dft/collect_ts_qc.py", "network/audit_missing_attempts.py",
                     "report/make_network_report.py"):
        file = tools / relative
        file.parent.mkdir(parents=True, exist_ok=True)
        file.touch()
    finalizer = tools / "finalize_network.sh"
    finalizer.write_text(
        f"#!{sys.executable}\n"
        "import json,os,pathlib,sys\n"
        "args=sys.argv[1:]\n"
        "pathlib.Path(os.environ['FAKE_FINALIZER_LOG']).write_text(json.dumps(args))\n"
        "out=pathlib.Path(args[args.index('--report')+1]); out.mkdir(parents=True)\n"
        "(out/'c2_gate.json').write_text('{\\\"gate_pass\\\": true}')\n"
    )
    finalizer.chmod(0o755)
    anchors = root / "anchors with spaces.json"
    anchors.write_text("{}")
    selection = root / "finalizer.json"
    env["FAKE_FINALIZER_LOG"] = str(selection)
    report = root / "report"
    result = invoke("05_finalize_c2.sbatch", env, "--root", root / "pilot",
                    "--root", root / "campaign", "--expected-root", root / "campaign",
                    "--report", report, "--anchors", anchors, "--coverage-threshold", "0.91")
    assert result.returncode == 0, result.stderr
    args = json.loads(selection.read_text())
    assert args.count("--root") == 2
    assert args[args.index("--anchors") + 1] == str(anchors)
    assert args[args.index("--coverage-threshold") + 1] == "0.91"
    assert (report / "TOOL_SHA256SUMS").is_file()


def test_single_model_without_head_reaches_native_backend(fake_site):
    _, root, log, env = fake_site
    model = root / "single.pt"
    model.touch()
    env.update(RECNET_RUN_ROOT=str(root / "runs"), RECNET_DP_MODEL=str(model))
    env.pop("RECNET_DP_HEAD", None)
    result = invoke("00_backend_check.sbatch", env)
    assert result.returncode == 0, result.stderr
    assert calls(log)[-1]["head"] == "none"


def test_env_setup_head_is_resolved_after_source(fake_site):
    _, root, log, env = fake_site
    model = root / "multi.pt"
    model.touch()
    setup = root / "env.sh"
    setup.write_text("export RECNET_DP_HEAD=branch-from-env\n")
    env.update(RECNET_RUN_ROOT=str(root / "runs"), RECNET_DP_MODEL=str(model),
               RECNET_ENV_SCRIPT=str(setup))
    env.pop("RECNET_DP_HEAD", None)
    result = invoke("00_backend_check.sbatch", env)
    assert result.returncode == 0, result.stderr
    assert calls(log)[-1]["head"] == "branch-from-env"


def test_relative_project_config_survives_run_directory_change(fake_site):
    _, root, log, env = fake_site
    model = root / "single.pt"
    model.touch()
    project = root / "project.json"
    project.write_text(json.dumps({"root": str(root), "model": str(model)}))
    env.update(RECNET_RUN_ROOT=str(root / "runs"), RECNET_PROJECT_CONFIG="project.json")
    env.pop("RECNET_DP_HEAD", None)
    env.pop("RECNET_DP_MODEL", None)
    result = subprocess.run(["bash", str(SAI / "00_backend_check.sbatch")], cwd=root,
                            env=env, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert calls(log)[-1]["project"] == str(project)
    assert calls(log)[-1]["model"] == str(model)


def test_relative_model_config_resolves_after_run_directory_change(fake_site):
    _, root, log, env = fake_site
    model = root / "selected.pt"
    model.touch()
    selector = root / "models.json"
    selector.write_text(json.dumps({"model": "selected.pt", "head": "chosen"}))
    env.update(RECNET_RUN_ROOT=str(root / "runs"), RECNET_MODEL_CONFIG="models.json",
               FAKE_RESOLVE_MODEL="1")
    env.pop("RECNET_DP_MODEL", None)
    env.pop("RECNET_DP_HEAD", None)
    result = subprocess.run(["bash", str(SAI / "00_backend_check.sbatch")], cwd=root,
                            env=env, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert calls(log)[-2]["model_config"] == str(selector)
    assert calls(log)[-1] == {"resolved_model": str(model), "resolved_head": "chosen"}


def test_seed_scan_reads_per_arm_heads_from_real_runtime(fake_site):
    code, root, log, env = fake_site
    plan = root / "scan.tsv"
    plan.write_text("S\t1\t0\nM\t2\t1\n")
    for name in ("S.pt", "M.pt"):
        (root / name).touch()
    project = root / "project.json"
    project.write_text(json.dumps({"root": str(root), "models": {
        "S": {"path": "S.pt", "head": "none"},
        "M": {"path": "M.pt", "head": "branch-M"}}}))
    env.update(RECNET_PROJECT_CONFIG=str(project), RECNET_RUN_ROOT=str(root / "runs"),
               RECNET_SCAN_CASE_ROOT=str(root / "cases"), RECNET_SCAN_SPECIES="CO",
               RECNET_RXN="reaction", RECNET_SCAN_CONCURRENCY="1",
               RECNET_SCAN_THREADS="2", RECNET_SCAN_TIMEOUT_S="10",
               RECNET_SCAN_FREEZE_THRESHOLD="12.5", RECNET_SCAN_DRY_RUN="1")
    for key in ("RECNET_MODEL_S", "RECNET_MODEL_M", "RECNET_SCAN_HEAD_S",
                "RECNET_SCAN_HEAD_M", "RECNET_DP_HEAD"):
        env.pop(key, None)
    result = invoke("04_seed_scan.sbatch", env, plan, "trial")
    assert result.returncode == 0, result.stderr
    scans = [record["argv"] for record in calls(log)
             if record["argv"] and record["argv"][0] == str(code / "recnet_adapter/tools/make_seeds_scan.py")]
    assert [args[args.index("--head") + 1] for args in scans] == ["none", "branch-M"]


def test_farm_project_default_model_and_head_are_forwarded(fake_site):
    _, root, log, env = fake_site
    manifest = root / "cases.tsv"
    manifest.write_text("case\t/path\t\t\t\t\n")
    model = root / "default.pt"
    model.touch()
    project = root / "project.json"
    project.write_text(json.dumps({"root": str(root), "models": {
        "default": {"path": "default.pt", "head": "chosen"}}}))
    env.update(RECNET_PROJECT_CONFIG=str(project), RECNET_RUN_ROOT=str(root / "runs"),
               RECNET_CONCURRENCY="1", RECNET_THREADS="2")
    env.pop("RECNET_DP_HEAD", None)
    result = invoke("03_case_farm.sbatch", env, manifest, "trial")
    assert result.returncode == 0, result.stderr
    argv = calls(log)[-1]["argv"]
    assert argv[argv.index("--model") + 1] == str(model)
    assert argv[argv.index("--head") + 1] == "chosen"


@pytest.mark.parametrize("gate_text", ["{", "{}", '{"gate_pass": "false"}'])
def test_round2_rejects_invalid_gate_before_output(fake_site, gate_text):
    _, root, log, env = fake_site
    gate = root / "gate.json"
    gate.write_text(gate_text)
    manifest = root / "cases.tsv"
    manifest.write_text("case\t/path\t/model.pt\tnone\t\t\n")
    env.update(RECNET_RUN_ROOT=str(root / "runs"), RECNET_CONCURRENCY="1",
               RECNET_THREADS="2", RECNET_PYTHON=sys.executable)
    result = invoke("06_round2_if_needed.sbatch", env, "--gate", gate,
                    "--cases", manifest, "--run-id", "trial")
    assert result.returncode != 0
    assert not (root / "runs").exists()
    assert not calls(log)
