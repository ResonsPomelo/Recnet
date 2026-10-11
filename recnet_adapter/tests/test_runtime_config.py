"""Configuration contracts; no GPU, checkpoint or private campaign data needed."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace

import pytest

import recnet_adapter as adapter
from recnet_adapter.tools import farm_runner
import synth_case

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def clean_config(monkeypatch, tmp_path_factory):
    for name in list(os.environ):
        if name.startswith("RECNET_") or name == "DPEVA_PYTHON":
            monkeypatch.delenv(name)
    config = tmp_path_factory.mktemp("config") / "model_config.json"
    config.write_text('{"model": null, "head": "ft2dp"}')
    monkeypatch.setattr(adapter, "CONFIG_PATH", config)


def test_adapter_existing_priority_and_single_head(monkeypatch, tmp_path):
    adapter.CONFIG_PATH.write_text('{"model": "config.pt", "head": "cfg"}')
    monkeypatch.setattr(adapter, "REPO_ROOT", tmp_path)
    assert adapter.resolve().model == tmp_path / "config.pt"
    monkeypatch.setenv("RECNET_DP_MODEL", "env.pt")
    monkeypatch.setenv("RECNET_DP_HEAD", "env")
    assert adapter.resolve().model == tmp_path / "env.pt"
    explicit = adapter.resolve(model="explicit.pt", head="none")
    assert explicit.model == tmp_path / "explicit.pt"
    assert explicit.head is None
    assert explicit.source == {"model": "argument", "head": "argument"}


def test_adapter_scoped_config_without_global_mutation(tmp_path):
    other = tmp_path / "other.json"
    other.write_text('{"model": "other.pt", "head": "none"}')
    result = adapter.resolve(config_path=other, repo_root=tmp_path)
    assert result.model == tmp_path / "other.pt"
    assert result.head is None
    assert adapter.resolve().head == "ft2dp"


def test_runtime_arm_priority_and_separate_single_model(monkeypatch, tmp_path):
    from recnet_adapter.runtime_config import resolve_runtime

    monkeypatch.setenv("RECNET_WORKSPACE_ROOT", str(tmp_path / "env-root"))
    monkeypatch.setenv("RECNET_MODEL_S", "env-S.pt")
    monkeypatch.setenv("RECNET_MODEL_M", "env-M.pt")
    monkeypatch.setenv("RECNET_DP_MODEL", "do-not-use-for-both.pt")
    monkeypatch.setenv("RECNET_DP_HEAD", "env-head")
    runtime = resolve_runtime(root=tmp_path, model_s="explicit-S.pt", head="none")
    assert runtime.root == tmp_path
    assert runtime.models == {"S": tmp_path / "explicit-S.pt", "M": tmp_path / "env-M.pt"}
    assert runtime.head is None
    assert resolve_runtime().head == "env-head"
    assert resolve_runtime().root == tmp_path / "env-root"


def test_runtime_defaults_and_lazy_env(monkeypatch, tmp_path):
    from recnet_adapter.runtime_config import resolve_runtime

    runtime = resolve_runtime(root=tmp_path)
    assert runtime.models == {}
    assert runtime.heads == {}
    assert runtime.head is None
    assert runtime.model_s is None and runtime.model_m is None
    assert runtime.recnet_dir == ROOT
    assert runtime.python == sys.executable
    monkeypatch.setenv("RECNET_PYTHON", "/custom/python")
    monkeypatch.setenv("RECNET_DIR", str(tmp_path / "code"))
    assert resolve_runtime().python == "/custom/python"
    assert resolve_runtime().recnet_dir == tmp_path / "code"
    assert resolve_runtime(python="/explicit/python", recnet_dir=ROOT).recnet_dir == ROOT
    monkeypatch.chdir(tmp_path)
    assert resolve_runtime(python="bin/python").python == str(tmp_path / "bin/python")
    link = tmp_path / "venv/bin/python"
    link.parent.mkdir(parents=True)
    link.symlink_to(sys.executable)
    assert resolve_runtime(python=link).python == str(link)


def test_farm_single_head_not_overridden_by_parent_env(monkeypatch, tmp_path):
    monkeypatch.setenv("RECNET_DP_HEAD", "ft2dp")
    monkeypatch.setenv("RECNET_DP_MODEL", "/parent.pt")
    case = farm_runner.Case(0, "single", tmp_path)
    args = SimpleNamespace(threads=None, recnet_dir=ROOT, run_id="probe")
    env = farm_runner.child_env(case, args, "0", None, None)
    assert env["RECNET_DP_HEAD"] == "none"
    assert env["RECNET_DP_MODEL"] == "none"


def test_campaign_slab_and_config_are_preserved(tmp_path):
    case = synth_case.create_case(tmp_path / "template")
    (case.dir / "Fe5C2_510.vasp").rename(case.dir / "Fe5C2_111.vasp")
    (case.dir / "c_vacancy_structures").mkdir()
    out = tmp_path / "campaign with spaces"
    proc = subprocess.run([
        sys.executable, str(ROOT / "recnet_adapter/tools/build_campaign.py"),
        "--root", str(tmp_path), "--out", str(out), "--template-case", str(case.dir),
        "--prepared", str(case.dir / "prepared_data/prepared_rmg_data.yaml"),
        "--slab-name", "Fe5C2_111.vasp", "--freeze-threshold", "9.542",
        "--model-s", "S.pt", "--model-m", "M.pt", "--head", "none",
    ], capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    rows = [line.split("\t") for line in (out / "cases.tsv").read_text().splitlines() if not line.startswith("#")]
    assert len(rows) == 2
    for row in rows:
        assert row[2] == str(tmp_path / f"{row[0][0]}.pt")
        assert row[3] == "none"
        tokens = shlex.split(row[4])
        assert tokens[tokens.index("--slab") + 1] == str(out / row[0] / "Fe5C2_111.vasp")
        assert tokens[tokens.index("--bottom-freeze-threshold") + 1] == "9.542"
        assert (out / row[0] / "prepared_data/prepared_rmg_data.yaml").is_file()


def test_shell_output_quotes_paths_and_keeps_legacy_override(monkeypatch, tmp_path):
    malicious = tmp_path / "space ' $(touch UNEXPECTED)"
    monkeypatch.setenv("RECNET_WORKSPACE_ROOT", str(malicious))
    monkeypatch.setenv("DPEVA_PYTHON", "/custom/legacy python")
    script = ROOT / "recnet_adapter/runtime_config.py"
    proc = subprocess.run([sys.executable, str(script), "--shell"], capture_output=True, text=True, check=True)
    # Parse generated assignments in bash, then expose values without eval of their contents.
    shell = proc.stdout + '\nexport RECNET_WORKSPACE_ROOT DPEVA_PYTHON\n' + shlex.quote(sys.executable) + ' -c ' + shlex.quote(
        "import os,json; print(json.dumps([os.environ['RECNET_WORKSPACE_ROOT'],os.environ['DPEVA_PYTHON']]))"
    )
    result = subprocess.run(["bash", "-c", shell], cwd=tmp_path, capture_output=True, text=True, check=True)
    assert json.loads(result.stdout) == [str(malicious), "/custom/legacy python"]
    assert not (tmp_path / "UNEXPECTED").exists()


@pytest.mark.parametrize("form", ["name", "tilde"])
def test_finalize_uses_central_interpreter_resolution(form, monkeypatch, tmp_path):
    executable = tmp_path / "bin/probe_python"
    executable.parent.mkdir()
    # Stop at the first scientific-tool invocation: no aggregation or calculation.
    executable.write_text("#!/bin/sh\necho FINALIZE_INTERPRETER_OK >&2\nexit 29\n")
    executable.chmod(0o755)
    anchors = tmp_path / "anchors.json"
    anchors.write_text("{}")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PATH", str(executable.parent) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("RECNET_CONFIG_PYTHON", sys.executable)
    monkeypatch.setenv("RECNET_WORKSPACE_ROOT", str(tmp_path))
    monkeypatch.setenv("RECNET_PYTHON", "probe_python" if form == "name" else "~/bin/probe_python")
    proc = subprocess.run(["bash", str(ROOT / "recnet_adapter/tools/finalize_network.sh"),
                           "--root", str(tmp_path), "--anchors", str(anchors),
                           "--report", str(tmp_path / "report")], cwd=tmp_path,
                          capture_output=True, text=True, timeout=15)
    assert proc.returncode == 29, proc.stderr
    assert "FINALIZE_INTERPRETER_OK" in proc.stderr


@pytest.mark.parametrize("name", ["eval_geometry_both", "co_thermo", "channel_thermo"])
def test_analysis_help_is_lightweight_and_has_no_side_effects(name, tmp_path):
    # -S hides site-packages: help must not require ASE/DeepMD or evaluate a model.
    proc = subprocess.run([sys.executable, "-S", str(ROOT / f"recnet_adapter/tools/{name}.py"), "--help"],
                          cwd=tmp_path, capture_output=True, text=True, timeout=15,
                          env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "usage:" in proc.stdout
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("source", ["cli", "env", "config"])
def test_farm_dry_run_priority_and_evidence(source, monkeypatch, tmp_path):
    code = tmp_path / "code"
    (code / "recnet_adapter").mkdir(parents=True)
    (code / "recnet_adapter/run_dp_ts_ft2dp.py").write_text("raise AssertionError('dry-run must not launch')")
    (code / "recnet_adapter/model_config.json").write_text('{"model": "config.pt", "head": "config"}')
    args = ["--recnet-dir", str(code), "--concurrency", "1", "--devices", "0", "--run-id", "probe", "--dry-run"]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    for key in ("SLURM_JOB_GPUS", "SLURM_STEP_GPUS", "SLURM_GPUS_ON_NODE"):
        monkeypatch.delenv(key, raising=False)
    if source in ("cli", "env"):
        monkeypatch.setenv("RECNET_DP_MODEL", "env.pt")
        monkeypatch.setenv("RECNET_DP_HEAD", "env")
    if source == "cli":
        args += ["--model", "cli.pt", "--head", "cli"]
    first = synth_case.create_case(tmp_path / "row")
    second = synth_case.create_case(tmp_path / "fallback")
    table = tmp_path / "cases.tsv"
    table.write_text(f"row\t{first.dir}\trow.pt\tnone\t\n" + f"fallback\t{second.dir}\t\t\t\n")
    monkeypatch.chdir(tmp_path)
    assert farm_runner.main(["--cases", str(table), *args]) == 0
    first_manifest, = first.dir.rglob("RUN_MANIFEST.txt")
    second_manifest, = second.dir.rglob("RUN_MANIFEST.txt")
    assert f"model: {code / 'row.pt'}\n" in first_manifest.read_text()
    assert "head: none\n" in first_manifest.read_text()
    assert f"model: {code / (source + '.pt')}\n" in second_manifest.read_text()
    assert f"head: {source}\n" in second_manifest.read_text()


def test_farm_selected_interpreter_and_single_head_reach_child(tmp_path):
    code = tmp_path / "code"
    (code / "recnet_adapter").mkdir(parents=True)
    (code / "recnet_adapter/run_dp_ts_ft2dp.py").write_text("import os; print('HEAD=' + os.environ['RECNET_DP_HEAD'])")
    python = tmp_path / "selected python"
    python.write_text('#!/bin/sh\necho SELECTED_PYTHON\nexec ' + shlex.quote(sys.executable) + ' "$@"\n')
    python.chmod(0o755)
    args = SimpleNamespace(recnet_dir=code, python=str(python), default_time_budget_s=None)
    case = farm_runner.Case(0, "child", tmp_path)
    log = tmp_path / "child.log"
    rc, timed_out = farm_runner.run_case_subprocess(case, args, tmp_path, log,
                                                    {**os.environ, "RECNET_DP_HEAD": "none"}, [])
    assert (rc, timed_out) == (0, False)
    assert "SELECTED_PYTHON\nHEAD=none" in log.read_text()


def test_thermo_and_geometry_keep_results_with_configured_models(monkeypatch, tmp_path, capsys):
    from ase import Atoms
    from ase.io import write
    from ase.calculators.calculator import Calculator, all_changes
    import deepmd.calculator
    import recnet_adapter.model_selection as selection
    from recnet_adapter.tools import co_thermo, channel_thermo, eval_geometry_both

    calls = []

    class ProbeDP(Calculator):
        implemented_properties = ["energy"]

        def __init__(self, model, head=None):
            super().__init__()
            calls.append((model, head))

        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            self.results = {"energy": float(atoms.info["probe_energy"])}

    monkeypatch.setattr(deepmd.calculator, "DP", ProbeDP)
    monkeypatch.setattr(selection, "inspect_model", lambda model: {"type_map": ["C", "O", "H"]})

    def structure(path, energy):
        path.parent.mkdir(parents=True, exist_ok=True)
        atoms = Atoms("CO", positions=[(0, 0, 0), (1.2, 0, 0)], cell=[10, 10, 10])
        atoms.info["probe_energy"] = energy
        write(path, atoms)

    energies = {"c": 103, "o": 105, "co": 107, "h": 102, "ch": 106,
                "cc": 108, "cco": 110, "cch": 109, "chco": 114}
    for arm in ("S", "M"):
        (tmp_path / f"{arm}.pt").touch()
        case = tmp_path / f"recnet-runs/campaign-c2/{arm}-c0"
        for species, value in energies.items():
            structure(case / f"rxn/Adsorbates/{species}_probe/0_vg0_opt.xyz", value)
        structure(case / "rxn/FS_energy/slab_opt_vg0.xyz", 100)
        structure(case / "prepared_data/h2.xyz", 1)
        (case / "prepared_data/prepared_rmg_data.yaml").write_text(
            'species:\n  h2:\n    name: H2\n    template_xyz: h2.xyz\n')
    options = ["--root", str(tmp_path), "--model-s", "S.pt", "--model-m", "M.pt", "--head", "none"]
    assert co_thermo.main([str(tmp_path / "co"), *options]) == 0
    assert [r["dE_diss"] for r in json.loads((tmp_path / "co/co_thermo.json").read_text())["results"]] == [1, 1]
    assert channel_thermo.main([str(tmp_path / "channels"), *options]) == 0
    rows = json.loads((tmp_path / "channels/channel_thermo.json").read_text())["rows"]
    assert [r["dE_eV"] for r in rows] == [-1, 3, 0, 0, 3] * 2
    structure(tmp_path / "geo.xyz", 7)
    structure(tmp_path / "ref.xyz", 5)
    assert eval_geometry_both.main([str(tmp_path / "geo.xyz"), str(tmp_path / "ref.xyz"), *options]) == 0
    assert capsys.readouterr().out.count("dE = +2.000 eV") == 2
    assert set(calls) == {(str(tmp_path / "S.pt"), None), (str(tmp_path / "M.pt"), None)}


def test_farm_cli_project_selects_interpreter_before_env(monkeypatch, tmp_path):
    from recnet_adapter.tools.campaign import farm_runner as farm
    profile = tmp_path / 'selected.json'
    profile.write_text(json.dumps({'python':'/project/bin/python'}))
    monkeypatch.setenv('RECNET_PROJECT_CONFIG', str(tmp_path / 'missing.json'))
    class ReachedLaunch(Exception):
        pass
    def stop_at_devices(devices):
        raise ReachedLaunch()
    observed = []
    original = farm.resolve_runtime
    def resolve_runtime(**kwargs):
        result = original(**kwargs)
        observed.append(result.python)
        return result
    monkeypatch.setattr(farm, 'resolve_runtime', resolve_runtime)
    monkeypatch.setattr(farm, 'read_device_view', stop_at_devices)
    with pytest.raises(ReachedLaunch):
        farm.main(['--cases',str(tmp_path / 'unused.tsv'), '--project-config',str(profile)])
    assert observed[-1] == '/project/bin/python'


@pytest.mark.parametrize('selector', ['RECNET_PROJECT_CONFIG', 'RECNET_MODEL_CONFIG'])
def test_farm_relative_config_survives_child_cwd(selector, monkeypatch, tmp_path):
    from recnet_adapter.tools.campaign import farm_runner as farm
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'settings.json').write_text(json.dumps({'model':'model.pt','head':'science'}))
    monkeypatch.setenv(selector, 'settings.json')
    selected = adapter.resolve()
    code = tmp_path / 'code'
    entry = code / 'recnet_adapter/run_dp_ts_ft2dp.py'
    entry.parent.mkdir(parents=True)
    entry.write_text('from recnet_adapter import resolve\nr=resolve()\nprint(str(r.model), r.head)\n')
    work_dir = tmp_path / 'case-output'
    work_dir.mkdir()
    args = SimpleNamespace(recnet_dir=code, threads=1, run_id='probe', python=sys.executable, default_time_budget_s=None)
    case = farm.Case(0, 'probe', work_dir)
    monkeypatch.setenv('PYTHONPATH', str(ROOT))
    env = farm.child_env(case, args, '0', str(selected.model), selected.head)
    log = tmp_path / 'child.log'
    assert farm.run_case_subprocess(case, args, work_dir, log, env, []) == (0, False), log.read_text()
    assert f'{selected.model} science' in log.read_text()
