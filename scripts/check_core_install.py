"""Core/model-input/builder wheel and editable install gate.

Builds the wheel from this worktree, asserts neutral backend and adapter
and builder/tool members and absence of tests, evidence and deployment payloads, then probes installs
in throwaway venvs. Dependency directories come from the selected interpreter, without
executing parent editable hooks. This checks installed package behavior, not
fresh dependency resolution or scientific model execution.
"""
from __future__ import annotations

import argparse
from email import message_from_bytes
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]

PACKAGES = ("handlers", "ccqn", "utils", "recnet_backend", "recnet_adapter", "network_builder")
PY_MODULES = ("run_dp_ts.py", "slabsite.py")
EXCLUDED_PARTS = {"tests", "evidence", "sai", "__pycache__"}


def expected_members():
    members = set(PY_MODULES) | {"recnet_adapter/model_config.json", "recnet_adapter/tools/finalize_network.sh"}
    for package in PACKAGES:
        for path in (ROOT / package).rglob("*.py"):
            relative = path.relative_to(ROOT)
            if not EXCLUDED_PARTS.intersection(relative.parts):
                members.add(relative.as_posix())
    return members


def validate_members(names):
    expected = expected_members()
    missing = expected - names
    if missing:
        raise RuntimeError(f"Missing wheel members: {sorted(missing)}")
    allowed = set(PACKAGES) | set(PY_MODULES) | {"recnet-0.0.1.dist-info"}
    bad = {name for name in names if name.split("/", 1)[0] not in allowed}
    bad.update(name for name in names
               if not name.startswith("recnet-0.0.1.dist-info/") and name not in expected)
    if bad:
        raise RuntimeError(f"Unexpected wheel members: {sorted(bad)}")


def run(command, cwd, env):
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), cwd=cwd, env=env, check=True, timeout=180)


def stage_source(destination: Path) -> Path:
    """Build from a fresh source snapshot, excluding prior setuptools output."""
    source = destination / "source"
    def ignored(directory, names):
        excluded = {name for name in names if name in {"__pycache__", ".pytest_cache", ".ruff_cache"}
                    or name.endswith(".pyc")}
        if Path(directory).resolve() == ROOT.resolve():
            excluded.update(name for name in names if name in {".git", "build", "dist"}
                            or name.endswith(".egg-info"))
        return excluded

    shutil.copytree(
        ROOT,
        source,
        ignore=ignored,
    )
    return source


def make_probe_env(interpreter, directory: Path, scratch: Path, env: dict, dependencies):
    """Use the selected environment, not the base Python of a nested venv.

    Plain .pth directory entries expose third-party files without processing the
    source environment's .pth files (which may point editable Recnet to source).
    """
    query = "import json,sysconfig; print(json.dumps(sorted({sysconfig.get_path('purelib'), sysconfig.get_path('platlib')})))"
    source_dirs = json.loads(subprocess.check_output(
        [str(interpreter), "-I", "-c", query], cwd=scratch, env=env, text=True))
    run([interpreter, "-m", "venv", directory], scratch, env)
    python = directory / "bin/python"
    destination, *_ = json.loads(subprocess.check_output(
        [str(python), "-I", "-c", query], cwd=scratch, env=env, text=True))
    Path(destination, "recnet_probe_dependencies.pth").write_text(
        "\n".join(source_dirs) + "\n", encoding="utf-8")
    probe = (
        "from pathlib import Path; import importlib; "
        f"roots=[Path(p).resolve() for p in {source_dirs!r}]; "
        f"modules=[importlib.import_module(n) for n in {dependencies!r}]; "
        "assert all(any(Path(m.__file__).resolve().is_relative_to(r) for r in roots) "
        "for m in modules), [(m.__name__, m.__file__) for m in modules]; "
        "print('PASS selected dependency origins:', [(m.__name__, m.__file__) for m in modules])"
    )
    run([python, "-c", probe], scratch, env)
    return python


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pipeline-python",
        help="Also probe installed pipeline imports in this environment",
    )
    parser.add_argument("--builder-python", default=sys.executable,
                        help="Interpreter providing RDKit for installed builder execution")
    args = parser.parse_args()
    env = dict(os.environ)
    for key in ("PYTHONPATH", "PYTHONHOME", "DISPLAY"):
        env.pop(key, None)
    for key in list(env):
        if key.startswith("RECNET_"):
            env.pop(key)
    env.update(
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONNOUSERSITE="1",
        MPLBACKEND="Agg",
        PIP_DISABLE_PIP_VERSION_CHECK="1",
    )
    with tempfile.TemporaryDirectory(prefix="recnet-model-install-") as directory:
        scratch = Path(directory)
        wheel_dir = scratch / "wheels"
        source = stage_source(scratch)
        run(
            [
                sys.executable,
                "-m",
                "pip",
                "wheel",
                "--no-deps",
                "--no-build-isolation",
                "--no-index",
                "--wheel-dir",
                wheel_dir,
                source,
            ],
            scratch,
            env,
        )
        wheel, = wheel_dir.glob("*.whl")
        with zipfile.ZipFile(wheel) as archive:
            names = set(archive.namelist())
            metadata = message_from_bytes(archive.read("recnet-0.0.1.dist-info/METADATA"))
            base_requirements = [value for value in metadata.get_all("Requires-Dist", []) if ";" not in value]
            for dependency in ("ase", "numpy", "scipy", "pyyaml", "sella", "deepmd-kit"):
                if not any(value.lower().split("[")[0].split(">")[0].split("=")[0] == dependency for value in base_requirements):
                    raise RuntimeError(f"Missing base runtime dependency: {dependency}")
        validate_members(names)

        for mode in ("wheel", "editable"):
            venv = scratch / mode
            python = make_probe_env(sys.executable, venv, scratch, env,
                                    ["ase", "numpy", "scipy", "yaml", "sella", "deepmd"])
            target = [wheel] if mode == "wheel" else ["--editable", ROOT]
            run(
                [
                    python,
                    "-m",
                    "pip",
                    "install",
                    "--ignore-installed",
                    "--no-deps",
                    "--no-build-isolation",
                    "--no-index",
                    *target,
                ],
                scratch,
                env,
            )
            expected = venv if mode == "wheel" else ROOT
            probe = (
                "from pathlib import Path; "
                "import importlib.metadata as metadata; "
                "import sys; import handlers, ccqn, utils, slabsite, recnet_adapter, recnet_backend; "
                "import recnet_adapter.calculator as ac, recnet_adapter.model_selection as am; "
                "import recnet_backend.calculator as bc, recnet_backend.model_selection as bm; "
                "assert ac is bc and am is bm; "
                "assert recnet_adapter.AdapterError is recnet_backend.AdapterError; "
                "assert not any(n.split('.')[0] == 'network_builder' for n in sys.modules); "
                f"root=Path({str(expected)!r}).resolve(); "
                "assert all(Path(mod.__file__).resolve().is_relative_to(root) "
                "for mod in (handlers, ccqn, utils, slabsite, recnet_adapter, recnet_backend)); "
                "assert recnet_adapter.CONFIG_PATH.is_file(); "
                "assert recnet_adapter.resolve().model is None; "
                "assert metadata.version('recnet') == '0.0.1'; "
                "print('PASS installed paths, aliases, and no implicit model')"
            )
            run([python, "-c", probe], scratch, env)
            help_dir = scratch / f"help-{mode}"
            help_dir.mkdir()
            for command in (["-m", "run_dp_ts", "--help"],
                            ["-m", "recnet_adapter", "--help"],
                            ["-m", "recnet_adapter", "run", "--help"]):
                result = subprocess.run([str(python), *command],
                                        cwd=help_dir, env=env, capture_output=True, text=True, timeout=60)
                if result.returncode or "usage:" not in result.stdout.lower() or list(help_dir.iterdir()):
                    raise RuntimeError(f"Installed CLI help failed: {command} {result.stdout} {result.stderr}")
            missing = subprocess.run([str(python), "-m", "run_dp_ts"],
                                     cwd=help_dir, env=env, capture_output=True, text=True, timeout=60)
            if missing.returncode != 2 or not any(word in missing.stderr.lower() for word in ("model", "模型")) or list(help_dir.iterdir()):
                raise RuntimeError(f"Installed CLI accepted missing model: {missing.stdout} {missing.stderr}")
            print(f"PASS {mode} installed CLI help and missing-model rejection without outputs", flush=True)
            module_root = subprocess.check_output(
                [str(python), "-c", "import pathlib,recnet_adapter; print(pathlib.Path(recnet_adapter.__file__).resolve().parent.parent)"],
                cwd=scratch, env=env, text=True).strip()
            for suite in ("adapter", "builder"):
                run([python, ROOT / "scripts/check_help.py", suite, "--root", module_root], scratch, env)
            result = subprocess.run(["bash", str(Path(module_root) / "recnet_adapter/tools/finalize_network.sh"), "--help"],
                                    cwd=help_dir, env=env, capture_output=True, text=True, timeout=30)
            if result.returncode or "usage:" not in result.stdout.lower() or list(help_dir.iterdir()):
                raise RuntimeError("Installed finalizer help failed or wrote output")
            print(f"PASS {mode} installed tool wrappers/modules and finalizer")


        if args.pipeline_python:
            venv = scratch / "pipeline"
            python = make_probe_env(args.pipeline_python, venv, scratch, env,
                                    ["ase", "numpy", "scipy", "yaml", "sella", "deepmd"])
            run([python, "-m", "pip", "install", "--ignore-installed", "--no-deps", "--no-index", wheel], scratch, env)
            probe = (
                "from pathlib import Path; "
                "import handlers, ccqn, utils, slabsite, run_dp_ts, recnet_adapter, recnet_backend; "
                f"root=Path({str(venv)!r}).resolve(); "
                "assert all(Path(mod.__file__).resolve().is_relative_to(root) "
                "for mod in (handlers, ccqn, utils, slabsite, run_dp_ts, recnet_adapter, recnet_backend)); "
                "print('PASS pipeline imports (no model execution)')"
            )
            run([python, "-c", probe], scratch, env)
        builder_venv = scratch / "builder"
        builder_python = make_probe_env(args.builder_python, builder_venv, scratch, env,
                                        ["ase", "numpy", "rdkit", "yaml"])
        run([builder_python, "-m", "pip", "install", "--ignore-installed", "--no-deps",
             "--no-index", wheel], scratch, env)
        builder_probe = (
            "from pathlib import Path; import sys; "
            "import network_builder, network_builder.cli as cli; "
            f"root=Path({str(builder_venv)!r}).resolve(); "
            "assert all(Path(mod.__file__).resolve().is_relative_to(root) "
            "for mod in (network_builder, cli)); "
            "assert not any(n.split('.')[0] in {'recnet_adapter', 'recnet_backend'} "
            "for n in sys.modules); "
            "print('PASS installed builder paths and boundaries')"
        )
        run([builder_python, "-c", builder_probe], scratch, env)

        builder_help = scratch / "builder-help"
        builder_help.mkdir()
        result = subprocess.run([str(builder_python), "-m", "network_builder.cli", "--help"],
                                cwd=builder_help, env=env, capture_output=True, text=True, timeout=60)
        if result.returncode or "usage:" not in result.stdout.lower() or list(builder_help.iterdir()):
            raise RuntimeError(f"Installed builder help failed: {result.stdout} {result.stderr}")
        print("PASS installed builder help without outputs", flush=True)

        builder_run = scratch / "builder-run"
        builder_run.mkdir()
        run([builder_python, "-m", "network_builder.cli"], builder_run, env)
        output = builder_run / "network_inputs"
        manifest_path = output / "MANIFEST.json"
        yaml_path = output / "fe5c2_510_c2.prepared_rmg_data.yaml"
        if not (manifest_path.is_file() and yaml_path.is_file()):
            raise RuntimeError("Installed builder did not create the expected dataset")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        count = manifest["counts"]["rxns"]
        bond_check = manifest["checks"]["broken_bond_visible_to_pipeline"]
        if (count <= 0 or bond_check["checked"] != count or bond_check["failures"]
                or not Path(bond_check["source"]).resolve().is_relative_to(builder_venv.resolve())):
            raise RuntimeError("Installed builder did not use installed core geometry")
        print(f"PASS installed builder generated {count} checked channels outside source", flush=True)
    print("PASS core/model-input/builder package smoke")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
