"""Core and explicit model-input wheel/editable installation gate.

Builds the wheel from this worktree, asserts every supported source member is
shipped and that the wheel excludes builder, tools and historical scripts, then
probes wheel and editable installs in throwaway venvs. Dependency directories
come from the selected interpreter, without executing parent editable hooks.
This checks installed package behavior, not fresh dependency resolution or
scientific model execution.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]

PACKAGES = ("handlers", "ccqn", "utils", "recnet_backend", "recnet_adapter")
PY_MODULES = ("run_dp_ts.py", "slabsite.py")
PACKAGE_DATA = {"recnet_adapter/model_config.json"}
EXCLUDED_PARTS = {"tests", "tools", "sai", "evidence", "__pycache__"}
EXCLUDED_FILES = {"recnet_adapter/runtime_config.py"}  # introduced with tool consumers
DIST_INFO = "recnet-0.0.1.dist-info/"


def expected_members() -> set[str]:
    """Source-tree ground truth for the wheel member set (core only)."""
    members: set[str] = set(PACKAGE_DATA)
    for package in PACKAGES:
        for path in sorted((ROOT / package).rglob("*.py")):
            relative = path.relative_to(ROOT)
            if not EXCLUDED_PARTS.intersection(relative.parts) and relative.as_posix() not in EXCLUDED_FILES:
                members.add(relative.as_posix())
    for module in PY_MODULES:
        if (ROOT / module).is_file():
            members.add(module)
    return members


def validate_wheel_members(names: set[str]) -> None:
    """Reject missing source files and any non-metadata payload not in scope."""
    expected = expected_members()
    missing = expected - names
    if missing:
        raise RuntimeError(f"Missing wheel members: {sorted(missing)}")
    unexpected = sorted(name for name in names if name not in expected and not name.startswith(DIST_INFO))
    if unexpected:
        raise RuntimeError(f"Unexpected wheel members: {unexpected}")


def run(command, cwd, env):
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), cwd=cwd, env=env, check=True, timeout=180)


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
                ROOT,
            ],
            scratch,
            env,
        )
        wheel, = wheel_dir.glob("*.whl")
        with zipfile.ZipFile(wheel) as archive:
            names = set(archive.namelist())
        validate_wheel_members(names)

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
                result = subprocess.run([str(python), *command], cwd=help_dir,
                                        env=env, capture_output=True, text=True, timeout=60)
                if result.returncode or "usage:" not in result.stdout.lower() or list(help_dir.iterdir()):
                    raise RuntimeError(f"Installed CLI help failed: {command} {result.stdout} {result.stderr}")
            missing_model = subprocess.run([str(python), "-m", "run_dp_ts"], cwd=help_dir,
                                           env=env, capture_output=True, text=True, timeout=60)
            if (missing_model.returncode != 2
                    or not any(word in missing_model.stderr.lower() for word in ("model", "模型"))
                    or list(help_dir.iterdir())):
                raise RuntimeError(f"Installed CLI accepted missing model: {missing_model.stdout} {missing_model.stderr}")
            print(f"PASS {mode} installed CLI help and missing-model rejection without outputs", flush=True)

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
    print("PASS core/model-input package smoke")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
