"""Resolve user-supplied workspace, model and interpreter settings.

A project file is loaded only when --project-config or RECNET_PROJECT_CONFIG
selects it. Importing this module does not read project data or load a model.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import shlex
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recnet_adapter import REPO_ROOT  # noqa: E402


def _path(value, base: Path) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(str(value))))
    return (path if path.is_absolute() else base / path).resolve()


def _python(value) -> str:
    executable = os.path.expandvars(os.path.expanduser(str(value)))
    # Do not dereference a venv's python symlink: its path selects the environment.
    return os.path.abspath(executable) if os.sep in executable else executable


def _head(value) -> str | None:
    if value is None or str(value).strip().lower() in {"", "none", "null", "off", "false"}:
        return None
    return str(value).strip()


def _label(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value) or value in {".", ".."}:
        raise ValueError(f"model label must be a safe directory name: {value!r}")
    return value


def _assignments(values, option: str) -> dict[str, str]:
    if isinstance(values, dict):
        for label in values:
            _label(label)
        return dict(values)
    result = {}
    for value in values or ():
        label, separator, setting = value.partition("=")
        if not separator or not label.strip() or not setting.strip():
            raise ValueError(f"{option} requires LABEL=VALUE: {value!r}")
        _label(label)
        if label in result:
            raise ValueError(f"duplicate {option} label: {label}")
        result[label] = setting
    return result


def _load_project(value) -> tuple[dict, Path | None]:
    if value is None:
        return {}, None
    path = _path(value, Path.cwd())
    try:
        config = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read project config {path}: {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError(f"project config must be a JSON object: {path}")
    allowed = {"root", "python", "head", "models", "model"}
    unknown = set(config) - allowed
    if unknown:
        raise ValueError(f"unsupported project config keys in {path}: {sorted(unknown)}")
    if "models" in config and not isinstance(config["models"], dict):
        raise ValueError(f"project config models must be an object: {path}")
    return config, path


@dataclass(frozen=True)
class RuntimeConfig:
    root: Path
    recnet_dir: Path
    models: dict[str, Path]
    heads: dict[str, str | None]
    head: str | None
    python: str

    @property
    def model_s(self) -> Path | None:
        return self.models.get("S")

    @property
    def model_m(self) -> Path | None:
        return self.models.get("M")


def resolve_runtime(*, root=None, recnet_dir=None, models=None, model_heads=None,
                    model_s=None, model_m=None, head=None, python=None,
                    project_config=None) -> RuntimeConfig:
    """Resolve overrides in CLI > environment > selected project file order."""
    project, project_path = _load_project(
        project_config if project_config is not None else os.environ.get("RECNET_PROJECT_CONFIG"))
    config_base = project_path.parent if project_path else Path.cwd()
    external_root = root or os.environ.get("RECNET_WORKSPACE_ROOT")
    workspace = (_path(external_root, Path.cwd()) if external_root else
                 _path(project.get("root") or config_base, config_base))
    code = _path(recnet_dir or os.environ.get("RECNET_DIR") or REPO_ROOT, Path.cwd())
    global_head = _head(head if head is not None else os.environ.get("RECNET_DP_HEAD", project.get("head")))
    configured = project.get("models", {})
    if "models" not in project and project.get("model") is not None:
        configured = {"default": project["model"]}
    paths: dict[str, Path] = {}
    heads: dict[str, str | None] = {}
    for label, record in configured.items():
        _label(label)
        if isinstance(record, str):
            record = {"path": record}
        if not isinstance(record, dict) or set(record) - {"path", "head"} or not record.get("path"):
            raise ValueError(f"project model {label!r} requires path and optional head")
        paths[label] = _path(record["path"], workspace)
        heads[label] = _head(record.get("head", global_head))

    explicit = _assignments(models, "--model")
    for label, value in (("S", model_s), ("M", model_m)):
        if value is not None:
            explicit[label] = value
        elif label not in explicit and os.environ.get(f"RECNET_MODEL_{label}"):
            explicit[label] = os.environ[f"RECNET_MODEL_{label}"]
    for label, value in explicit.items():
        paths[label] = _path(value, workspace)
        heads.setdefault(label, global_head)

    overrides = _assignments(model_heads, "--model-head")
    for label in paths:
        env_head = os.environ.get(f"RECNET_MODEL_HEAD_{label}")
        if label in overrides:
            heads[label] = _head(overrides[label])
        elif head is not None:
            heads[label] = global_head
        elif env_head is not None:
            heads[label] = _head(env_head)
        elif "RECNET_DP_HEAD" in os.environ:
            heads[label] = global_head
    unknown = set(overrides) - set(paths)
    if unknown:
        raise ValueError(f"--model-head has no matching model: {sorted(unknown)}")

    executable = _python(python or os.environ.get("RECNET_PYTHON") or project.get("python") or sys.executable)
    return RuntimeConfig(workspace, code, paths, heads, global_head, executable)


def add_runtime_arguments(parser: argparse.ArgumentParser, *, code=False, interpreter=False) -> None:
    parser.add_argument("--project-config", help="explicit JSON project settings (RECNET_PROJECT_CONFIG)")
    parser.add_argument("--root", help="workspace root (RECNET_WORKSPACE_ROOT or current directory)")
    parser.add_argument("--model", action="append", default=[], metavar="LABEL=PATH",
                        help="named model; repeat for each model")
    parser.add_argument("--model-head", action="append", default=[], metavar="LABEL=HEAD",
                        help="head for one named model; repeat as needed")
    parser.add_argument("--model-s", help="S checkpoint (legacy explicit alias)")
    parser.add_argument("--model-m", help="M checkpoint (legacy explicit alias)")
    parser.add_argument("--head", help="head shared by selected models; none clears it")
    if code:
        parser.add_argument("--recnet-dir", help="code root (RECNET_DIR or installed source root)")
    if interpreter:
        parser.add_argument("--python", help="child Python (RECNET_PYTHON or current interpreter)")


def runtime_from_args(args: argparse.Namespace) -> RuntimeConfig:
    names = ("root", "recnet_dir", "model_s", "model_m", "head", "python", "project_config")
    options = {name: getattr(args, name, None) for name in names}
    options["models"] = getattr(args, "model", [])
    options["model_heads"] = getattr(args, "model_head", [])
    return resolve_runtime(**options)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_runtime_arguments(parser, code=True, interpreter=True)
    parser.add_argument("--shell", action="store_true", help="emit quoted bash assignments")
    args = parser.parse_args(argv)
    cfg = runtime_from_args(args)
    values = {
        "RECNET_WORKSPACE_ROOT": str(cfg.root), "RECNET_DIR": str(cfg.recnet_dir),
        "RECNET_DP_HEAD": cfg.head if cfg.head is not None else "none",
        "RECNET_PYTHON": cfg.python,
        "DPEVA_PYTHON": _python(os.environ.get("DPEVA_PYTHON") or cfg.python),
    }
    for label in ("S", "M"):
        if label in cfg.models:
            values[f"RECNET_MODEL_{label}"] = str(cfg.models[label])
    if args.shell:
        for key, value in values.items():
            print(f"{key}={shlex.quote(str(value))}")
    else:
        print(json.dumps({**values, "models": {k: str(v) for k, v in cfg.models.items()},
                          "heads": cfg.heads}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
