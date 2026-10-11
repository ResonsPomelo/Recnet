"""Model inputs shared by the scientific core and the Recnet CLI adapter.

This module is independent of the adapter and of the scientific backend. A
multi-head checkpoint still requires its own explicit head at construction.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
# Keep the installed legacy empty config as the default compatibility location.
CONFIG_PATH = REPO_ROOT / "recnet_adapter" / "model_config.json"
ENV_MODEL = "RECNET_DP_MODEL"
ENV_HEAD = "RECNET_DP_HEAD"


class AdapterError(RuntimeError):
    """Invalid model configuration or model-to-head mapping."""


@dataclass
class Resolution:
    """Resolved model inputs and their origins."""

    model: Optional[Path]
    head: Optional[str]
    source: dict

    def summary(self) -> str:
        m = str(self.model) if self.model is not None else "<unconfigured>"
        h = repr(self.head) if self.head else "None (no head selected)"
        return f"model={m}\nhead={h}\nsource={self.source}"


def _load_config(config_path: Any = None, *, default_config_path: Any = None) -> dict:
    path = Path(config_path) if config_path is not None else Path(default_config_path or CONFIG_PATH)
    if config_path is None and not path.exists():
        return {}
    try:
        cfg = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(cfg, dict):
            raise ValueError("configuration must be a JSON object")
        return cfg
    except (OSError, ValueError) as exc:
        raise AdapterError(f"Cannot read configuration {path}: {exc}") from exc


def _clean_head(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in ("", "none", "null", "off", "false"):
        return None
    return text


def _resolve_model_path(value: Any, repo_root: Any = None) -> Optional[Path]:
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in ("", "none", "null"):
        return None
    path = Path(os.path.expandvars(os.path.expanduser(text)))
    if not path.is_absolute():
        path = ((Path(repo_root) if repo_root is not None else REPO_ROOT) / path).resolve()
    return path


def resolve(model: Any = None, head: Any = None, *, config_path: Any = None,
            repo_root: Any = None, project_config: Any = None,
            default_config_path: Any = None, default_repo_root: Any = None) -> Resolution:
    """Resolve explicit model inputs without loading a scientific backend."""
    fallback_config = Path(default_config_path or CONFIG_PATH)
    fallback_root = Path(default_repo_root or REPO_ROOT)
    selected_config = (config_path or os.environ.get("RECNET_MODEL_CONFIG")
                       or project_config or os.environ.get("RECNET_PROJECT_CONFIG"))
    if selected_config:
        selected_config = Path(os.path.expandvars(str(selected_config))).expanduser().resolve()
    cfg = _load_config(selected_config, default_config_path=fallback_config)
    config_base = Path(repo_root) if repo_root is not None else fallback_root
    if selected_config and (project_config or os.environ.get("RECNET_PROJECT_CONFIG")) and not (config_path or os.environ.get("RECNET_MODEL_CONFIG")):
        root = Path(os.path.expandvars(str(cfg.get("root") or "."))).expanduser()
        config_base = root if root.is_absolute() else selected_config.parent / root
    elif selected_config and repo_root is None:
        config_base = selected_config.parent
    source = {"model": "unset", "head": "unset"}

    raw_model = None
    if model is not None:
        raw_model, source["model"] = model, "argument"
    elif os.environ.get(ENV_MODEL):
        raw_model, source["model"] = os.environ[ENV_MODEL], f"env:{ENV_MODEL}"
    elif cfg.get("model"):
        raw_model, source["model"] = cfg["model"], str(selected_config or fallback_config)

    raw_head = None
    if head is not None:
        raw_head, source["head"] = head, "argument"
    elif os.environ.get(ENV_HEAD):
        raw_head, source["head"] = os.environ[ENV_HEAD], f"env:{ENV_HEAD}"
    elif cfg.get("head") is not None:
        raw_head, source["head"] = cfg.get("head"), str(selected_config or fallback_config)

    return Resolution(
        model=_resolve_model_path(raw_model, config_base),
        head=_clean_head(raw_head),
        source=source,
    )
