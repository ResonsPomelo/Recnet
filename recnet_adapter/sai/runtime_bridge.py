#!/usr/bin/env python3
"""Emit quoted SAI assignments from Recnet's actual runtime/model resolvers."""
from __future__ import annotations

import os
import shlex

from recnet_adapter.runtime_config import resolve_runtime
from recnet_backend import CONFIG_PATH, resolve as resolve_model


def main() -> None:
    runtime = resolve_runtime()
    selected_project = os.environ.get("RECNET_PROJECT_CONFIG")
    model = resolve_model(project_config=selected_project)
    implicit_source = str(CONFIG_PATH)
    selected_model = model.model if model.source["model"] != implicit_source else None
    selected_head = model.head if model.source["head"] != implicit_source else None
    default = selected_model or runtime.models.get("default")
    # Per-label heads can differ from the global head. A missing head remains
    # empty; wrappers pass "none" only where a legacy CLI needs a sentinel.
    default_head = (selected_head if selected_model is not None else
                    runtime.heads.get("default", runtime.head))
    values = {
        "RECNET_WORKSPACE_ROOT": runtime.root,
        "RECNET_DIR": runtime.recnet_dir,
        "RECNET_PYTHON": runtime.python,
        "DPEVA_PYTHON": os.environ.get("DPEVA_PYTHON") or runtime.python,
        "RECNET_SAI_DEFAULT_MODEL": default or "",
        "RECNET_SAI_DEFAULT_HEAD": default_head or "",
        "RECNET_SAI_GLOBAL_HEAD": runtime.head or "",
    }
    for label in ("S", "M"):
        values[f"RECNET_SAI_MODEL_{label}"] = runtime.models.get(label, "")
        values[f"RECNET_SAI_HEAD_{label}"] = runtime.heads.get(label) or ""
    for key, value in values.items():
        print(f"{key}={shlex.quote(str(value))}")


if __name__ == "__main__":
    main()
