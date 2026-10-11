#!/bin/bash
# Shared SAI launcher preflight. Source this from a wrapper after its --help path.

recnet_sai_error() { echo "[recnet-sai] ERROR: $*" >&2; return 2; }

recnet_sai_init() {
    # Code and scientific environment are deployment inputs, never personal defaults.
    [[ -n "${RECNET_DIR:-}" ]] || { recnet_sai_error "set RECNET_DIR to the deployed Recnet checkout"; return 2; }
    [[ -f "$RECNET_DIR/recnet_adapter/runtime_config.py" ]] || {
        recnet_sai_error "missing $RECNET_DIR/recnet_adapter/runtime_config.py"; return 2;
    }
    if [[ -n "${RECNET_ENV_SCRIPT:-}" && ! -f "$RECNET_ENV_SCRIPT" ]]; then
        recnet_sai_error "missing RECNET_ENV_SCRIPT: $RECNET_ENV_SCRIPT"; return 2
    fi
    # SAI may provide a partition-specific MPS mapping. Do not fabricate a GPU mask.
    if [[ -n "${SLURM_JOB_PARTITION:-}" && -f "/opt/sai_config/mps_mapping.d/${SLURM_JOB_PARTITION}.bash" ]]; then
        source "/opt/sai_config/mps_mapping.d/${SLURM_JOB_PARTITION}.bash"
    fi
    if [[ -n "${RECNET_ENV_SCRIPT:-}" ]]; then source "$RECNET_ENV_SCRIPT"; fi
    [[ -n "${RECNET_PROJECT_CONFIG:-}${RECNET_WORKSPACE_ROOT:-}" ]] || {
        recnet_sai_error "set RECNET_PROJECT_CONFIG or RECNET_WORKSPACE_ROOT"; return 2;
    }
    # Both selectors are consumed after wrappers enter output folders.
    local selector value absolute
    for selector in RECNET_PROJECT_CONFIG RECNET_MODEL_CONFIG; do
        value="${!selector:-}"
        [[ -n "$value" ]] || continue
        absolute="$("${RECNET_CONFIG_PYTHON:-python3}" -c \
            'import os,sys; print(os.path.abspath(os.path.expanduser(os.path.expandvars(sys.argv[1]))))' \
            "$value")" || return 2
        printf -v "$selector" '%s' "$absolute"
        export "$selector"
    done
    export PYTHONNOUSERSITE=1
    export PYTHONPATH="$RECNET_DIR${PYTHONPATH:+:$PYTHONPATH}"
    local assignments
    assignments="$("${RECNET_CONFIG_PYTHON:-python3}" "$RECNET_DIR/recnet_adapter/sai/runtime_bridge.py")" || return 2
    eval "$assignments"
    export RECNET_WORKSPACE_ROOT RECNET_DIR RECNET_PYTHON DPEVA_PYTHON
    [[ -x "$RECNET_PYTHON" ]] || {
        # A command name is also a valid runtime_config interpreter.
        command -v -- "$RECNET_PYTHON" >/dev/null || {
            recnet_sai_error "runtime Python unavailable: $RECNET_PYTHON"; return 2;
        }
    }
}

recnet_sai_run_root() {
    RECNET_RUN_ROOT="${RECNET_RUN_ROOT:-${RUN_ROOT:-}}"
    [[ -n "$RECNET_RUN_ROOT" ]] || {
        recnet_sai_error "set RECNET_RUN_ROOT (legacy RUN_ROOT is accepted)"; return 2;
    }
}

recnet_sai_validate_cases() {
    # Keep the legacy farm's model_config.json fallback from silently selecting
    # a scientific model/head for rows left blank in a deployment manifest.
    local manifest="$1" fallback_model="$2"
    "${RECNET_CONFIG_PYTHON:-python3}" - "$manifest" "$fallback_model" <<'PY'
import sys
from pathlib import Path

manifest, fallback_model = sys.argv[1:]
for line_number, raw in enumerate(Path(manifest).read_text().splitlines(), 1):
    if not raw.strip() or raw.lstrip().startswith("#"):
        continue
    fields = raw.split("\t") + [""] * 4
    if not (fields[2].strip() or fallback_model):
        raise SystemExit(f"{manifest}:{line_number}: model must be explicit in row or wrapper input")
PY
}
