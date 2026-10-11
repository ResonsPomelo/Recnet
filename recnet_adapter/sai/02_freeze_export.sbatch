#!/bin/bash
# Export a selected checkpoint to a single-head .pt2 and smoke-test loading.
#SBATCH --job-name=recnet-freeze
set -euo pipefail

if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    echo 'Usage: sbatch [resource options] 02_freeze_export.sbatch [checkpoint]'
    echo 'Requires project root/config, OUT_DIR and a selected checkpoint; multi-head needs a valid head.'
    exit 0
fi
[[ $# -le 1 ]] || { echo '[recnet-freeze] expected at most one checkpoint argument' >&2; exit 2; }
RECNET_DIR="${RECNET_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}"
source "$RECNET_DIR/recnet_adapter/sai/runtime.sh"
recnet_sai_init
OUT_DIR="${OUT_DIR:?set OUT_DIR for the frozen model}"
CKPT="${1:-${RECNET_SAI_DEFAULT_MODEL:-}}"
[[ -n "$CKPT" && -f "$CKPT" ]] || { recnet_sai_error "checkpoint is required and must exist: $CKPT"; exit 2; }
if [[ $# -eq 1 ]]; then
    HEAD="${RECNET_SAI_GLOBAL_HEAD:-}"
else
    HEAD="${RECNET_SAI_DEFAULT_HEAD:-}"
fi
# The existing backend validator inspects checkpoint metadata and rejects a
# multi-head model without a real selected branch before invoking DeepMD.
validated="$("$RECNET_PYTHON" - "$CKPT" "$HEAD" <<'PY'
import shlex
import sys
from recnet_backend import AdapterError
from recnet_backend.model_selection import validate_model_head

try:
    path, selected = validate_model_head(sys.argv[1], sys.argv[2] or None)
except AdapterError as exc:
    raise SystemExit(str(exc)) from exc
print("CKPT=" + shlex.quote(str(path)))
print("SELECTED_HEAD=" + shlex.quote(selected or ""))
PY
)" || exit 2
eval "$validated"
OUT_DIR="$("${RECNET_CONFIG_PYTHON:-python3}" -c \
    'import os,sys; print(os.path.abspath(os.path.expanduser(os.path.expandvars(sys.argv[1]))))' \
    "$OUT_DIR")"
TAG="$(basename "$CKPT" .pt)"
OUT="$OUT_DIR/frozen_${TAG}_${SELECTED_HEAD:-single}.pt2"
mkdir -p "$OUT_DIR"
cd "$OUT_DIR"

"$RECNET_PYTHON" -c 'import torch, deepmd; print("[freeze] deepmd", deepmd.__version__, "| torch", torch.__version__, "| cuda", torch.cuda.is_available())'
echo "[freeze] ckpt=$CKPT head=${SELECTED_HEAD:-<native>} out=$OUT"
freeze_args=("$RECNET_PYTHON" -m deepmd --pt freeze -c "$CKPT" -o "$OUT")
if [[ -n "$SELECTED_HEAD" ]]; then freeze_args+=(--head "$SELECTED_HEAD"); fi
time "${freeze_args[@]}"
ls -lh "$OUT"

"$RECNET_PYTHON" - "$OUT" <<'PY'
import math
import sys
from ase import Atoms
from deepmd.calculator import DP

model = sys.argv[1]
co = Atoms("CO", positions=[(0, 0, 0), (0, 0, 1.13)])
co.set_cell([20.0, 20.0, 20.0])
co.set_pbc([False, False, False])
co.center()
co.calc = DP(model=model)
energy = float(co.get_potential_energy())
forces = co.get_forces()
assert math.isfinite(energy) and all(math.isfinite(float(x)) for row in forces for x in row)
print("[freeze] load smoke: finite energy and forces", energy)
PY
echo "[freeze] done"
