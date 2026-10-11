#!/bin/bash
# SAI 后端冒烟：在计算节点（单卡 V100）运行 recnet_adapter check。
# Deployment inputs and sbatch resources are described in README.md.
#SBATCH --job-name=recnet-check

set -euo pipefail

if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    echo 'Usage: sbatch [resource options] 00_backend_check.sbatch'
    echo 'Requires project root/config, RECNET_RUN_ROOT and selected model; RECNET_ENV_SCRIPT is optional.'
    exit 0
fi
[[ $# -eq 0 ]] || { echo '[recnet-sai] unexpected arguments' >&2; exit 2; }
RECNET_DIR="${RECNET_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}"
source "$RECNET_DIR/recnet_adapter/sai/runtime.sh"
recnet_sai_init
recnet_sai_run_root
MODEL="${RECNET_SAI_DEFAULT_MODEL:?select RECNET_DP_MODEL or a project default model}"
HEAD="${RECNET_SAI_DEFAULT_HEAD:-}"
[[ -f "$MODEL" ]] || { recnet_sai_error "missing model: $MODEL"; exit 2; }

# Recnet 的 Sella/CCQN 路径必须用 conda env 内的 Sella/JAX/NumPy；
# 用户 site-packages 中旧 Sella + 新 JAX + 旧 NumPy 会在 TS 初始化时失败。
"$RECNET_PYTHON" - <<'PY'
import importlib.metadata as metadata
import jax
import numpy
import sella

try:
    sella_version = metadata.version("Sella")
except Exception:
    sella_version = getattr(sella, "__version__", "?")
print(f"[recnet-sai] runtime: sella={sella_version} ({sella.__file__}) "
      f"jax={jax.__version__} numpy={numpy.__version__}")
PY

export RECNET_DP_MODEL="$MODEL"
# "none" clears a legacy model_config head; the backend still rejects multi-head.
export RECNET_DP_HEAD="${HEAD:-none}"

RUN_DIR="$RECNET_RUN_ROOT/backend-check-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RUN_DIR"
cd "$RUN_DIR"

echo "[recnet-sai] job=${SLURM_JOB_ID} node=$(hostname) partition=${SLURM_JOB_PARTITION}"
echo "[recnet-sai] code=$(git -C "$RECNET_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "[recnet-sai] model=$MODEL head=$HEAD"
nvidia-smi -L || true

"$RECNET_PYTHON" -m recnet_adapter check --repeats 5
