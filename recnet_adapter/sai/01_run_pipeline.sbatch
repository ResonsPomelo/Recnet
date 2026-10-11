#!/bin/bash
# Recnet 管线（FT2DP/DPA4 后端）——SAI 单卡 V100 模板。
#
# 前置：CASE 目录内已备好 slab（cif/poscar）与 prepared_data/prepared_rmg_data.yaml
#（RMG 数据准备步骤见 docs/tasks/reaction-network-dpa4.md；本脚本不执行 prepare_rmg_data.py）。
#
# Deployment inputs and sbatch resources are described in README.md.
#SBATCH --job-name=recnet-rxn

set -euo pipefail

if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    echo 'Usage: sbatch [resource options] 01_run_pipeline.sbatch [pipeline arguments]'
    echo 'Requires project root/config, RECNET_CASE and selected model; RECNET_ENV_SCRIPT is optional.'
    exit 0
fi
RECNET_DIR="${RECNET_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}"
source "$RECNET_DIR/recnet_adapter/sai/runtime.sh"
recnet_sai_init
CASE="${RECNET_CASE:?set RECNET_CASE to a prepared case directory}"
SLAB="${RECNET_SLAB:-}"
PREPARED="${RECNET_PREPARED:-$CASE/prepared_data/prepared_rmg_data.yaml}"
MODEL="${RECNET_SAI_DEFAULT_MODEL:?select RECNET_DP_MODEL or a project default model}"
HEAD="${RECNET_SAI_DEFAULT_HEAD:-}"
[[ -f "$MODEL" ]] || { recnet_sai_error "missing model: $MODEL"; exit 2; }

if [ ! -f "$PREPARED" ]; then
    echo "[recnet-sai] ERROR: prepared 数据不存在: $PREPARED" >&2
    echo "[recnet-sai] Prepare the case input before submission." >&2
    exit 3
fi

# TS/CCQN 依赖 conda env 内的 Sella/JAX/NumPy；用户 site 旧包会导致
# Sella 初始化失败（np.asarray copy keyword/JAX 兼容性错误）。
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
export RECNET_DP_HEAD="${HEAD:-none}"

cd "$CASE"
echo "[recnet-sai] job=${SLURM_JOB_ID} node=$(hostname) case=$CASE"
echo "[recnet-sai] code=$(git -C "$RECNET_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "[recnet-sai] model=$MODEL head=$HEAD"
nvidia-smi -L || true

ARGS=(--path "$CASE" --prepared "$PREPARED" --top-x 1)
if [ -n "$SLAB" ]; then
    ARGS+=(--slab "$SLAB")
fi
if [ -n "${RECNET_EXTRA_ARGS:-}" ]; then
    # shellcheck disable=SC2206
    ARGS+=(${RECNET_EXTRA_ARGS})
fi

"$RECNET_PYTHON" "$RECNET_DIR/recnet_adapter/run_dp_ts_ft2dp.py" "${ARGS[@]}" "$@"

echo "[recnet-sai] finished at $(date '+%F %T')"
