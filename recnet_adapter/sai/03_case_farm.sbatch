#!/bin/bash
# Recnet case farm: multiple cases in the caller's selected allocation.
#
# 与 01_run_pipeline.sbatch 的关系：01 是「一作业一 case 一卡」，本脚本把 N 个 case
# 收进一个 allocation，用 case 级并发换调度效率。
#
# 用法（登录节点、干净 shell；SAI 禁 sbatch --export，参数请用位置参数/环境变量）：
#   sbatch 03_case_farm.sbatch <cases.tsv> [run_id]
#   可选环境变量：
#     RECNET_CASES_PY  覆盖 farm_runner.py 路径
#     RECNET_CONCURRENCY  explicit process count
#     RECNET_THREADS      explicit per-case thread budget
#     RECNET_DEVICES      optional logical index/UUID subset of inherited CUDA_VISIBLE_DEVICES
#     RECNET_DP_MODEL / RECNET_DP_HEAD  清单列留空时的兜底
#     RECNET_DEFAULT_TIME_BUDGET_S      单 case 墙钟预算兜底（秒）
#     RECNET_FARM_SUMMARY               farm_summary.tsv 落盘路径
#   清单格式（TSV，`#` 注释与空行允许）：
#     case_name<TAB>case_dir<TAB>model<TAB>head<TAB>extra_args<TAB>time_budget_s
#
# Select account, partition, QOS and resources at submission; see README.md.
#SBATCH --job-name=recnet-farm

set -euo pipefail

if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    echo 'Usage: sbatch [resource options] 03_case_farm.sbatch <cases.tsv> [run_id]'
    echo 'Requires project root/config, RECNET_RUN_ROOT, RECNET_CONCURRENCY and RECNET_THREADS.'
    exit 0
fi
RECNET_DIR="${RECNET_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}"
source "$RECNET_DIR/recnet_adapter/sai/runtime.sh"
recnet_sai_init
recnet_sai_run_root
CASES="${1:-${RECNET_CASES:-}}"
RUN_ID="${2:-${RECNET_RUN_ID:-farm-${SLURM_JOB_ID:-$$}}}"
CONCURRENCY="${RECNET_CONCURRENCY:?set RECNET_CONCURRENCY}"
THREADS="${RECNET_THREADS:?set RECNET_THREADS}"
MODEL="${RECNET_SAI_DEFAULT_MODEL:-}"
HEAD="${RECNET_SAI_DEFAULT_HEAD:-none}"
FARM_RUNNER="${RECNET_CASES_PY:-$RECNET_DIR/recnet_adapter/tools/farm_runner.py}"
[[ "$CONCURRENCY" =~ ^[1-9][0-9]*$ && "$THREADS" =~ ^[1-9][0-9]*$ ]] || {
    recnet_sai_error "RECNET_CONCURRENCY and RECNET_THREADS must be positive integers"; exit 2;
}

if [ -z "$CASES" ]; then
    echo "[recnet-farm] ERROR: 未给清单文件。用法：sbatch 03_case_farm.sbatch <cases.tsv> [run_id]" >&2
    exit 3
fi
if [ ! -f "$CASES" ]; then
    echo "[recnet-farm] ERROR: 清单不存在: $CASES" >&2
    exit 3
fi

# R541：wrapper 稍后 `cd "$RUN_DIR"`（run_dir 内启动），相对路径清单会解析错位
# （曾致 1530085 在依赖解锁后 10 秒 exit 2）⇒ 先把清单规范化成绝对路径，工位无关。
CASES="$(readlink -f "$CASES")"
recnet_sai_validate_cases "$CASES" "$MODEL"
if [ ! -f "$FARM_RUNNER" ]; then
    echo "[recnet-farm] ERROR: farm_runner.py 不存在: $FARM_RUNNER" >&2
    exit 3
fi

# TS/CCQN 依赖 conda env 内的 Sella/JAX/NumPy；用户 site 旧包会导致
# Sella 初始化失败（np.asarray copy keyword/JAX 兼容性错误）。
export PYTHONNOUSERSITE=1

# 运行时版本打印 + Sella 初始化门禁（复用 01/00 的做法）：初始化失败就别开始烧卡。
"$RECNET_PYTHON" - <<'PY'
import importlib.metadata as metadata
import jax
import numpy
import sella
from sella import Sella

try:
    sella_version = metadata.version("Sella")
except Exception:
    sella_version = getattr(sella, "__version__", "?")
print(f"[recnet-farm] runtime: sella={sella_version} ({sella.__file__}) "
      f"jax={jax.__version__} numpy={numpy.__version__}")

import ase
import ase.build
import torch

atoms = ase.build.molecule("H2O")
print(f"[recnet-farm] sella-init: OK (ASE {ase.__version__}, torch {torch.__version__}, "
      f"cuda_available={torch.cuda.is_available()})")
PY
echo "[recnet-farm] sella-init check: PASS"

# 作业内必须能证明「哪些卡属于本 allocation」：设备绑定是 fail-closed 的，
# 空掩码一律拒绝，不猜 0..N-1（否则可能绑到别的作业的卡上）。
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "[recnet-farm] ERROR: 作业内 CUDA_VISIBLE_DEVICES 为空 —— 无法确定本 allocation 的卡" >&2
    echo "[recnet-farm]        检查提交资源和站点设备绑定；RECNET_DEVICES 不能代替 allocation mask。" >&2
    exit 4
fi
RUN_DIR="$RECNET_RUN_ROOT/$RUN_ID"
mkdir -p "$RUN_DIR"
cd "$RUN_DIR"
GPU_COUNT="$(printf '%s' "$CUDA_VISIBLE_DEVICES" | awk -F, '{print NF}')"
if [ "$CONCURRENCY" -gt "$GPU_COUNT" ]; then
    echo "[recnet-farm] WARNING: concurrency=$CONCURRENCY > 可见卡数 $GPU_COUNT；并发会压到 $GPU_COUNT" >&2
    echo "[recnet-farm]          注意：多个 case 可能共享同一张卡的显存。" >&2
fi

echo "[recnet-farm] job=${SLURM_JOB_ID} node=$(hostname) partition=${SLURM_JOB_PARTITION}"
echo "[recnet-farm] run_dir=$RUN_DIR cases=$CASES run_id=$RUN_ID"
echo "[recnet-farm] code=$(git -C "$RECNET_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "[recnet-farm] cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "[recnet-farm] concurrency=$CONCURRENCY threads=$THREADS"
echo "[recnet-farm] model=${MODEL:-<inherit>} head=${HEAD:-<inherit>}"
nvidia-smi -L || true

ARGS=(
    --cases "$CASES"
    --concurrency "$CONCURRENCY"
    --threads "$THREADS"
    --recnet-dir "$RECNET_DIR"
    --run-id "$RUN_ID"
    --summary "${RECNET_FARM_SUMMARY:-$RUN_DIR/farm_summary.tsv}"
)
if [ -n "${RECNET_DEVICES:-}" ]; then
    ARGS+=(--devices "$RECNET_DEVICES")
fi
if [ -n "$MODEL" ]; then
    ARGS+=(--model "$MODEL")
fi
if [ -n "$HEAD" ]; then
    ARGS+=(--head "$HEAD")
fi
if [ -n "${RECNET_DEFAULT_TIME_BUDGET_S:-}" ]; then
    ARGS+=(--default-time-budget-s "$RECNET_DEFAULT_TIME_BUDGET_S")
fi

echo "[recnet-farm] started at $(date '+%F %T')"
"$RECNET_PYTHON" "$FARM_RUNNER" "${ARGS[@]}"
rc=$?
echo "[recnet-farm] farm_runner exit=$rc"
echo "[recnet-farm] manifest 汇总: $RUN_DIR/<case>/RUN_MANIFEST.txt"
echo "[recnet-farm] finished at $(date '+%F %T')"
exit $rc
