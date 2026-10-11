#!/bin/bash
# RecNet 启动方案支撑实验：一张 4V100 卡上「几个 case 进程」的稳态吞吐。
#
# 结论用途：为指定项目决定 farm 并发；探针只读模型与结构，不写 case 目录。
#
# Scheduler resources are explicit sbatch options (README.md).
#SBATCH --job-name=recnet-gpu-probe

set -euo pipefail

if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    echo 'Usage: sbatch [resource options] 07_gpu_concurrency_probe.sbatch'
    echo 'Requires project root/config, run root, model, two frames, worker counts and repeats; head for multi-head.'
    exit 0
fi
[[ $# -eq 0 ]] || { echo '[gpu-probe] unexpected arguments' >&2; exit 2; }
RECNET_DIR="${RECNET_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}"
source "$RECNET_DIR/recnet_adapter/sai/runtime.sh"
recnet_sai_init
recnet_sai_run_root
MODEL="${RECNET_SAI_DEFAULT_MODEL:?select RECNET_DP_MODEL or a project default model}"
HEAD="${RECNET_SAI_DEFAULT_HEAD:-none}"
TS_FRAME="${RECNET_TS_FRAME:?set RECNET_TS_FRAME}"
MD_FRAME="${RECNET_MD_FRAME:?set RECNET_MD_FRAME}"
WORKERS="${RECNET_PROBE_WORKERS:?set RECNET_PROBE_WORKERS}"
REPEATS="${RECNET_PROBE_REPEATS:?set RECNET_PROBE_REPEATS}"
for input in "$MODEL" "$TS_FRAME" "$MD_FRAME"; do
    [[ -f "$input" ]] || { recnet_sai_error "missing input: $input"; exit 2; }
done
[[ "$REPEATS" =~ ^[1-9][0-9]*$ ]] || { recnet_sai_error 'RECNET_PROBE_REPEATS must be positive'; exit 2; }

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "[gpu-probe] ERROR: CUDA_VISIBLE_DEVICES 为空，无法确定本 allocation 的卡" >&2
    exit 4
fi
RUN_DIR="$RECNET_RUN_ROOT/gpu-probe-${SLURM_JOB_ID:-$$}"
mkdir -p "$RUN_DIR"
cd "$RUN_DIR"

echo "[gpu-probe] job=${SLURM_JOB_ID} node=$(hostname) partition=${SLURM_JOB_PARTITION}"
echo "[gpu-probe] cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
echo "[gpu-probe] model=${MODEL} head=${HEAD} workers=${WORKERS} repeats=${REPEATS}"
echo "[gpu-probe] run_dir=$RUN_DIR"

"$RECNET_PYTHON" - <<'PY'
import importlib.metadata as metadata
import torch, ase, deepmd
print(f"[gpu-probe] torch={torch.__version__} cuda_available={torch.cuda.is_available()} "
      f"ase={ase.__version__} deepmd={metadata.version('deepmd-kit')}")
if torch.cuda.is_available():
    print(f"[gpu-probe] device0={torch.cuda.get_device_name(0)}")
PY

PROBE="$RECNET_DIR/recnet_adapter/tools/gpu_concurrency_probe.py"

echo '=== A. selected TS frame ==='
"$RECNET_PYTHON" "$PROBE" --structure "$TS_FRAME" --model "$MODEL" --head "$HEAD" \
    --workers "$WORKERS" --repeats "$REPEATS" --out "$RUN_DIR/probe_ts.tsv" \
    2>&1 | tee "$RUN_DIR/probe_ts.log"

echo '=== B. selected MD frame ==='
"$RECNET_PYTHON" "$PROBE" --structure "$MD_FRAME" --model "$MODEL" --head "$HEAD" \
    --workers "$WORKERS" --repeats "$REPEATS" --out "$RUN_DIR/probe_md.tsv" \
    2>&1 | tee "$RUN_DIR/probe_md.log"

echo "[gpu-probe] done; results in $RUN_DIR"
