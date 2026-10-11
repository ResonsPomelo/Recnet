#!/bin/bash
# Historical Fe5C2 CO dissociation D2S recovery; external reaction runner.
#SBATCH --job-name=recnet-co-d2s
set -euo pipefail

if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    cat <<'HELP'
Usage: sbatch [resource options] 06_co_d2s.sbatch <cases.tsv>
Requires RECNET_REACTION_DIR (contains d2s_run.py), RECNET_D2S_OUT_ROOT,
RECNET_D2S_CONCURRENCY and allocated CUDA_VISIBLE_DEVICES. Manifest rows are
name<TAB>case<TAB>model<TAB>head<TAB>outdir. Models and heads must be explicit.
HELP
    exit 0
fi
[[ $# -eq 1 && -f "$1" ]] || { echo '[co-d2s] require existing cases.tsv' >&2; exit 2; }
CASES="$(readlink -f "$1")"
RECNET_DIR="${RECNET_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}"
REACTION="${RECNET_REACTION_DIR:?set RECNET_REACTION_DIR}"
OUT_ROOT="${RECNET_D2S_OUT_ROOT:?set RECNET_D2S_OUT_ROOT}"
CONC="${RECNET_D2S_CONCURRENCY:?set RECNET_D2S_CONCURRENCY}"
[[ -f "$REACTION/d2s_run.py" && "$CONC" =~ ^[1-9][0-9]*$ ]] || {
    echo '[co-d2s] require d2s_run.py and positive concurrency' >&2; exit 2;
}
source "$RECNET_DIR/recnet_adapter/sai/runtime.sh"
recnet_sai_init
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || { recnet_sai_error 'CUDA allocation unknown'; exit 4; }
IFS=',' read -r -a DEVICES <<< "$CUDA_VISIBLE_DEVICES"
(( CONC <= ${#DEVICES[@]} )) || { recnet_sai_error 'concurrency exceeds visible GPUs'; exit 4; }
mkdir -p "$OUT_ROOT"
cd "$REACTION"

echo "[co-d2s] job=${SLURM_JOB_ID:-local} concurrency=$CONC"
pids=()
i=0
rc=0
while IFS=$'\t' read -r name case model head outdir extra; do
    [[ -n "${name:-}" && "${name:0:1}" != '#' ]] || continue
    [[ -z "${extra:-}" && -n "$case" && -f "$model" && -n "$head" && -n "$outdir" && "$name" =~ ^[A-Za-z0-9_.-]+$ ]] || {
        echo "[co-d2s] invalid manifest row: $name" >&2; exit 2;
    }
    device="${DEVICES[$((i % ${#DEVICES[@]}))]}"
    echo "[co-d2s] START $name device=$device"
    ( CUDA_VISIBLE_DEVICES="$device" "$RECNET_PYTHON" -u "$REACTION/d2s_run.py" \
        "$case" "$model" "$head" "$outdir" > "$OUT_ROOT/${name}.run.log" 2>&1 ) &
    pids+=("$!")
    i=$((i + 1))
    if (( ${#pids[@]} >= CONC )); then
        for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
        pids=()
    fi
done < "$CASES"
for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
(( i > 0 )) || { echo '[co-d2s] empty manifest' >&2; exit 2; }
echo "[co-d2s] completed cases=$i rc=$rc; inspect $OUT_ROOT"
exit "$rc"
