#!/bin/bash
# Historical Fe5C2 C2 round2, conditional on the selected final gate.
#SBATCH --job-name=recnet-round2
set -euo pipefail

if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    cat <<'HELP'
Usage: sbatch [resource options] 06_round2_if_needed.sbatch --gate FILE --cases TSV --run-id ID
Requires RECNET_DIR, project root/config, RECNET_RUN_ROOT, RECNET_CONCURRENCY,
RECNET_THREADS and allocated CUDA_VISIBLE_DEVICES. The gate and cases are
project-specific inputs; submit only after a finalizer with chosen dependency.
HELP
    exit 0
fi
RECNET_DIR="${RECNET_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}"
GATE="${RECNET_ROUND2_GATE:-}"
CASES="${RECNET_ROUND2_CASES:-}"
RUN_ID="${RECNET_RUN_ID:-}"
while (($#)); do
    case "$1" in
        --gate) GATE="${2:?--gate needs FILE}"; shift 2 ;;
        --cases) CASES="${2:?--cases needs TSV}"; shift 2 ;;
        --run-id) RUN_ID="${2:?--run-id needs ID}"; shift 2 ;;
        *) echo "[round2] unexpected argument: $1" >&2; exit 2 ;;
    esac
done
[[ -f "$GATE" && -f "$CASES" && -n "$RUN_ID" ]] || {
    echo '[round2] require an existing gate, cases manifest and explicit run ID' >&2; exit 2;
}
source "$RECNET_DIR/recnet_adapter/sai/runtime.sh"
recnet_sai_init
recnet_sai_run_root
recnet_sai_validate_cases "$CASES" ""
CONCURRENCY="${RECNET_CONCURRENCY:?set RECNET_CONCURRENCY}"
THREADS="${RECNET_THREADS:?set RECNET_THREADS}"
[[ "$CONCURRENCY" =~ ^[1-9][0-9]*$ && "$THREADS" =~ ^[1-9][0-9]*$ ]] || {
    recnet_sai_error 'concurrency and threads must be positive integers'; exit 2;
}

gate_state="$("$RECNET_PYTHON" - "$GATE" <<'PY'
import json
import sys

try:
    with open(sys.argv[1]) as stream:
        gate = json.load(stream)
    state = gate["gate_pass"]
    if type(state) is not bool:
        raise ValueError("gate_pass must be a JSON boolean")
except (OSError, KeyError, TypeError, ValueError) as exc:
    raise SystemExit(f"invalid C2 gate: {exc}") from exc
print("PASS" if state else "FAIL")
PY
)" || { recnet_sai_error "refusing invalid gate: $GATE"; exit 2; }
if [[ "$gate_state" == PASS ]]; then
    echo '[round2] final C2 gate PASS; no round2 launch'
    exit 0
fi
[[ "$gate_state" == FAIL ]] || { recnet_sai_error "unexpected gate state: $gate_state"; exit 2; }
AUDIT="$(dirname "$GATE")/missing_attempt_audit.json"
if [[ -f "$AUDIT" ]]; then
    if "$RECNET_PYTHON" -c 'import json,sys; xs=json.load(open(sys.argv[1])).get("items",[]); sys.exit(0 if xs and all(x.get("status")=="attempted_rejected" for x in xs) else 1)' "$AUDIT"; then
        echo '[round2] all missing channels already attempted and rejected; skip duplicate run'
        exit 0
    fi
fi
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || { recnet_sai_error 'CUDA allocation unknown'; exit 4; }
RUN_DIR="$RECNET_RUN_ROOT/$RUN_ID"
mkdir -p "$RUN_DIR"
echo "[round2] gate=$GATE cases=$CASES run_dir=$RUN_DIR devices=$CUDA_VISIBLE_DEVICES"
"$RECNET_PYTHON" "$RECNET_DIR/recnet_adapter/tools/farm_runner.py" \
    --cases "$CASES" --concurrency "$CONCURRENCY" --threads "$THREADS" \
    --recnet-dir "$RECNET_DIR" --run-id "$RUN_ID" --summary "$RUN_DIR/farm_summary.tsv" \
    --head "${RECNET_SAI_DEFAULT_HEAD:-none}"
echo '[round2] farm complete; regenerate final C2 gate before C3 decision'
