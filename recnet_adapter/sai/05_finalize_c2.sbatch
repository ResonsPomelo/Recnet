#!/bin/bash
# Historical Fe5C2 C2 finalization. Select all campaign inputs explicitly.
#SBATCH --job-name=recnet-c2-final
set -euo pipefail

usage() {
    cat <<'HELP'
Usage: sbatch [resource options] 05_finalize_c2.sbatch --root DIR [--root DIR ...]
       --expected-root DIR --report DIR --anchors FILE --coverage-threshold NUMBER
       [--title TEXT]
Historical Fe5C2 C2 workflow. Requires RECNET_DIR and project root/config;
RECNET_ENV_SCRIPT may select a scientific environment. C2_REPORT is a legacy
alias for --report. Scheduler account/resources/dependency are sbatch inputs.
HELP
}
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then usage; exit 0; fi
RECNET_DIR="${RECNET_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}"
ROOTS=()
EXPECTED=()
REPORT="${C2_REPORT:-}"
ANCHORS=""
COVERAGE="${C2_COVERAGE_THRESHOLD:-}"
TITLE="Fe5C2 C2 network report"
while (($#)); do
    case "$1" in
        --root) ROOTS+=("${2:?--root needs DIR}"); shift 2 ;;
        --expected-root) EXPECTED+=("${2:?--expected-root needs DIR}"); shift 2 ;;
        --report) REPORT="${2:?--report needs DIR}"; shift 2 ;;
        --anchors) ANCHORS="${2:?--anchors needs FILE}"; shift 2 ;;
        --coverage-threshold) COVERAGE="${2:?--coverage-threshold needs NUMBER}"; shift 2 ;;
        --title) TITLE="${2:?--title needs TEXT}"; shift 2 ;;
        *) echo "[c2-final] unexpected argument: $1" >&2; exit 2 ;;
    esac
done
[[ ${#ROOTS[@]} -gt 0 && ${#EXPECTED[@]} -gt 0 && -n "$REPORT" && -f "$ANCHORS" && -n "$COVERAGE" ]] || {
    echo '[c2-final] require root(s), expected root(s), report, anchors file and coverage threshold' >&2; exit 2;
}
source "$RECNET_DIR/recnet_adapter/sai/runtime.sh"
recnet_sai_init
FINALIZER="$RECNET_DIR/recnet_adapter/tools/finalize_network.sh"
[[ -f "$FINALIZER" ]] || { recnet_sai_error "missing finalizer: $FINALIZER"; exit 2; }

root_args=()
for root in "${ROOTS[@]}"; do root_args+=(--root "$root"); done
expected_args=()
for root in "${EXPECTED[@]}"; do expected_args+=(--expected-root "$root"); done
"$FINALIZER" "${root_args[@]}" "${expected_args[@]}" \
    --report "$REPORT" --anchors "$ANCHORS" --title "$TITLE" \
    --coverage-threshold "$COVERAGE"

# The legacy paths remain compatibility wrappers. Record the actual delivered files.
files=(
    "$RECNET_DIR/recnet_adapter/tools/c2_gate.py"
    "$RECNET_DIR/recnet_adapter/tools/finalize_network.sh"
    "$RECNET_DIR/recnet_adapter/tools/network/c2_gate.py"
    "$RECNET_DIR/recnet_adapter/tools/network/aggregate_network.py"
    "$RECNET_DIR/recnet_adapter/tools/dft/collect_ts_qc.py"
    "$RECNET_DIR/recnet_adapter/tools/network/audit_missing_attempts.py"
    "$RECNET_DIR/recnet_adapter/tools/report/make_network_report.py"
)
sha256sum "${files[@]}" | sort -k2 > "$REPORT/TOOL_SHA256SUMS"
sha256sum -c "$REPORT/TOOL_SHA256SUMS"
echo "[c2-final] report=$REPORT"
"$RECNET_PYTHON" -c 'import json,sys; print("[c2-final] gate=", json.load(open(sys.argv[1]))["gate_pass"])' "$REPORT/c2_gate.json"
