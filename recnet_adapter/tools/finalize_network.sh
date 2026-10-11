#!/bin/bash
# Finalize a Recnet C-network exploration: aggregate + TS QC + gate + anchors.
#
# Example:
#   finalize_network.sh \
#     --root $R/recnet-runs/pilot \
#     --root $R/recnet-runs/campaign-c2 \
#     --expected-root $R/recnet-runs/campaign-c2 \
#     --report $R/recnet-runs/reports/c2-final-v1 \
#     --anchors $R/validation-pipeline/reaction/paper_network_table.json

set -euo pipefail

usage() {
    cat <<'HELP'
Usage: finalize_network.sh --root PATH [--root PATH ...] --report PATH --anchors FILE
       [--expected-root PATH ...] [--coverage-threshold NUMBER] [--title TEXT]

Run the historical C2 aggregation, QC, gate and paper-comparison workflow.
--anchors must explicitly select the project's reference table. The paper
comparison uses the existing Fe5C2 channel mapping; a title does not change it.
Runtime Python: DPEVA_PYTHON, RECNET_PYTHON or the selected project configuration.
--help does not load configuration or create outputs.
HELP
}

ANCHORS=""
TITLE="Reaction network report"
COVERAGE=0.95
ROOTS=()
EXPECTED_ROOTS=()
REPORT=""

while (($#)); do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --title)
            [[ -n "${2:-}" ]] || { echo "ERROR: --title needs text" >&2; exit 2; }
            TITLE="$2"; shift 2 ;;
        --root)
            [[ -n "${2:-}" ]] || { echo "ERROR: --root needs a path" >&2; exit 2; }
            ROOTS+=("$2"); shift 2 ;;
        --expected-root)
            [[ -n "${2:-}" ]] || { echo "ERROR: --expected-root needs a path" >&2; exit 2; }
            EXPECTED_ROOTS+=("$2"); shift 2 ;;
        --report)
            [[ -n "${2:-}" ]] || { echo "ERROR: --report needs a path" >&2; exit 2; }
            REPORT="$2"; shift 2 ;;
        --anchors)
            [[ -n "${2:-}" ]] || { echo "ERROR: --anchors needs a path" >&2; exit 2; }
            ANCHORS="$2"; shift 2 ;;
        --coverage-threshold)
            [[ -n "${2:-}" ]] || { echo "ERROR: --coverage-threshold needs a number" >&2; exit 2; }
            COVERAGE="$2"; shift 2 ;;
        *) echo "ERROR: unknown argument: $1" >&2; exit 2 ;;
    esac
done

[[ ${#ROOTS[@]} -gt 0 ]] || { echo "ERROR: at least one --root is required" >&2; exit 2; }
[[ ${#EXPECTED_ROOTS[@]} -gt 0 ]] || EXPECTED_ROOTS=("${ROOTS[@]}")
[[ -n "$REPORT" ]] || { echo "ERROR: --report is required" >&2; exit 2; }
[[ -n "$ANCHORS" ]] || { echo "ERROR: --anchors is required" >&2; exit 2; }
[[ -f "$ANCHORS" ]] || { echo "ERROR: anchors not found: $ANCHORS" >&2; exit 2; }

CONFIG_SCRIPT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/runtime_config.py"
# The resolver is stdlib-only; the selected scientific Python is used below.
runtime_assignments="$("${RECNET_CONFIG_PYTHON:-python3}" "$CONFIG_SCRIPT" --shell)"
eval "$runtime_assignments"
TOOLDIR="$RECNET_DIR/recnet_adapter/tools"

if [[ "$DPEVA_PYTHON" != */* ]]; then
    runtime_python_path="$(command -v -- "$DPEVA_PYTHON")" || {
        echo "ERROR: runtime python not found on PATH: $DPEVA_PYTHON" >&2; exit 2;
    }
    DPEVA_PYTHON="$runtime_python_path"
fi
[[ -x "$DPEVA_PYTHON" ]] || { echo "ERROR: runtime python not executable: $DPEVA_PYTHON" >&2; exit 2; }
export PYTHONNOUSERSITE=1

mkdir -p "$REPORT"
export PYTHONPATH="$RECNET_DIR${PYTHONPATH:+:$PYTHONPATH}"

root_args=()
for root in "${ROOTS[@]}"; do root_args+=(--root "$root"); done
expected_args=()
for root in "${EXPECTED_ROOTS[@]}"; do expected_args+=(--campaign-root "$root"); done
data_args=()
for root in "${ROOTS[@]}"; do data_args+=(--data-root "$root"); done

python_canon() { "$DPEVA_PYTHON" "$@"; }
python_canon "$TOOLDIR/aggregate_network.py" "${root_args[@]}" --out "$REPORT"
python_canon "$TOOLDIR/collect_ts_qc.py" "${root_args[@]}" --out "$REPORT/ts_qc.json"
python_canon "$TOOLDIR/c2_gate.py" \
    "${expected_args[@]}" \
    "${data_args[@]}" \
    --aggregate-json "$REPORT/network_summary.json" \
    --qc-json "$REPORT/ts_qc.json" \
    --out "$REPORT/c2_gate.json" \
    --coverage-threshold "$COVERAGE"
python_canon "$TOOLDIR/audit_missing_attempts.py" \
    "${expected_args[@]}" \
    --gate-json "$REPORT/c2_gate.json" \
    --out "$REPORT/missing_attempt_audit.json"
python_canon "$TOOLDIR/make_network_report.py" \
    --summary "$REPORT/network_summary.json" \
    --anchors "$ANCHORS" \
    --out "$REPORT/paper_anchored_network.md" \
    --title "$TITLE"

{
    sha256sum \
        "$REPORT/network_summary.json" \
        "$REPORT/network_summary.md" \
        "$REPORT/ts_qc.json" \
        "$REPORT/c2_gate.json" \
        "$REPORT/c2_gate.md" \
        "$REPORT/missing_attempt_audit.json" \
        "$REPORT/paper_anchored_network.md"
} | sort -k2 > "$REPORT/SHA256SUMS"
sha256sum -c "$REPORT/SHA256SUMS"

echo "[finalize] report=$REPORT"
cat "$REPORT/c2_gate.md"
