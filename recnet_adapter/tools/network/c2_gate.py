#!/usr/bin/env python3
"""Evaluate the C2 exploration gate before allowing C3 work.

Inputs are already-produced artifacts: prepared campaign YAML files define the
expected channel universe, ``aggregate_network.py`` JSON defines accepted rows,
``collect_ts_qc.py`` JSON defines saddle-point evidence, and farm
``RUN_MANIFEST.txt`` files define case terminality.  The output is a
machine-readable decision report plus a short Markdown summary.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


def canonical_channel(rxn: dict[str, Any]) -> tuple[str, str, tuple[Any, ...]]:
    bb = rxn.get("broken_bond")
    if isinstance(bb, int):
        bb = [bb]
    if isinstance(bb, (list, tuple)):
        try:
            bb_key = tuple(int(x) for x in bb)
        except (TypeError, ValueError):
            bb_key = tuple(str(x) for x in bb)
    else:
        bb_key = (str(bb),)
    return (str(rxn.get("reactant")), str(rxn.get("product")), bb_key)


def case_roots(roots: list[str]) -> list[Path]:
    cases: list[Path] = []
    for root in roots:
        root_path = Path(root).resolve()
        if not root_path.exists():
            raise SystemExit(f"root not found: {root}")
        hits = sorted({p.parent.parent for p in root_path.rglob("prepared_data/*prepared_rmg_data.yaml")})
        if not hits:
            hits = sorted({p.parent for p in root_path.rglob("prepared_rmg_data.yaml")})
        cases.extend(hits)
    seen: set[Path] = set()
    out: list[Path] = []
    for case in cases:
        if case not in seen:
            seen.add(case)
            out.append(case)
    return out


def expected_channels(roots: list[str]) -> tuple[set[tuple[str, str, tuple[Any, ...]]], Counter, int]:
    expected: set[tuple[str, str, tuple[Any, ...]]] = set()
    per_case: Counter = Counter()
    n_rxns = 0
    for case in case_roots(roots):
        files = sorted(case.glob("prepared_data/*prepared_rmg_data.yaml"))
        for f in files:
            payload = yaml.safe_load(f.read_text()) or {}
            rxns = payload.get("rxns") or []
            per_case[case.name] += len(rxns)
            n_rxns += len(rxns)
            for rxn in rxns:
                expected.add(canonical_channel(rxn))
    return expected, per_case, n_rxns


def parse_manifest(path: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    for raw in path.read_text(errors="replace").splitlines():
        line = raw.strip()
        if line.startswith("===== RUN_MANIFEST record"):
            current = {}
            continue
        if line == "===== END record =====":
            if current is not None:
                records.append(current)
            current = None
            continue
        if current is not None and ":" in line:
            key, _, value = line.partition(":")
            current.setdefault(key.strip(), value.strip())
    return records


def latest_real_record(case: Path) -> dict[str, str] | None:
    records: list[dict[str, str]] = []
    for manifest in sorted(case.glob("farm/runs/*/*/RUN_MANIFEST.txt")):
        records.extend(parse_manifest(manifest))
    real = [r for r in records if r.get("dry_run", "").lower() == "false"]
    real.sort(key=lambda r: (r.get("ended_at", ""), r.get("started_at", "")))
    return real[-1] if real else None


def prepared_rxn_channels(cases: list[Path]) -> dict[tuple[str, int], tuple[str, str, tuple[Any, ...]]]:
    out: dict[tuple[str, int], tuple[str, str, tuple[Any, ...]]] = {}
    for case in cases:
        for f in sorted(case.glob("prepared_data/*prepared_rmg_data.yaml")):
            payload = yaml.safe_load(f.read_text()) or {}
            for idx, rxn in enumerate(payload.get("rxns") or []):
                out[(str(case), idx)] = canonical_channel(rxn)
    return out


def qc_index(qc_payload: dict[str, Any]) -> dict[tuple[str, int, Any, str], dict[str, Any]]:
    """Index QC records by case path, not display name.

    Campaign and retry chunks often reuse display names such as ``S-c0``;
    display-name indexing lets round-2 overwrite campaign QC records.
    """
    out: dict[tuple[str, int, Any, str], dict[str, Any]] = {}
    for case in qc_payload.get("cases", []):
        for rec in case.get("ts_records", []):
            key = (
                str(case.get("case_dir")),
                int(rec.get("rxn_idx", -1)),
                rec.get("site"),
                str(rec.get("vg")),
            )
            out[key] = rec.get("qc") or {}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--campaign-root", action="append", required=True)
    ap.add_argument("--data-root", action="append",
                    help="root containing accepted rows; defaults to campaign roots")
    ap.add_argument("--aggregate-json", required=True)
    ap.add_argument("--qc-json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--coverage-threshold", type=float, default=0.95)
    args = ap.parse_args(argv)

    expected, rxns_per_case, n_rxns = expected_channels(args.campaign_root)
    cases = case_roots(args.campaign_root)
    data_cases = case_roots(args.data_root or args.campaign_root)
    rxn_channels = prepared_rxn_channels(data_cases)
    aggregate = json.loads(Path(args.aggregate_json).read_text())
    qc_payload = json.loads(Path(args.qc_json).read_text())
    qidx = qc_index(qc_payload)

    case_states: dict[str, dict[str, Any]] = {}
    terminal_count = 0
    for case in cases:
        rec = latest_real_record(case)
        terminal = rec is not None and rec.get("status") in {"SUCCESS", "FAILURE"}
        terminal_count += int(terminal)
        case_states[case.name] = {
            "case_dir": str(case),
            "terminal": terminal,
            "status": rec.get("status") if rec else "RUNNING",
            "failure_class": rec.get("failure_class") if rec else None,
            "ended_at": rec.get("ended_at") if rec else None,
            "manifest": rec.get("log") if rec else None,
        }

    accepted_rows: list[dict[str, Any]] = []
    accepted_qc_rows: list[dict[str, Any]] = []
    accepted_review_rows: list[dict[str, Any]] = []
    for case_name, info in aggregate.get("cases", {}).items():
        case_dir = info.get("case_dir")
        for row in info.get("rows", []):
            if not row.get("accepted"):
                continue
            rxn_idx = int(row.get("rxn_idx", -1))
            enriched = dict(row)
            enriched["case"] = case_name
            enriched["case_dir"] = case_dir
            channel = rxn_channels.get((case_dir, rxn_idx))
            if channel is not None:
                enriched["reactant"], enriched["product"], enriched["broken_bond"] = channel
            key = (case_dir, rxn_idx, row.get("site"), str(row.get("vg")))
            qc = qidx.get(key, {})
            enriched["qc_pass"] = bool(qc.get("qc_pass"))
            enriched["qc_fail_reasons"] = qc.get("qc_fail_reasons", ["qc_record_missing"])
            accepted_rows.append(enriched)
            if qc.get("qc_pass"):
                accepted_qc_rows.append(enriched)
            else:
                accepted_review_rows.append(enriched)

    accepted_channels = {
        (r["reactant"], r["product"], tuple(r["broken_bond"]))
        for r in accepted_rows if r.get("broken_bond") is not None
    }
    accepted_qc_channels = {
        (r["reactant"], r["product"], tuple(r["broken_bond"]))
        for r in accepted_qc_rows if r.get("broken_bond") is not None
    }
    missing = sorted(expected - accepted_qc_channels)
    unexpected = sorted(accepted_qc_channels - expected)
    coverage = len(accepted_qc_channels) / len(expected) if expected else 0.0
    failures = [s for s in case_states.values() if s["status"] == "FAILURE"]
    all_terminal = terminal_count == len(cases) and len(cases) > 0
    checks = {
        "all_cases_terminal": all_terminal,
        "failures_classified": all(s["failure_class"] not in {None, ""} for s in case_states.values()),
        "coverage_at_least_threshold": coverage >= args.coverage_threshold,
        "all_accepted_rows_have_full_ts_qc": len(accepted_review_rows) == 0,
        "no_unexpected_accepted_channels": len(unexpected) == 0,
    }

    result = {
        "schema": "recnet_c2_gate/1",
        "coverage_threshold": args.coverage_threshold,
        "expected_channels": len(expected),
        "expected_rxn_rows_before_dedup": n_rxns,
        "rxns_per_case": dict(sorted(rxns_per_case.items())),
        "cases": len(cases),
        "terminal_cases": terminal_count,
        "case_states": dict(sorted(case_states.items())),
        "aggregate_accepted_rows": len(accepted_rows),
        "qc_pass_accepted_rows": len(accepted_qc_rows),
        "qc_review_accepted_rows": len(accepted_review_rows),
        "qc_review_rows": accepted_review_rows,
        "accepted_channels_without_qc_filter": len(accepted_channels),
        "accepted_qc_channels": len(accepted_qc_channels),
        "coverage_qc_pass": coverage,
        "missing_qc_pass_channels": [
            {"reactant": x[0], "product": x[1], "broken_bond": list(x[2])} for x in missing
        ],
        "unexpected_channels": [
            {"reactant": x[0], "product": x[1], "broken_bond": list(x[2])} for x in unexpected
        ],
        "checks": checks,
        "gate_pass": all(checks.values()),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    md = out.with_suffix(".md")
    status = "PASS" if result["gate_pass"] else "FAIL"
    lines = [
        "# C2 gate report",
        "",
        f"- decision: **{status}**",
        f"- cases: {terminal_count}/{len(cases)} terminal",
        f"- QC-pass channel coverage: {len(accepted_qc_channels)}/{len(expected)} = {coverage:.1%}",
        f"- accepted rows: {len(accepted_rows)} (QC pass {len(accepted_qc_rows)}, review {len(accepted_review_rows)})",
        f"- failures: {len(failures)}",
        "",
        "| check | pass |",
        "|---|---|",
    ]
    lines += [f"| {k} | {'YES' if v else 'NO'} |" for k, v in checks.items()]
    md.write_text("\n".join(lines) + "\n")
    print(
        f"wrote {out}: gate={status} cases={terminal_count}/{len(cases)} "
        f"coverage={coverage:.1%} rows={len(accepted_rows)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
