#!/usr/bin/env python3
"""Collect machine-readable TS QC evidence from RecNet TS archives.

This complements ``aggregate_network.py``: the aggregator proves that an energy
row has a TS record, while this collector extracts the saddle-point checks from
the optimization summary for each accepted record (one significant imaginary
frequency, bond-breaking character, endpoint energies, and bond-length guards).
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import yaml


FLOAT = r"(-?\d+(?:\.\d+)?)"


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


def last_float(text: str, pattern: str) -> float | None:
    hits = re.findall(pattern, text)
    return float(hits[-1]) if hits else None


def qc_from_summary(path: Path) -> dict[str, Any]:
    text = path.read_text(errors="replace")
    # A file may retain rejected orientation attempts. The accepted QC is the
    # final segment after the last rejection; if none was rejected, the whole
    # log is the accepted attempt.
    marker = "Rejected this orientation:"
    accepted_text = text.rsplit(marker, 1)[-1]

    initial_imags = re.findall(r"Initial imaginary frequency count:\s*(\d+)", accepted_text)
    significant_imags = re.findall(
        r"Significant imaginary frequency count \(\\|nu\\| >= 20\.0 cm\^-1\):\s*(\d+)",
        accepted_text,
    )
    if not significant_imags:
        significant_imags = re.findall(
            r"Significant imaginary frequency count .*?:\s*(\d+)",
            accepted_text,
        )
    frequencies = [
        float(x)
        for x in re.findall(r"Imaginary frequencies found:.*?([0-9.]+)j", accepted_text)
    ]
    mode_match = re.search(
        r"Imag mode \d+: f=" + FLOAT + r"j, bond_proj=" + FLOAT
        + r", ads_drift_proj=" + FLOAT + r", max_atom=(\S+)",
        accepted_text,
    )
    dominated = re.findall(
        r"Primary imaginary mode is (bond-breaking|adsorbate-drift) dominated",
        accepted_text,
    )

    after_opt = last_float(accepted_text, r"distance between broken bond atoms after opt:\s*" + FLOAT)
    after_ccqn = last_float(accepted_text, r"distance between broken bond atoms after ccqn:\s*" + FLOAT)
    rollback = last_float(accepted_text, r"distance between broken bond atoms after rollback ccqn:\s*" + FLOAT)
    # If the final accepted sequence used a rollback, that is the final guard
    # distance; otherwise ordinary CCQN is authoritative.
    rollback_recovered = "Rollback retry recovered acceptable TS bond length." in accepted_text
    final_distance = rollback if rollback_recovered and rollback is not None else after_ccqn
    max_len = last_float(accepted_text, r"TS reactive-bond max length threshold:\s*" + FLOAT)
    min_len = last_float(accepted_text, r"TS reactive-bond dissociation min threshold:\s*" + FLOAT)
    unexpected = re.findall(r"Unexpected broken bonds detected after CCQN:\s*(\[[^\]]*\])", accepted_text)

    endpoints: dict[str, Any] = {}
    for sign in ("+", "-"):
        m = re.search(
            rf"Imag{re.escape(sign)} endpoint: E={FLOAT} eV, d_break={FLOAT} A, h_ads={FLOAT} A",
            accepted_text,
        )
        if m:
            endpoints[sign] = {
                "energy_eV": float(m.group(1)),
                "d_break_A": float(m.group(2)),
                "h_ads_A": float(m.group(3)),
            }

    sig_count = int(significant_imags[-1]) if significant_imags else None
    bond_dominated = bool(dominated and dominated[-1] == "bond-breaking")
    mode = None
    if mode_match:
        mode = {
            "frequency_cm_1": -abs(float(mode_match.group(1))),
            "bond_proj": float(mode_match.group(2)),
            "ads_drift_proj": float(mode_match.group(3)),
            "max_atom": mode_match.group(4),
        }
    length_in_guard = (
        final_distance is not None
        and min_len is not None
        and max_len is not None
        and min_len <= final_distance <= max_len
    )
    qc_pass = sig_count == 1 and bond_dominated and length_in_guard and len(frequencies) == 1
    return {
        "summary_log": str(path),
        "accepted_segment_after_last_rejection": marker in text,
        "broken_bond_after_opt_A": after_opt,
        "broken_bond_after_ccqn_A": after_ccqn,
        "broken_bond_after_rollback_A": rollback,
        "broken_bond_final_A": final_distance,
        "bond_length_min_A": min_len,
        "bond_length_max_A": max_len,
        "bond_length_in_guard": length_in_guard,
        "initial_imaginary_count": int(initial_imags[-1]) if initial_imags else None,
        "significant_imaginary_count": sig_count,
        "imaginary_frequencies_cm_1": [-abs(x) for x in frequencies],
        "primary_imag_mode": mode,
        "primary_mode_bond_breaking_dominated": bond_dominated,
        "unexpected_broken_bonds": unexpected[-1] if unexpected else None,
        "imaginary_endpoints": endpoints,
        "qc_pass": qc_pass,
        "qc_fail_reasons": [
            reason
            for reason, ok in [
                ("significant_imaginary_count_is_not_one", sig_count == 1),
                ("primary_mode_is_not_bond_breaking_dominated", bond_dominated),
                ("bond_length_outside_guard", length_in_guard),
                ("imaginary_frequency_parse_failed", len(frequencies) == 1),
            ]
            if not ok
        ],
    }


def endpoint_ts_minus_check(qc: dict[str, Any], e_ts: float | None) -> None:
    """Both imaginary-mode endpoints should be lower than the saddle energy."""
    if e_ts is None:
        qc["endpoints_below_ts"] = None
        return
    values = [v.get("energy_eV") for v in qc.get("imaginary_endpoints", {}).values()]
    qc["endpoints_below_ts"] = bool(values) and all(
        v is not None and math.isfinite(v) and v < e_ts for v in values
    )
    if qc["endpoints_below_ts"] is False and "endpoint_energy_not_below_ts" not in qc["qc_fail_reasons"]:
        qc["qc_fail_reasons"].append("endpoint_energy_not_below_ts")
        qc["qc_pass"] = False


def collect(case: Path) -> dict[str, Any]:
    prepared_candidates = sorted((case / "prepared_data").glob("*prepared_rmg_data.yaml"))
    labels: dict[int, dict[str, Any]] = {}
    if prepared_candidates:
        payload = yaml.safe_load(prepared_candidates[0].read_text()) or {}
        labels = {
            i: {"reactant": r.get("reactant"), "product": r.get("product")}
            for i, r in enumerate(payload.get("rxns") or [])
        }
    records: list[dict[str, Any]] = []
    seen: set[tuple[int, int | None, str]] = set()
    for path in sorted(case.glob("rxn/ts_records*.yaml")) + sorted(case.glob("rxn/**/ts_records*.yaml")):
        vg = path.stem.split("_vg")[-1] if "_vg" in path.stem else None
        payload = yaml.safe_load(path.read_text()) or {}
        for rec in payload.get("ts_records") or []:
            key = (int(rec.get("rxn_idx", -1)), rec.get("site"), str(vg))
            if key in seen:
                continue
            seen.add(key)
            summary = Path(rec["summary_log"])
            qc = qc_from_summary(summary) if summary.exists() else {
                "summary_log": str(summary),
                "qc_pass": False,
                "qc_fail_reasons": ["summary_log_missing"],
            }
            records.append({
                "case": case.name,
                "case_dir": str(case),
                "tag": rec.get("tag"),
                "rxn_idx": rec.get("rxn_idx"),
                "label": labels.get(int(rec.get("rxn_idx", -1)), {}),
                "site": rec.get("site"),
                "vg": vg,
                "species": rec.get("species"),
                "species_key": rec.get("species_key"),
                "rec_bond_local": rec.get("rec_bond_local"),
                "ts_xyz": rec.get("ts_xyz"),
                "ts_record_file": str(path),
                "qc": qc,
            })
    return {"case": case.name, "case_dir": str(case), "ts_records": records}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--energy-glob", default="final_state_energy*.yaml")
    args = ap.parse_args(argv)

    cases = case_roots(args.root)
    if not cases:
        raise SystemExit("no case directories found")
    all_cases = [collect(case) for case in cases]
    for item in all_cases:
        energies: dict[tuple[int, int | None, str], float] = {}
        case_dir = Path(item["case_dir"])
        for path in sorted(case_dir.glob(f"rxn/**/{args.energy_glob}")):
            vg = path.stem.split("_vg")[-1] if "_vg" in path.stem else None
            payload = yaml.safe_load(path.read_text()) or {}
            for row in payload.get("final_state_energy") or []:
                try:
                    key = (int(row.get("rxn_idx", -1)), row.get("site"), str(vg))
                    energies[key] = float(row["e_ts"])
                except (KeyError, TypeError, ValueError):
                    continue
        for rec in item["ts_records"]:
            key = (int(rec["rxn_idx"]), rec["site"], str(rec["vg"]))
            endpoint_ts_minus_check(rec["qc"], energies.get(key))

    records = [rec for item in all_cases for rec in item["ts_records"]]
    passed = sum(bool(rec["qc"]["qc_pass"]) for rec in records)
    result = {
        "schema": "recnet_ts_qc/1",
        "totals": {
            "cases": len(all_cases),
            "ts_records": len(records),
            "qc_pass": passed,
            "qc_fail": len(records) - passed,
        },
        "cases": all_cases,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(
        f"wrote {out}: cases={len(all_cases)} ts_records={len(records)} "
        f"qc_pass={passed} qc_fail={len(records)-passed}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
