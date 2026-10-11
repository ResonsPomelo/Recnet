#!/usr/bin/env python3
"""Export boundary-profile scan barriers (constrained-scan caliber) — 2026-09-25, R213.

The campaign table has no accepted TS for a few boundary channels (notably `CO* -> C* O*`, the FT
rate-determining step): their barrier columns are empty.  The constrained **boundary-profile scans**
(`recnet-runs/boundary-profile-20260922/*/SCAN_MANIFEST_rxn_*.json`) do carry a per-frame energy
curve, so this tool exports the *profile* barrier with its own caliber label:

    barrier_profile = max(E over frames) − E(first frame)      # molecular state → dissociated
    barrier_span    = max(E) − min(E)

The scan steps the bond by 0.1 Å and relaxes the rest with the model, so the profile barrier is a
**scan-based estimate** (upper-bound-ish) — *not* an accepted TS, and it must not be mixed into the
`barrier_best_eV` column of the campaign caliber.  Output: ``profile_barriers.{json,csv}``.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import yaml


def channel_label(case_dir: str, rxn_index) -> str | None:
    if not case_dir or rxn_index is None:
        return None
    for cand in (Path(case_dir) / "prepared_data" / "prepared_rmg_data.yaml",
                 Path(case_dir) / "prepared_data" / "fe5c2_510_c2.prepared_rmg_data.yaml"):
        if cand.exists():
            rxns = (yaml.safe_load(cand.read_text()) or {}).get("rxns") or []
            if 0 <= rxn_index < len(rxns):
                r = rxns[rxn_index]
                return f"{r['reactant']} -> {r['product']}"
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Export boundary-profile scan barriers")
    ap.add_argument("--scans-dir", action="append", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    rows = []
    for d in args.scans_dir:
        for man in sorted(Path(d).glob("*/SCAN_MANIFEST_rxn_*.json")):
            payload = json.loads(man.read_text())
            case = payload.get("case") or ""
            for scan in payload.get("scans", []):
                recs = [r for r in (scan.get("records") or []) if r.get("E") is not None]
                if not recs:
                    continue
                recs.sort(key=lambda r: r["d"])
                es = [r["E"] for r in recs]
                rows.append({
                    "channel": channel_label(case, scan.get("rxn_index")),
                    "reactant": scan.get("reactant"),
                    "arm": "M" if "multi200k" in str(scan.get("model")) else "S",
                    "site": scan.get("site"), "vg": str(scan.get("vg")),
                    "scan_dir": str(man.parent), "case": case,
                    "d_min": round(recs[0]["d"], 3), "d_max": round(recs[-1]["d"], 3),
                    "n_frames": len(recs),
                    "all_converged": all(bool(r.get("converged")) for r in recs),
                    "monotonic_increasing": scan.get("monotonic_increasing"),
                    "barrier_profile_eV": round(max(es) - es[0], 3),
                    "barrier_span_eV": round(max(es) - min(es), 3),
                })
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "profile_barriers.json").write_text(json.dumps(
        {"schema": "profile-barriers-v1",
         "caliber": ("constrained scan (0.1 Å steps, rest relaxed): barrier_profile = max(E) − "
                     "E(first frame); not an accepted TS — do not merge into barrier_best_eV"),
         "rows": rows}, ensure_ascii=False, indent=1))
    with (out / "profile_barriers.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["channel"])
        w.writeheader()
        w.writerows(rows)
    print(f"[profile-barriers] {len(rows)} scans -> {out}/profile_barriers.{{json,csv}}")
    for r in rows:
        print("   %-34s %s site%-3s d %.2f-%.2f  barrier=%.3f  (converged=%s)" %
              (str(r["channel"])[:34], r["arm"], r["site"], r["d_min"], r["d_max"],
               r["barrier_profile_eV"], r["all_converged"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
