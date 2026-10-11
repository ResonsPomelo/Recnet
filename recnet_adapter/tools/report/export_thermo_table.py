#!/usr/bin/env python3
"""Per-channel thermodynamics from an aggregate summary (2026-09-25, R186).

The delivered channel table is deliberately **barrier-only** (its schema is frozen against the
delivered c2r2/c2r3 snapshots).  The reaction energies are already computed by the pipeline
(``rxn/FS_energy/final_state_energy_vg*.yaml`` → ``delta_e_fs_minus_is`` and its ZPE/G companions
at the campaign temperature) and are carried into ``network_summary.json`` by
``aggregate_network.py`` as additive fields; this tool turns them into a **side table** so the
network can be discussed thermodynamically without moving a delivered cell.

    export_thermo_table.py --summary <report>/network_summary.json --out-dir <dir>

The Ea/ΔE pair reported per channel comes from the **same site** (the accepted site with the lowest
barrier), so BEP-style analysis is self-consistent.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import statistics
from pathlib import Path


def arm_of(case: str) -> str | None:
    for arm in ("S", "M"):
        if case.startswith(f"{arm}-"):
            return arm
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Export per-channel thermodynamics (ΔE / ΔE_ZPE / ΔG)")
    ap.add_argument("--summary", required=True, help="network_summary.json from aggregate_network.py")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args(argv)

    summary = json.loads(Path(args.summary).read_text())
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    per_channel: dict[str, dict] = collections.defaultdict(dict)
    for channel, per_case in summary.get("channels", {}).items():
        for case, rec in per_case.items():
            arm = arm_of(case)
            if arm is None:
                continue
            accepted = [s for s in rec.get("sites", [])
                        if s.get("accepted") and s.get("barrier") is not None
                        and s.get("delta_e") is not None]
            if not accepted:
                continue
            best = min(accepted, key=lambda s: s["barrier"])
            rows.append({
                "channel": channel, "arm": arm, "case": case,
                "site": best.get("site"), "vg": best.get("vg"),
                "barrier_eV": round(best["barrier"], 4),
                "delta_e_eV": round(best["delta_e"], 4),
                "delta_e_zpe_eV": None if best.get("delta_e_zpe") is None else round(best["delta_e_zpe"], 4),
                "delta_g_eV": None if best.get("delta_g") is None else round(best["delta_g"], 4),
            })
            per_channel[channel][arm] = rows[-1]

    de = [r["delta_e_eV"] for r in rows]
    stats = {
        "n_pairs": len(rows),
        "channels": len(per_channel),
        "delta_e": {
            "min": min(de) if de else None,
            "median": round(statistics.median(de), 3) if de else None,
            "max": max(de) if de else None,
            "exothermic": sum(1 for v in de if v < -0.05),
            "near_thermoneutral": sum(1 for v in de if -0.05 <= v <= 0.05),
            "endothermic": sum(1 for v in de if v > 0.05),
        },
    }
    # arm-level comparison on channels that have both arms (same site choice rule per arm)
    both = {c: v for c, v in per_channel.items() if "S" in v and "M" in v}
    if both:
        stats["arm_delta_e_median"] = {
            arm: round(statistics.median(v[arm]["delta_e_eV"] for v in both.values()), 3)
            for arm in ("S", "M")}

    payload = {"schema": "thermo-table-v1", "summary": args.summary, "stats": stats, "rows": rows}
    (out / "thermo_table.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1))
    with (out / "thermo_table.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else
                           ["channel", "arm", "case", "site", "vg", "barrier_eV",
                            "delta_e_eV", "delta_e_zpe_eV", "delta_g_eV"])
        w.writeheader()
        w.writerows(rows)

    md = ["# 通道热力学（ΔE = E_FS − E_IS，取各自臂「最优势垒位点」）", "",
          f"- 可用配对：**{stats['n_pairs']}**（{stats['channels']} 个通道）",
          f"- ΔE：min **{stats['delta_e']['min']}** / 中位 **{stats['delta_e']['median']}** / "
          f"max **{stats['delta_e']['max']}** eV；放热 {stats['delta_e']['exothermic']} / "
          f"近热中性 {stats['delta_e']['near_thermoneutral']} / 吸热 {stats['delta_e']['endothermic']}"]
    if "arm_delta_e_median" in stats:
        md.append(f"- 双臂 ΔE 中位：S **{stats['arm_delta_e_median']['S']}** / "
                  f"M **{stats['arm_delta_e_median']['M']}** eV")
    md += ["", "| 通道 | 臂 | site | Ea (eV) | ΔE (eV) | ΔE_ZPE | ΔG |", "|---|---|---|---|---|---|---|"]
    for ch in sorted(per_channel):
        for arm in ("S", "M"):
            r = per_channel[ch].get(arm)
            if r:
                md.append(f"| `{ch}` | {arm} | {r['site']} | {r['barrier_eV']} | {r['delta_e_eV']} | "
                          f"{r['delta_e_zpe_eV']} | {r['delta_g_eV']} |")
    (out / "thermo_table.md").write_text("\n".join(md) + "\n")
    print(f"[thermo] {stats['n_pairs']} pairs / {stats['channels']} channels -> "
          f"{out}/thermo_table.{{json,csv,md}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
