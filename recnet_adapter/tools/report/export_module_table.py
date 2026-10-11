#!/usr/bin/env python3
"""Module-facing channel table: one row per channel with every caliber written out (R206).

The FTS module must not have to guess what a delivered number means, so this export flattens the
snapshot table together with the gate summary (site/vg detail) and the thermo side table into a
single machine-readable artifact whose README states the derivation rules.  It is a **derived
view**: the snapshot table remains the source of truth and its sha256 is carried in every output.

    export_module_table.py --table <snapshot>/network_channels_final.json \
        --summary <snapshot>/gate/network_summary.json \
        --thermo <snapshot>/thermo/thermo_table.json --out <dir>

Caliber fields per row (all explicitly named so nothing has to be inferred):

  * ``barrier_best_eV`` = min over arms of that arm's *best accepted site*; ``arm_of_best`` says
    which arm it came from; ``site_caliber`` records the site/vg and the **site dispersion**
    (worst − best over accepted sites of the same channel, the R197 reading caveat);
  * ``zpe_available`` says whether a ZPE-corrected barrier exists for that best site;
  * ``thermo_*`` are ΔE/ΔE_ZPE/ΔG at the same best site (per arm, from the side table);
  * ``slab_state`` = ``campaign-vacancy`` when the accepted sites come from the vg0/vg1 campaign
    (i.e. the C-vacancy slab; see the metastability warning) — the module must treat those numbers
    as the **vacancy branch**, not as clean-surface reactivity;
  * every evidence column is copied verbatim (``arbitrated``/``dft_closer_arm``/… ) so a consumer
    can filter on "has DFT evidence" without re-deriving anything.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def _profile_scalar(per_arm: dict | None, best_arm: str | None):
    """Profile barrier for the best arm, falling back to the lower arm when the campaign has none."""
    if not per_arm:
        return None
    if best_arm and per_arm.get(best_arm):
        return per_arm[best_arm]["barrier_profile_eV"]
    vals = [v["barrier_profile_eV"] for v in per_arm.values() if v.get("barrier_profile_eV") is not None]
    return min(vals) if vals else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Export the module-facing channel table")
    ap.add_argument("--table", required=True)
    ap.add_argument("--summary", action="append", default=None,
                    help="gate network_summary.json (repeatable — one per wave, merged for site/vg detail)")
    ap.add_argument("--thermo", action="append", default=None,
                    help="thermo_table.json (repeatable — rows merged, first occurrence wins)")
    ap.add_argument("--profile", default=None,
                    help="profile_barriers.json — adds the constrained-scan barrier columns (R213)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    table_path = Path(args.table)
    rows_in = json.loads(table_path.read_text())["channels"]
    merged_channels: dict = {}
    for sp in (args.summary or []):
        p = Path(sp)
        if not p.exists():
            continue
        for chan, cases in (json.loads(p.read_text()).get("channels") or {}).items():
            merged_channels.setdefault(chan, {}).update(cases)
    summary = {"channels": merged_channels}
    thermo_rows, seen_thermo = [], set()
    for tp in (args.thermo or []):
        p = Path(tp)
        if not p.exists():
            continue
        for r in json.loads(p.read_text()).get("rows", []):
            key = (r.get("channel"), r.get("arm"))
            if r.get("arm") is not None and key in seen_thermo:
                continue
            seen_thermo.add(key)
            thermo_rows.append(r)
    thermo = {"rows": thermo_rows}
    prof = {}
    if args.profile and Path(args.profile).exists():
        for r in json.loads(Path(args.profile).read_text()).get("rows", []):
            if r.get("channel"):
                prof.setdefault(r["channel"], {}).setdefault(r["arm"], r)
    thermo_by = {(r["channel"], r["arm"]): r for r in thermo.get("rows", [])}

    rows = []
    for c in rows_in:
        chan = c["channel"]
        # site detail from the gate summary (all cases/arms of this channel)
        sites = []
        for case, rec in (summary["channels"].get(chan) or {}).items():
            arm = "S" if case.startswith("S-") else "M" if case.startswith("M-") else "?"
            for e in rec.get("sites", []):
                if e.get("accepted") and e.get("barrier") is not None:
                    sites.append({"arm": arm, "case": case, "site": e.get("site"), "vg": str(e.get("vg")),
                                  "barrier": e["barrier"], "barrier_zpe": e.get("barrier_zpe")})
        s_vals = [s["barrier"] for s in sites if s["arm"] == "S"]
        m_vals = [s["barrier"] for s in sites if s["arm"] == "M"]
        best_arm = None
        if c.get("barrier_S_best_eV") is not None and c.get("barrier_M_best_eV") is not None:
            best_arm = "S" if c["barrier_S_best_eV"] <= c["barrier_M_best_eV"] else "M"
        elif c.get("barrier_S_best_eV") is not None:
            best_arm = "S"
        elif c.get("barrier_M_best_eV") is not None:
            best_arm = "M"
        best_site = next((s for s in sorted(sites, key=lambda s: s["barrier"])
                          if s["arm"] == best_arm), None)
        disp = round(max(s["barrier"] for s in sites) - min(s["barrier"] for s in sites), 3) if sites else None
        t = thermo_by.get((chan, best_arm)) if best_arm else None
        rows.append({
            "channel": chan,
            "wave": c.get("wave"),
            "qc_status": c.get("qc_status"),
            "barrier_best_eV": c.get("barrier_best_eV"),
            "arm_of_best": best_arm,
            # R252: `barrier_best_eV` stays empty when one arm has no accepted site, which is
            # easy to misread as "no barrier known".  This column makes the shape explicit
            # (2 = both arms, 1 = single-arm barrier only, 0 = none).
            "n_arms_with_barrier": sum(
                1 for v in (c.get("barrier_S_best_eV"), c.get("barrier_M_best_eV"))
                if v is not None),
            "barrier_S_best_eV": c.get("barrier_S_best_eV"),
            "barrier_M_best_eV": c.get("barrier_M_best_eV"),
            "ddE_best_M_minus_S_eV": c.get("ddE_best_M_minus_S_eV"),
            "arm_sensitive": bool(c.get("arm_sensitive")),
            "site_caliber": None if best_site is None else {
                "site": best_site["site"], "vg": best_site["vg"], "case": best_site["case"],
                "n_accepted_sites": len(sites),
                "site_spread_eV": disp,
                "zpe_available": best_site.get("barrier_zpe") is not None,
            },
            "slab_state": "campaign-vacancy" if sites else None,
            "thermo_delta_e_eV": None if t is None else t.get("delta_e_eV"),
            "thermo_delta_e_zpe_eV": None if t is None else t.get("delta_e_zpe_eV"),
            "thermo_delta_g_eV": None if t is None else t.get("delta_g_eV"),
            # R213: constrained-scan barriers.  Per-arm columns are always written (so a channel with
            # no campaign TS still exposes both arms); the scalar falls back to the lower arm.
            "profile_barrier_S_eV": (prof.get(chan, {}).get("S") or {}).get("barrier_profile_eV"),
            "profile_barrier_M_eV": (prof.get(chan, {}).get("M") or {}).get("barrier_profile_eV"),
            "profile_barrier_eV": _profile_scalar(prof.get(chan), best_arm),
            "profile_arm": best_arm if prof.get(chan, {}).get(best_arm) else
                           (min(prof[chan], key=lambda a: prof[chan][a].get("barrier_profile_eV") or 1e9)
                            if chan in prof else None),
            "arbitrated": bool(c.get("arbitrated")),
            "dft_closer_arm": c.get("dft_closer_arm"),
            "arb_err_S_eV": c.get("arb_err_S_eV"),
            "arb_err_M_eV": c.get("arb_err_M_eV"),
            "dft_pairs": c.get("dft_pairs"),
            "dft_profile_arms": c.get("dft_profile_arms"),
        })

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "module-channel-table-v1",
        "source_table": str(table_path),
        "source_table_sha256": hashlib.sha256(table_path.read_bytes()).hexdigest(),
        "source_summary": args.summary,
        "source_thermo": args.thermo,
        "n_channels": len(rows),
        "caliber": [
            "barrier_best_eV = min(best accepted site of S, best accepted site of M); arm_of_best names it",
            "n_arms_with_barrier = 2/1/0：当某臂没有接受位点时 barrier_best_eV 留空（不是零、也不是"
            "「无势垒」）——此时以 barrier_<arm>_best_eV 为准，arm_of_best 指有势垒的那一臂；"
            "约束扫描（profile_*）可为缺臂提供参考值但属另一口径",
            "site_caliber.site_spread_eV = max−min accepted-site barrier of the same channel (R197 caveat)",
            "slab_state=campaign-vacancy ⇒ numbers are the C-vacancy branch (metastability warning)",
            "thermo_* are ΔE/ΔE_ZPE/ΔG at the same best site (per arm) — arms are not comparable",
            "profile_barrier_eV = constrained-scan estimate (0.1 Å steps) of the arm named by "
            "profile_arm (per-arm columns profile_barrier_S/M_eV are always written); a different "
            "caliber from barrier_best_eV, present only for the six boundary-profile channels — "
            "notably CO* -> C* O*, whose campaign columns are empty on the S arm",
            "join key: channel labels carry '*' (adsorbed) and '(g)' (gas) while the structure "
            "library/adsorbate names do not — strip those suffixes before joining to "
            "adsorbate_library.json (verified: 87/87 species of this table are covered)",
        ],
        "rows": rows,
    }
    (out / "module_table.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1))
    with (out / "module_table.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["channel", "wave", "qc_status", "barrier_best_eV", "arm_of_best",
                    "n_arms_with_barrier",
                    "barrier_S_best_eV", "barrier_M_best_eV", "ddE_best_M_minus_S_eV", "arm_sensitive",
                    "site", "vg", "n_accepted_sites", "site_spread_eV", "zpe_available", "slab_state",
                    "thermo_delta_e_eV", "thermo_delta_e_zpe_eV", "thermo_delta_g_eV",
                    "arbitrated", "dft_closer_arm", "arb_err_S_eV", "arb_err_M_eV", "dft_pairs",
                    "dft_profile_arms"])
        for r in rows:
            sc = r["site_caliber"] or {}
            w.writerow([r["channel"], r["wave"], r["qc_status"], r["barrier_best_eV"], r["arm_of_best"],
                        r["n_arms_with_barrier"],
                        r["barrier_S_best_eV"], r["barrier_M_best_eV"], r["ddE_best_M_minus_S_eV"],
                        r["arm_sensitive"], sc.get("site"), sc.get("vg"), sc.get("n_accepted_sites"),
                        sc.get("site_spread_eV"), sc.get("zpe_available"), r["slab_state"],
                        r["thermo_delta_e_eV"], r["thermo_delta_e_zpe_eV"], r["thermo_delta_g_eV"],
                        r["arbitrated"], r["dft_closer_arm"], r["arb_err_S_eV"], r["arb_err_M_eV"],
                        r["dft_pairs"], r["dft_profile_arms"]])
    md = ["# 模块面向的通道表（module_table）", "",
          f"- 源：`{table_path}`（sha256 `{payload['source_table_sha256'][:16]}…`）；"
          f"gate `{args.summary}`；thermo `{args.thermo}`",
          f"- 行数 **{len(rows)}**；列定义（口径）——", ""]
    md += [f"  - {x}" for x in payload["caliber"]]
    (out / "README.md").write_text("\n".join(md) + "\n")

    # ---- species_branches: what each intermediate can do (selectivity-relevant cut) ----------
    # Derived from the same rows — no new facts.  "Outgoing" channels are the scission channels
    # whose reactant is that species; the module needs this to reason about branching.
    branches: dict[str, dict] = {}
    for r in rows:
        lhs = r["channel"].split(" -> ")[0]
        b = branches.setdefault(lhs, {"species": lhs, "n_out": 0, "out": [],
                                      "min_barrier_eV": None, "n_as_product": 0})
        b["n_out"] += 1
        b["out"].append({"channel": r["channel"], "barrier_best_eV": r["barrier_best_eV"],
                         "arm_of_best": r["arm_of_best"], "qc_status": r["qc_status"],
                         "thermo_delta_e_eV": r["thermo_delta_e_eV"],
                         "site_spread_eV": (r["site_caliber"] or {}).get("site_spread_eV")})
        if r["barrier_best_eV"] is not None:
            cur = b["min_barrier_eV"]
            b["min_barrier_eV"] = r["barrier_best_eV"] if cur is None else min(cur, r["barrier_best_eV"])
    for r in rows:
        for prod in r["channel"].split(" -> ", 1)[1].split():
            if prod in branches:
                branches[prod]["n_as_product"] += 1
    for b in branches.values():
        b["out"].sort(key=lambda e: (e["barrier_best_eV"] is None, e["barrier_best_eV"]))
    (out / "species_branches.json").write_text(
        json.dumps({"schema": "species-branches-v1",
                    "note": ("per-species outgoing channels sorted by barrier; derived from the "
                             "same module rows (see module_table.json). 'min_barrier_eV' is the "
                             "kinetic escape barrier used in network_analysis §7."),
                    "species": [branches[k] for k in sorted(branches)]},
                   ensure_ascii=False, indent=1))
    with (out / "species_branches.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["species", "n_out", "n_as_product", "min_barrier_eV", "channel",
                    "barrier_best_eV", "arm_of_best", "qc_status", "thermo_delta_e_eV", "site_spread_eV"])
        for b in (branches[k] for k in sorted(branches)):
            for e in b["out"]:
                w.writerow([b["species"], b["n_out"], b["n_as_product"], b["min_barrier_eV"],
                            e["channel"], e["barrier_best_eV"], e["arm_of_best"], e["qc_status"],
                            e["thermo_delta_e_eV"], e["site_spread_eV"]])
    print(f"[module-table] species_branches: {len(branches)} species -> "
          f"{out}/species_branches.{{json,csv}}")
    print(f"[module-table] {len(rows)} channels -> {out}/module_table.{{json,csv}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
