#!/usr/bin/env python3
"""ABACUS cross-validation cases from a RecNet campaign (2026-09-22).

For the closure-expanded network (``c2r2``) the verification flow is the same one
used for the frozen 42-channel network: take a model-accepted transition state,
recompute ``IS`` and ``TS`` with ABACUS at the production DFT settings, and compare
``E_DFT(TS) - E_DFT(IS)`` with the model barrier
``barrier_ts_minus_is`` recorded by ``handlers/energy.py``.

Inputs are already-produced artifacts::

    <case>/rxn/FS_energy/final_state_energy_vg*.yaml   # e_is / e_ts / barrier_ts_minus_is
    <case>/rxn/TS_archive/ts_records_vg*.yaml          # tag -> initial.xyz / ts.xyz

Output: an ABACUS batch cases file (``<label> <structure> <frame>`` lines, two per
pair) plus a JSON summary carrying the model barrier for each pair, so the DFT
comparison can be rebuilt without re-deriving anything.

Usage::

    build_abacus_cases_from_campaign.py --campaign-root $R/recnet-runs/c2r2-campaign \\
        --species sp_029 --species sp_030 --max-pairs 12 \\
        --out abacus_batch_cases_xval.txt --summary abacus_xval_summary.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import yaml

ARM_TOKEN = {"S": "single100k", "M": "multi200k"}


def _clean(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(text).replace("*", "")).strip("_")


def load_case(case: Path):
    """Return {(tag): record} for FS energies and TS archive entries."""
    fs = {}
    for path in sorted((case / "rxn" / "FS_energy").glob("final_state_energy_vg*.yaml")):
        payload = yaml.safe_load(path.read_text()) or {}
        for record in payload.get("final_state_energy", []) or []:
            fs[record.get("tag")] = record
    ts_records = {}
    for path in sorted((case / "rxn" / "TS_archive").glob("ts_records_vg*.yaml")):
        payload = yaml.safe_load(path.read_text()) or {}
        for record in payload.get("ts_records", []) or []:
            ts_records[record.get("tag")] = record
    return fs, ts_records


def arm_of_case(case: Path) -> str:
    name = case.name
    for arm in ("S", "M"):
        if name.startswith(f"{arm}-"):
            return arm
    return "?"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="ABACUS cross-validation cases from a campaign")
    ap.add_argument("--campaign-root", action="append", required=True)
    ap.add_argument("--species", action="append", default=None,
                    help="species keys to include (default: every expansion species sp_029+)")
    ap.add_argument("--max-pairs", type=int, default=12,
                    help="maximum number of (case, tag) IS/TS pairs to emit")
    ap.add_argument("--out", required=True, help="ABACUS cases file")
    ap.add_argument("--summary", required=True, help="JSON summary with model barriers")
    args = ap.parse_args(argv)

    cases = []
    for root in args.campaign_root:
        cases += sorted(p for p in Path(root).iterdir()
                        if p.is_dir() and (p / "rxn").is_dir())

    wanted = None if args.species is None else {s.strip() for s in args.species if s.strip()}
    candidates = []
    for case in cases:
        fs, ts_records = load_case(case)
        for tag, energy in fs.items():
            barrier = energy.get("barrier_ts_minus_is")
            ts_rec = ts_records.get(tag)
            if barrier is None or ts_rec is None:
                continue
            species = ts_rec.get("species")
            if wanted is not None and species not in wanted:
                continue
            is_xyz, ts_xyz = ts_rec.get("initial_xyz"), ts_rec.get("ts_xyz")
            if not is_xyz or not ts_xyz or not Path(is_xyz).exists() or not Path(ts_xyz).exists():
                continue
            candidates.append({
                "case": case.name,
                "arm": arm_of_case(case),
                "tag": tag,
                "species": species,
                "rxn_idx": ts_rec.get("rxn_idx"),
                "site": ts_rec.get("site"),
                "model_barrier_eV": round(float(barrier), 4),
                "is_xyz": is_xyz,
                "ts_xyz": ts_xyz,
            })

    # spread the sample over channels: one record per (species, rxn) group in
    # round-robin, lowest barrier first inside each group. Sorting purely by
    # (species, barrier) used to fill the quota from the first two species only,
    # which under-samples the expanded network.
    candidates.sort(key=lambda r: (str(r["species"]), r["model_barrier_eV"]))
    groups: dict = {}
    for record in candidates:
        groups.setdefault((record["species"], record["rxn_idx"]), []).append(record)
    order = sorted(groups)
    selected, seen = [], set()
    round_index = 0
    while len(selected) < args.max_pairs:
        progressed = False
        for key in order:
            bucket = groups[key]
            if round_index >= len(bucket):
                continue
            record = bucket[round_index]
            label_key = (record["case"], record["tag"])
            if label_key in seen:
                continue
            seen.add(label_key)
            selected.append(record)
            progressed = True
            if len(selected) >= args.max_pairs:
                break
        if not progressed:
            break
        round_index += 1

    lines = []
    for record in selected:
        label = (f"XVAL_{_clean(record['species'])}_{record['arm']}"
                 f"_rxn{record['rxn_idx']}_site{record['site']}")
        record["label"] = label
        lines.append(f"{label}_IS {record['is_xyz']} -1")
        lines.append(f"{label}_TS {record['ts_xyz']} -1")

    Path(args.out).write_text("\n".join(lines) + "\n" if lines else "")
    Path(args.summary).write_text(json.dumps({
        "schema": "ft2dp_campaign_xval/1",
        "campaign_roots": args.campaign_root,
        "species_filter": sorted(wanted) if wanted else None,
        "candidates": len(candidates),
        "selected": selected,
    }, ensure_ascii=False, indent=1) + "\n")

    print(f"candidates={len(candidates)} selected={len(selected)} cases={len(lines)}")
    for record in selected:
        print(f"  {record['case']:6s} {record['species']:8s} {record['arm']} "
              f"rxn{record['rxn_idx']:<3} site{record['site']:<3} "
              f"model barrier {record['model_barrier_eV']:.3f} eV")
    print(f"wrote {args.out} and {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
