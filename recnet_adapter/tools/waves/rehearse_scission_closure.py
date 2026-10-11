#!/usr/bin/env python3
"""Zero-GPU rehearsal of a *scission-closed* universe (wave-6 decision material).

`closure_audit.py` closes the species table under **bonding** (radical-mediated combination).
`audit_dropped_cuts.py` measures the residue on the **scission** side: cuts whose fragment
graph is not in the table are dropped.  This tool iterates the other direction:

    repeat:
        enumerate channels over the current table
        for every dropped cut, materialise its missing fragment(s) as *synthetic* species
        (graph-level: symbols + intra-fragment bonds, anchor = most-unsaturated heavy atom)
    until no new missing identities appear (or the caps are hit)

and reports how the channel count grows per round, which of the original drops get resolved,
and what residue remains.  **The synthetic species are graph-level only** (no SMILES / 3D /
valence validation) ⇒ the counts are an *upper bound*; a real wave-6 still needs reviewed
`SpeciesSpec` entries (and may reject some fragments as non-chemical).

Zero GPU; rdkit environment (`recnet-prep`).

Usage::

    PYTHONPATH=Recnet /home/james/apps/miniforge3/envs/recnet-prep/bin/python \
        Recnet/recnet_adapter/tools/rehearse_scission_closure.py \
        --out-dir Recnet/network_inputs/scission-rehearsal
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
VALENCE = {"C": 4, "O": 2, "H": 1}


def _formula(symbols) -> str:
    counts = collections.Counter(symbols)
    order = [el for el in ("C", "H", "O") if el in counts] + \
            sorted(el for el in counts if el not in ("C", "H", "O"))
    return "".join(f"{el}{counts[el] if counts[el] > 1 else ''}" for el in order)


def _fragment_species(index: int, symbols, bonds, round_no: int):
    """Build a graph-level Species for one missing fragment (synthetic key/name)."""
    from network_builder.species import Species, identity_key

    degree = collections.Counter()
    for i, j, order in bonds:
        degree[i] += order
        degree[j] += order
    dangling = {i: max(0.0, VALENCE[symbols[i]] - degree[i]) for i in range(len(symbols))}
    heavy = [i for i, s in enumerate(symbols) if s != "H"]
    anchor = max(heavy or [0], key=lambda i: (dangling.get(i, 0.0), -i))
    counts = collections.Counter(symbols)
    # 2026-09-25: keys must be collision-free — an earlier version keyed on the identity tail and
    # collided (140 species / 114 unique keys), which corrupted the round-loop's parent lookup
    # and left 7 phantom residual drops.  Use a running index instead.
    key = f"SCIS{round_no}_{index:03d}"
    return Species(
        key=key, name=key, group="scission-rehearsal", smiles="", source="rehearsal",
        note="graph-level synthetic fragment (rehearsal only)",
        symbols=tuple(symbols),
        positions=tuple((0.0, 0.0, 0.0) for _ in symbols),
        ad_idx=(int(anchor),),
        bonds=tuple((int(i), int(j), float(o)) for i, j, o in bonds),
        identity=identity_key(symbols, [(i, j) for i, j, _ in bonds]),
        counts=dict(counts),
    )


def _verdict_of(parent, component):
    """rdkit verdict of one missing fragment, using the parent's own template coordinates."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from triage_dropped_fragments import _perceive

    return _perceive([parent.symbols[a] for a in component],
                     [parent.positions[a] for a in component])[0]


def rehearse(max_rounds: int = 6, max_species: int = 300, verdicts=None) -> dict:
    sys.path.insert(0, str(ROOT / "Recnet"))
    from network_builder.channels import enumerate_channels
    from network_builder.species import (SEED, build_species_table, fragment_identity,
                                         split_fragments)

    baseline = build_species_table(seed=SEED, expansion=True, expansion2=True,
                                   expansion3=True, expansion4=True)
    table = list(baseline)
    channels, dropped = enumerate_channels(table)
    baseline_channels = len(channels)
    baseline_drops = [(d["reactant"], tuple(sorted(d["broken_bond"])))
                      for d in dropped]
    baseline_drop_set = set(baseline_drops)

    rounds = []
    skipped_by_verdict = collections.Counter()
    seen_identities = {s.identity for s in table}
    counter = 0
    for round_no in range(1, max_rounds + 1):
        by_key = {s.key: s for s in table}
        added, resolved = {}, set()
        for drop in dropped:
            parent = by_key[drop["reactant"]]
            cut = tuple(sorted(drop["broken_bond"]))
            for component in split_fragments(parent.natoms, parent.edges, cut):
                fid = fragment_identity(parent.symbols, parent.edges, component)
                if fid in seen_identities:
                    continue
                if verdicts:
                    verdict = _verdict_of(parent, component)
                    if verdict not in verdicts:
                        skipped_by_verdict[verdict] += 1
                        continue
                local = {atom: idx for idx, atom in enumerate(component)}
                sub_bonds = [(local[i], local[j], order) for (i, j, order) in parent.bonds
                             if i in local and j in local]
                counter += 1
                species = _fragment_species(
                    counter, [parent.symbols[a] for a in component], sub_bonds, round_no)
                added[fid] = species
                resolved.add((parent.key, cut))
        if not added:
            rounds.append({"round": round_no, "added_species": 0, "channels": len(channels),
                           "dropped": len(dropped), "note": "no new fragments — closed"})
            break
        if len(table) + len(added) > max_species:
            rounds.append({"round": round_no, "added_species": len(added),
                           "channels": len(channels), "dropped": len(dropped),
                           "note": f"species cap {max_species} reached — stopped"})
            break
        table = table + list(added.values())
        seen_identities.update(added.keys())
        channels, dropped = enumerate_channels(table)
        rounds.append({
            "round": round_no,
            "added_species": len(added),
            "added_formulas": dict(collections.Counter(s.formula for s in added.values())),
            "channels": len(channels),
            "channels_new": len(channels) - baseline_channels if round_no == 1
            else None,
            "dropped": len(dropped),
            "resolved_original_cuts": len(resolved & baseline_drop_set),
        })
        if not dropped:
            break

    final_channels, final_dropped = channels, dropped
    report = {
        "schema": "ft2dp_scission_rehearsal/1",
        "baseline": {"species_entries": len(baseline), "channels": baseline_channels,
                     "dropped_cuts": len(baseline_drops)},
        "rounds": rounds,
        "final": {"species_entries": len(table), "channels": len(final_channels),
                  "dropped_cuts": len(final_dropped),
                  "channels_vs_baseline": len(final_channels) - baseline_channels},
        "verdict_filter": sorted(verdicts) if verdicts else None,
        "skipped_by_verdict": dict(skipped_by_verdict),
        "caveat": "synthetic species are graph-level only (no SMILES/3D/valence review); "
                  "counts are an upper bound for a reviewed wave-6.",
    }
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Rehearse a scission-closed species universe")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-rounds", type=int, default=6)
    parser.add_argument("--max-species", type=int, default=300)
    parser.add_argument("--verdicts", default="",
                        help="comma list of rdkit verdicts to admit (closed_shell,"
                             "zwitterionic,radical_or_open_shell); empty = admit every fragment")
    args = parser.parse_args(argv)

    verdicts = {v.strip() for v in args.verdicts.split(",") if v.strip()} or None
    report = rehearse(max_rounds=args.max_rounds, max_species=args.max_species, verdicts=verdicts)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "scission_rehearsal.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# Scission-closure rehearsal (zero-GPU upper bound)", ""]
    b = report["baseline"]
    lines.append(f"- baseline: {b['species_entries']} species entries / "
                 f"**{b['channels']} channels** / {b['dropped_cuts']} dropped cuts")
    for r in report["rounds"]:
        lines.append(f"- round {r['round']}: +{r['added_species']} species → "
                     f"channels **{r['channels']}** / dropped {r['dropped']}"
                     + (f" / resolved original cuts {r['resolved_original_cuts']}"
                        if "resolved_original_cuts" in r else "")
                     + (f" ({r['note']})" if "note" in r else ""))
    f = report["final"]
    lines.append(f"- final: {f['species_entries']} species entries / "
                 f"**{f['channels']} channels** (+{f['channels_vs_baseline']} vs baseline) / "
                 f"{f['dropped_cuts']} dropped")
    lines += ["", f"> caveat: {report['caveat']}", ""]
    (out / "scission_rehearsal.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[rehearsal] baseline {b['channels']} ch / {b['dropped_cuts']} drops → "
          f"final {f['channels']} ch (+{f['channels_vs_baseline']}) / {f['dropped_cuts']} drops")
    for r in report["rounds"]:
        print("   ", json.dumps(r, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
