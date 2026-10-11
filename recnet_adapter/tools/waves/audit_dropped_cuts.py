#!/usr/bin/env python3
"""Dropped-cut audit: which single-bond scissions the generator cannot resolve, and why.

The generator (`network_builder.channels.enumerate_channels`) enumerates one-bond scission
channels for every species of the closure table and **transparently records** every cut whose
fragment identity is not in that table (`dropped_candidates` in the dataset MANIFEST).  Those
dropped cuts are the residue of the "全面" claim: the closure was closed under *bonding*
(`closure_audit.py`, radical-mediated combination), **not** under *scission*.

This tool re-derives the drops and classifies each missing fragment:

* formula (C/H/O), heavy-atom skeleton and an indicative dangling-electron count
  (valence deficit computed from the parent bond orders);
* whether the fragment is inside the closure element budget (C<=2, O<=2, no O-O bond);
* aggregates per reactant and per formula, so the "how big is the hole" question has numbers.

Zero GPU; rdkit environment (`recnet-prep`).

Usage::

    PYTHONPATH=Recnet /home/james/apps/miniforge3/envs/recnet-prep/bin/python \
        Recnet/recnet_adapter/tools/audit_dropped_cuts.py --out-dir Recnet/network_inputs/audit-dropped-cuts
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


def _fragment_stats(species, component):
    """(symbols, formula, dangling electrons, has O-O) for one fragment of ``species``."""
    local = {atom: idx for idx, atom in enumerate(component)}
    symbols = [species.symbols[atom] for atom in component]
    edges = [(local[i], local[j], order) for (i, j, order) in species.bonds
             if i in local and j in local]
    degree = collections.Counter()
    for i, j, order in edges:
        degree[i] += order
        degree[j] += order
    dangling = int(round(sum(max(0.0, VALENCE[symbols[i]] - degree[i])
                             for i in range(len(symbols)))))
    has_oo = any(symbols[i] == "O" and symbols[j] == "O" for i, j, _ in edges)
    return symbols, _formula(symbols), dangling, has_oo


def audit() -> dict:
    sys.path.insert(0, str(ROOT / "Recnet"))
    from network_builder.channels import enumerate_channels, preferred_entries
    from network_builder.species import (SEED, build_species_table, fragment_identity,
                                         split_fragments)

    species = build_species_table(seed=SEED, expansion=True, expansion2=True,
                                  expansion3=True, expansion4=True)
    channels, dropped = enumerate_channels(species)
    by_key = {s.key: s for s in species}
    preferred = preferred_entries(species)
    identity_to_key = {entry.identity: entry.key for entry in preferred.values()}
    identity_all = collections.defaultdict(list)
    for s in species:
        identity_all[s.identity].append(s.key)

    rows = []
    nonpreferred = []
    for drop in dropped:
        parent = by_key[drop["reactant"]]
        cut = tuple(sorted(drop["broken_bond"]))
        for component in split_fragments(parent.natoms, parent.edges, cut):
            fid = fragment_identity(parent.symbols, parent.edges, component)
            if fid in identity_to_key:
                continue
            if fid in identity_all:
                nonpreferred.append({"reactant": parent.key, "cut": list(cut),
                                     "identities": identity_all[fid]})
                continue
            _, formula, dangling, has_oo = _fragment_stats(parent, component)
            counts = collections.Counter(formula)
            n_c, n_o = counts.get("C", 0), counts.get("O", 0)
            in_budget = (n_c <= 2 and n_o <= 2 and not has_oo)
            rows.append({
                "reactant": parent.key, "cut": list(cut), "fragment": formula,
                "dangling_electrons": dangling, "has_O_O": has_oo,
                "within_element_budget": in_budget,
                "identity": fid,
            })

    by_reactant = collections.Counter(r["reactant"] for r in rows)
    by_fragment = collections.Counter(r["fragment"] for r in rows)
    report = {
        "schema": "ft2dp_dropped_cut_audit/1",
        "channels": len(channels),
        "dropped_cuts": len(dropped),
        "missing_fragment_rows": len(rows),
        "nonpreferred_identity_rows": len(nonpreferred),
        "within_element_budget": sum(1 for r in rows if r["within_element_budget"]),
        "max_dangling_electrons": max((r["dangling_electrons"] for r in rows), default=0),
        "by_reactant": dict(by_reactant.most_common()),
        "by_fragment_formula": dict(by_fragment.most_common()),
        "rows": rows,
        "nonpreferred": nonpreferred,
    }
    return report


def _markdown(report: dict) -> str:
    lines = [
        "# Dropped-cut audit (scission residue of the closure universe)",
        "",
        f"- channels enumerated: **{report['channels']}**",
        f"- dropped cuts (transparently recorded): **{report['dropped_cuts']}**",
        f"- missing-fragment rows: **{report['missing_fragment_rows']}** "
        f"(within C/O element budget: **{report['within_element_budget']}**)",
        f"- non-preferred-identity drops (resolution-policy, not missing species): "
        f"**{report['nonpreferred_identity_rows']}**",
        f"- max indicative dangling electrons: {report['max_dangling_electrons']}",
        "",
        "## By reactant",
        "",
        "| reactant | dropped cuts |",
        "|---|---|",
    ]
    for name, n in report["by_reactant"].items():
        lines.append(f"| `{name}` | {n} |")
    lines += ["", "## By missing-fragment formula", "", "| formula | rows |", "|---|---|"]
    for formula, n in report["by_fragment_formula"].items():
        lines.append(f"| `{formula}` | {n} |")
    lines += ["", "## Rows", "",
              "| reactant | cut | fragment | dangling e⁻ | O–O | in element budget |",
              "|---|---|---|---|---|---|"]
    for r in report["rows"]:
        lines.append(f"| `{r['reactant']}` | {r['cut']} | `{r['fragment']}` | "
                     f"{r['dangling_electrons']} | {'yes' if r['has_O_O'] else 'no'} | "
                     f"{'yes' if r['within_element_budget'] else 'no'} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Audit the generator's dropped single-bond cuts")
    parser.add_argument("--out-dir", required=True, help="write dropped_cuts.{json,md} here")
    parser.add_argument("--json-only", action="store_true")
    args = parser.parse_args(argv)

    report = audit()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "dropped_cuts.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not args.json_only:
        (out / "dropped_cuts.md").write_text(_markdown(report), encoding="utf-8")
    print(f"[drops] channels={report['channels']} dropped_cuts={report['dropped_cuts']} "
          f"missing_rows={report['missing_fragment_rows']} "
          f"within_budget={report['within_element_budget']} "
          f"nonpreferred={report['nonpreferred_identity_rows']}")
    print(f"[drops] wrote {out}/dropped_cuts.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
