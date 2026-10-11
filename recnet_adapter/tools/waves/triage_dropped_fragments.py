#!/usr/bin/env python3
"""Triage the 32 missing fragment identities behind the 76 dropped cuts (wave-6 ①b material).

R224 measured the residue, R225 bounded the full scission closure (+46 species / +244 channels,
graph-level).  This tool turns the remaining "needs a chemistry review" into a per-fragment table:

for every unique missing fragment identity (deduplicated from the drops) it reports
  * formula, heavy-atom skeleton and indicative dangling electrons,
  * how many dropped cuts it blocks,
  * rdkit's neutral bond perception on the *parent's own 3D template coordinates*
    (``rdDetermineBonds(charge=0)``) → canonical SMILES when a charge-neutral assignment exists,
  * a disposition hint: ``closed_shell`` (clean neutral molecule, like CO2) /
    ``zwitterionic`` (charge-separated neutral form) / ``radical_or_open_shell``.

Nothing is submitted and no species is added — this is review material for the ①b decision.
Zero GPU; rdkit environment (`recnet-prep`).

Usage::

    PYTHONPATH=Recnet /home/james/apps/miniforge3/envs/recnet-prep/bin/python \
        Recnet/recnet_adapter/tools/triage_dropped_fragments.py \
        --out-dir Recnet/network_inputs/audit-dropped-cuts
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


def _perceive(symbols, coords):
    """(verdict, smiles) — rdkit neutral bond perception on the fragment's own coordinates."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdDetermineBonds

    RDLogger.DisableLog("rdApp.*")
    rw = Chem.RWMol()
    for symbol in symbols:
        rw.AddAtom(Chem.Atom(symbol))
    conf = Chem.Conformer(len(symbols))
    for i, (x, y, z) in enumerate(coords):
        conf.SetAtomPosition(i, (float(x), float(y), float(z)))
    mol = rw.GetMol()
    mol.AddConformer(conf)
    try:
        rdDetermineBonds.DetermineBonds(mol, charge=0)
        Chem.SanitizeMol(mol)
    except Exception:  # noqa: BLE001
        return "radical_or_open_shell", ""
    smiles = Chem.MolToSmiles(Chem.RemoveHs(mol))
    # 2026-09-25: radicals/carbenes must not be labelled closed-shell — count the perceived
    # unpaired electrons as well (an earlier version only looked for explicit + / - charges and
    # mislabelled `[CH]O[CH]` as closed-shell).
    n_radicals = sum(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms())
    if n_radicals:
        verdict = "radical_or_open_shell"
    elif "+" in smiles and "-" in smiles:
        verdict = "zwitterionic"
    else:
        verdict = "closed_shell"
    return verdict, smiles


def triage() -> dict:
    sys.path.insert(0, str(ROOT / "Recnet"))
    from network_builder.channels import enumerate_channels, preferred_entries
    from network_builder.species import (SEED, build_species_table, fragment_identity,
                                         split_fragments)

    species = build_species_table(seed=SEED, expansion=True, expansion2=True,
                                  expansion3=True, expansion4=True, expansion5=True)
    channels, dropped = enumerate_channels(species)
    by_key = {s.key: s for s in species}
    identity_to_key = {e.identity: e.key for e in preferred_entries(species).values()}

    blocked = collections.Counter()
    examples = collections.defaultdict(list)
    fragments = {}
    for drop in dropped:
        parent = by_key[drop["reactant"]]
        cut = tuple(sorted(drop["broken_bond"]))
        for component in split_fragments(parent.natoms, parent.edges, cut):
            fid = fragment_identity(parent.symbols, parent.edges, component)
            if fid in identity_to_key:
                continue
            blocked[fid] += 1
            if len(examples[fid]) < 3:
                examples[fid].append(f"{parent.key}:{list(cut)}")
            if fid not in fragments:
                symbols = [parent.symbols[a] for a in component]
                coords = [parent.positions[a] for a in component]
                fragments[fid] = {"symbols": symbols, "coords": coords}

    rows = []
    for fid, info in fragments.items():
        verdict, smiles = _perceive(info["symbols"], info["coords"])
        rows.append({
            "formula": _formula(info["symbols"]),
            "verdict": verdict,
            "smiles": smiles,
            "blocked_cuts": blocked[fid],
            "examples": examples[fid],
            "identity": fid,
        })
    rows.sort(key=lambda r: (-r["blocked_cuts"], r["formula"]))
    return {
        "schema": "ft2dp_dropped_fragment_triage/1",
        "dropped_cuts": len(dropped),
        "unique_fragments": len(rows),
        "verdict_counts": dict(collections.Counter(r["verdict"] for r in rows)),
        "rows": rows,
    }


def _markdown(report: dict) -> str:
    lines = [
        "# Dropped-fragment triage (wave-6 ①b review material)",
        "",
        f"- dropped cuts: **{report['dropped_cuts']}** over **{report['unique_fragments']}** "
        f"unique missing fragment identities",
        f"- verdicts: {report['verdict_counts']}",
        "",
        "| formula | verdict | rdkit neutral SMILES | blocked cuts | examples |",
        "|---|---|---|---|---|",
    ]
    for r in report["rows"]:
        lines.append(f"| `{r['formula']}` | {r['verdict']} | "
                     f"{('`' + r['smiles'] + '`') if r['smiles'] else '—'} | {r['blocked_cuts']} | "
                     f"{', '.join(r['examples'])} |")
    lines += ["", "> `closed_shell` = charge-neutral molecule on the parent's own template "
                  "geometry (like CO2); `zwitterionic` = only a charge-separated neutral form; "
                  "`radical_or_open_shell` = no neutral closed assignment (may still be a valid "
                  "adsorbate under the closure's dangling<=2 rule — needs human review).", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Triage the missing scission fragments")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)

    report = triage()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "fragment_triage.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "fragment_triage.md").write_text(_markdown(report), encoding="utf-8")
    print(f"[triage] fragments={report['unique_fragments']} verdicts={report['verdict_counts']}")
    for r in report["rows"][:12]:
        print(f"   {r['formula']:9s} {r['verdict']:22s} cuts={r['blocked_cuts']:2d} "
              f"{r['smiles'][:40] if r['smiles'] else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
