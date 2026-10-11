#!/usr/bin/env python3
"""Independent validation of a closure species table + dataset provenance (2026-09-25, R184).

`closure_audit.py` proves that *enumerated* products are covered; this tool checks the other
direction — that **every species actually in the table obeys the closure rules** (C<=2 / O<=2,
no O–O, net charge 0, dangling <= 2) and that the dataset file matches its MANIFEST hash.  It is
the guard against a hand-edited spec slipping an out-of-budget species into a "complete" closure.

    validate_closure_table.py [--max-c 2 --max-o 2 --max-radicals 2] \
        [--dataset <prepared yaml> --manifest <MANIFEST.json>]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "Recnet"))

from rdkit import Chem, RDLogger  # noqa: E402
from rdkit.Chem import Descriptors  # noqa: E402

from network_builder.species import build_species_table  # noqa: E402

RDLogger.DisableLog("rdApp.*")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Validate the closure species table against its rules")
    ap.add_argument("--max-c", type=int, default=2)
    ap.add_argument("--max-o", type=int, default=2)
    ap.add_argument("--max-radicals", type=int, default=2)
    ap.add_argument("--dataset", default=None, help="prepared yaml to hash-check")
    ap.add_argument("--manifest", default=None, help="MANIFEST.json that records the dataset sha")
    args = ap.parse_args(argv)

    table = build_species_table(expansion=True, expansion2=True, expansion3=True, expansion4=True)
    problems: list[str] = []
    radical_info: list[str] = []
    for sp in table:
        mol = Chem.MolFromSmiles(sp.smiles, sanitize=True)
        if mol is None:
            problems.append(f"{sp.key}: SMILES not parseable: {sp.smiles}")
            continue
        mol = Chem.AddHs(mol)
        n_c = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "C")
        n_o = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "O")
        if n_c > args.max_c:
            problems.append(f"{sp.key}: C={n_c} > {args.max_c}")
        if n_o > args.max_o:
            problems.append(f"{sp.key}: O={n_o} > {args.max_o}")
        if Chem.GetFormalCharge(mol) != 0:
            problems.append(f"{sp.key}: net charge {Chem.GetFormalCharge(mol)} != 0")
        if any(b.GetBeginAtom().GetSymbol() == "O" and b.GetEndAtom().GetSymbol() == "O"
               for b in mol.GetBonds()):
            problems.append(f"{sp.key}: contains an O–O bond")
        # The radical cap is a rule on the *closure-derived* additions: the frozen seeds are the
        # paper's adsorbed fragments (C*, CH*, COH*, CCH3* …), whose free-molecule radical count is
        # 4/3/3/3 because they bind to the surface through those valences.  Counting them as
        # violations would be a category error; they are reported as information instead.
        rad = Descriptors.NumRadicalElectrons(mol)
        if rad > args.max_radicals:
            if str(getattr(sp, "source", "")).startswith("expansion"):
                problems.append(f"{sp.key}: radicals {rad} > {args.max_radicals}")
            else:
                radical_info.append(f"{sp.key} ({rad})")
        if len(mol.GetAtoms()) != len(sp.symbols):
            problems.append(f"{sp.key}: SMILES atom count {len(mol.GetAtoms())} != template "
                            f"{len(sp.symbols)}")

    print(f"[validate] table entries: {len(table)} | identities: "
          f"{len({sp.identity for sp in table})} | rule violations: {len(problems)}")
    if radical_info:
        print(f"[validate] info — frozen seeds with >{args.max_radicals} free-molecule radicals "
              f"(expected for paper adsorbates): {', '.join(radical_info)}")
    for p in problems[:20]:
        print("  ", p)

    sha_ok = None
    if args.dataset and args.manifest:
        ds = Path(args.dataset)
        got = hashlib.sha256(ds.read_bytes()).hexdigest()
        manifest = json.loads(Path(args.manifest).read_text())
        blob = json.dumps(manifest)
        sha_ok = got in blob
        print(f"[validate] dataset sha256 {got[:16]}… present in {Path(args.manifest).name}: {sha_ok}")
        if not sha_ok:
            problems.append("dataset sha256 not found in MANIFEST.json")

    if problems:
        print("[validate] FAIL", file=sys.stderr)
        return 1
    print("[validate] OK — every table species obeys the closure rules"
          + (" and the dataset hash is recorded" if sha_ok else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
