#!/usr/bin/env python3
"""Prepare a wave-3 dataset + the subset of *new* channels, for one of the frontier options.

Options (see ``Recnet/network_inputs/expansion_frontier.md``):

  b1 : the four species with audit-ready >= 5
  b2 : the nine species with audit-ready >= 4
  d  : chemistry filter — carbonyl/acid/ester/acyl + hydroxy-alkyl/enol families
       (excludes the O-rich hydroxy/diol/ether family)
  c  : all 30 frontier species

The tool
  1. filters ``network_builder.expansion3.EXPANSION3_SPECS`` accordingly and rebuilds the
     prepared dataset (frozen 29 + wave-1 + wave-2 + the chosen wave-3 subset),
  2. diffs the resulting channel labels against the delivered 138-channel table so only the
     **new** channels go into the campaign subset,
  3. writes ``<out>/prepared_rmg_data.yaml`` (+ ``ads_templates/``) for the campaign.

Nothing here touches the GPU or the cluster: it is the zero-cost half of the launch; the
campaign itself is built with ``build_campaign.py --prepared <out>/prepared_rmg_data.yaml``.

Usage:
  PYTHONPATH=Recnet python recnet_adapter/tools/prepare_wave3.py --option d \\
      --table validation-pipeline/summary/c2r2-network-20260922/network_channels_final.json \\
      --out ~/scratch/wave3-d
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import yaml
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[4]
NI = ROOT / "Recnet" / "network_inputs"


def family_is_chem(smiles: str) -> bool:
    """True for the carbonyl/acid/ester/acyl and hydroxy-alkyl/enol families (option d)."""
    mol = Chem.MolFromSmiles(smiles)
    has_co = any(
        b.GetBondTypeAsDouble() == 2
        and {b.GetBeginAtom().GetSymbol(), b.GetEndAtom().GetSymbol()} == {"C", "O"}
        for b in mol.GetBonds()
    )
    n_o = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "O")
    n_c = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "C")
    o_rich = (not has_co) and (n_o >= 2 or (n_o >= 1 and n_c == 1))
    return not o_rich


def main() -> int:
    ap = argparse.ArgumentParser(description="Prepare a wave-3 dataset + new-channel subset")
    ap.add_argument("--option", required=True, choices=("b1", "b2", "d", "c"))
    ap.add_argument("--table", required=True, help="delivered network_channels_final.json")
    ap.add_argument("--out", required=True, help="output dir for the campaign subset")
    ap.add_argument("--workdir", default=None, help="dir for the full dataset (default <out>/dataset)")
    args = ap.parse_args()

    sys.path.insert(0, str(ROOT / "Recnet"))
    import network_builder.expansion3 as e3
    from network_builder import build as builder
    from network_builder.expansion import anchor_of
    from network_builder.species import SpeciesSpec

    audit = json.loads((NI / "closure_audit-expanded" / "closure_audit.json").read_text())
    ready = {Chem.MolToSmiles(Chem.MolFromSmiles(g["smiles"])): len(g["new_channels_ready"])
             for g in audit["gaps"].values()}
    canon = lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s))  # noqa: E731

    raw = list(e3._RAW)
    if args.option == "b1":
        sel = [t for t in raw if ready.get(canon(t[1]), 0) >= 5]
    elif args.option == "b2":
        sel = [t for t in raw if ready.get(canon(t[1]), 0) >= 4]
    elif args.option == "d":
        sel = [t for t in raw if family_is_chem(t[1])]
    else:
        sel = raw
    print(f"[wave3] option={args.option}: {len(sel)}/{len(raw)} frontier species")

    e3.EXPANSION3_SPECS = tuple(
        SpeciesSpec(k, k, "expansion3", s, anchor_of(s), "closure-round2-gap", f,
                    f"closure-audit route: {r}") for k, s, f, r in sel)

    # NOTE: keep the throw-away dataset OUT of --out: the subset step re-creates --out.
    workdir = (Path(args.workdir).expanduser() if args.workdir
               else Path(tempfile.mkdtemp(prefix="wave3-dataset-")))
    workdir.mkdir(parents=True, exist_ok=True)
    manifest = builder.build_dataset(out_dir=workdir, yaml_name="fe5c2_510_c2r3.prepared_rmg_data.yaml",
                                     expansion=True, expansion2=True, expansion3=True)
    print(f"[wave3] dataset: {manifest['counts']['species_entries']} species entries, "
          f"{manifest['counts']['rxns']} reaction rows")

    payload = yaml.safe_load((workdir / "fe5c2_510_c2r3.prepared_rmg_data.yaml").read_text())
    # channel identity must use the reaction-record display fields ("CH*", "H2(g)"), which are
    # what the pipeline and the delivered table use — the species table only carries bare keys.
    labels = [f"{r['reactant']} -> {r['product']}" for r in payload["rxns"]]
    delivered = {c["channel"] for c in json.loads(Path(args.table).read_text())["channels"]}
    new_idx = [i for i, label in enumerate(labels) if label not in delivered]
    print(f"[wave3] channels: {len(labels)} total, {len(new_idx)} new vs the delivered "
          f"{len(delivered)}-channel table")

    out = Path(args.out).expanduser()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    # reuse the campaign-subset tool (keeps the exact schema the pipeline expects)
    import subprocess
    subset_tool = ROOT / "Recnet" / "recnet_adapter" / "tools" / "subset_prepared.py"
    subprocess.run([sys.executable, str(subset_tool),
                    "--src", str(workdir / "fe5c2_510_c2r3.prepared_rmg_data.yaml"),
                    "--out", str(out),
                    "--rxn-index", ",".join(str(i) for i in new_idx)],
                   check=True)
    print(f"[wave3] campaign subset -> {out} ({len(new_idx)} new channels)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
