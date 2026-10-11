#!/usr/bin/env python3
"""Prepare the wave-6 (scission-residue) dataset + the subset of *new* channels.

wave-6 = the scission-closure round opened by R224/R225.  This first (minimal) increment adds
**CO2 (gas)** only — the single clean closed-shell fragment among the 32 missing fragment
identities of the 76 dropped cuts (`network_builder/expansion5.py`).  It unlocks 4 channels
(measured locally, 294 → 298 reaction rows):

  * ``CHO2_w2_1* -> CO2(g) H*`` / ``CHO2_w2_2* -> CO2(g) H*`` (formate decarboxylation ×2)
  * ``C2H3O2_w3_2* -> CH3* CO2(g)`` (acetate decarboxylation)
  * ``CO2(g) -> CO* O*`` (CO2 dissociation; reverse of CO* + O* -> CO2)

The tool mirrors ``prepare_wave5.py``: rebuild the full dataset with every expansion flag,
diff the channel labels against the wave-5 dataset so only genuinely new channels go to the
campaign, and write the subset (+ templates) for ``build_campaign.py``.

Nothing here touches the GPU.

Usage::

  PYTHONPATH=Recnet python recnet_adapter/tools/prepare_wave6.py \
      --out ~/scratch/wave6-subset --workdir Recnet/network_inputs/c2r6
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[4]
NI = ROOT / "Recnet" / "network_inputs"
REF_DEFAULT = NI / "c2r5" / "fe5c2_510_c2r5.prepared_rmg_data.yaml"


def labels_of(yaml_path: Path) -> list[str]:
    payload = yaml.safe_load(yaml_path.read_text())
    return [f"{r['reactant']} -> {r['product']}" for r in payload["rxns"]]


def main() -> int:
    ap = argparse.ArgumentParser(description="Prepare the wave-6 scission-residue dataset")
    ap.add_argument("--ref-dataset", default=str(REF_DEFAULT),
                    help="prepared yaml whose channel labels are already covered "
                         "(default: the wave-5 dataset c2r5)")
    ap.add_argument("--out", required=True, help="output dir for the campaign subset")
    ap.add_argument("--workdir", default=None, help="dir for the full dataset (default <out>/dataset)")
    ap.add_argument("--yaml-name", default="fe5c2_510_c2r6.prepared_rmg_data.yaml")
    args = ap.parse_args()

    sys.path.insert(0, str(ROOT / "Recnet"))
    from network_builder import build as builder
    from network_builder.expansion5 import EXPANSION5_SPECS

    workdir = (Path(args.workdir).expanduser() if args.workdir
               else Path(args.out).expanduser() / "dataset")
    workdir.mkdir(parents=True, exist_ok=True)
    manifest = builder.build_dataset(out_dir=workdir, yaml_name=args.yaml_name,
                                     expansion=True, expansion2=True, expansion3=True,
                                     expansion4=True, expansion5=True)
    print(f"[wave6] dataset: {manifest['counts']['species_entries']} species entries, "
          f"{manifest['counts']['rxns']} reaction rows, "
          f"{manifest['checks']['fragment_closure']['dropped_count']} dropped cuts")

    full_yaml = workdir / args.yaml_name
    labels = labels_of(full_yaml)
    ref = set(labels_of(Path(args.ref_dataset)))
    new_idx = [i for i, label in enumerate(labels) if label not in ref]
    print(f"[wave6] channels: {len(labels)} total, {len(new_idx)} new vs the reference "
          f"{len(ref)}-channel dataset ({Path(args.ref_dataset).name})")
    for i in new_idx:
        print(f"[wave6]   + {labels[i]}")
    if len(new_idx) != 4:
        print(f"[wave6] WARNING: expected 4 new channels (CO2 only) — got {len(new_idx)}")

    out = Path(args.out).expanduser()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    subset_tool = ROOT / "Recnet" / "recnet_adapter" / "tools" / "subset_prepared.py"
    subprocess.run([sys.executable, str(subset_tool),
                    "--src", str(full_yaml), "--out", str(out),
                    "--rxn-index", ",".join(str(i) for i in new_idx)], check=True)
    print(f"[wave6] campaign subset -> {out} ({len(new_idx)} new channels); "
          f"specs={[s.key for s in EXPANSION5_SPECS]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
