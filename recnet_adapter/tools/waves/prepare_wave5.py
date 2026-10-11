#!/usr/bin/env python3
"""Prepare the wave-5 (closure-completion) dataset + the subset of *new* channels.

wave-5 = the **7 residual products** left by the wave-4 (option c) audit
(``network_builder/expansion4.py``, generated from
``Recnet/network_inputs/closure_audit-wave4c/closure_audit.json``).  With them the mechanical
closure (C <= 2 / O <= 2, no O-O, dangling <= 2) is **complete**: 94/94 species, 0 gaps
(``closure_audit-wave5``).

The tool
  1. rebuilds the full closure dataset (frozen 29 + wave-1 + wave-2 + wave-3 (all 30) +
     wave-4 (7) = 96 entries),
  2. diffs the resulting channel labels against a **reference dataset** (default: the wave-4
     dataset ``c2r3all30``) so only the channels unlocked by the new species go to the campaign,
  3. writes ``<out>/prepared_rmg_data.yaml`` (+ ``ads_templates/``) for the campaign, and the
     full dataset under ``--workdir`` (same convention as prepare_wave3).

Nothing here touches the GPU.

Usage:
  PYTHONPATH=Recnet python recnet_adapter/tools/prepare_wave5.py \
      --out ~/scratch/wave5-final --workdir Recnet/network_inputs/c2r5
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
REF_DEFAULT = NI / "c2r3all30" / "fe5c2_510_c2r3.prepared_rmg_data.yaml"


def labels_of(yaml_path: Path) -> list[str]:
    payload = yaml.safe_load(yaml_path.read_text())
    return [f"{r['reactant']} -> {r['product']}" for r in payload["rxns"]]


def main() -> int:
    ap = argparse.ArgumentParser(description="Prepare the wave-5 closure-completion dataset")
    ap.add_argument("--ref-dataset", default=str(REF_DEFAULT),
                    help="prepared yaml whose channel labels are already covered "
                         "(default: the wave-4 dataset c2r3all30)")
    ap.add_argument("--out", required=True, help="output dir for the campaign subset")
    ap.add_argument("--workdir", default=None, help="dir for the full dataset (default <out>/dataset)")
    ap.add_argument("--yaml-name", default="fe5c2_510_c2r5.prepared_rmg_data.yaml")
    args = ap.parse_args()

    sys.path.insert(0, str(ROOT / "Recnet"))
    from network_builder import build as builder
    from network_builder.expansion4 import EXPANSION4_SPECS

    workdir = (Path(args.workdir).expanduser() if args.workdir
               else Path(args.out).expanduser() / "dataset")
    workdir.mkdir(parents=True, exist_ok=True)
    manifest = builder.build_dataset(out_dir=workdir, yaml_name=args.yaml_name,
                                     expansion=True, expansion2=True, expansion3=True,
                                     expansion4=True)
    print(f"[wave5] dataset: {manifest['counts']['species_entries']} species entries, "
          f"{manifest['counts']['rxns']} reaction rows")

    full_yaml = workdir / args.yaml_name
    labels = labels_of(full_yaml)
    ref = set(labels_of(Path(args.ref_dataset)))
    new_idx = [i for i, label in enumerate(labels) if label not in ref]
    print(f"[wave5] channels: {len(labels)} total, {len(new_idx)} new vs the reference "
          f"{len(ref)}-channel dataset ({Path(args.ref_dataset).name})")

    # cross-check against the audit's "ready" count (rule-based, not hand-picked)
    audit_path = NI / "closure_audit-wave4c" / "closure_audit.json"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
        ready = sum(len(g["new_channels_ready"]) for g in audit["gaps"].values())
        print(f"[wave5] closure-audit 'ready' channels for these 7 species: {ready}"
              + ("  (matches)" if ready == len(new_idx) else
                 f"  (differs by {len(new_idx) - ready}: the new species also unlock channels "
                 "among themselves and from existing reactants — mutual unlocking, same effect "
                 "as the wave-3 prediction (+126) vs measurement (+69))"))

    out = Path(args.out).expanduser()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    subset_tool = ROOT / "Recnet" / "recnet_adapter" / "tools" / "subset_prepared.py"
    subprocess.run([sys.executable, str(subset_tool),
                    "--src", str(full_yaml), "--out", str(out),
                    "--rxn-index", ",".join(str(i) for i in new_idx)], check=True)
    print(f"[wave5] campaign subset -> {out} ({len(new_idx)} new channels); "
          f"specs={[s.key for s in EXPANSION4_SPECS]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
