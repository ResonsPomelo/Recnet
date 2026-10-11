#!/usr/bin/env python3
"""Prepare the C3 wave-1 dataset + the subset of *new* channels (R309).

C3 wave-1 = the C3 closure round opened by the `--max-c 3` audit (241 missing C3 species).
This tool mirrors ``prepare_wave6.py``: rebuild the full dataset with every expansion flag
**plus ``expansion6``**, diff the channel labels against the delivered c2r6 dataset so only
genuinely new channels go to the campaign, and write the subset (+ templates) for
``build_campaign.py``.

The species tier is a **maintainer decision** (see ``network_builder/expansion6.py``):

  * ``closed-shell`` (45 species, +91 channels) — no chemical review needed (default);
  * ``o1`` (85, +347) / ``ready2`` (136, +505) / ``all`` (241, +1072) — contain dangling-valence
    species whose "surface bond" assumption is still unreviewed (R225/R249 precedent).

Nothing here touches the GPU.

Usage::

  PYTHONPATH=Recnet python recnet_adapter/tools/prepare_c3wave1.py \
      --out ~/scratch/c3w1-subset --tier closed-shell
"""

from __future__ import annotations

import argparse
import json

import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[4]
NI = ROOT / "Recnet" / "network_inputs"
REF_DEFAULT = NI / "c2r6" / "fe5c2_510_c2r6.prepared_rmg_data.yaml"

#: 各档的预期新增通道数（R309 实测；用于自检，防止静默口径漂移）
EXPECTED_NEW = {"closed-shell": 91, "o1": 347, "ready2": 505, "all": 1072}


def labels_of(yaml_path: Path) -> list[str]:
    payload = yaml.safe_load(yaml_path.read_text())
    return [f"{r['reactant']} -> {r['product']}" for r in payload["rxns"]]


def main() -> int:
    ap = argparse.ArgumentParser(description="Prepare the C3 wave-1 dataset + new-channel subset")
    ap.add_argument("--tier", default="closed-shell",
                    choices=("closed-shell", "o1", "ready2", "all"),
                    help="expansion6 物种档（默认 closed-shell；其它档含未评审的悬挂价物种）")
    ap.add_argument("--ref-dataset", default=str(REF_DEFAULT),
                    help="已交付渠道的 prepared yaml（默认 c2r6）")
    ap.add_argument("--out", required=True, help="campaign 子集输出目录")
    ap.add_argument("--workdir", default=None, help="完整数据集目录（默认 <out>/dataset）")
    ap.add_argument("--yaml-name", default="fe5c2_510_c3w1.prepared_rmg_data.yaml")
    args = ap.parse_args()

    sys.path.insert(0, str(ROOT / "Recnet"))
    from network_builder import build as builder

    # 注意：完整数据集**不能**默认落在 --out 之内——旧写法 <out>/dataset 会被下面的输出目录
    # 清理连带删除（prepare_wave6.py 靠调用方显式 --workdir 才没暴露）。这里默认放同级兄弟目录。
    out = Path(args.out).expanduser()
    workdir = (Path(args.workdir).expanduser() if args.workdir
               else out.parent / f"{out.name}-dataset")
    workdir.mkdir(parents=True, exist_ok=True)
    manifest = builder.build_dataset(out_dir=workdir, yaml_name=args.yaml_name,
                                     expansion=True, expansion2=True, expansion3=True,
                                     expansion4=True, expansion5=True,
                                     expansion6=True, expansion6_tier=args.tier)
    counts = manifest["counts"]
    print(f"[c3w1] tier={args.tier} dataset: {counts['species_entries']} species entries, "
          f"{counts['rxns']} reaction rows, "
          f"{manifest['checks']['fragment_closure']['dropped_count']} dropped cuts")

    full_yaml = workdir / args.yaml_name
    labels = labels_of(full_yaml)
    ref = set(labels_of(Path(args.ref_dataset)))
    new_idx = [i for i, label in enumerate(labels) if label not in ref]
    expected = EXPECTED_NEW[args.tier]
    print(f"[c3w1] channels: {len(labels)} total, {len(new_idx)} new vs the reference "
          f"{len(ref)}-channel dataset ({Path(args.ref_dataset).name}); expected {expected}")
    if len(new_idx) != expected:
        print(f"[c3w1] WARNING: 新增通道数与 R309 实测不符（{len(new_idx)} != {expected}）"
              f" ⇒ 先核对口径再投作业")
    for i in new_idx:
        print(f"[c3w1]   + {labels[i]}")

    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"[c3w1] 输出目录非空，拒绝覆盖：{out}"
                         f"（本工作区纪律：只移不删；请先移走旧产物或换 --out）")
    out.mkdir(parents=True, exist_ok=True)
    subset_tool = ROOT / "Recnet" / "recnet_adapter" / "tools" / "subset_prepared.py"
    subprocess.run([sys.executable, str(subset_tool),
                    "--src", str(full_yaml), "--out", str(out),
                    "--rxn-index", ",".join(str(i) for i in new_idx)], check=True)
    (out / "c3wave1_meta.json").write_text(json.dumps({
        "schema": "c3wave1_subset/1", "tier": args.tier,
        "species_entries": counts["species_entries"], "rxns": counts["rxns"],
        "new_channels": len(new_idx), "expected_new_channels": expected,
        "ref_dataset": str(args.ref_dataset), "full_yaml": str(full_yaml),
    }, indent=1) + "\n")
    print(f"[c3w1] campaign subset -> {out} ({len(new_idx)} new channels)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
