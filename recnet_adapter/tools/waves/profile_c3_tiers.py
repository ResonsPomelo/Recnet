#!/usr/bin/env python3
"""四档 C3 扩网的"新增通道画像"（R316）：按键型 / 链长 / 反应物身份分布。

定档需要的不只是"加多少通道"，还有"加的是什么化学"。本工具对每个档位：
  1. 建全量数据集（expansion..expansion5 + expansion6 指定档）；
  2. 与已交付 c2r6 渠道（299 条 yaml 行）diff 出**新增通道**；
  3. 按键型（C–H/C–C/C–O/O–H）、反应物碳数、产物碳数拆分、C3 反应物占比、气体反应物占比做画像。

零 GPU、零集群写；只读 c2r6 交付 yaml。

用法::
    PYTHONPATH=Recnet python recnet_adapter/tools/profile_c3_tiers.py --out Recnet/network_inputs/c3-tier-profile.json
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[4]
REF = ROOT / "Recnet" / "network_inputs" / "c2r6" / "fe5c2_510_c2r6.prepared_rmg_data.yaml"
TIERS = ("closed-shell", "o1", "ready2", "all")
BASE_FLAGS = dict(expansion=True, expansion2=True, expansion3=True, expansion4=True, expansion5=True)


def _n_carbons(key: str) -> int:
    """从物种 key（如 C2H4O2_w3_4 或 CH3O_1）数碳原子。"""
    m = re.match(r"C(\d*)", key)
    if not m:
        return 0
    return int(m.group(1) or 1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, help="画像 JSON 输出")
    ap.add_argument("--ref", default=str(REF), help="已交付渠道 yaml（默认 c2r6）")
    args = ap.parse_args()

    sys.path.insert(0, str(ROOT / "Recnet"))
    from network_builder.build import build_dataset
    from network_builder.channels import enumerate_channels
    from network_builder.species import build_species_table

    def labels(table):
        chans, _ = enumerate_channels(table)
        return {c.reactant_label + " -> " + c.product_label: c for c in chans}

    base_table = build_species_table(**BASE_FLAGS)
    base = labels(base_table)
    ref_keys = {f"{r['reactant']} -> {r['product']}"
                for r in yaml.safe_load(Path(args.ref).read_text())["rxns"]}
    print(f"[profile] c2r6 交付 yaml 渠道 {len(ref_keys)}；基线枚举 {len(base)}")

    report = {"schema": "c3_tier_profile/1", "reference": str(args.ref),
              "baseline_channels": len(base), "tiers": {}}
    for tier in TIERS:
        table = build_species_table(expansion6=True, expansion6_tier=tier, **BASE_FLAGS)
        cur = labels(table)
        new = {k: v for k, v in cur.items() if k not in base}
        by_bond = collections.Counter(c.bond_type for c in new.values())
        by_rc = collections.Counter(_n_carbons(c.reactant) for c in new.values())
        gas = sum(1 for c in new.values() if c.reactant_is_gas)
        c3_reactant = sum(1 for c in new.values() if _n_carbons(c.reactant) == 3)
        # 链断裂型：C3 反应物 → 两个碎片
        scission_c3 = sum(1 for c in new.values()
                          if _n_carbons(c.reactant) == 3 and len(c.products) == 2)
        report["tiers"][tier] = {
            "species_entries": len(table), "channels_total": len(cur), "new_channels": len(new),
            "new_by_bond_type": dict(sorted(by_bond.items())),
            "new_by_reactant_carbons": {str(k): v for k, v in sorted(by_rc.items())},
            "new_reactant_is_gas": gas, "new_reactant_c3": c3_reactant,
            "new_c3_scission_channels": scission_c3,
            "examples": sorted(new)[:12],
        }
        print(f"[profile] {tier:13s} +{len(new):4d} 通道 | 键型 {dict(sorted(by_bond.items()))} | "
              f"C3 反应物 {c3_reactant} | C3 断裂型 {scission_c3} | 气体反应物 {gas}")
    Path(args.out).write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    print(f"[profile] -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
