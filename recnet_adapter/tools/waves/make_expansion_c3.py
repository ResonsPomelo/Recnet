#!/usr/bin/env python
"""从 C3 闭包审计（closure_audit-c3/closure_audit.json）生成 network_builder/expansion6.py。

为什么这么做：wave-1..wave-6 的每次扩网都是"审计 → 冻结物种表 → 建数据集"三步，本工具把第二步机械化，
保证 expansion6.py 的物种清单**可复算、不是人工挑选**（同 make_expansion_frontier.py 的做法）。

分档（机械判据，写死在产物里；选择权留给维护者）：
  * closed-shell（默认）：残留悬挂价 = 0 —— 与 wave-6 收编 CO2 的判据同类（干净闭壳层）；
  * o1：O <= 1（贫氧碳化物表面的核心族）；
  * ready2：入表后立即可枚举的新通道 >= 2（性价比档）；
  * all：审计的全部 C3 缺口（含双悬挂价物种，需化学评审，见台账 R225/R249 同类先例）。

用法：
    PYTHONPATH=Recnet ~/apps/miniforge3/envs/recnet-prep/bin/python \
        Recnet/recnet_adapter/tools/make_expansion_c3.py \
        --audit Recnet/network_inputs/closure_audit-c3/closure_audit.json \
        --out Recnet/network_builder/expansion6.py
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors

RDLogger.DisableLog("rdApp.*")


def _o_count(formula: str) -> int:
    match = re.search(r"O(\d*)", formula)
    if not match:
        return 0
    return int(match.group(1) or 1)


def _identity(smiles: str) -> str:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    if mol is None:
        raise ValueError("SMILES 无法解析: %r" % smiles)
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    edges = [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in mol.GetBonds()]
    return Chem.MolToSmiles(mol, canonical=True) + "|" + ",".join(
        sorted("%s-%s" % (symbols[i], symbols[j]) for i, j in edges))


def build_rows(audit_path: Path) -> List[Dict]:
    report = json.loads(audit_path.read_text())
    rows: List[Dict] = []
    seen: Dict[str, str] = {}
    for gap in report["gaps"].values():
        smiles = gap["smiles"]
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        formula_rdkit = rdMolDescriptors.CalcMolFormula(mol)
        if formula_rdkit != gap["formula"]:
            raise AssertionError("%s: 审计 formula %s != rdkit %s" % (smiles, gap["formula"], formula_rdkit))
        n_rad = sum(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms())
        if n_rad != gap["residual_radicals"]:
            raise AssertionError("%s: 审计悬挂价 %s != rdkit %s" % (smiles, gap["residual_radicals"], n_rad))
        identity = _identity(smiles)
        if identity in seen:
            raise AssertionError("身份重复：%s 与 %s" % (smiles, seen[identity]))
        seen[identity] = smiles
        route = gap["routes"][0]
        rows.append({
            "smiles": smiles,
            "formula": gap["formula"],
            "radicals": int(gap["residual_radicals"]),
            "o_count": _o_count(gap["formula"]),
            "ready": len(gap["new_channels_ready"]),
            "blocked": len(gap["new_channels_blocked"]),
            "route": " + ".join(route["components"]) + " (o=%s)" % route["bond_order"],
        })
    rows.sort(key=lambda row: (row["formula"], row["smiles"]))
    return rows


def _key(row: Dict, index: int) -> str:
    return "%s_c3w1_%03d" % (row["formula"], index)


def _enumerated_counts() -> Dict[str, int]:
    """真实枚举的通道数（baseline + 四档）——比审计的"可立即枚举"口径更硬，直接进产物 docstring。"""
    from network_builder.build import build_species_table
    from network_builder.channels import enumerate_channels

    base = dict(expansion=True, expansion2=True, expansion3=True, expansion4=True, expansion5=True)
    counts = {"baseline": len(enumerate_channels(build_species_table(**base))[0])}
    for tier in ("closed-shell", "o1", "ready2", "all"):
        species = build_species_table(expansion6=True, expansion6_tier=tier, **base)
        counts[tier] = len(enumerate_channels(species)[0])
    return counts


def _raw_block(rows: Sequence[Dict], keys: Dict[str, str], name: str) -> str:
    lines = ["_RAW_%s: Tuple[Tuple[str, str, str, str], ...] = (" % name]
    for row in rows:
        lines.append('    ("%s", "%s", "%s", "%s"),' % (
            keys[row["smiles"]], row["smiles"], row["formula"], row["route"]))
    lines.append(")")
    return "\n".join(lines)


DOC_TEMPLATE = '''"""C3 扩网物种表（wave-1，机械生成，**默认关闭**）。

来源：recnet_adapter/tools/closure_audit.py --max-c 3 --max-o 2 --expansion..--expansion5 --iterate
（产物 network_inputs/closure_audit-c3/closure_audit.json），由
recnet_adapter/tools/make_expansion_c3.py 生成。判据与 wave-1..wave-6 完全一致（连接点 = 悬挂价原子 ∪
吸附锚点、成键阶 1/2/3、rdkit 收敛、净电荷 0、C <= 3 / O <= 2、禁 O-O、残留悬挂价 <= 2）——
**规则客观，不是人工挑选**；吸附锚点由 expansion.anchor_of 机械推导。

规模（生成时**真实枚举**实测；基线 = wave-1..wave-6 交付态 97 物种 / %(baseline)d 通道）：

| 档 | 新增物种 | 物种条目 | 通道数 | 通道增量 |
|---|---|---|---|---|
| closed-shell（rad=0；**默认档**） | %(cs)d | %(cs_entries)d | %(cs_channels)d | +%(cs_delta)d |
| o1（O <= 1） | %(o1)d | %(o1_entries)d | %(o1_channels)d | +%(o1_delta)d |
| ready2（ready >= 2） | %(r2)d | %(r2_entries)d | %(r2_channels)d | +%(r2_delta)d |
| all（全部 C3 缺口） | %(all)d | %(all_entries)d | %(all_channels)d | +%(all_delta)d |

冻结路径约束：本模块只在显式 expansion6=True 时被引入；无 flag 的构建必须保持字节一致
（network_builder/tests/test_schema.py::test_rebuild_is_byte_identical 等既有测试兜底）。
expansion6_tier 可切线（默认 closed-shell）；**all 含双悬挂价物种，与台账 R225/R249 的同类先例一样
需化学评审后再用**。
"""

from __future__ import annotations

from typing import Dict, Tuple

from .expansion import anchor_of
from .species import SpeciesSpec


'''

TAIL = '''

def _specs(raw: Tuple[Tuple[str, str, str, str], ...]) -> Tuple[SpeciesSpec, ...]:
    return tuple(
        SpeciesSpec(key, key, "expansion6", smiles, anchor_of(smiles), "closure-c3-gap",
                    formula, "closure-audit route: " + route)
        for key, smiles, formula, route in raw
    )


EXPANSION6_SPECS: Tuple[SpeciesSpec, ...] = _specs(_RAW_CLOSED_SHELL)

#: 档名 → 物种表；EXPANSION6_SPECS 恒等于 EXPANSION6_TIERS["closed-shell"]。
EXPANSION6_TIERS: Dict[str, Tuple[SpeciesSpec, ...]] = {
    "closed-shell": EXPANSION6_SPECS,
    "o1": _specs(_RAW_O1),
    "ready2": _specs(_RAW_READY2),
    "all": _specs(_RAW_ALL),
}

DEFAULT_TIER = "closed-shell"


def specs_for_tier(tier: str) -> Tuple[SpeciesSpec, ...]:
    """取某一档的物种表（未知档名显式报错，不静默回退）。"""
    try:
        return EXPANSION6_TIERS[tier]
    except KeyError as exc:
        raise ValueError("未知的 expansion6 档：%r（可选：%s）" % (tier, sorted(EXPANSION6_TIERS))) from exc
'''


def render(rows: Sequence[Dict], channel_counts: Dict[str, int]) -> str:
    keys = {row["smiles"]: _key(row, index + 1) for index, row in enumerate(rows)}
    tiers = {
        "CLOSED_SHELL": [row for row in rows if row["radicals"] == 0],
        "O1": [row for row in rows if row["o_count"] <= 1],
        "READY2": [row for row in rows if row["ready"] >= 2],
        "ALL": list(rows),
    }
    base = channel_counts["baseline"]
    base_entries = 97
    counts = {
        "baseline": base,
        "cs": len(tiers["CLOSED_SHELL"]),
        "o1": len(tiers["O1"]),
        "r2": len(tiers["READY2"]),
        "all": len(tiers["ALL"]),
    }
    for tag, tier_key, raw_name in (("cs", "closed-shell", "CLOSED_SHELL"),
                                    ("o1", "o1", "O1"),
                                    ("r2", "ready2", "READY2"),
                                    ("all", "all", "ALL")):
        counts[f"{tag}_entries"] = base_entries + len(tiers[raw_name])
        counts[f"{tag}_channels"] = channel_counts[tier_key]
        counts[f"{tag}_delta"] = channel_counts[tier_key] - base
    parts = [DOC_TEMPLATE % counts]
    for name in ("CLOSED_SHELL", "O1", "READY2", "ALL"):
        parts.append(_raw_block(tiers[name], keys, name))
        parts.append("")
    parts.append(TAIL)
    return "\n".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--audit", required=True, help="closure_audit.json（--max-c 3 的产物）")
    parser.add_argument("--out", required=True, help="写出的 expansion6.py")
    parser.add_argument("--check", action="store_true", help="只校验产物是否与现文件一致")
    args = parser.parse_args()

    rows = build_rows(Path(args.audit))
    text = render(rows, _enumerated_counts())
    out = Path(args.out)
    if args.check:
        current = out.read_text() if out.exists() else ""
        if current != text:
            print("[check] %s 与审计产物不一致（需重新生成）" % out, flush=True)
            return 1
        print("[check] %s 与审计产物一致（%d 行）" % (out, len(rows)), flush=True)
        return 0
    out.write_text(text, encoding="utf-8")
    print("[make-expansion-c3] %d 个 C3 缺口 → %s" % (len(rows), out))
    for tier, subset in (("closed-shell", [r for r in rows if r["radicals"] == 0]),
                         ("o1", [r for r in rows if r["o_count"] <= 1]),
                         ("ready2", [r for r in rows if r["ready"] >= 2]),
                         ("all", rows)):
        print("  %-13s %4d 物种 | 可立即枚举通道 %4d | 待解锁 %4d" % (
            tier, len(subset), sum(r["ready"] for r in subset), sum(r["blocked"] for r in subset)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
