#!/usr/bin/env python3
"""Regenerate ``Recnet/network_inputs/expansion_frontier.md`` (the expansion decision memo).

Consumes the machine-readable output of::

    closure_audit.py --out-dir Recnet/network_inputs/closure_audit-expanded --expansion2 --iterate

plus ``c2r2w2/MANIFEST.json`` (to verify the round-0 leftover candidate list against what is
actually in the network) and ``wave2_candidates.md`` (the old candidate list itself).

The memo is decision material only: it states the closure arithmetic, resolves the stale
candidate list, and lays out options (a/b1/b2/c/d) with cost and closure-coverage readings.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[4]          # workspace root
NI = ROOT / "Recnet" / "network_inputs"


def measure_options(ready_by_smiles, is_orich):
    """Build each candidate wave-3 subset and measure the *actual* new channels.

    The audit's ``new_channels_ready`` counts only scissions whose fragments are already in
    the current table, so it is a **lower bound**: once several wave-3 species come in
    together they unlock each other's channels.  Measuring is cheap (seconds, CPU only) and
    is what the decision table quotes.
    """
    import shutil
    import sys
    import tempfile

    sys.path.insert(0, str(ROOT / "Recnet"))
    import yaml
    import network_builder.expansion3 as e3
    from network_builder import build as builder
    from network_builder.expansion import anchor_of
    from network_builder.species import SpeciesSpec

    raw = [(k, s, f, r) for k, s, f, r in e3._RAW]          # real, unique keys
    ready = lambda smi: ready_by_smiles.get(canon(smi), 0)   # noqa: E731

    def build(sel):
        e3.EXPANSION3_SPECS = tuple(
            SpeciesSpec(k, k, "expansion3", s, anchor_of(s), "closure-round2-gap", f,
                        f"closure-audit route: {r}") for k, s, f, r in sel)
        out = Path(tempfile.mkdtemp(prefix="c2r3-measure-"))
        builder.build_dataset(out_dir=out, yaml_name="x.yaml", expansion=True,
                              expansion2=True, expansion3=True)
        data = yaml.safe_load((out / "x.yaml").read_text())
        sp = data["species"]
        labels = {f"{sp[r['reactant_species'][0]]['name']} -> "
                  f"{' '.join(sp[p]['name'] for p in r['product_species'])}" for r in data["rxns"]}
        deltas = (len(data["rxns"]) - 139, len(labels) - 138)
        shutil.rmtree(out, ignore_errors=True)
        return deltas

    return {
        "b1": (len([t for t in raw if ready(t[1]) >= 5]), build([t for t in raw if ready(t[1]) >= 5])),
        "b2": (len([t for t in raw if ready(t[1]) >= 4]), build([t for t in raw if ready(t[1]) >= 4])),
        "d": (len([t for t in raw if not is_orich(t[1])]),
              build([t for t in raw if not is_orich(t[1])])),
        "c": (len(raw), build(raw)),
    }


def canon(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol else None


def family(mol: Chem.Mol) -> str:
    """Coarse functional-group family used to split the frontier chemically."""
    has_co = any(
        b.GetBondTypeAsDouble() == 2
        and {b.GetBeginAtom().GetSymbol(), b.GetEndAtom().GetSymbol()} == {"C", "O"}
        for b in mol.GetBonds()
    )
    n_o = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "O")
    n_c = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "C")
    if has_co:
        return "carbonyl/acid/ester/acyl (C=O)"
    if n_o >= 2 or (n_o >= 1 and n_c == 1):
        return "O-rich hydroxy/diol/ether (no C=O)"
    return "hydroxy-alkyl / enol (other)"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)  # Handle --help before reading or writing campaign data.
    audit = json.loads((NI / "closure_audit-expanded" / "closure_audit.json").read_text())
    gaps = audit["gaps"]

    # --- 1) verify the stale round-0 candidate list against the current network ---
    cands = []
    for line in (NI / "wave2_candidates.md").read_text().splitlines():
        if not line.startswith("| ") or line.startswith("| # "):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 5 or not cells[0].isdigit():
            continue
        cands.append({"idx": int(cells[0]), "formula": cells[1], "smiles": cells[2].strip("`"),
                      "route": cells[3], "ready0": cells[5]})
    man = json.loads((NI / "c2r2w2" / "MANIFEST.json").read_text())
    in_net: dict = {}
    for entry in man["species_entries"]:
        in_net.setdefault(canon(entry["smiles"]) or entry["smiles"], []).append(entry["name"])
    cand_rows = [(c, in_net.get(canon(c["smiles"]))) for c in cands]
    leftovers = [(c, n) for c, n in cand_rows if not n]

    # --- 2) frontier table + family split ---
    front = sorted(gaps.values(), key=lambda g: (-len(g["new_channels_ready"]), g["formula"]))
    tot_ready = sum(len(g["new_channels_ready"]) for g in front)
    fam: dict = collections.defaultdict(lambda: {"n": 0, "ready": 0, "sp": []})
    for g in front:
        mol = Chem.MolFromSmiles(g["smiles"])
        key = family(mol)
        fam[key]["n"] += 1
        fam[key]["ready"] += len(g["new_channels_ready"])
        fam[key]["sp"].append((g["formula"], g["smiles"], len(g["new_channels_ready"])))
    chem_family = next(k for k in fam if k.startswith("carbonyl"))
    other_family = next(k for k in fam if k.startswith("hydroxy-alkyl"))
    o_rich_family = next(k for k in fam if k.startswith("O-rich"))
    chem_sp = fam[chem_family]["n"] + fam[other_family]["n"]
    chem_ready = fam[chem_family]["ready"] + fam[other_family]["ready"]

    # measured yields per option (the audit's ready counts are only a lower bound)
    measured = measure_options({canon(g["smiles"]): len(g["new_channels_ready"]) for g in front},
                               lambda s: family(Chem.MolFromSmiles(s)).startswith("O-rich"))

    def route_of(g) -> str:
        c = g["routes"][0]["components"]
        return f"{c[0]} + {c[1]} (o={g['routes'][0]['bond_order']})"

    L: list = []
    L.append("# 扩网前沿：当前物种表还缺什么（2026-09-23）\n")
    L.append("**一句话**：机械闭包（C ≤ 2 / O ≤ 2，禁 O–O，悬挂价 ≤ 2）总规模 **94 物种**；"
             "三轮网络已入 **57**（29 frozen + 16 wave-1 + 14 wave-2，条目 59 含气相参考）⇒ "
             "**闭包覆盖率 57/94 = 60.6%**，剩余 **37 = 30（下一轮，85 条可立即枚举通道）+ 7（再下一轮，24 条）**。\n")
    L.append("> 来源：`closure_audit-expanded/closure_audit.{json,md}`"
             "（`closure_audit.py --expansion2 --iterate`，3 s 复算）。"
             "本文件由 `recnet_adapter/tools/make_expansion_frontier.py` 生成，可重跑。\n")
    L.append("## 1. 旧“15 个 ready≤1 候选”清单的核对（该清单已过期）\n")
    L.append(f"- **{sum(1 for _, n in cand_rows if n)}/{len(cand_rows)} 已随 wave-2 入库**"
             "（RDKit SMILES 规范化后逐条精确匹配；即 `*_w2_*` 系物种）；")
    if leftovers:
        c, _ = leftovers[0]
        L.append(f"- 仅剩 **{len(leftovers)} 条**未入库：`{c['smiles']}`（{c['formula']}，路线 {c['route']}）"
                 f"——它在**当前表**下 ready 已从 {c['ready0']} 升到 **4**。")
    L.append("- ⇒ 旧清单**不再是待裁决项**；裁决对象改为 §3 的新前沿。\n")
    L.append("| 旧候选 | 式 | SMILES | 已入库名 |")
    L.append("|---|---|---|---|")
    for c, names in cand_rows:
        L.append(f"| #{c['idx']} | {c['formula']} | `{c['smiles']}` | "
                 f"{'、'.join(names) if names else '**未入库**'} |")
    L.append("")
    L.append(f"## 2. 下一轮前沿（{len(front)} 物种 / {tot_ready} 条可立即枚举通道）\n")
    L.append("阈值读数：" + "；".join(
        f"ready ≥ {t}: {sum(1 for g in front if len(g['new_channels_ready']) >= t)} 物种 / "
        f"{sum(len(g['new_channels_ready']) for g in front if len(g['new_channels_ready']) >= t)} 通道"
        for t in (5, 4, 3, 2, 1)) + "。\n")
    L.append("**官能团族分解**（本文件的化学取舍依据）：\n")
    L.append("| 族 | 物种 | ready 通道 | 代表 |")
    L.append("|---|---|---|---|")
    for key in (chem_family, other_family, o_rich_family):
        v = fam[key]
        rep = "、".join(f"`{s}`" for _, s, r in sorted(v["sp"], key=lambda x: -x[2])[:2])
        L.append(f"| {key} | {v['n']} | {v['ready']} | {rep} |")
    L.append("")
    L.append("> **审计的 ready 数是下界**：它只统计『碎片已在当前表内』的断裂；若干 wave-3 物种同时入表后会互相解锁，"
             f"实测（见 §4）加入全部 30 物种实际新增 **{measured['c'][1][1]} 条通道**（审计 ready 只数到 {tot_ready}）。\n")
    L.append("| # | 式 | SMILES | 族 | 代表路线 | ready | blocked |")
    L.append("|---|---|---|---|---|---|---|")
    for i, g in enumerate(front, 1):
        mol = Chem.MolFromSmiles(g["smiles"])
        L.append(f"| {i} | {g['formula']} | `{g['smiles']}` | {family(mol).split(' (')[0]} | "
                 f"{route_of(g)} | {len(g['new_channels_ready'])} | {len(g['new_channels_blocked'])} |")
    L.append("")
    L.append("## 3. 决策要点（成本 / 精度）\n")
    L.append("- **精度**：§10 已定标“闭包越外层误差越大”（wave-1 中位 Δ −0.28 eV → wave-2 **−0.78 eV**）。"
             "再外扩一轮，预期中位误差继续上升；下游要定量用势垒须走 §14 的复核流程。")
    L.append("- **成本（用实测 farm 墙钟标定，不拍脑袋）**：")
    L.append("  - wave-1：farm `1438888` 6.6 h × 4 卡 ≈ 26 GPU·h / 50 通道 ≈ **0.53 GPU·h/通道**；")
    L.append("  - wave-2：farm `1442243` 8.6 h + round-2 `1446165` 2.2 h，合计 10.8 h × 4 卡 ≈ 43 GPU·h / 46 通道 ≈ **0.94 GPU·h/通道**"
             "（含一次失败重跑，属保守上限）；")
    L.append("  - 若选 (b1)/(b2)/(c)：(b1) 22 通道 ≈ **12–21 GPU·h**；(b2) 42 ≈ 22–40；(d) 40 ≈ 21–38；"
             "(c) 85 ≈ **45–80 GPU·h**（4 卡 ≈ 半天–1 天 farm）。")
    L.append("  - 注意这是 **farm 计算时间**；端到端还要加 gate/汇总（分钟级）与抽样 DFT 锚定（每通道 2–4 SP），"
             "wave-2 的日历时间 ≈ 1 天量级。")
    L.append("- **O-rich 一族的存疑点**：`CH3O2`/`CH4O2`/`C2H5O2`/`C2H6O2` 等羟基/二醇/醚类在**贫氧碳化物表面**"
             "的丰度与寿命存疑，其势垒的物理意义弱于羰基/酸/酯族；这一族恰好占前沿的一半（16/30 物种、45/85 通道）。\n")
    L.append("## 4. 选项\n")
    L.append("| 选项 | 规模 | **实测新增通道**（含互相解锁） | 粗估算力 | 闭包覆盖率 |")
    L.append("|---|---|---|---|")
    L.append(f"| (a) 停止 | — | 0 | 0 | 57/94 = 60.6%（按现状交付并注明） |")
    for tag, label, nsp in (("b1", "(b1) 补 ready ≥ 5", 4), ("b2", "(b2) 补 ready ≥ 4", 9),
                            ("d", "(d) 化学筛选：羰基/酸/酯/酰基 + 羟基烷基族", chem_sp),
                            ("c", "(c) 补满下一轮", len(front))):
        n_sp, (d_rows, d_labels) = measured[tag]
        L.append(f"| {label} | {n_sp} 物种 | **+{d_labels}**（+{d_rows} 行） | "
                 f"≈{d_labels * 0.53:.0f}–{d_labels * 0.94:.0f} GPU·h | {57 + n_sp}/94 = "
                 f"{100 * (57 + n_sp) / 94:.1f}% |")
    L.append("")
    L.append("> (b2) 与 (d) 规模相近但成分不同：(b2) 按 ready 阈值选，9 物种里 5 个属 O-rich 族；"
             "(d) 按化学族选，保留羰基/酸/酯/酰基与羟基烷基族，把 O-rich 一族整体留待将来。\n")
    L.append("## 5. 若选 (b)/(c)/(d) 的执行入口（复用现有链，新增一个 wave 定义文件）\n")
    L.append("```bash")
    L.append("# 1) 数据集 + “新增通道”子集（一条命令；选项 d/b1/b2/c）")
    L.append("PYTHONPATH=Recnet python Recnet/recnet_adapter/tools/prepare_wave3.py --option d \\")
    L.append("    --table validation-pipeline/summary/c2r2-network-20260922/network_channels_final.json \\")
    L.append("    --out ~/scratch/wave3-d")
    L.append("# 2) campaign 计划（--dry-run 零 GPU 先看 chunk/臂分布）")
    L.append("python Recnet/recnet_adapter/tools/build_campaign.py --root $R --out $R/recnet-runs/c2r3-d \\")
    L.append("    --template-case $R/recnet-runs/c2r2w2-campaign/S-c0 --prepared <上一步>/prepared_rmg_data.yaml \\")
    L.append("    --chunk-size 8 --top-x 2 --dry-run")
    L.append("# 3) farm（去掉 --dry-run）→ finalize_network.sh → gate → export_network_table.py 并表 → 图重出")
    L.append("```")
    L.append("")
    L.append("> **已实测（2026-09-23，零 GPU）**：`prepare_wave3.py --option d` ⇒ 数据集 73 物种条目 / 197 行，")
    L.append("> 与已交付 138 通道比对得 **58 行 / 57 条新通道**（与 §4 的 Δ 一致）；build_campaign dry-run ⇒ **16 个 case**")
    L.append("> （8 chunk × 2 臂）。即『用户点头即可提交 farm』，生成/子集/计划三步都已验证。")
    L.append("")
    L.append("> **本文件是决策材料**：不含结论。裁决（做/不做/做到哪一档）请写入 "
             "`docs/tasks/reaction-network-dpa4.md` 的 Ruling，并同步报告 §9 的覆盖度读数。\n")

    (NI / "expansion_frontier.md").write_text("\n".join(L) + "\n")
    print(f"wrote {NI / 'expansion_frontier.md'} ({len(L)} blocks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
