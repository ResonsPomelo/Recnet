#!/usr/bin/env python
"""网络「全面性」审计：物种表的成键封闭性 + 缺失通道清单。

背景（为什么需要它）：``network_builder`` 的通道空间定义是
「29 物种的**全部单键断裂**」（42 条）。于是"网络是否全面"完全等价于
「29 物种表对 C2 以下的 C–C / C–O / C–H 成键是否**封闭**」——
表里缺一个中间体，就会连带缺掉它的一整组通道。

本工具做机械审计（不依赖任何文献清单的完备性主张）：

1. 取物种表（``network_builder.species.build_species_table``）；
2. 对每个物种定义**连接点** = 未成对电子所在的原子 ∪ 表面吸附锚点 ``ad_idx``
   （吸附物在表面上正是以这些原子成键的，例如 CO 的 C 是锚点，
   所以 CH3 + CO → CH3CO 这类表面介导偶联要被枚举到）；
3. 枚举所有 (A, B) 组合（含 A == B）与所有连接点对，尝试成键阶 1/2/3，
   要求产物**可被 rdkit 收敛（净电荷 0）**且在元素预算内；
   **注意**：产物允许保留悬挂价（未成对电子）——表内的吸附物本来就带悬挂价
   （methoxy `[CH3][O]`、formyl `[CH]=O`…），残留的自由基正是新吸附物的
   表面成键价；只保留 0 自由基的产物会把 formate/acetyl/ethoxy 这类关键
   中间体全部漏掉（首版即犯此错，已修正，`--max-radicals` 可调）；
4. 按 H 显式图身份（``identity_key``，与生成器同一判据）去重后与物种表比对：
   * 命中 → 该组合**已覆盖**；
   * 未命中 → **缺失物种**（gap），并进一步算它一旦入表会新增多少条
     "两碎片都在表里"的可枚举通道。

``--iterate`` 模式把闭包推到不动点：每轮把新物种并入宇宙再枚举，
直到没有新物种（或达到 ``--max-rounds``），给出"**C2 以下物种宇宙的闭包规模**"
与逐轮新增清单——这是"网络是否全面"的直接答案。

输出 JSON + Markdown，给出「缺失物种 → 其可立即枚举的通道数」的可执行清单。

用法::

    PYTHONPATH=Recnet /home/james/apps/miniforge3/envs/recnet-prep/bin/python \
        Recnet/recnet_adapter/tools/closure_audit.py --out-dir Recnet/network_inputs
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations_with_replacement
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors

RDLogger.DisableLog("rdApp.*")

from network_builder.species import BOND_TYPES, build_species_table, identity_key  # noqa: E402


def _mol_with_h(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"SMILES 无法解析: {smiles!r}")
    return Chem.AddHs(mol)


def _graph_identity(mol: Chem.Mol) -> str:
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    edges = [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in mol.GetBonds()]
    return identity_key(symbols, edges)


def _budget_ok(mol: Chem.Mol, max_c: int, max_o: int) -> bool:
    counts = rdMolDescriptors.CalcMolFormula(mol)
    formula = Chem.rdMolDescriptors.CalcMolFormula(mol)
    n_c = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "C")
    n_o = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "O")
    del formula, counts
    return n_c <= max_c and n_o <= max_o


def _connection_points(mol: Chem.Mol, ad_idx: Sequence[int]) -> List[int]:
    """未成对电子原子 ∪ 吸附锚点（adsorbate 在表面上的成键原子）。"""
    points = {atom.GetIdx() for atom in mol.GetAtoms() if atom.GetNumRadicalElectrons() > 0}
    points.update(int(i) for i in ad_idx)
    return sorted(points)


def _try_combine(mol_a: Chem.Mol, mol_b: Chem.Mol, i: int, j: int, order: int,
                 max_c: int, max_o: int, max_radicals: int,
                 allow_oo: bool = False) -> Chem.Mol | None:
    if not allow_oo and mol_a.GetAtomWithIdx(i).GetSymbol() == "O" \
            and mol_b.GetAtomWithIdx(j).GetSymbol() == "O":
        # Fe5C2 碳化物在 FT 条件下是贫氧表面：O–O 键（O2 / H2O2 / 过氧物）
        # 不是该体系的基元步骤。这条约束是可证的物理过滤，不是给结果凑数：
        # 不加它，闭包会沿 O+O → O2 → 过氧化物级联炸出与体系无关的化学。
        return None
    combined = Chem.CombineMols(mol_a, mol_b)
    rw = Chem.RWMol(combined)
    offset = mol_a.GetNumAtoms()
    atom_i = rw.GetAtomWithIdx(i)
    atom_j = rw.GetAtomWithIdx(offset + j)
    if atom_i.GetNumRadicalElectrons() + atom_j.GetNumRadicalElectrons() < order:
        return None
    try:
        rw.AddBond(i, offset + j, {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE,
                                   3: Chem.BondType.TRIPLE}[order])
    except Exception:  # noqa: BLE001
        return None
    atom_i.SetNumRadicalElectrons(max(0, atom_i.GetNumRadicalElectrons() - order))
    atom_j.SetNumRadicalElectrons(max(0, atom_j.GetNumRadicalElectrons() - order))
    mol = rw.GetMol()
    try:
        Chem.SanitizeMol(mol)
    except Exception:  # noqa: BLE001 — 价键不可满足即丢弃
        return None
    n_radicals = sum(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms())
    if max_radicals >= 0 and n_radicals > max_radicals:
        return None
    if Chem.GetFormalCharge(mol) != 0:
        return None
    if not _budget_ok(mol, max_c, max_o):
        return None
    return mol


def _single_bond_scissions(mol: Chem.Mol) -> List[Tuple[Tuple[str, ...], Tuple[str, ...], str]]:
    """产物的所有单键断裂（H 显式），返回 (frag1_symbols, frag2_symbols, bond_type)。"""
    out = []
    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.SINGLE:
            continue
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        sym_i = bond.GetBeginAtom().GetSymbol()
        sym_j = bond.GetEndAtom().GetSymbol()
        if tuple(sorted((sym_i, sym_j))) not in [tuple(sorted(pair)) for pair in BOND_TYPES]:
            continue
        frags = Chem.FragmentOnBonds(mol, [bond.GetIdx()], addDummies=False)
        pieces = Chem.GetMolFrags(frags)
        if len(pieces) != 2:
            continue
        syms = [atom.GetSymbol() for atom in mol.GetAtoms()]
        f1 = tuple(syms[k] for k in pieces[0])
        f2 = tuple(syms[k] for k in pieces[1])
        edges = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds() if b.GetIdx() != bond.GetIdx()]
        out.append((_frag_key(syms, edges, pieces[0]), _frag_key(syms, edges, pieces[1]),
                    f"{sym_i}-{sym_j}"))
    return out


def _frag_key(symbols: Sequence[str], edges: Iterable[Tuple[int, int]], keep: Sequence[int]) -> str:
    keep = list(keep)
    local = {atom: idx for idx, atom in enumerate(keep)}
    sub_symbols = [symbols[atom] for atom in keep]
    sub_edges = [(local[i], local[j]) for i, j in edges if i in local and j in local]
    return identity_key(sub_symbols, sub_edges)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="C2 网络物种表成键封闭性审计")
    parser.add_argument("--out-dir", required=True, help="输出目录（写 closure_audit.{json,md}）")
    parser.add_argument("--max-c", type=int, default=2, help="元素预算：最大碳数")
    parser.add_argument("--max-o", type=int, default=2, help="元素预算：最大氧数")
    parser.add_argument("--max-radicals", type=int, default=2,
                        help="产物允许的最大未成对电子数（-1 = 不限；吸附物靠悬挂价表面成键）")
    parser.add_argument("--iterate", action="store_true",
                        help="闭包迭代到不动点：每轮把新物种并入宇宙再枚举")
    parser.add_argument("--max-rounds", type=int, default=6, help="闭包迭代轮数上限")
    parser.add_argument("--allow-oo", action="store_true",
                        help="允许形成 O–O 键（默认禁止：贫氧碳化物表面无过氧化学）")
    parser.add_argument("--expansion", action="store_true",
                        help="种子表 = frozen 29 + wave-1（16）")
    parser.add_argument("--expansion2", action="store_true",
                        help="种子表 = frozen 29 + wave-1（16）+ wave-2（14）："
                             "审计『当前网络还缺什么』（默认 = 只审 frozen 29）")
    # 2026-09-24 (reaction-network R143 follow-up): wave-3 closure expansion.  --expansion3 alone
    # seeds the full 30-species next-round frontier; --wave3-option narrows it to one of the
    # frontier options with exactly the rules prepare_wave3.py used, so a post-wave-3 audit
    # reports the *true* remaining gap (the delivered campaign covered option d only).
    parser.add_argument("--expansion3", action="store_true",
                        help="种子表再加 wave-3 闭包缺口（network_builder/expansion3.py，30 物种）")
    # 2026-09-25 (R175): wave-5 closure completion — the 7 residual products left after the
    # wave-4 (option c) audit (network_builder/expansion4.py).
    parser.add_argument("--expansion4", action="store_true",
                        help="种子表再加 wave-5 闭包收口物种（network_builder/expansion4.py，7 物种）")
    parser.add_argument("--wave3-option", choices=("b1", "b2", "d", "c"), default=None,
                        help="只计入 wave-3 的某一档（规则同 prepare_wave3）："
                             "b1=ready>=5、b2=ready>=4、d=化学族筛选（已交付档）、c=全 30")
    parser.add_argument("--expansion5", action="store_true",
                         help="种子表再加断键残差物种（network_builder/expansion5.py，CO2 气相；R224/R225）")
    parser.add_argument("--expansion6", action="store_true",
                         help="种子表再加 C3 扩网物种（network_builder/expansion6.py；R309）——"
                              "用来审计『C3 wave-1 之后还缺什么』")
    parser.add_argument("--expansion6-tier", default="closed-shell",
                         choices=("closed-shell", "o1", "ready2", "all"),
                         help="expansion6 的物种档（默认 closed-shell）")
    args = parser.parse_args(argv)

    if args.wave3_option:
        # filter EXPANSION3_SPECS the same way prepare_wave3.py does (module attribute patched
        # before build_species_table imports it) before materialising the seed table
        import network_builder.expansion3 as e3
        from network_builder.expansion import anchor_of
        from network_builder.species import SpeciesSpec
        from recnet_adapter.tools.waves.prepare_wave3 import family_is_chem

        NI = Path(__file__).resolve().parents[3] / "network_inputs"
        prev = json.loads((NI / "closure_audit-expanded" / "closure_audit.json").read_text())
        canon = lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s))  # noqa: E731
        ready = {canon(g["smiles"]): len(g["new_channels_ready"]) for g in prev["gaps"].values()}
        raw = list(e3._RAW)
        if args.wave3_option == "b1":
            sel = [t for t in raw if ready.get(canon(t[1]), 0) >= 5]
        elif args.wave3_option == "b2":
            sel = [t for t in raw if ready.get(canon(t[1]), 0) >= 4]
        elif args.wave3_option == "d":
            sel = [t for t in raw if family_is_chem(t[1])]
        else:
            sel = raw
        print(f"[closure_audit] wave3 option={args.wave3_option}: {len(sel)}/{len(raw)} species")
        e3.EXPANSION3_SPECS = tuple(
            SpeciesSpec(k, k, "expansion3", s, anchor_of(s), "closure-round2-gap", f,
                        f"closure-audit route: {r}") for k, s, f, r in sel)
        args.expansion3 = True

    table = build_species_table(expansion=args.expansion, expansion2=args.expansion2,
                                expansion3=args.expansion3, expansion4=args.expansion4,
                                expansion5=args.expansion5,
                                expansion6=args.expansion6,
                                expansion6_tier=args.expansion6_tier)
    by_identity: Dict[str, object] = {}
    for species in table:
        by_identity.setdefault(species.identity, species)

    entries = []
    for species in by_identity.values():
        mol = _mol_with_h(species.smiles)
        symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
        if len(symbols) != len(species.symbols):
            raise AssertionError(
                f"{species.key}: SMILES 重解析的原子数 {len(symbols)} != 模板 {len(species.symbols)}"
            )
        entries.append({"species": species, "mol": mol,
                        "points": _connection_points(mol, species.ad_idx),
                        "label": species.key, "identity": species.identity})

    print(f"[closure] 表内唯一身份物种数 = {len(entries)}（总条目 {len(table)}）", file=sys.stderr)

    known_identities = set(by_identity)
    covered: Dict[str, Dict[str, object]] = {}
    gaps: Dict[str, Dict[str, object]] = {}
    rounds: List[Dict[str, object]] = []

    def enumerate_round(universe: List[Dict[str, object]], known: set,
                        round_index: int) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
        """一轮枚举：返回（本轮新物种，本轮记录）。"""
        new_products: Dict[str, Dict[str, object]] = {}
        new_entries: List[Dict[str, object]] = []
        for a, b in combinations_with_replacement(range(len(universe)), 2):
            entry_a, entry_b = universe[a], universe[b]
            mol_a, mol_b = entry_a["mol"], entry_b["mol"]
            for i in entry_a["points"]:
                for j in entry_b["points"]:
                    for order in (1, 2, 3):
                        product = _try_combine(mol_a, mol_b, i, j, order,
                                               args.max_c, args.max_o, args.max_radicals,
                                               args.allow_oo)
                        if product is None:
                            continue
                        identity = _graph_identity(product)
                        route = {
                            "components": [entry_a["label"], entry_b["label"]],
                            "atoms": [i, j],
                            "bond_order": order,
                            "smiles": Chem.MolToSmiles(product),
                            "formula": rdMolDescriptors.CalcMolFormula(product),
                            "round": round_index,
                        }
                        if identity in known:
                            if round_index == 0:
                                hit = covered.setdefault(identity, {
                                    "product": (entry_a["label"] if identity == entry_a["identity"]
                                                else entry_b["label"]),
                                    "routes": [],
                                })
                                hit["routes"].append(route)
                            continue
                        bucket = gaps if round_index == 0 else new_products
                        gap = bucket.setdefault(identity, {
                            "smiles": route["smiles"],
                            "formula": route["formula"],
                            "residual_radicals": sum(
                                atom.GetNumRadicalElectrons() for atom in product.GetAtoms()),
                            "routes": [],
                            "new_channels_ready": [],
                            "new_channels_blocked": [],
                        })
                        gap["routes"].append(route)
                        if len(gap["routes"]) == 1:
                            for frag_a, frag_b, bond_type in _single_bond_scissions(product):
                                record = {"bond_type": bond_type, "fragments": [frag_a, frag_b]}
                                if frag_a in known and frag_b in known:
                                    gap["new_channels_ready"].append(record)
                                else:
                                    gap["new_channels_blocked"].append(record)
                            new_entries.append({
                                "label": str(route["smiles"]),
                                "identity": identity,
                                "mol": product,
                                "points": _connection_points(product, ()),
                            })
        return new_products, new_entries

    # ---- 第 0 轮：现有物种表 ------------------------------------------------
    enumerate_round(entries, known_identities, 0)
    rounds.append({"round": 0,
                   "universe": len(entries),
                   "covered_pairs": len(covered),
                   "new_species": len(gaps),
                   "new_channels_ready": sum(len(g["new_channels_ready"]) for g in gaps.values())})

    closure_universe = list(entries)
    if args.iterate:
        known = set(known_identities)
        for round_index in range(1, args.max_rounds + 1):
            new_products, new_entries = enumerate_round(closure_universe, known, round_index)
            if not new_entries:
                break
            rounds.append({
                "round": round_index,
                "universe": len(closure_universe) + len(new_entries),
                "new_species": len(new_entries),
                "new_channels_ready": sum(len(g["new_channels_ready"]) for g in new_products.values()),
                "new_channels_blocked": sum(len(g["new_channels_blocked"]) for g in new_products.values()),
                "species": [
                    {"smiles": g["smiles"], "formula": g["formula"],
                     "residual_radicals": g["residual_radicals"],
                     "routes": len(g["routes"]),
                     "ready": len(g["new_channels_ready"]),
                     "blocked": len(g["new_channels_blocked"])}
                    for g in sorted(new_products.values(),
                                    key=lambda g: (-len(g["new_channels_ready"]), g["formula"]))
                ],
            })
            known |= {entry["identity"] for entry in new_entries}
            closure_universe += new_entries

    # 把 gap 里的物种片段身份翻译成 key（可读）
    def label(frag_identity: str) -> str:
        species = by_identity.get(frag_identity)
        return species.key if species else "<不在表内>"

    for gap in gaps.values():
        for bucket in ("new_channels_ready", "new_channels_blocked"):
            for record in gap[bucket]:
                record["fragments"] = [label(record["fragments"][0]), label(record["fragments"][1])]

    report = {
        "schema": "ft2dp_c2_closure_audit/1",
        "budget": {"max_C": args.max_c, "max_O": args.max_o,
                   "max_radicals": args.max_radicals, "allow_OO": bool(args.allow_oo)},
        "iterate": bool(args.iterate),
        "species_unique": len(entries),
        "covered_pairs": len(covered),
        "gap_products": len(gaps),
        "rounds": rounds,
        "gaps": gaps,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "closure_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# C2 网络成键封闭性审计（机械枚举）",
        "",
        f"- 元素预算：C ≤ {args.max_c}、O ≤ {args.max_o}",
        f"- 产物允许残留悬挂价（未成对电子）≤ {args.max_radicals}（吸附物靠悬挂价表面成键）",
        f"- O–O 键：{'允许' if args.allow_oo else '**禁止**（贫氧碳化物表面无过氧化学）'}",
        f"- 表内唯一身份物种：**{len(entries)}**（外加入表条目 {len(table)}，含同分子气相参考）",
        f"- 已覆盖组合（产物命中表内物种）：**{len(covered)}**",
        f"- **缺失产物物种（gap）：{len(gaps)}**",
        "",
        "| 缺失产物 | 式 | 代表路线 | 可立即枚举的新通道 | 碎片未入表的通道 |",
        "|---|---|---|---|---|",
    ]
    for gap in sorted(gaps.values(), key=lambda g: -len(g["new_channels_ready"])):
        route = gap["routes"][0]
        route_text = " + ".join(route["components"]) + f" (o={route['bond_order']})"
        lines.append(
            f"| `{gap['smiles']}` | {gap['formula']}(rad {gap['residual_radicals']}) | {route_text} | "
            f"{len(gap['new_channels_ready'])} | {len(gap['new_channels_blocked'])} |"
        )
    lines += ["", "## 立即可枚举的新通道", ""]
    for gap in sorted(gaps.values(), key=lambda g: -len(g["new_channels_ready"])):
        if not gap["new_channels_ready"]:
            continue
        lines.append(f"### `{gap['smiles']}`（{gap['formula']}）")
        for record in gap["new_channels_ready"]:
            lines.append(f"- {record['bond_type']}：{record['fragments'][0]} + {record['fragments'][1]}")
        lines.append("")

    if args.iterate:
        lines += ["", "## 闭包迭代（到不动点）", "",
                  "| 轮 | 宇宙规模 | 新增物种 | 新增可枚举通道 | 新增待解锁通道 |", "|---|---|---|---|---|"]
        for record in rounds:
            lines.append(
                f"| {record['round']} | {record.get('universe', '')} | {record.get('new_species', '')} | "
                f"{record.get('new_channels_ready', 0)} | {record.get('new_channels_blocked', 0)} |")
        lines += ["", "### 各轮新增物种", ""]
        for record in rounds:
            if record["round"] == 0 or not record.get("species"):
                continue
            lines.append(f"**round {record['round']}（{len(record['species'])} 个）**")
            for item in record["species"][:40]:
                lines.append(f"- `{item['smiles']}` — {item['formula']} "
                             f"(悬挂价 {item['residual_radicals']}，路线 {item['routes']} 条，"
                             f"可枚举通道 {item.get('ready', 0)}，待解锁 {item.get('blocked', 0)})")
            lines.append("")
    (out_dir / "closure_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[closure] 缺失产物 {len(gaps)} 个；报告写入 {out_dir}/closure_audit.{{json,md}}",
          file=sys.stderr)
    for gap in sorted(gaps.values(), key=lambda g: -len(g["new_channels_ready"]))[:12]:
        route = gap["routes"][0]
        print(f"  - {gap['formula']:8s} {gap['smiles']:20s} via {' + '.join(route['components'])}"
              f"  可枚举通道={len(gap['new_channels_ready'])}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
