#!/usr/bin/env python3
"""提交前预检：校验一个 Recnet case 目录（slab + prepared_data）是否可跑。

检查项（PASS/WARN/FAIL）：
  1. 目录结构：slab 文件、prepared_data/prepared_rmg_data.yaml 是否存在
  2. yaml 结构：rxns / species 字段完整（对照 handlers/context.py 的读取口径）
  3. 模板：每个 species 的 template_xyz 存在、可读；ad_idx 索引合法
  4. 反应：broken_bond 为 2 个整数，且能在 reactant_species[0] 的模板内索引
  5. 元素：slab+模板的元素集合 vs 模型域（C/Fe/H/O）
  6. 几何：slab 最近原子距离、层结构、表层元素（Recnet 默认按 Fe 表层识别位点）、
     法向是否对齐 z 轴、建议的 --bottom-freeze-threshold
  7. 可选（--dp-check）：用 FT2DP 模型对 slab 做单点，检查能量/力是否有限

退出码：0 = 无 FAIL；1 = 存在 FAIL；2 = 参数/依赖错误。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from ase.io import read as ase_read

try:
    from slab_utils import (
        MODEL_DOMAIN_ELEMENTS,
        element_coverage,
        group_layers,
        min_distance,
        normal_axis_alignment,
        suggest_bottom_freeze_threshold,
        top_layer_elements,
    )
except ImportError:
    from .slab_utils import (  # type: ignore[no-redef]
        MODEL_DOMAIN_ELEMENTS,
        element_coverage,
        group_layers,
        min_distance,
        normal_axis_alignment,
        suggest_bottom_freeze_threshold,
        top_layer_elements,
    )

SLAB_SUFFIXES = (".cif", ".vasp", ".poscar", ".POSCAR", ".xyz", ".str", ".stru")


class Reporter:
    def __init__(self) -> None:
        self.n_fail = 0
        self.n_warn = 0

    def ok(self, msg: str) -> None:
        print(f"  [PASS] {msg}")

    def warn(self, msg: str) -> None:
        self.n_warn += 1
        print(f"  [WARN] {msg}")

    def fail(self, msg: str) -> None:
        self.n_fail += 1
        print(f"  [FAIL] {msg}")


def _find_slab(case: Path) -> Path | None:
    candidates = [
        p for p in sorted(case.iterdir())
        if p.is_file() and (p.suffix in SLAB_SUFFIXES or p.name.upper().startswith("POSCAR"))
    ]
    return candidates[0] if len(candidates) >= 1 else None


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description="Recnet case 提交前预检")
    parser.add_argument("case", help="case 目录")
    parser.add_argument("--slab", default=None, help="slab 文件（默认在 case 内自动寻找）")
    parser.add_argument("--prepared", default=None, help="prepared yaml（默认 <case>/prepared_data/prepared_rmg_data.yaml）")
    parser.add_argument("--dp-check", action="store_true", help="额外做一次 FT2DP 单点（需 deepmd + GPU/CPU）")
    parser.add_argument("--model", default=None, help="覆盖模型路径（默认按适配器解析）")
    parser.add_argument("--head", default=None, help="覆盖 head（默认按适配器解析）")
    args = parser.parse_args(argv)

    case = Path(args.case).expanduser().resolve()
    rep = Reporter()
    print(f"== recnet case preflight ==\ncase: {case}\n")

    # ---- 1. 目录结构 ----
    print("[1] 目录结构")
    if not case.is_dir():
        rep.fail(f"case 目录不存在: {case}")
        return 1
    slab_path = Path(args.slab).expanduser() if args.slab else _find_slab(case)
    prepared_path = (
        Path(args.prepared).expanduser() if args.prepared else case / "prepared_data" / "prepared_rmg_data.yaml"
    )
    if slab_path and slab_path.exists():
        rep.ok(f"slab: {slab_path}")
    else:
        rep.fail(f"未找到 slab 文件（用 --slab 指定）: {slab_path}")
    if prepared_path.exists():
        rep.ok(f"prepared: {prepared_path}")
    else:
        rep.fail(f"prepared 数据不存在: {prepared_path}")
    if rep.n_fail:
        print(f"\nresult: FAIL (fail={rep.n_fail}, warn={rep.n_warn})")
        return 1

    # ---- 2. yaml 结构 ----
    print("\n[2] prepared yaml 结构")
    with open(prepared_path, "r", encoding="utf-8") as fh:
        payload = yaml.safe_load(fh) or {}
    rxns = payload.get("rxns")
    species = payload.get("species")
    if not isinstance(rxns, list) or not rxns:
        rep.fail("rxns 为空或缺失")
        rxns = []
    else:
        rep.ok(f"rxns: {len(rxns)} 条反应")
    if not isinstance(species, dict) or not species:
        rep.fail("species 为空或缺失")
        species = {}
    else:
        n_gas = sum(1 for rec in species.values() if not rec.get("ad_idx"))
        rep.ok(f"species: {len(species)} 个（其中 ad_idx 为空的“气相”物种 {n_gas} 个）")

    # ---- 3. 模板 ----
    print("\n[3] 吸附模板")
    base_dir = prepared_path.parent
    templates: dict[str, object] = {}
    element_pool: set[str] = set()
    worst = (np.inf, "")
    for sp_id, rec in species.items():
        if not isinstance(rec, dict) or "template_xyz" not in rec:
            rep.fail(f"{sp_id}: 缺少 template_xyz 字段")
            continue
        tpl_path = (base_dir / rec["template_xyz"]).resolve()
        if not tpl_path.exists():
            rep.fail(f"{sp_id}: 模板不存在 {tpl_path}")
            continue
        atoms = ase_read(str(tpl_path))
        templates[sp_id] = atoms
        element_pool |= set(atoms.get_chemical_symbols())
        dmin, _ = min_distance(atoms, mic=bool(atoms.pbc.any()))
        if dmin < worst[0]:
            worst = (dmin, f"{sp_id} ({tpl_path.name})")
        ad_idx = rec.get("ad_idx", [])
        if not isinstance(ad_idx, list):
            rep.fail(f"{sp_id}: ad_idx 不是列表")
        elif any((not isinstance(i, int)) or i < 0 or i >= len(atoms) for i in ad_idx):
            rep.fail(f"{sp_id}: ad_idx 越界 {ad_idx}（模板 {len(atoms)} 原子）")
    if templates:
        rep.ok(f"模板可读: {len(templates)}/{len(species)}")
        if worst[0] < 0.9:
            rep.warn(f"模板内最近原子距离过小: {worst[0]:.3f} Å（{worst[1]}）→ 可能是构建错误")
        else:
            rep.ok(f"模板最小原子间距 {worst[0]:.3f} Å（{worst[1]}）")

    # ---- 4. 反应完整性 ----
    print("\n[4] 反应完整性")
    bad = 0
    for k, rxn in enumerate(rxns):
        if not isinstance(rxn, dict):
            rep.fail(f"rxn[{k}] 不是字典")
            bad += 1
            continue
        rs, ps, bb = rxn.get("reactant_species"), rxn.get("product_species"), rxn.get("broken_bond")
        if not rs:
            rep.fail(f"rxn[{k}]: reactant_species 缺失")
            bad += 1
            continue
        if not ps or len(ps) < 2:
            rep.warn(f"rxn[{k}]: product_species < 2（{ps}）→ 能量评估阶段会跳过该反应")
        if not (isinstance(bb, list) and len(bb) == 2 and all(isinstance(i, int) for i in bb)):
            rep.fail(f"rxn[{k}]: broken_bond 非法: {bb}")
            bad += 1
            continue
        tpl = templates.get(rs[0])
        if tpl is None:
            rep.fail(f"rxn[{k}]: reactant_species[0]={rs[0]} 无可用模板")
            bad += 1
            continue
        if max(bb) >= len(tpl):
            rep.fail(
                f"rxn[{k}]: broken_bond {bb} 超出模板范围（{rs[0]} 只有 {len(tpl)} 原子）"
                " → 运行时会 IndexError（检查 prepare 时是否把表面位点也编进了索引）"
            )
            bad += 1
    if not bad and rxns:
        rep.ok(f"{len(rxns)} 条反应的 species/broken_bond 字段与索引均合法")

    # ---- 5. 元素覆盖 ----
    print("\n[5] 元素覆盖（模型域）")
    slab_atoms = ase_read(str(slab_path))
    slab_elements = set(slab_atoms.get_chemical_symbols())
    element_pool |= slab_elements
    outside = element_pool - set(MODEL_DOMAIN_ELEMENTS)
    rep.ok(f"slab+模板涉及元素: {sorted(element_pool)}")
    if outside:
        rep.warn(
            f"域外元素 {sorted(outside)}：模型 type_map 含全周期表可以跑，"
            f"但 FT2DP 训练域只有 {list(MODEL_DOMAIN_ELEMENTS)}，精度无担保"
        )
    else:
        rep.ok("全部元素在 C/Fe/H/O 域内")

    # ---- 6. slab 几何 ----
    print("\n[6] slab 几何")
    layers = group_layers(slab_atoms)
    rep.ok(f"原子数 {len(slab_atoms)}，层数 {len(layers)}：{slab_atoms.get_chemical_formula()}")
    dmin, (i, j) = min_distance(slab_atoms)
    if dmin < 1.2:
        rep.fail(f"最近原子距离过小: {dmin:.3f} Å (#{i}-#{j}) → 结构可能重叠")
    else:
        rep.ok(f"最近原子距离 {dmin:.3f} Å (#{i} {slab_atoms[i].symbol} - #{j} {slab_atoms[j].symbol})")
    top = top_layer_elements(slab_atoms, window=1.5)
    if "Fe" in top:
        rep.ok(f"表层 1.5 Å 内元素: {top}（含 Fe，符合位点识别约定）")
    else:
        rep.warn(f"表层 1.5 Å 内元素: {top}（无 Fe）→ Voronoi 位点识别以 Fe 表层为基准，可能给不出位点")
    idx, cos = normal_axis_alignment(slab_atoms)
    if cos < 0.98:
        rep.warn(
            f"表面法向未对齐 z 轴（cos={cos:.3f}，晶格向量 {idx}）→ 需要 --surface-normal/--normal-axis 参数"
        )
    else:
        rep.ok(f"表面法向对齐 z 轴（cos={cos:.3f}）")
    threshold, _ = suggest_bottom_freeze_threshold(slab_atoms, n_freeze_layers=2)
    if threshold is None:
        rep.warn("层数不足 3，无法建议冻结阈值（--bottom-freeze-threshold）")
    else:
        rep.ok(f"建议 --bottom-freeze-threshold {threshold:.3f}（冻结底部 2 层）")
    if len(slab_atoms) > 255:
        rep.warn(
            f"原子数 {len(slab_atoms)} > 255：v1 QC gate 是 255；当前管线本身无硬上限，"
            "但注意单卡显存与耗时"
        )

    # ---- 7. 可选 DP 单点 ----
    if args.dp_check:
        print("\n[7] DP 单点（FT2DP 后端）")
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
            import recnet_adapter

            report = recnet_adapter.install(model=args.model, head=args.head, verbose=False)
            res = report["resolution"]
            from recnet_adapter.model_selection import create_calculator

            atoms = slab_atoms.copy()
            atoms.calc = create_calculator(res.model, res.head)
            t0 = time.perf_counter()
            energy = float(atoms.get_potential_energy())
            forces = atoms.get_forces()
            dt = time.perf_counter() - t0
            fmax = float(np.sqrt((forces ** 2).sum(axis=1).max()))
            if np.isfinite(energy) and np.isfinite(forces).all():
                rep.ok(f"E={energy:.4f} eV，Fmax={fmax:.3f} eV/Å，{dt:.2f}s（{len(atoms)} 原子）")
                if fmax > 30:
                    rep.warn(f"Fmax={fmax:.1f} eV/Å 偏大：结构可能远离平衡或超出训练域，建议先做短弛豫")
            else:
                rep.fail("能量/力非有限值")
        except Exception as exc:  # noqa: BLE001
            rep.fail(f"DP 单点失败: {type(exc).__name__}: {exc}")

    print(f"\nresult: {'FAIL' if rep.n_fail else 'PASS'} (fail={rep.n_fail}, warn={rep.n_warn})")
    if not rep.n_fail:
        print(
            "提交示例:\n"
            f"  RECNET_CASE={case} RECNET_SLAB={slab_path} \\\n"
            f"  RECNET_EXTRA_ARGS=\"--use-c-vacancy-io --bottom-freeze-threshold "
            f"{threshold:.3f}\" \\\n    sbatch 01_run_pipeline.sbatch"
            if threshold
            else "提交示例: 见 recnet_adapter/sai/README.md"
        )
    return 1 if rep.n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
