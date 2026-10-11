#!/usr/bin/env python3
"""从 bulk 结构切出任意 Miller 指数的表面板（Recnet 可用的 slab CIF）。

依赖 pymatgen（SAI 的 `ase` / `atst` env 已含；本机可 `pip install pymatgen`）。

示例::

    # 1) 列出该 Miller 指数下所有可能终止，挑一个
    python build_slab.py Fe5C2_bulk.vasp --miller 5 1 0 --list

    # 2) 生成 slab（3x2 面内扩胞、约 4 层、12 Å 真空）
    python build_slab.py Fe5C2_bulk.vasp --miller 5 1 0 --termination 0 \
        --size 3 2 --layers 8 --vacuum 12 --out Fe5C2_510.cif

输出报告包含：化学式/原子数、分层与层间距、表层元素（Recnet 位点识别默认按 Fe 表层）、
最近原子距离、以及建议的 `--bottom-freeze-threshold` 与可直接粘贴的 sbatch 参数。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read as ase_read, write as ase_write

try:  # 直接运行（python build_slab.py）时
    from slab_utils import (
        group_layers,
        min_distance,
        normal_axis_alignment,
        orient_normal_to_z,
        suggest_bottom_freeze_threshold,
        top_layer_elements,
    )
except ImportError:  # 作为包导入时
    from .slab_utils import (  # type: ignore[no-redef]
        group_layers,
        min_distance,
        normal_axis_alignment,
        orient_normal_to_z,
        suggest_bottom_freeze_threshold,
        top_layer_elements,
    )


def _load_bulk(path: Path):
    from pymatgen.core import Structure

    return Structure.from_file(str(path))


def _slabs(bulk, miller, min_slab: float, vacuum: float, lll: bool, center: bool):
    from pymatgen.core.surface import SlabGenerator

    gen = SlabGenerator(
        bulk,
        miller_index=miller,
        min_slab_size=min_slab,
        min_vacuum_size=vacuum,
        lll_reduce=lll,
        center_slab=center,
    )
    return list(gen.get_slabs())


def _slab_report(atoms: Atoms) -> str:
    from collections import Counter

    lines = []
    comp = Counter(atoms.get_chemical_symbols())
    lines.append(f"  化学式        : {atoms.get_chemical_formula()}  ({dict(comp)})")
    lines.append(f"  原子数        : {len(atoms)}")
    cell = atoms.get_cell()
    lines.append(
        "  晶胞 (Å)      : a=%.3f b=%.3f c=%.3f  (|c|=%.2f)"
        % (np.linalg.norm(cell[0]), np.linalg.norm(cell[1]), np.linalg.norm(cell[2]), np.linalg.norm(cell[2]))
    )
    idx, cos = normal_axis_alignment(atoms)
    lines.append(
        f"  法向对齐      : 第 {idx} 个晶格向量与 z 轴夹角余弦 = {cos:.3f}"
        + ("" if cos > 0.98 else "  [警告] 表面法向未对齐 z 轴，需用 --surface-normal/--normal-axis")
    )
    layers = group_layers(atoms)
    lines.append(f"  层数          : {len(layers)}")
    for k, layer in enumerate(layers):
        elem = ",".join(layer["elements"])
        lines.append(
            f"    L{k:<2d} z=[{layer['z_min']:7.3f},{layer['z_max']:7.3f}] n={layer['n_atoms']:<4d} {elem}"
        )
    if len(layers) >= 2:
        spacings = [layers[k + 1]["center"] - layers[k]["center"] for k in range(len(layers) - 1)]
        lines.append("  层间距 (Å)    : " + ", ".join(f"{s:.2f}" for s in spacings))
    top = top_layer_elements(atoms, window=1.5)
    lines.append(
        f"  表层 1.5Å 内  : {top}"
        + ("" if "Fe" in top else "  [警告] 表层无 Fe —— Recnet 位点识别默认以 Fe 表层为基准")
    )
    dmin, (i, j) = min_distance(atoms)
    flag = "" if dmin > 1.2 else "  [警告] 原子过近，构建可能有误"
    lines.append(f"  最近原子距离  : {dmin:.3f} Å (#{i} {atoms[i].symbol} - #{j} {atoms[j].symbol}){flag}")
    threshold, _ = suggest_bottom_freeze_threshold(atoms, n_freeze_layers=2)
    if threshold is not None:
        lines.append(f"  建议冻结阈值  : --bottom-freeze-threshold {threshold:.3f}   (冻结底部 2 层)")
    else:
        lines.append("  [警告] 层数不足 3，无法给出冻结阈值建议")
    return "\n".join(lines)


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description="切 surface slab（pymatgen）并输出 Recnet 就绪的 CIF")
    parser.add_argument("bulk", help="bulk 结构文件（cif/vasp/poscar，ASE/pymatgen 可读）")
    parser.add_argument("--miller", nargs=3, type=int, required=True, metavar=("H", "K", "L"))
    parser.add_argument("--size", nargs=2, type=int, default=(1, 1), metavar=("U", "V"), help="面内扩胞 (默认 1 1)")
    parser.add_argument("--layers", type=float, default=8.0, help="最小板厚 (Å，默认 8)")
    parser.add_argument("--vacuum", type=float, default=12.0, help="最小真空层 (Å，默认 12)")
    parser.add_argument("--termination", type=int, default=None, help="终止方式索引（先用 --list 查看）")
    parser.add_argument("--list", action="store_true", help="只列出所有终止方式")
    parser.add_argument("--no-lll", action="store_true", help="禁用 LLL 约化")
    parser.add_argument("--out", default=None, help="输出 CIF 路径（默认 <bulk>_<hkl>.cif）")
    args = parser.parse_args(argv)

    bulk_path = Path(args.bulk).expanduser()
    if not bulk_path.exists():
        print(f"ERROR: bulk 不存在: {bulk_path}", file=sys.stderr)
        return 2
    try:
        bulk = _load_bulk(bulk_path)
    except ImportError:
        print("ERROR: 需要 pymatgen（SAI: 用 ase/atst env；本机: pip install pymatgen）", file=sys.stderr)
        return 2

    miller = tuple(int(x) for x in args.miller)
    slabs = _slabs(bulk, miller, args.layers, args.vacuum, not args.no_lll, True)
    if not slabs:
        print(f"ERROR: Miller {miller} 未生成任何 slab", file=sys.stderr)
        return 2

    print(f"bulk      : {bulk_path}")
    print(f"Miller    : {miller}   终止数: {len(slabs)}")
    for i, slab in enumerate(slabs):
        atoms = _to_ase(slab)
        layers = group_layers(atoms)
        top = top_layer_elements(atoms, window=1.5)
        print(f"  [{i}] {slab.composition.reduced_formula:<14s} n={len(atoms):<4d} layers={len(layers):<3d} top={top}")
    if args.list:
        print("\n用 --termination <索引> 选定终止。")
        return 0

    term = 0 if args.termination is None else int(args.termination)
    if term < 0 or term >= len(slabs):
        print(f"ERROR: --termination {term} 超出范围 0..{len(slabs)-1}", file=sys.stderr)
        return 2
    slab = slabs[term]
    if args.size != [1, 1]:
        slab = slab.make_supercell([args.size[0], args.size[1], 1])

    # 统一对齐：法向 = +z；写 VASP（显式笛卡尔，取向无损）而不是 CIF
    atoms = orient_normal_to_z(_to_ase(slab))
    if args.out:
        out = Path(args.out)
    else:
        out = bulk_path.with_name(f"{bulk_path.stem}_{miller[0]}{miller[1]}{miller[2]}.vasp")
    suffix = out.suffix.lower()
    if suffix in (".vasp", ".poscar"):
        ase_write(out, atoms, format="vasp", direct=False, vasp5=True)
    else:
        ase_write(out, atoms)
        if suffix == ".cif":
            print(
                "[警告] CIF 只存晶胞参数+分数坐标，会丢失笛卡尔取向；"
                "管线按「z 轴=表面法向」读取，建议改用 .vasp（POSCAR）输出。"
            )

    atoms = ase_read(str(out))  # 用「回读结果」做报告，保证与管线看到的一致
    print(f"\n终止 [{term}] -> 写出: {out}\n")
    print(_slab_report(atoms))
    print(
        "\n提交示例:\n"
        f"  RECNET_CASE=<case目录> RECNET_SLAB={out} \\\n"
        "  RECNET_EXTRA_ARGS=\"--use-c-vacancy-io --gas-species-whitelist sp_002 "
        "--bottom-freeze-threshold <上面建议值>\" \\\n"
        "    sbatch 01_run_pipeline.sbatch"
    )
    return 0


def _to_ase(slab) -> Atoms:
    """pymatgen Structure -> ASE Atoms（避免 pymatgen 的 ASE 适配器版本差异）。"""
    symbols = [str(site.specie) for site in slab]
    positions = np.array([site.coords for site in slab], dtype=float)
    cell = np.array(slab.lattice.matrix, dtype=float)
    atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)
    return atoms


if __name__ == "__main__":
    raise SystemExit(main())
