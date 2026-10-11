#!/usr/bin/env python3
"""用 DP 模型生成 per-site TS seed：断键距离约束扫描。

为什么需要它
------------
C-O 断裂族（CO* -> C* + O* 等）在多位点上让 CCQN 沿解离坐标滑走（末步 d(reactive)
远超过拉伸上限）→ 所有取向被拒 → 该族拿不到 TS。本工具从**已弛豫吸附结构**出发，
对 ``broken_bond`` 做固定键长扫描（ASE ``FixBondLength``，每步弛豫其余原子），
取扫描中能量最高（最接近鞍点）的一帧写成 per-site TS seed::

    <case>/rxn/seeds/<rxn_key>_site_<site>[_vg<k>].xyz

管线侧（``handlers/ts.py``）默认会发现并使用该文件作为初始 TS guess（``--no-ts-seed``
可关闭）。seed 的原子顺序与 case 的组装结构一致（slab + adsorbate）。

输入
----
* ``<case>/prepared_data/prepared_rmg_data.yaml``（反应/物种表）
* ``<case>/rxn/Adsorbates/<species_key>/<site>[_vg<k>]_opt.xyz``（已弛豫吸附结构）
* 模型：``--model/--head`` > 环境变量 ``RECNET_DP_MODEL`` / ``RECNET_DP_HEAD`` >
  ``recnet_adapter/model_config.json``（与 ``recnet_adapter.run`` 同一解析链）

输出
----
* seed 结构：``<out-dir>/<rxn_key>_site_<site>[_vg<k>].xyz``（extxyz）
* 扫描台账（两份，内容一致）：
  ``<out-dir>/SCAN_MANIFEST.json``（规范账本，按 ``<rxn_key>_site_<site>[_vg<k>]`` 合并）
  与 ``<out-dir>/SCAN_MANIFEST_<rxn_key>_site_<site>[_vg<k>].json``（per-target 文件，
  多 site 并发扫描时互不争用）。每步 d/E/fmax/是否收敛 + 所取 seed 帧索引。

依赖与运行环境
--------------
集群 ``dpeva-dpa4`` 环境（ase / deepmd / torch）；**不需要** GPU 也能起（CPU 推理慢）。
``--dry-run`` 只做路径解析与扫描点规划（不建 DP 计算器、不写任何文件）。

例::

    python recnet_adapter/tools/make_seeds_scan.py $R/recnet-runs/pilot/CO-S \\
        --rxn 'CO*' --site 10 --vg 0 --bottom-freeze-threshold 13.456
    # 只规划，不下场算：
    python recnet_adapter/tools/make_seeds_scan.py <case> --rxn 'CO*' --site 10 --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from ase.constraints import FixAtoms, FixBondLength
from ase.data import covalent_radii
from ase.io import read, write
from ase.optimize import QuasiNewton

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_PIPELINE_KEYS = None


def _pipeline_keys():
    """惰性导入管线的 key 工具（2026-09-21：模块级 import 会把 deepmd/sella 拖进
    pytest collection，使无 deepmd 的 env 直接 INTERNALERROR；改为首次调用时导入，
    这样本模块在纯 CPU/无 DP 环境下仍可 import 与做计划类单测）。"""
    global _PIPELINE_KEYS
    if _PIPELINE_KEYS is None:
        try:
            from handlers.context import compute_reaction_key, compute_species_key
        except ImportError as exc:  # noqa: BLE001 - 依赖缺失要给可操作的提示
            raise SystemExit(
                "[make_seeds_scan] ERROR: 无法导入 Recnet 管线包 handlers "
                f"({exc.__class__.__name__}: {exc})；请激活 dpeva-dpa4 环境后重试"
            )
        _PIPELINE_KEYS = (compute_reaction_key, compute_species_key)
    return _PIPELINE_KEYS


def compute_reaction_key(*args, **kwargs):  # noqa: D103 - 转调管线实现，保持调用点不变
    return _pipeline_keys()[0](*args, **kwargs)


def compute_species_key(*args, **kwargs):  # noqa: D103
    return _pipeline_keys()[1](*args, **kwargs)

MANIFEST_NAME = "SCAN_MANIFEST.json"


class ToolError(RuntimeError):
    """工具级错误（用法/输入/模型问题）。"""


# ----------------------------------------------------------------------
#  TS 断键上限阈值（与 handlers.context.WorkflowContext 的默认值同口径）
# ----------------------------------------------------------------------

class _TsBondGateCtx:
    """只读阈值载体：复刻 ``handlers/ts.py:_ts_bond_upper_bound`` 所需的 ctx 字段。

    数值来源 = ``handlers/context.py`` 中 WorkflowContext 的默认值
    (``ts_bond_max_scale_ref`` / ``ts_bond_max_scale_covalent`` / ``ts_bond_max_additive_cap``)。
    ``recnet_adapter/tests/test_make_seeds_scan.py`` 里有一条测试断言本工具的上限公式
    与管线实现一致，避免两边漂移。
    """

    ts_bond_max_scale_ref = 1.9
    ts_bond_max_scale_covalent = 1.6
    ts_bond_max_additive_cap = 1.2


def ts_bond_upper_bound(d_ref: float, atom_i_number: int, atom_j_number: int) -> float:
    """过拉伸上限（与 ``handlers/ts.py:_ts_bond_upper_bound`` 同公式）。"""
    d_ref = float(d_ref)
    r_sum = float(covalent_radii[int(atom_i_number)] + covalent_radii[int(atom_j_number)])
    base_limit = max(
        _TsBondGateCtx.ts_bond_max_scale_ref * d_ref,
        _TsBondGateCtx.ts_bond_max_scale_covalent * r_sum,
    )
    additive_cap = d_ref + _TsBondGateCtx.ts_bond_max_additive_cap
    return float(min(base_limit, additive_cap))


# ----------------------------------------------------------------------
#  扫描点规划 / 选帧（纯函数，便于离线验证）
# ----------------------------------------------------------------------

def plan_scan_distances(d0: float, d_step: float, d_limit: float) -> list[float]:
    """从 d0 起按 d_step 递增到 d_limit（含起点与 d_limit）。

    若 ``d0`` 已超过 ``d_limit``，将其夹到 ``d_limit``。这样解离吸附态（如 H2 的
    backup-site 弛豫产物）不会直接产出超 QC 门槛的 seed；调用方可用模板 ``d_ref``
    作为更合理的重入起点。
    """
    d0 = float(d0)
    d_step = float(d_step)
    d_limit = float(d_limit)
    if d_step <= 0.0:
        raise ToolError(f"d-step must be > 0, got {d_step}")
    if d_limit <= d0:
        return [d_limit]
    points = [d0]
    while points[-1] + d_step <= d_limit + 1e-9:
        points.append(round(points[-1] + d_step, 6))
    if points[-1] < d_limit - 1e-9:
        points.append(round(d_limit, 6))
    return points


def choose_seed_frame(energies: list[float], *, tol: float = 1e-6) -> tuple[int, bool]:
    """取扫描中能量最高的一帧；若能量单调上升则取最后一帧（最接近上限）。

    返回 ``(帧索引, 是否单调上升)``。
    """
    if not energies:
        raise ToolError("empty scan: no energy to choose a seed frame from")
    monotonic = all(b >= a - tol for a, b in zip(energies, energies[1:]))
    if monotonic:
        return len(energies) - 1, True
    return int(np.argmax(np.asarray(energies, dtype=float))), False


# ----------------------------------------------------------------------
#  case / 反应解析
# ----------------------------------------------------------------------

@dataclass
class ScanTarget:
    """一次扫描的目标（反应 + 位点 + 输入/输出路径）。"""

    case_dir: Path
    prepared: Path
    slab_path: Path
    rxn_index: int
    rxn: dict
    rxn_key: str
    species_id: str
    species_key: str
    template_atoms: object
    broken_bond_local: tuple[int, int]
    site: int
    vg: int | None
    products: list[str]
    adsorbate_opt: Path
    seed_path: Path
    manifest_path: Path


def load_prepared(prepared_path: Path) -> dict:
    if not prepared_path.exists():
        raise ToolError(f"prepared data not found: {prepared_path}")
    with open(prepared_path, "r") as fh:
        payload = yaml.safe_load(fh) or {}
    if "rxns" not in payload or "species" not in payload:
        raise ToolError(f"prepared data missing 'rxns'/'species': {prepared_path}")
    return payload


def load_templates(prepared: dict, base_dir: Path) -> dict:
    """读入所有吸附模板（口径同 WorkflowContext.ads_templates）。"""
    templates = {}
    for sp_id, rec in prepared["species"].items():
        template_path = base_dir / rec["template_xyz"]
        if not template_path.exists():
            raise ToolError(f"adsorbate template not found: {template_path}")
        templates[sp_id] = {
            "atoms": read(template_path),
            "ad_idx": rec.get("ad_idx", []),
            "name": rec.get("name", sp_id),
        }
    return templates


def match_reaction(rxns: list[dict], species: dict, selector: str) -> int:
    """把 ``--rxn`` 选择器解析成反应索引（支持索引 / reactant 标签 / reactant 物种名）。"""
    text = str(selector).strip()
    if re.fullmatch(r"\d+", text):
        idx = int(text)
        if not 0 <= idx < len(rxns):
            raise ToolError(f"--rxn index {idx} out of range (0..{len(rxns) - 1})")
        return idx

    hits = []
    wanted = text.rstrip("*").lower()
    for idx, rxn in enumerate(rxns):
        names = [str(rxn.get("reactant", ""))]
        names += [str(species[sp].get("name", sp)) for sp in rxn.get("reactant_species", []) if sp in species]
        if wanted in [name.rstrip("*").lower() for name in names]:
            hits.append(idx)
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise ToolError(f"--rxn {selector!r} matched no reaction; use --list-rxns to inspect")
    raise ToolError(
        f"--rxn {selector!r} matched {len(hits)} reactions ({hits}); pass an explicit index"
    )


def list_reactions(prepared: dict, templates: dict) -> None:
    """打印反应表（含断键元素对），便于挑选 --rxn。"""
    species_key_by_id = {sp: compute_species_key(tpl) for sp, tpl in templates.items()}
    print(f"{'idx':>4}  {'reactant':<12}{'bond':<6}{'rxn_key':<22}broken_bond")
    for idx, rxn in enumerate(prepared["rxns"]):
        sp_id = rxn["reactant_species"][0]
        symbols = templates[sp_id]["atoms"].get_chemical_symbols()
        i, j = (int(x) for x in rxn["broken_bond"][:2])
        pair = f"{symbols[i]}-{symbols[j]}"
        print(
            f"{idx:>4}  {str(rxn.get('reactant', '')):<12}{pair:<6}"
            f"{compute_reaction_key(rxn, species_key_by_id):<22}{[i, j]}"
        )


def resolve_target(args) -> ScanTarget:
    """解析 case 目录 → 反应 → 吸附结构 → seed/manifest 路径。"""
    case_dir = Path(args.case).expanduser().resolve()
    if not case_dir.is_dir():
        raise ToolError(f"case directory not found: {case_dir}")

    prepared = Path(args.prepared).expanduser() if args.prepared else case_dir / "prepared_data" / "prepared_rmg_data.yaml"
    if not prepared.is_absolute():
        prepared = (case_dir / prepared).resolve()
    payload = load_prepared(prepared)
    templates = load_templates(payload, prepared.parent)

    if args.list_rxns:
        list_reactions(payload, templates)
        raise SystemExit(0)
    if args.rxn is None:
        raise ToolError("--rxn is required (reactant label like 'CO*', species name, or index)")

    rxn_index = match_reaction(payload["rxns"], payload["species"], args.rxn)
    rxn = payload["rxns"][rxn_index]
    sp_id = rxn["reactant_species"][0]
    broken_bond = rxn.get("broken_bond") or []
    if len(broken_bond) < 2:
        raise ToolError(f"reaction {rxn_index} has no valid broken_bond: {broken_bond}")
    i, j = int(broken_bond[0]), int(broken_bond[1])
    template_atoms = templates[sp_id]["atoms"]
    for idx in (i, j):
        if not 0 <= idx < len(template_atoms):
            raise ToolError(
                f"broken_bond {[i, j]} out of range for template {sp_id} "
                f"({len(template_atoms)} atoms)"
            )

    species_key_by_id = {sp: compute_species_key(tpl) for sp, tpl in templates.items()}
    species_key = species_key_by_id[sp_id]
    rxn_key = compute_reaction_key(rxn, species_key_by_id)

    slab_path = Path(args.slab).expanduser() if args.slab else case_dir / "Fe5C2_510.vasp"
    if not slab_path.is_absolute():
        slab_path = (case_dir / slab_path).resolve()

    site = int(args.site)
    vg = None if args.vg is None else int(args.vg)
    out_dir = Path(args.out_dir).expanduser() if args.out_dir else case_dir / "rxn" / "seeds"
    if not out_dir.is_absolute():
        out_dir = (case_dir / out_dir).resolve()

    tagged_site = f"{site}" + (f"_vg{vg}" if vg is not None else "")
    seed_name = f"{rxn_key}_site_{tagged_site}"
    manifest_name = f"SCAN_MANIFEST_{seed_name}.json"

    # 吸附结构：口径同 WorkflowContext._adsorbate_dir_candidates
    candidates = [
        case_dir / "rxn" / "Adsorbates" / species_key / f"{tagged_site}_opt.xyz",
        case_dir / "rxn" / "Adsorbates" / str(sp_id) / f"{tagged_site}_opt.xyz",
    ]
    adsorbate_opt = next((p for p in candidates if p.exists()), candidates[0])

    return ScanTarget(
        case_dir=case_dir,
        prepared=prepared.resolve(),
        slab_path=slab_path,
        rxn_index=rxn_index,
        rxn=rxn,
        rxn_key=rxn_key,
        species_id=sp_id,
        species_key=species_key,
        template_atoms=template_atoms,
        broken_bond_local=(i, j),
        site=site,
        vg=vg,
        products=[str(payload["species"][sp].get("name", sp)) for sp in rxn.get("product_species", [])],
        adsorbate_opt=adsorbate_opt,
        seed_path=out_dir / f"{seed_name}.xyz",
        manifest_path=out_dir / manifest_name,
    )


def resolve_model(args) -> tuple[Path | None, str | None, dict]:
    """复用 recnet_adapter 的配置解析链（参数 > 环境变量 > model_config.json）。"""
    from recnet_adapter import resolve as resolve_adapter

    res = resolve_adapter(model=args.model, head=args.head)
    return res.model, res.head, res.source


def assert_model_exists(model: Path | None, source: dict) -> Path:
    if model is None:
        raise ToolError(
            "未配置 DP 模型：请设置 RECNET_DP_MODEL 环境变量或传 --model <检查点路径>"
        )
    model = Path(model)
    if not model.exists():
        raise ToolError(
            f"模型检查点不存在: {model}（来源 {source.get('model')}）；"
            "可用 RECNET_DP_MODEL 或 --model 覆盖"
        )
    return model


# ----------------------------------------------------------------------
#  几何 / 扫描
# ----------------------------------------------------------------------

def load_adsorbate(target: ScanTarget):
    """读入已弛豫吸附结构，并定位 slab/吸附质边界。"""
    if not target.adsorbate_opt.exists():
        raise ToolError(
            f"已弛豫吸附结构不存在: {target.adsorbate_opt}\n"
            "（先跑完吸附阶段，或检查 --site/--vg 与 case 目录）"
        )
    atoms = read(target.adsorbate_opt)
    n_slab = len(atoms) - len(target.template_atoms)
    if n_slab <= 0:
        raise ToolError(
            f"组装结构原子数({len(atoms)}) 小于模板原子数({len(target.template_atoms)})："
            f"{target.adsorbate_opt}"
        )
    return atoms, n_slab


def cross_check_slab(atoms, n_slab: int, slab_path: Path) -> str | None:
    """组装结构前 n_slab 个原子与 case slab 文件比对（vacancy case 允许不一致）。"""
    if not slab_path.exists():
        return f"slab 文件不存在（忽略比对）: {slab_path}"
    try:
        slab = read(slab_path)
    except Exception as exc:  # noqa: BLE001 - 只警告
        return f"slab 文件不可读（忽略比对）: {slab_path} ({exc})"
    if len(slab) == n_slab and list(slab.get_chemical_symbols()) == list(
        atoms.get_chemical_symbols()[:n_slab]
    ):
        return None
    return (
        f"case slab 文件与组装结构前 {n_slab} 个原子不一致"
        f"（slab={len(slab)} atoms；--use-c-vacancy-io 生成的空位板属正常情况，"
        "以组装结构为准）"
    )


def plan_scan(target: ScanTarget, atoms, n_slab: int, args) -> tuple[float, list[float], float, float]:
    """读 d0、算上限、规划扫描点。

    上限用**模板里断键对的距离**作 ``d_ref``：``handlers/ts.py`` 的过拉伸判据
    (``_ts_bond_upper_bound``) 用的就是模板距离，seed 距离超过该门槛会在 CCQN 之后
    被判 over-elongated 而拒收。所以扫描上限 = ``min(--d-max, 该门槛)``，
    保证产出的 seed 落在管线自己的 QC 区间内。
    """
    i, j = target.broken_bond_local
    gi, gj = i + n_slab, j + n_slab
    pos = atoms.get_positions()
    d0 = float(np.linalg.norm(pos[gj] - pos[gi]))
    tpl_pos = target.template_atoms.get_positions()
    d_ref = float(np.linalg.norm(tpl_pos[j] - tpl_pos[i]))
    d_cap = ts_bond_upper_bound(d_ref, int(atoms[gi].number), int(atoms[gj].number))
    d_limit = min(float(args.d_max), d_cap)
    scan_start = d0 if d0 <= d_limit else min(d_ref, d_limit)
    return d0, plan_scan_distances(scan_start, args.d_step, d_limit), d_cap, d_ref


def run_scan(atoms, target: ScanTarget, n_slab: int, d_points: list[float],
             calc, args) -> tuple[list[dict], list]:
    """固定键长扫描：每步弛豫其余原子，记录 d/E/fmax/是否收敛。"""
    i, j = target.broken_bond_local
    gi, gj = i + n_slab, j + n_slab

    freeze_indices: list[int] = []
    if args.bottom_freeze_threshold is not None:
        axis_id = {"x": 0, "y": 1, "z": 2}[args.normal_axis]
        freeze_indices = [
            atom.index for atom in atoms
            if float(atom.position[axis_id]) < float(args.bottom_freeze_threshold)
        ]

    records: list[dict] = []
    frames: list = []
    current = atoms
    for d_target in d_points:
        frame = current
        # 先卸掉上一步的约束，否则 set_positions 会立刻把新坐标投影回旧键长
        frame.set_constraint([])
        pos = frame.get_positions()
        vec = pos[gj] - pos[gi]
        norm = float(np.linalg.norm(vec))
        if norm < 1e-9:
            raise ToolError(f"反应键长过短（{norm}）：无法做约束扫描")
        unit = vec / norm
        delta = (float(d_target) - norm) / 2.0
        pos[gi] -= unit * delta
        pos[gj] += unit * delta
        frame.set_positions(pos)

        constraints = []
        if freeze_indices:
            constraints.append(FixAtoms(indices=freeze_indices))
        constraints.append(FixBondLength(gi, gj))
        frame.set_constraint(constraints)
        frame.calc = calc

        record = {"d": round(float(d_target), 6), "E": None, "fmax": None,
                  "d_final": None, "converged": None, "steps": None, "error": None}
        try:
            opt = QuasiNewton(frame, logfile=None)
            record["converged"] = bool(opt.run(fmax=float(args.fmax), steps=int(args.steps)))
            record["steps"] = int(opt.get_number_of_steps())
            record["E"] = float(frame.get_potential_energy())
            record["d_final"] = float(
                np.linalg.norm(frame.get_positions()[gj] - frame.get_positions()[gi])
            )
            forces = frame.get_forces()
            norms = np.sqrt((forces ** 2).sum(axis=1))
            if freeze_indices:
                norms = np.delete(norms, freeze_indices)
            record["fmax"] = float(norms.max()) if norms.size else 0.0
        except Exception as exc:  # noqa: BLE001 - 单点失败不终止整条扫描
            record["error"] = f"{exc.__class__.__name__}: {exc}"
        records.append(record)
        frames.append(frame.copy())
        current = frame
    return records, frames


# ----------------------------------------------------------------------
#  输出
# ----------------------------------------------------------------------

def print_scan_table(records: list[dict], seed_index: int, monotonic: bool,
                     seed_path: Path) -> None:
    print(f"\n[make_seeds_scan] 扫描结果 ({len(records)} 帧)")
    print(f"{'#':>3}  {'d(Å)':>7}  {'E(eV)':>14}  {'fmax':>9}  {'conv':>5}  note")
    for idx, rec in enumerate(records):
        e = "" if rec["E"] is None else f"{rec['E']:.6f}"
        f = "" if rec["fmax"] is None else f"{rec['fmax']:.4f}"
        note = rec["error"] or ("<-- seed" if idx == seed_index else "")
        print(f"{idx:>3}  {rec['d']:>7.3f}  {e:>14}  {f:>9}  {str(rec['converged']):>5}  {note}")
    print(f"[make_seeds_scan] 能量{'单调上升' if monotonic else '非单调'}；"
          f"取帧 #{seed_index} (d={records[seed_index]['d']:.3f} Å)")
    print(f"[make_seeds_scan] seed -> {seed_path}")


def _merge_manifest_scan(path, case_dir, scan_entry: dict) -> None:
    """把一条扫描记录按 id 合并进 manifest（同一 id 覆盖，其它保留）。"""
    payload = {"case": str(case_dir), "scans": []}
    if path.exists():
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            print(f"[make_seeds_scan] WARNING: 现有 manifest 不可解析，将覆盖: {path}")
            payload = {"case": str(case_dir), "scans": []}
    payload["scans"] = [
        s for s in payload.get("scans", []) if s.get("id") != scan_entry["id"]
    ]
    payload["scans"].append(scan_entry)
    payload["case"] = str(case_dir)
    payload["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def write_manifest(target: ScanTarget, records: list[dict], *, seed_index: int,
                   monotonic: bool, model, head, source: dict, args,
                   d0: float, d_cap: float, d_ref: float, files: dict) -> None:
    """写出扫描台账。

    两份：
      * ``<out-dir>/SCAN_MANIFEST.json``（规范账本，按 ``<rxn_key>_site_<site>[_vg<k>]`` 合并）；
      * ``<out-dir>/SCAN_MANIFEST_<rxn_key>_site_<site>[_vg<k>].json``（per-target 副本，
        多 site 并发扫描时互不争用）。
    """
    target.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    scan_entry = {
        "id": files["id"],
        "rxn_index": target.rxn_index,
        "rxn_key": target.rxn_key,
        "reactant": target.rxn.get("reactant"),
        "site": target.site,
        "vg": target.vg,
        "species_id": target.species_id,
        "species_key": target.species_key,
        "broken_bond_local": list(target.broken_bond_local),
        "broken_bond_global": list(files["bond_global"]),
        "adsorbate_opt": str(target.adsorbate_opt),
        "slab_file": str(target.slab_path),
        "n_slab": files["n_slab"],
        "d0": round(float(d0), 6),
        "d_ref_template": files["d_ref_template"],
        "d_cap": round(float(d_cap), 6),
        "d_step": float(args.d_step),
        "d_max": float(args.d_max),
        "fmax": float(args.fmax),
        "steps": int(args.steps),
        "bottom_freeze_threshold": (
            None if args.bottom_freeze_threshold is None else float(args.bottom_freeze_threshold)
        ),
        "normal_axis": args.normal_axis,
        "model": None if model is None else str(model),
        "head": head,
        "model_source": source,
        "records": records,
        "seed_frame": int(seed_index),
        "monotonic_increasing": bool(monotonic),
        "seed_path": str(target.seed_path),
        "command": " ".join(sys.argv),
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    _merge_manifest_scan(target.manifest_path, target.case_dir, scan_entry)
    print(f"[make_seeds_scan] manifest -> {target.manifest_path}")

    canonical = target.manifest_path.parent / MANIFEST_NAME
    if canonical != target.manifest_path:
        _merge_manifest_scan(canonical, target.case_dir, scan_entry)
        print(f"[make_seeds_scan] manifest (canonical) -> {canonical}")


# ----------------------------------------------------------------------
#  CLI
# ----------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="make_seeds_scan.py",
        description="断键距离约束扫描生成 per-site TS seed（Recnet 管线 rxn/seeds/）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("case", help="case 目录（含 prepared_data/ 与 rxn/Adsorbates/）")
    parser.add_argument("--prepared", default=None, help="prepared yaml（默认 <case>/prepared_data/prepared_rmg_data.yaml）")
    parser.add_argument("--slab", default=None, help="slab 文件（默认 <case>/Fe5C2_510.vasp；仅用于一致性核对）")
    parser.add_argument("--rxn", default=None, help="反应选择器：reactant 标签（如 'CO*'）、物种名（'CO'）或索引")
    parser.add_argument("--list-rxns", action="store_true", help="打印反应表（含断键元素对）后退出")
    parser.add_argument("--site", type=int, required=False, default=None, help="位点索引（Adsorbates 目录里的 <site>_opt.xyz）")
    parser.add_argument("--vg", type=int, default=None, help="vacancy group 索引（对应文件名后缀 _vg<k>）")
    parser.add_argument("--out-dir", default=None, help="seed/manifest 输出目录（默认 <case>/rxn/seeds）")

    parser.add_argument("--model", default=None, help="DP 检查点路径（默认 环境变量/配置）")
    parser.add_argument("--head", default=None, help="多头检查点的 head 名（单头模型传 none）")

    parser.add_argument("--d-step", type=float, default=0.2, help="扫描步长（Å，默认 0.2）")
    parser.add_argument("--d-max", type=float, default=2.2, help="扫描上限（Å，默认 2.2；实际再取 min(与过拉伸上限)）")
    parser.add_argument("--fmax", type=float, default=0.05, help="每步约束弛豫的 fmax（默认 0.05）")
    parser.add_argument("--steps", type=int, default=200, help="每步约束弛豫的最大步数（默认 200）")
    parser.add_argument("--bottom-freeze-threshold", type=float, default=None,
                        help="冻结低于该法向坐标的原子（与管线 --bottom-freeze-threshold 同义；默认不冻结）")
    parser.add_argument("--normal-axis", choices=["x", "y", "z"], default="z", help="法向轴（默认 z）")
    parser.add_argument("--dry-run", action="store_true", help="只做路径解析与扫描点规划，不建 DP、不写文件")
    parser.add_argument("--dump-frames", action="store_true",
                        help="把每个 d 上的约束弛豫帧也写盘（<stem>_d<d>.xyz），"
                             "用于把模型扫描升级成 DFT 剖面；默认只写 seed 帧")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.list_rxns:
            resolve_target(args)  # 内部打印反应表后 SystemExit(0)
            return 0
        if args.site is None:
            parser.error("--site is required (或者用 --list-rxns 先看反应表)")

        model, head, source = resolve_model(args)
        if not args.dry_run:
            model = assert_model_exists(model, source)

        target = resolve_target(args)
        print(f"[make_seeds_scan] case      : {target.case_dir}")
        print(f"[make_seeds_scan] prepared  : {target.prepared}")
        print(f"[make_seeds_scan] reaction  : #{target.rxn_index} {target.rxn.get('reactant')} "
              f"-> {' + '.join(target.products)}")
        print(f"[make_seeds_scan] keys      : rxn_key={target.rxn_key} species_key={target.species_key}")
        print(f"[make_seeds_scan] adsorbate : {target.adsorbate_opt} "
              f"({'found' if target.adsorbate_opt.exists() else 'MISSING'})")
        print(f"[make_seeds_scan] model     : "
              f"{model if model is not None else '<unset>'} "
              f"{'' if model is not None and Path(model).exists() else '[MISSING]'} "
              f"(source={source['model']}) head={head!r} (source={source['head']})")

        if not target.adsorbate_opt.exists():
            raise ToolError(f"已弛豫吸附结构不存在: {target.adsorbate_opt}")

        atoms, n_slab = load_adsorbate(target)
        warning = cross_check_slab(atoms, n_slab, target.slab_path)
        if warning:
            print(f"[make_seeds_scan] WARNING: {warning}")
        d0, d_points, d_cap, d_ref = plan_scan(target, atoms, n_slab, args)
        gi, gj = target.broken_bond_local[0] + n_slab, target.broken_bond_local[1] + n_slab
        print(f"[make_seeds_scan] bond      : local={list(target.broken_bond_local)} "
              f"global=[{gi}, {gj}] {atoms[gi].symbol}-{atoms[gj].symbol} "
              f"d0={d0:.3f} Å d_ref(template)={d_ref:.3f} Å "
              f"d_cap={d_cap:.3f} Å (管线过拉伸门槛)")
        print(f"[make_seeds_scan] plan      : {len(d_points)} points -> "
              + ", ".join(f"{d:.3f}" for d in d_points))
        if args.bottom_freeze_threshold is not None:
            axis_id = {"x": 0, "y": 1, "z": 2}[args.normal_axis]
            n_frozen = sum(
                1 for atom in atoms
                if float(atom.position[axis_id]) < float(args.bottom_freeze_threshold)
            )
            print(f"[make_seeds_scan] freeze    : {n_frozen}/{len(atoms)} atoms below "
                  f"{args.normal_axis}={args.bottom_freeze_threshold}")
        print(f"[make_seeds_scan] seed      : {target.seed_path}")
        print(f"[make_seeds_scan] manifest  : {target.manifest_path}")

        if args.dry_run:
            print("[make_seeds_scan] dry-run: 不做 DP 计算、不写文件")
            return 0

        from recnet_adapter.model_selection import create_calculator
        calc = create_calculator(model, head)

        records, frames = run_scan(atoms, target, n_slab, d_points, calc, args)
        energies = [rec["E"] for rec in records if rec["E"] is not None]
        if not energies:
            for rec in records:
                print(f"[make_seeds_scan] d={rec['d']:.3f} 失败: {rec['error']}")
            raise ToolError("所有扫描点都失败：未写出 seed")

        done = [idx for idx, rec in enumerate(records) if rec["E"] is not None]
        valid_energies = [records[idx]["E"] for idx in done]
        seed_index, monotonic = choose_seed_frame(valid_energies)
        seed_index = done[seed_index]

        target.seed_path.parent.mkdir(parents=True, exist_ok=True)
        # seed 只保留几何（剥掉扫描用的 FixBondLength/FixAtoms，避免下游误继承）
        seed_frame = frames[seed_index].copy()
        seed_frame.set_constraint([])
        write(target.seed_path, seed_frame)

        if getattr(args, "dump_frames", False):
            # 2026-09-22: 逐点落帧。默认只存 seed（峰值）帧；当要把模型的约束扫描
            # 升级成 DFT 剖面时，需要每个 d 上的约束弛豫帧（"DFT-along-model-path"）。
            dumped = []
            for rec, frame in zip(records, frames):
                if rec.get("E") is None:
                    continue
                out = target.seed_path.with_name(
                    f"{target.seed_path.stem}_d{rec['d']:.3f}.xyz")
                clean = frame.copy()
                clean.set_constraint([])
                write(out, clean)
                dumped.append(str(out))
            print(f"[make_seeds_scan] dump-frames -> {len(dumped)} 帧写入 "
                  f"{target.seed_path.parent}", flush=True)

        print_scan_table(records, seed_index, monotonic, target.seed_path)
        write_manifest(
            target, records, seed_index=seed_index, monotonic=monotonic,
            model=model, head=head, source=source, args=args,
            d0=d0, d_cap=d_cap, d_ref=d_ref,
            files={"id": f"{target.rxn_key}_site_{target.site}"
                        + (f"_vg{target.vg}" if target.vg is not None else ""),
                   "bond_global": [gi, gj], "n_slab": n_slab,
                   "d_ref_template": round(float(d_ref), 6)},
        )
        return 0
    except ToolError as exc:
        print(f"[make_seeds_scan] ERROR: {exc}", file=sys.stderr)
        return 2
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 - 干净报错；RECNET_TOOLS_DEBUG=1 看栈
        if os.environ.get("RECNET_TOOLS_DEBUG") == "1":
            raise
        print(f"[make_seeds_scan] ERROR: {exc.__class__.__name__}: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
