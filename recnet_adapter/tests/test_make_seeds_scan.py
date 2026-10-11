"""``recnet_adapter/tools/make_seeds_scan.py`` 的离线（CPU/无 DP）测试。

覆盖：扫描点规划、选帧规则、与管线 TS 断键上限公式的一致性、约束扫描机制
（用 EMT 替身计算器，不需要 GPU/DP）、seed 写出格式与 manifest 结构。
"""
from __future__ import annotations

import json

import numpy as np
import pytest
from ase.calculators.lj import LennardJones
from ase.io import read

import synth_case as sc

# 2026-09-21: 本文件的多数用例依赖管线 key 工具（惰性导入 handlers.context → deepmd）。
# 无 deepmd 的环境（如 recnet-prep）应整体 skip，而不是在 collection 阶段 INTERNALERROR。
pytest.importorskip("deepmd", reason="make_seeds_scan 的管线 key 工具需要 deepmd")

from recnet_adapter.tools import make_seeds_scan as mss  # noqa: E402


# ----------------------------------------------------------------------
#  纯函数：扫描点规划 / 选帧
# ----------------------------------------------------------------------

def test_plan_scan_distances_includes_both_endpoints():
    assert mss.plan_scan_distances(1.158, 0.2, 2.2) == [
        1.158, 1.358, 1.558, 1.758, 1.958, 2.158, 2.2,
    ]


def test_plan_scan_distances_degenerate_cases():
    # 已解离/超过过拉伸门槛的反应物必须被夹回门槛内，不能产出超 QC 上限的 seed。
    assert mss.plan_scan_distances(2.3, 0.2, 2.2) == [2.2]
    # 步长大于区间：只保留 d0 与上限
    assert mss.plan_scan_distances(1.0, 5.0, 2.2) == [1.0, 2.2]
    with pytest.raises(mss.ToolError):
        mss.plan_scan_distances(1.0, 0.0, 2.2)


def test_choose_seed_frame_monotonic_takes_last():
    energies = [0.0, 0.4, 0.9, 1.4]
    assert mss.choose_seed_frame(energies) == (3, True)


def test_choose_seed_frame_takes_energy_maximum():
    energies = [0.0, 0.4, 1.4, 0.9, 0.2]
    assert mss.choose_seed_frame(energies) == (2, False)


def test_ts_bond_upper_bound_matches_pipeline_formula(case, ts_module):
    """工具里的上限公式必须与 handlers/ts.py:_ts_bond_upper_bound 一致。"""
    from ase.data import atomic_numbers

    ctx = sc.make_context(case)
    ctx_args = dict(
        ts_bond_max_scale_ref=ctx.ts_bond_max_scale_ref,
        ts_bond_max_scale_covalent=ctx.ts_bond_max_scale_covalent,
        ts_bond_max_additive_cap=ctx.ts_bond_max_additive_cap,
    )
    # 1) 常量与 WorkflowContext 默认值一致
    assert mss._TsBondGateCtx.ts_bond_max_scale_ref == pytest.approx(ctx_args["ts_bond_max_scale_ref"])
    assert mss._TsBondGateCtx.ts_bond_max_scale_covalent == pytest.approx(
        ctx_args["ts_bond_max_scale_covalent"]
    )
    assert mss._TsBondGateCtx.ts_bond_max_additive_cap == pytest.approx(
        ctx_args["ts_bond_max_additive_cap"]
    )
    # 2) 函数输出一致（C-O / C-H 两对元素、几个 d_ref）
    z_c, z_o, z_h = (atomic_numbers[s] for s in ("C", "O", "H"))
    for d_ref in (1.0, 1.158, 1.5, 2.0):
        for pair in ((z_c, z_o), (z_c, z_h)):
            assert mss.ts_bond_upper_bound(d_ref, *pair) == pytest.approx(
                ts_module._ts_bond_upper_bound(ctx, d_ref=d_ref, atom_i_number=pair[0],
                                               atom_j_number=pair[1])
            )


def test_scan_cap_uses_template_distance_as_d_ref(case):
    """扫描上限的口径 = handlers/ts.py 的过拉伸门槛（d_ref 取模板距离，而非已弛豫结构的距离）。"""
    from ase.data import atomic_numbers

    target = mss.resolve_target(_parse([
        str(case.dir), "--rxn", "CO*", "--site", str(case.site), "--vg", "0",
    ]))
    args = _parse([str(case.dir), "--rxn", "CO*", "--site", str(case.site)])
    for d0 in (1.16, 1.30):
        atoms = case.assembly(d0)
        _, d_points, d_cap, d_ref = mss.plan_scan(target, atoms, case.n_slab, args)
        assert d_ref == pytest.approx(sc.D_TEMPLATE)
        expected_cap = mss.ts_bond_upper_bound(
            sc.D_TEMPLATE, atomic_numbers["C"], atomic_numbers["O"]
        )
        assert d_cap == pytest.approx(expected_cap)
        assert max(d_points) <= min(2.2, expected_cap) + 1e-9


# ----------------------------------------------------------------------
#  case 解析（--rxn / 路径 / key）
# ----------------------------------------------------------------------

def _parse(args_list):
    return mss.build_parser().parse_args(args_list)


def test_resolve_target_matches_pipeline_naming(case):
    args = _parse([str(case.dir), "--rxn", "CO*", "--site", str(case.site), "--vg", "0"])
    target = mss.resolve_target(args)
    assert target.rxn_index == 0
    assert target.rxn_key == case.rxn_key
    assert target.species_key == case.species_key
    assert target.broken_bond_local == (0, 1)
    assert target.seed_path.name == f"{case.rxn_key}_site_{case.site}_vg0.xyz"
    assert target.seed_path.parent == case.dir / "rxn" / "seeds"
    assert target.manifest_path.name == f"SCAN_MANIFEST_{case.rxn_key}_site_{case.site}_vg0.json"


def test_rxn_selector_accepts_label_species_and_index(case, capsys):
    for selector in ("CO*", "CO", "co", "0"):
        args = _parse([str(case.dir), "--rxn", selector, "--site", str(case.site)])
        assert mss.resolve_target(args).rxn_index == 0
    with pytest.raises(mss.ToolError):
        mss.resolve_target(_parse([str(case.dir), "--rxn", "NOPE", "--site", "0"]))


def test_list_rxns_prints_rxn_key(case, capsys):
    args = _parse([str(case.dir), "--list-rxns"])
    with pytest.raises(SystemExit) as excinfo:
        mss.resolve_target(args)
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert case.rxn_key in out and "C-O" in out


def test_dry_run_plans_without_model_and_writes_nothing(case, capsys, monkeypatch):
    """--dry-run：不建 DP 计算器、不依赖 GPU/模型、不写任何文件。"""
    case.adsorbate_opt(1.158)  # 组装结构（已弛豫吸附结构）
    monkeypatch.setenv("RECNET_DP_MODEL", "/tmp/missing-model.pt")

    rc = mss.main([
        str(case.dir), "--rxn", "CO*", "--site", str(case.site), "--vg", "0", "--dry-run",
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "dry-run: 不做 DP 计算、不写文件" in out
    assert "d0=1.158" in out
    assert f"seed      : {case.dir / 'rxn' / 'seeds' / f'{case.rxn_key}_site_{case.site}_vg0.xyz'}" in out
    assert "[MISSING]" in out  # 假模型在 dry-run 只标记，不报错
    assert not (case.dir / "rxn" / "seeds").exists()


def test_missing_model_fails_before_any_work(case, capsys):
    case.adsorbate_opt(1.158)
    rc = mss.main([
        str(case.dir), "--rxn", "CO*", "--site", str(case.site),
        "--model", "/tmp/definitely-missing-model.pt",
    ])
    assert rc == 2
    err = capsys.readouterr().err
    assert "模型检查点不存在" in err and "/tmp/definitely-missing-model.pt" in err
    assert not (case.dir / "rxn" / "seeds").exists()


def test_missing_adsorbate_is_reported(case, capsys):
    rc = mss.main([
        str(case.dir), "--rxn", "CO*", "--site", str(case.site), "--dry-run",
    ])
    assert rc == 2
    assert "已弛豫吸附结构不存在" in capsys.readouterr().err


# ----------------------------------------------------------------------
#  约束扫描机制（EMT 替身计算器；不需要 DP/GPU）
# ----------------------------------------------------------------------

def test_run_scan_holds_distance_and_records_every_step(case):
    target = mss.resolve_target(_parse([
        str(case.dir), "--rxn", "CO*", "--site", str(case.site), "--vg", "0",
    ]))
    atoms = case.assembly(1.16)
    initial_positions = atoms.get_positions().copy()
    n_slab = case.n_slab
    d_points = [1.16, 1.56, 1.96]
    args = _parse([
        str(case.dir), "--rxn", "CO*", "--site", str(case.site),
        "--fmax", "0.5", "--steps", "50", "--bottom-freeze-threshold", "14.0",
    ])

    # 用 LennardJones 替身计算器（不需要 DP/GPU；EMT 不支持 Fe）
    records, frames = mss.run_scan(atoms, target, n_slab, d_points, LennardJones(), args)

    assert [rec["d"] for rec in records] == d_points
    assert all(rec["error"] is None for rec in records)
    assert all(rec["E"] is not None and rec["fmax"] is not None for rec in records)
    # 约束成立：每帧实际键长 ≈ 规划值
    for rec, frame in zip(records, frames):
        assert rec["d_final"] == pytest.approx(rec["d"], abs=1e-6)
        i, j = n_slab, n_slab + 1
        assert sc.distance_between(frame, i, j) == pytest.approx(rec["d"], abs=1e-6)
    # 冻结阈值生效：z=12/9/6 三个 Fe（索引 1..3）在扫描中不动，表层 Fe（索引 0）不冻结
    frozen = initial_positions[1:4]
    for frame in frames:
        assert np.allclose(frame.get_positions()[1:4], frozen)
        assert not np.allclose(frame.get_positions()[0], initial_positions[0])

    # 选帧 + 写出 seed（原子顺序与组装结构一致、只含几何）
    energies = [rec["E"] for rec in records]
    seed_index, monotonic = mss.choose_seed_frame(energies)
    assert 0 <= seed_index < len(records)
    assert bool(monotonic) in (True, False)

    out_dir = case.dir / "rxn" / "seeds"
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_path = out_dir / f"{target.rxn_key}_site_{case.site}_vg0.xyz"
    seed_frame = frames[seed_index].copy()
    seed_frame.set_constraint([])
    mss.write(seed_path, seed_frame)

    seed = read(seed_path)
    assert len(seed) == len(atoms)
    assert list(seed.get_chemical_symbols()) == list(atoms.get_chemical_symbols())
    # extxyz 文本精度 ~1e-8 Å，故给 1e-6 容差
    assert seed.get_positions() == pytest.approx(frames[seed_index].get_positions(), abs=1e-6)
    assert len(seed.constraints) == 0
    assert isinstance(float(seed.get_distance(n_slab, n_slab + 1)), float)


def test_write_manifest_records_every_step_and_merges(case):
    target = mss.resolve_target(_parse([
        str(case.dir), "--rxn", "CO*", "--site", str(case.site), "--vg", "0",
    ]))
    args = _parse([str(case.dir), "--rxn", "CO*", "--site", str(case.site)])
    records = [
        {"d": 1.16, "E": -1.0, "fmax": 0.01, "d_final": 1.16, "converged": True,
         "steps": 12, "error": None},
        {"d": 1.56, "E": 0.5, "fmax": 0.02, "d_final": 1.56, "converged": False,
         "steps": 50, "error": None},
    ]
    files = {"id": f"{target.rxn_key}_site_{case.site}_vg0",
             "bond_global": [case.n_slab, case.n_slab + 1], "n_slab": case.n_slab,
             "d_ref_template": sc.D_TEMPLATE}
    mss.write_manifest(target, records, seed_index=1, monotonic=False, model=None,
                       head="ft2dp", source={"model": "unset", "head": "env"},
                       args=args, d0=1.16, d_cap=2.27, d_ref=sc.D_TEMPLATE, files=files)

    payload = json.loads(target.manifest_path.read_text())
    assert payload["case"] == str(case.dir)
    scan = payload["scans"][0]
    assert scan["id"] == files["id"]
    assert scan["records"] == records
    assert scan["seed_frame"] == 1 and scan["monotonic_increasing"] is False
    assert scan["broken_bond_global"] == files["bond_global"]
    assert scan["seed_path"] == str(target.seed_path)
    assert scan["d_ref_template"] == pytest.approx(sc.D_TEMPLATE)
    # 规范账本（SCAN_MANIFEST.json）同时写入，内容一致
    canonical = target.manifest_path.parent / "SCAN_MANIFEST.json"
    assert canonical.exists()
    assert json.loads(canonical.read_text())["scans"][0]["records"] == records

    # 同一 (rxn, site, vg) 重跑：合并而不是追加
    mss.write_manifest(target, records, seed_index=0, monotonic=True, model=None,
                       head="ft2dp", source={"model": "unset", "head": "env"},
                       args=args, d0=1.16, d_cap=2.27, d_ref=sc.D_TEMPLATE, files=files)
    payload = json.loads(target.manifest_path.read_text())
    assert len(payload["scans"]) == 1
    assert payload["scans"][0]["seed_frame"] == 0
    assert len(json.loads(canonical.read_text())["scans"]) == 1


def test_slab_cross_check_warns_for_vacancy_case(case, tmp_path):
    """空位板（组装结构 slab 段 ≠ 原始 slab 文件）只警告不报错。"""
    target = mss.resolve_target(_parse([
        str(case.dir), "--rxn", "CO*", "--site", str(case.site),
    ]))
    atoms = case.assembly(1.6)
    n_slab = case.n_slab
    assert mss.cross_check_slab(atoms, n_slab, target.slab_path) is None  # 原始板：一致
    from ase import Atoms

    vacancy = Atoms("Fe3", positions=[(0, 0, 0)] * 3, cell=sc.SLAB_CELL, pbc=True)
    mss.write(tmp_path / "vac.vasp", vacancy)
    warning = mss.cross_check_slab(atoms, n_slab, tmp_path / "vac.vasp")
    assert warning is not None and "以组装结构为准" in warning
