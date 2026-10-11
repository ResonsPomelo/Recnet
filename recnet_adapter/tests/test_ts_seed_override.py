"""per-site TS seed 覆写测试（handlers/ts.py + handlers/context.py）。

覆盖交付物 A 的四项验收：
  1. 有 seed 时使用该几何（seed 即初始 TS guess，azimuth 只留 0°）；
  2. 无 seed 时行为不变（与 固定 f1c240f 的 ts.py 在合成 case 上逐位对比）；
  3. seed 原子数/元素不符时打印原因并回退默认 guess；
  4. seed 路径解析优先级（带 vg 优先于不带 vg）。

全部使用替身计算器，无需 GPU/DP 模型。
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import yaml
from ase import Atoms
from ase.io import read, write

# 2026-09-21: 这些用例构造 WorkflowContext（管线依赖 deepmd/sella）；无 deepmd 的环境
# 应整体 skip，而不是在 collection 阶段 INTERNALERROR。
pytest.importorskip("deepmd", reason="seed 覆写测试构建管线上下文，需要 deepmd")

import synth_case as sc

from conftest import GUI_SPAWN_ATTEMPTS  # noqa: E402


def test_no_gui_is_spawned_while_building_context(case):
    """回归守护：构造 WorkflowContext 不得 spawn ase gui（测试必须 headless）。"""
    import ase.visualize
    import slabsite

    assert slabsite.view.__name__ == "_no_view"
    assert ase.visualize.view.__name__ == "_no_view"
    assert "DISPLAY" not in __import__("os").environ
    before = len(GUI_SPAWN_ATTEMPTS)
    sc.make_context(case)
    # 默认路径不得调用 viewer，即使测试替身可阻止真正开窗。
    assert len(GUI_SPAWN_ATTEMPTS) == before


# ----------------------------------------------------------------------
#  运行助手
# ----------------------------------------------------------------------

def _run_ts(module, case, records, *, cwd=None, ccqn_target_d=None, imag_attempts=(1,),
            output_suffix="vg0", **ctx_overrides):
    """在合成 case 上跑一轮 TS 阶段（管线替身化）。

    ``optimization_summary_*.log`` 由管线写成**相对 CWD 的裸文件名**（既有行为），
    所以每次运行都切换到一个干净的临时 CWD，避免污染仓库根目录。
    """
    run_cwd = Path(cwd) if cwd is not None else case.dir
    run_cwd.mkdir(parents=True, exist_ok=True)
    restore = sc.patch_pipeline(
        module, records,
        ccqn_target_d=ccqn_target_d,
        imag_attempts=set(imag_attempts),
    )
    prev_cwd = os.getcwd()
    try:
        os.chdir(run_cwd)
        ctx = sc.make_context(case, output_suffix=output_suffix, **ctx_overrides)
        module.generate_rxn_ts_guesses_ccqn(ctx)
    finally:
        os.chdir(prev_cwd)
        restore()
    return ctx


def _stem(ctx, case):
    return ctx._tagged_stem(f"{case.rxn_key}_site_{case.site}")


def _summary(run_cwd) -> str:
    logs = sorted(Path(run_cwd).glob("optimization_summary_*.log"))
    assert logs, "expected one optimization_summary log in the run CWD"
    return logs[0].read_text()


def _broken_bond_distance(case, positions) -> float:
    i, j = case.n_slab, case.n_slab + 1
    return float(np.linalg.norm(positions[j] - positions[i]))


# ----------------------------------------------------------------------
#  1) 有 seed：seed 即初始 TS guess，且跳过方位角枚举
# ----------------------------------------------------------------------

def test_seed_geometry_is_used_as_initial_guess(tmp_path, case, ts_module):
    seed_path = case.seeds_file(2.00, vg=0)
    seed_positions = read(seed_path).get_positions()

    records = sc.Records()
    run_cwd = tmp_path / "run_cwd"
    ctx = _run_ts(ts_module, case, records, cwd=run_cwd, ccqn_target_d=None, imag_attempts=(1,))

    stem = _stem(ctx, case)
    stage = sc.read_stage_dir(case.dir)
    log = _summary(run_cwd)

    # 写出的初始 guess 就是 seed 几何（逐位相同）
    assert f"{stem}.xyz" in stage["guess_xyzs"]
    assert np.array_equal(stage["guess_xyzs"][f"{stem}.xyz"].get_positions(), seed_positions)

    # 预弛豫（MDMin/QN）与 CCQN 的起点几何 = seed 几何
    assert records.optimizer_starts, "pre-TS optimization was never invoked"
    assert np.array_equal(records.optimizer_starts[0], seed_positions)
    assert records.ccqn_calls, "CCQN was never invoked"
    assert np.array_equal(records.ccqn_calls[0], seed_positions)

    # azimuth 只留 0°，且完全没有做方位角枚举
    assert records.azimuth_calls == []
    assert "Attempt 1/1: azimuth=0.0°" in log
    assert "azimuth enumeration selected" not in log
    assert "azimuth enumeration failed" not in log

    # 日志里明确打印 seed override 路径
    assert f"seed override: {seed_path}" in log

    # 断键弹簧用 seed 自身的断键距离（2.00），而不是 1.1*模板距离（1.375）
    springs = [p for params in records.spring_params for p in params if "ind1" in p]
    assert springs, "reactive-bond restraint was never built"
    assert all(abs(p["deq"] - 2.00) < 1e-9 for p in springs)

    # 后续流程（过拉伸/解离判据 -> 虚频 -> 归档 -> ts_records）未被改动
    assert "TS reactive-bond max length threshold" in log
    assert "TS reactive-bond dissociation min threshold" in log
    assert "Significant imaginary frequency count" in log
    assert stem in stage["archives"]
    assert np.array_equal(stage["archives"][stem].get_positions(), seed_positions)
    records = yaml.safe_load(stage["records_yaml"]["ts_records_vg0.yaml"])["ts_records"]
    assert [rec["tag"] for rec in records] == [stem]


def test_seed_override_ignores_site_translation(tmp_path, case, ts_module):
    """seed 是完整组装结构：不应再被平移到位点（否则整块 slab 会漂移）。"""
    seed_path = case.seeds_file(2.00, vg=0)
    seed_positions = read(seed_path).get_positions()

    records = sc.Records()
    _run_ts(ts_module, case, records, cwd=tmp_path / "run_cwd", ccqn_target_d=None,
            imag_attempts=(1,))

    # 前若干 slab 原子位置保持 seed 原样（平移会整体改变它们）
    slab_positions = records.ccqn_calls[0][: case.n_slab]
    assert np.array_equal(slab_positions, seed_positions[: case.n_slab])
    assert abs(_broken_bond_distance(case, records.ccqn_calls[0]) - 2.00) < 1e-9


# ----------------------------------------------------------------------
#  2) 无 seed：与 固定 f1c240f 实现行为逐位一致（回归）
# ----------------------------------------------------------------------

def test_no_seed_matches_head_implementation(tmp_path, case, ts_module):
    head_ts = sc.load_baseline_module("handlers/ts.py", "_ts_head")
    new_case = sc.copy_case(case, tmp_path / "new_case")
    head_case = sc.copy_case(case, tmp_path / "head_case")
    for copied in (new_case, head_case):
        seeds = sorted((copied.seed_dir).glob("*.xyz")) if copied.seed_dir.exists() else []
        assert seeds == [], "regression run must not have any seed file"

    run_kwargs = {"ccqn_target_d": 1.8, "imag_attempts": {4}}
    records_new, records_head = sc.Records(), sc.Records()

    cwd_new, cwd_head = tmp_path / "cwd_new", tmp_path / "cwd_head"
    ctx_new = _run_ts(ts_module, new_case, records_new, cwd=cwd_new, **run_kwargs)
    ctx_head = _run_ts(head_ts, head_case, records_head, cwd=cwd_head, **run_kwargs)

    # 默认路径确实走了完整方位角候选循环（4 个候选），并触发了 Sella 重试
    assert records_new.azimuth_calls and records_head.azimuth_calls
    log_new = _summary(cwd_new)
    log_head = _summary(cwd_head)
    assert "Attempt 1/4: azimuth=0.0°" in log_new
    assert "run an extra Sella saddle optimization" in log_new
    assert "seed override" not in log_new

    stage_new = sc.read_stage_dir(new_case.dir)
    stage_head = sc.read_stage_dir(head_case.dir)

    # 汇总日志逐字节一致（剔除 C 批新增的阈值记录行：该行登记本次运行实际使用的
    # min bond projection 门值——旧基线 f1c240f 无此增量，其余内容必须逐字节相同）
    def _without_gate_lines(text: str) -> str:
        return "".join(
            line for line in text.splitlines(keepends=True)
            if not line.startswith("TS primary-mode min bond projection threshold:")
        )

    assert _without_gate_lines(log_new) == log_head
    # 写出的几何文件集合与坐标逐位一致
    assert set(stage_new["guess_xyzs"]) == set(stage_head["guess_xyzs"])
    for name, atoms in stage_new["guess_xyzs"].items():
        assert np.array_equal(atoms.get_positions(), stage_head["guess_xyzs"][name].get_positions()), name
    # 优化器起点序列（guess 预弛豫 -> CCQN -> Sella 重试 ...）一致
    assert len(records_new.optimizer_starts) == len(records_head.optimizer_starts)
    for a, b in zip(records_new.optimizer_starts, records_head.optimizer_starts):
        assert np.array_equal(a, b)
    # CCQN 起点序列一致
    assert len(records_new.ccqn_calls) == len(records_head.ccqn_calls)
    for a, b in zip(records_new.ccqn_calls, records_head.ccqn_calls):
        assert np.array_equal(a, b)
    # 断键弹簧参数一致（k/deq 都来自模板距离，未被 seed 分支影响）
    assert records_new.spring_params == records_head.spring_params
    # 归档与 ts_records 文件名一致（tag 命名未变）
    assert set(stage_new["archives"]) == set(stage_head["archives"])
    assert set(stage_new["records_yaml"]) == set(stage_head["records_yaml"])
    assert _stem(ctx_new, new_case) == _stem(ctx_head, head_case)


def test_species_and_reaction_keys_match_head_implementation(ts_module, tmp_path):
    """context.py 的 key 提取为模块级函数后，与 固定 f1c240f 实现输出完全一致。"""
    from handlers.context import compute_reaction_key, compute_species_key

    head_ctx = sc.load_baseline_module("handlers/context.py", "_context_head")
    # A clean checkout has no campaign data; use the existing synthetic case.
    case = sc.create_case(tmp_path / "key-regression")
    prepared = case.dir / "prepared_data" / "prepared_rmg_data.yaml"
    payload = yaml.safe_load(prepared.read_text())
    base_dir = prepared.parent

    templates = {
        sp_id: {
            "atoms": read(base_dir / rec["template_xyz"]),
            "ad_idx": rec["ad_idx"],
            "name": rec.get("name", sp_id),
        }
        for sp_id, rec in payload["species"].items()
    }
    # Preserve representative key branches without private campaign artifacts:
    # multi-atom bond graphs, adsorption-index sorting and repeated species.
    templates["water"] = {
        "atoms": Atoms("OHH", positions=[(0, 0, 0), (0.96, 0, 0), (-0.24, 0.93, 0)]),
        "ad_idx": [2, 0],
        "name": "H2O",
    }
    species_key_by_id = {sp: compute_species_key(tpl) for sp, tpl in templates.items()}

    # 用 固定 f1c240f 的 WorkflowContext 走 __new__（只喂 key 计算需要的属性，不建 SlabSite）
    stub = head_ctx.WorkflowContext.__new__(head_ctx.WorkflowContext)
    stub.ads_templates = templates
    stub.species_key_by_id = species_key_by_id

    for sp_id in templates:
        assert head_ctx.WorkflowContext._species_key(stub, sp_id) == compute_species_key(templates[sp_id])
    original = payload["rxns"][0]
    reversed_bond = {**original, "broken_bond": list(reversed(original["broken_bond"]))}
    repeated = {"reactant_species": ["water", "water"],
                "product_species": ["sp_002", "sp_002"], "broken_bond": [2, 0]}
    for rxn in [*payload["rxns"], reversed_bond, repeated]:
        assert head_ctx.WorkflowContext._reaction_key(stub, rxn) == compute_reaction_key(rxn, species_key_by_id)
    assert compute_reaction_key(original, species_key_by_id) == compute_reaction_key(reversed_bond, species_key_by_id)
    assert compute_reaction_key(repeated, species_key_by_id) != compute_reaction_key(
        {**repeated, "reactant_species": ["water"]}, species_key_by_id,
    )
    assert compute_species_key(templates["water"]) == compute_species_key(
        {**templates["water"], "ad_idx": [0, 2]},
    )


def test_ts_seed_can_be_disabled_by_flag(tmp_path, case, ts_module):
    """--no-ts-seed 等价路径：enable_ts_seed=False 时完全忽略 seed。"""
    case.seeds_file(2.00, vg=0)
    records = sc.Records()
    run_cwd = tmp_path / "run_cwd"
    _run_ts(ts_module, case, records, cwd=run_cwd, ccqn_target_d=None,
            imag_attempts=(1,), enable_ts_seed=False)

    log = _summary(run_cwd)
    assert "seed override" not in log
    # 默认 guess：方位角枚举照常执行，初始断键距离 = 模板距离（1.25 Å）
    assert records.azimuth_calls
    assert abs(_broken_bond_distance(case, records.optimizer_starts[0]) - sc.D_TEMPLATE) < 1e-9


def test_cli_flags_default_to_enabled():
    from utils import config as cfgmod

    parser = cfgmod.build_arg_parser()
    assert parser.parse_args([]).ts_seed is True
    assert parser.parse_args(["--no-ts-seed"]).ts_seed is False
    assert parser.parse_args(["--ts-seed"]).ts_seed is True
    assert cfgmod.config_from_args(parser.parse_args([])).ts_seed is True
    assert cfgmod.config_from_args(parser.parse_args(["--no-ts-seed"])).ts_seed is False


# ----------------------------------------------------------------------
#  3) 原子数 / 元素不符：打印原因并回退默认 guess
# ----------------------------------------------------------------------

def test_seed_atom_count_mismatch_is_rejected(tmp_path, case, ts_module, capsys):
    wrong = case.assembly(2.00)
    wrong += Atoms("H", positions=[[2.87, 2.87, 18.0]], cell=sc.SLAB_CELL, pbc=True)
    path = case.seed_dir / f"{case.rxn_key}_site_{case.site}_vg0.xyz"
    path.parent.mkdir(parents=True, exist_ok=True)
    write(path, wrong)

    records = sc.Records()
    run_cwd = tmp_path / "run_cwd"
    _run_ts(ts_module, case, records, cwd=run_cwd, ccqn_target_d=1.8, imag_attempts=(4,))

    log = _summary(run_cwd)
    stdout = capsys.readouterr().out

    assert "seed override rejected" in log
    assert "atom_count_mismatch" in log
    assert "seed override rejected" in stdout and "atom_count_mismatch" in stdout
    assert path.name in stdout
    # 回退到默认 guess：方位角枚举照常执行，几何不是 seed
    assert records.azimuth_calls
    assert abs(_broken_bond_distance(case, records.optimizer_starts[0]) - sc.D_TEMPLATE) < 1e-9


def test_seed_symbol_mismatch_is_rejected(tmp_path, case, ts_module, capsys):
    path = case.seeds_file(2.00, vg=0, order=("O", "C"))

    records = sc.Records()
    run_cwd = tmp_path / "run_cwd"
    _run_ts(ts_module, case, records, cwd=run_cwd, ccqn_target_d=1.8, imag_attempts=(4,))

    log = _summary(run_cwd)
    stdout = capsys.readouterr().out
    assert "symbol_mismatch" in log
    assert f"first_diff_index={case.n_slab}" in log
    assert "symbol_mismatch" in stdout
    assert records.azimuth_calls
    assert path.exists()


def test_unreadable_seed_falls_back_with_reason(tmp_path, case, ts_module, capsys):
    path = case.seed_dir / f"{case.rxn_key}_site_{case.site}_vg0.xyz"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("this is not an xyz file\n")

    records = sc.Records()
    run_cwd = tmp_path / "run_cwd"
    _run_ts(ts_module, case, records, cwd=run_cwd, ccqn_target_d=1.8, imag_attempts=(4,))

    log = _summary(run_cwd)
    assert "read_failed" in log
    assert "read_failed" in capsys.readouterr().out
    assert records.azimuth_calls


# ----------------------------------------------------------------------
#  4) 路径解析优先级：带 vg 优先于不带 vg
# ----------------------------------------------------------------------

def test_seed_path_priority_prefers_vg_suffixed_file(tmp_path, case, ts_module):
    plain = case.seeds_file(1.80, vg=None)
    tagged = case.seeds_file(2.00, vg=0)

    ctx = sc.make_context(case, output_suffix="vg0")
    paths = ctx._ts_seed_paths(case.rxn_key, case.site)
    assert paths == [str(tagged), str(plain)]

    atoms, used_path, reason = ctx.load_ts_seed(
        case.rxn_key, case.site, assembly_reference=case.assembly(2.00),
    )
    assert reason is None and atoms is not None
    assert Path(used_path) == tagged

    # 端到端：实际使用的是 vg0 的那份 2.00 Å 几何
    records = sc.Records()
    run_cwd = tmp_path / "run_cwd"
    _run_ts(ts_module, case, records, cwd=run_cwd, ccqn_target_d=None, imag_attempts=(1,))
    assert abs(_broken_bond_distance(case, records.ccqn_calls[0]) - 2.00) < 1e-9
    assert f"seed override: {tagged}" in _summary(run_cwd)


def test_seed_path_without_vg_suffix_uses_plain_file(tmp_path, case, ts_module):
    plain = case.seeds_file(1.80, vg=None)
    case.seeds_file(2.00, vg=0)

    ctx = sc.make_context(case, output_suffix="")
    assert ctx._ts_seed_paths(case.rxn_key, case.site) == [str(plain)]

    records = sc.Records()
    _run_ts(ts_module, case, records, cwd=tmp_path / "run_cwd", ccqn_target_d=None,
            imag_attempts=(1,), output_suffix="")
    assert abs(_broken_bond_distance(case, records.ccqn_calls[0]) - 1.80) < 1e-9


def test_seed_lookup_is_per_site_and_per_rxn(case, ts_module):
    """其它 site/rxn 的 seed 不会被误用。"""
    case.seeds_file(2.00, vg=0)
    ctx = sc.make_context(case, output_suffix="vg0")
    other_site = ctx._ts_seed_paths(case.rxn_key, case.site + 1)
    assert len(other_site) == 2
    atoms, path, reason = ctx.load_ts_seed(
        case.rxn_key, case.site + 1, assembly_reference=case.assembly(2.00),
    )
    assert (atoms, path, reason) == (None, None, None)

    other_rxn = ctx._ts_seed_paths("rxn_deadbeef0000", case.site)
    assert all(not Path(p).exists() for p in other_rxn)


def test_missing_name_key_preserves_baseline_fallback():
    from types import SimpleNamespace
    from handlers.context import WorkflowContext
    template = {"atoms": Atoms("H2", positions=[[0,0,0],[0,0,0.75]]), "ad_idx": [0]}
    ctx = SimpleNamespace(ads_templates={"legacy_id": template})
    ctx._stable_digest = WorkflowContext._stable_digest.__get__(ctx)
    ctx._slugify_token = WorkflowContext._slugify_token.__get__(ctx)
    old = sc.load_baseline_module("handlers/context.py", "baseline_missing_name")
    assert WorkflowContext._species_key(ctx, "legacy_id") == old.WorkflowContext._species_key(ctx, "legacy_id")
    assert "name" not in template

@pytest.mark.parametrize("variant", ["different-cell", "plain-xyz", "nonperiodic-slab"])
def test_seed_restraint_uses_normalized_cell(tmp_path, case, ts_module, monkeypatch, variant):
    seed_path = case.seeds_file(4.0, vg=0)
    atoms = read(seed_path)
    if variant == "different-cell":
        atoms.set_cell([10,10,30])
    if variant == "nonperiodic-slab":
        atoms.set_cell([5,5,30])
        original = sc.make_context
        def context(*args, **kwargs):
            ctx = original(*args, **kwargs)
            ctx.slab.stru.set_pbc(False)
            return ctx
        monkeypatch.setattr(sc, "make_context", context)
    write(seed_path, atoms, format="xyz" if variant == "plain-xyz" else "extxyz")
    records = sc.Records()
    ctx = _run_ts(ts_module, case, records, cwd=tmp_path / "run")
    guess = sc.read_stage_dir(case.dir)["guess_xyzs"][f"{_stem(ctx, case)}.xyz"]
    expected = guess.get_distance(case.n_slab, case.n_slab + 1, mic=True)
    springs = [p for params in records.spring_params for p in params if "ind1" in p]
    assert springs
    assert all(p["deq"] == pytest.approx(expected) for p in springs)


def test_workflow_config_legacy_positional_constructor():
    from dataclasses import fields
    from utils.config import WorkflowConfig, build_arg_parser, config_from_args
    default = config_from_args(build_arg_parser().parse_args([]))
    legacy_fields = [f.name for f in fields(WorkflowConfig) if f.name != "ts_seed"]
    config = WorkflowConfig(*(getattr(default, name) for name in legacy_fields))
    assert config.surface_normal == default.surface_normal
    assert config.gas_pressure_pa == default.gas_pressure_pa
    assert config.ts_seed is True
    assert fields(WorkflowConfig)[-1].name == "ts_seed"
