"""TS 键特征门阈值（``RECNET_TS_MIN_BOND_PROJ``）契约测试。

覆盖 C 批验收：
  1. 默认 0.16（未设置环境变量时行为不变）；
  2. 环境变量覆盖（含首尾空白与小数）；
  3. 无效/非有限/越界输入显式失败（不静默回退、不放宽门槛）；
  4. 判定日志记录**实际使用的**阈值（默认 0.16 与覆盖 0.30 均如实登记），
     且不把项目生产选择 0.30 写成通用默认。
全部使用替身计算器，无需 GPU/DP 模型。
"""
from __future__ import annotations

import pytest

# 与 seed 套件一致：用例构造 WorkflowContext（管线依赖 deepmd）。
pytest.importorskip("deepmd", reason="TS 门阈值测试构造管线上下文，需要 deepmd")

import synth_case as sc  # noqa: E402
from test_ts_seed_override import _run_ts, _summary  # noqa: E402

GATE_LINE_PREFIX = "TS primary-mode min bond projection threshold:"


def _gate_values(log_text: str) -> list[str]:
    return [
        line.split(":", 1)[1].strip()
        for line in log_text.splitlines()
        if line.startswith(GATE_LINE_PREFIX)
    ]


def test_default_threshold_is_0_16(case, monkeypatch):
    monkeypatch.delenv("RECNET_TS_MIN_BOND_PROJ", raising=False)
    ctx = sc.make_context(case)
    assert ctx.ts_primary_mode_min_bond_proj == pytest.approx(0.16)


def test_env_override_sets_threshold(case, monkeypatch):
    monkeypatch.setenv("RECNET_TS_MIN_BOND_PROJ", "0.30")
    ctx = sc.make_context(case)
    assert ctx.ts_primary_mode_min_bond_proj == pytest.approx(0.30)
    # 项目 0.30 是显式覆盖；默认仍是 0.16（见上一用例）。
    assert ctx.ts_primary_mode_min_bond_proj != pytest.approx(0.16)


def test_env_override_tolerates_whitespace(case, monkeypatch):
    monkeypatch.setenv("RECNET_TS_MIN_BOND_PROJ", " 0.25 ")
    assert sc.make_context(case).ts_primary_mode_min_bond_proj == pytest.approx(0.25)


@pytest.mark.parametrize(
    "raw",
    [
        "abc",
        "",
        "   ",
        "0.16abc",
        "nan",
        "NaN",
        "inf",
        "-inf",
        "Infinity",
        "-0.1",
        "0",
        "0.0",
        "1.0001",
        "1.5",
    ],
)
def test_invalid_values_fail_loudly(case, monkeypatch, raw):
    import handlers.context as hctx

    monkeypatch.setenv("RECNET_TS_MIN_BOND_PROJ", raw)
    with pytest.raises(ValueError):
        hctx._parse_ts_min_bond_proj_env()
    # 构造 WorkflowContext 时同样显式失败（失败发生在读数据/建板之前）。
    with pytest.raises(ValueError):
        sc.make_context(case)


def test_parser_accepts_open_interval_boundaries(monkeypatch):
    import handlers.context as hctx

    monkeypatch.setenv("RECNET_TS_MIN_BOND_PROJ", "1")
    assert hctx._parse_ts_min_bond_proj_env() == pytest.approx(1.0)
    monkeypatch.setenv("RECNET_TS_MIN_BOND_PROJ", "0.001")
    assert hctx._parse_ts_min_bond_proj_env() == pytest.approx(0.001)


def test_log_records_actual_threshold(tmp_path, case, ts_module, monkeypatch):
    """日志中的实际阈值：默认运行记 0.160，覆盖运行记 0.300。"""
    run_kwargs = {"ccqn_target_d": 1.8, "imag_attempts": {4}}

    monkeypatch.delenv("RECNET_TS_MIN_BOND_PROJ", raising=False)
    cwd_default = tmp_path / "cwd_default"
    _run_ts(
        ts_module,
        sc.copy_case(case, tmp_path / "case_default"),
        sc.Records(),
        cwd=cwd_default,
        **run_kwargs,
    )
    default_log = _summary(cwd_default)
    assert _gate_values(default_log), "默认运行日志应记录阈值行"
    assert set(_gate_values(default_log)) == {"0.16"}

    monkeypatch.setenv("RECNET_TS_MIN_BOND_PROJ", "0.30")
    cwd_override = tmp_path / "cwd_override"
    _run_ts(
        ts_module,
        sc.copy_case(case, tmp_path / "case_override"),
        sc.Records(),
        cwd=cwd_override,
        **run_kwargs,
    )
    override_log = _summary(cwd_override)
    assert set(_gate_values(override_log)) == {"0.3"}


@pytest.mark.parametrize("early", ["no-imag", "preopt-failure"])
@pytest.mark.parametrize("threshold", [0.16, 0.00005, 0.12345678901234567])
def test_each_attempt_records_roundtrip_threshold(tmp_path, case, ts_module, monkeypatch, early, threshold):
    monkeypatch.setenv("RECNET_TS_MIN_BOND_PROJ", repr(threshold))
    if early == "preopt-failure":
        original = sc.patch_pipeline
        def patch(*args, **kwargs):
            restore = original(*args, **kwargs)
            def fail(*args, **kwargs):
                raise RuntimeError("forced preopt failure")
            ts_module.MDMin = fail
            return restore
        monkeypatch.setattr(sc, "patch_pipeline", patch)
    _run_ts(ts_module, case, sc.Records(), cwd=tmp_path / "run", ccqn_target_d=1.8, imag_attempts=())
    log = _summary(tmp_path / "run")
    attempts = [line for line in log.splitlines() if line.startswith("Attempt ")]
    assert len(attempts) == 4
    values = _gate_values(log)
    assert len(values) == len(attempts)
    assert all(float(value) == threshold for value in values)
    if early == "preopt-failure":
        assert "forced preopt failure" in log


@pytest.mark.parametrize("threshold,projection,accepted", [(0.16,0.20,True),(0.30,0.20,False),(0.30,0.30,True),(0.30,0.299999,False)])
def test_threshold_changes_acceptance(tmp_path, case, ts_module, monkeypatch, threshold, projection, accepted):
    import numpy as np
    monkeypatch.setenv("RECNET_TS_MIN_BOND_PROJ", repr(threshold))
    def report(*args, **kwargs):
        primary = dict(mode_idx=0, freq=-450.0, bond_proj=projection, drift_proj=0.0,
                       max_atom_symbol="C", max_atom=case.n_slab, mode=np.zeros((case.n_slab+2,3)))
        return {"all": [primary], "primary": primary}
    monkeypatch.setattr(ts_module, "_get_imag_mode_report", report)
    ctx = _run_ts(ts_module, case, sc.Records(), cwd=tmp_path / "run", ccqn_target_d=1.8,
                  imag_attempts=(1,2,3,4), enable_imag_mode_check=False)
    assert bool(ctx.ts_records) is accepted
    rejected = "Rejected this orientation: primary imaginary mode does not show sufficiently strong bond-breaking character."
    assert (rejected in _summary(tmp_path / "run")) is (not accepted)
