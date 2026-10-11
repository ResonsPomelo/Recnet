"""`RECNET_TS_MIN_BOND_PROJ` 覆盖 TS 键特征门（R338，2026-09-27）。

背景：`handlers/context.py` 的 `ts_primary_mode_min_bond_proj` 原为硬编码 0.16；C3 探针实测表明
`bond_proj = 0.199/0.248` 的 TS 会被接受，而项目口径核对判定其虚频模**不沿被请求的断键**（R326/R336）。
本测试锁定新契约：**默认值不变（0.16）**，且可经环境变量覆盖。
"""
from __future__ import annotations


import pytest

import synth_case


def tmp_case_dir():
    import tempfile
    from pathlib import Path
    return Path(tempfile.mkdtemp(prefix="ts_gate_"))


def _ctx(monkeypatch, value=None):
    if value is None:
        monkeypatch.delenv("RECNET_TS_MIN_BOND_PROJ", raising=False)
    else:
        monkeypatch.setenv("RECNET_TS_MIN_BOND_PROJ", value)
    # env 在 WorkflowContext.__init__ 里读取 ⇒ 每次构造新实例即可生效（无需 reload 模块）
    case = synth_case.create_case(tmp_case_dir())
    return synth_case.make_context(case)


def test_default_threshold_is_016(monkeypatch):
    ctx = _ctx(monkeypatch)
    assert ctx.ts_primary_mode_min_bond_proj == pytest.approx(0.16)


def test_env_override_threshold(monkeypatch):
    ctx = _ctx(monkeypatch, "0.30")
    assert ctx.ts_primary_mode_min_bond_proj == pytest.approx(0.30)
