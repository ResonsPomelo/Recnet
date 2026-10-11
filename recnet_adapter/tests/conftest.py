"""recnet_adapter 测试的 sys.path、headless 守护与公共 fixture。"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
REPO_ROOT = _HERE.parents[1]

for _path in (str(REPO_ROOT), str(_HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

# ----------------------------------------------------------------------
#  headless 守护：测试进程绝不弹 GUI 窗口
# ----------------------------------------------------------------------
# 背景：``handlers/context.py`` 构造 WorkflowContext 时会调用
# ``SlabSite.voronoi(True, ["hollow", "bridge", "top"])``，而
# ``SlabSite.voronoi(self, show=False, type=[...])`` 的第一个位置参数是 ``show``
# —— 于是 ``slabsite.voronoi`` 会走到 ``ase.visualize.view(stru)`` 并 spawn
# ``python -m ase gui -``。有 DISPLAY 的机器上会连开窗口。
# 测试需要反复构造 ctx，因此这里在导入管线代码前把 GUI 入口封死：
import ase.visualize  # noqa: E402

GUI_SPAWN_ATTEMPTS: list[tuple] = []


def _no_view(atoms=None, *args, **kwargs):
    """``ase.visualize.view`` 的 no-op 替身（只记账，不 spawn）。"""
    GUI_SPAWN_ATTEMPTS.append((args, sorted(kwargs)))
    return None


ase.visualize.view = _no_view
os.environ["MPLBACKEND"] = "Agg"
os.environ.pop("DISPLAY", None)

import slabsite  # noqa: E402  (slabsite 在模块导入时 `from ase.visualize import view`)

slabsite.view = _no_view
slabsite.view.__doc__ = "no-op（测试禁止弹窗）"

import pytest  # noqa: E402

import synth_case  # noqa: E402


@pytest.fixture
def case(tmp_path):
    """最小合成 case（slab + prepared_data + 模板），位点索引已解析为实际存在的值。"""
    info = synth_case.create_case(tmp_path / "case")
    info.site = synth_case.resolve_site_index(info)
    return info


@pytest.fixture
def ts_module():
    """真实 handlers.ts 模块（DP 等重依赖由测试内的替身替换）。"""
    import handlers.ts

    return handlers.ts
