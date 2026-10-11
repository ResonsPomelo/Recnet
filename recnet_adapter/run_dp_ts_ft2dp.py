#!/usr/bin/env python3
"""等价的 run_dp_ts.py 入口：解析显式模型输入后执行管线。

用法与 ``run_dp_ts.py`` 完全一致（参数原样透传），例如::

    python recnet_adapter/run_dp_ts_ft2dp.py \
        --path ./ --prepared ./prepared_data/prepared_rmg_data.yaml \
        --slab "./surf_07_Fe2C(110)-3x2.cif" --top-x 1 --use-c-vacancy-io

模型来源：--model/--head、RECNET_DP_MODEL/RECNET_DP_HEAD 或显式项目配置。无内置模型/head。
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from recnet_adapter import run_pipeline  # noqa: E402


if __name__ == "__main__":
    run_pipeline(sys.argv[1:])
