"""兼容入口：实现已迁至 ``recnet_adapter.tools.report.co_thermo``（Phase 3 工具归组）。

保留旧文件路径、``python -m recnet_adapter.tools.co_thermo`` 与旧 import 名；
本文件只做 re-export、属性透传与 ``main()`` 委派，不含逻辑。
"""
from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from recnet_adapter.tools.report.co_thermo import *  # noqa: E402,F401,F403
from recnet_adapter.tools.report import co_thermo as _impl  # noqa: E402


def __getattr__(name: str):
    """透传未公开导出（含下划线名），保证旧 import 面完整可用。"""
    return getattr(_impl, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_impl)))

if __name__ == "__main__":
    from recnet_adapter.tools.report.co_thermo import main  # noqa: E402

    raise SystemExit(main())
