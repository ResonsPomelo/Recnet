"""Phase 3 工具归组兼容契约：旧路径 wrapper 与 canonical 子包模块等价。

自动发现 ``recnet_adapter/tools/<sub>/`` 下已迁移的模块（子包随批次增加），
对每个模块断言：

1. 旧路径 ``recnet_adapter.tools.<mod>`` 可导入（thin wrapper 存在）；
2. 旧路径暴露的公开属性与 canonical 模块身份一致（同一对象）；
3. 有 ``main`` 时，旧路径与 canonical 的 ``main`` 是同一入口。
"""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = REPO_ROOT / "recnet_adapter" / "tools"
SUBPACKAGES = ("surface", "dft", "campaign", "report", "network", "waves")
_MISSING = object()


def _migrated_modules() -> list[tuple[str, str]]:
    modules: list[tuple[str, str]] = []
    for sub in SUBPACKAGES:
        subdir = TOOLS_DIR / sub
        if not subdir.is_dir():
            continue
        for path in sorted(subdir.glob("*.py")):
            modules.append((sub, path.stem))
    return modules


MIGRATED = _migrated_modules()


def test_migration_has_started() -> None:
    assert MIGRATED, "Phase 3 迁移尚未开始：tools/ 下应有子包模块"


@pytest.mark.parametrize(("sub", "mod"), MIGRATED, ids=[f"{s}.{m}" for s, m in MIGRATED])
def test_old_path_reexports_canonical(sub: str, mod: str) -> None:
    try:
        new = importlib.import_module(f"recnet_adapter.tools.{sub}.{mod}")
    except ModuleNotFoundError as exc:
        # 可选依赖（如 rdkit）不在本环境时，旧路径必须同样缺失——等价行为本身是证据。
        with pytest.raises(ModuleNotFoundError) as old_exc:
            importlib.import_module(f"recnet_adapter.tools.{mod}")
        assert old_exc.value.name == exc.name
        pytest.skip(f"可选依赖 {exc.name} 不在本环境（旧路径同样缺失）")

    old = importlib.import_module(f"recnet_adapter.tools.{mod}")

    drifted = [
        name
        for name, value in vars(new).items()
        if not name.startswith("_") and getattr(old, name, _MISSING) is not value
    ]
    assert not drifted, f"recnet_adapter.tools.{mod} 与 canonical 不一致：{drifted}"

    if hasattr(new, "main"):
        assert old.main is new.main

    private = next(
        (name for name in vars(new) if name.startswith("_") and not name.startswith("__")),
        None,
    )
    if private is not None:
        assert getattr(old, private) is getattr(new, private), f"下划线名未透传：{private}"
