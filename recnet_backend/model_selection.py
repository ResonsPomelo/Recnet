"""Validate model-native branch metadata before constructing a calculator.

Metadata inspection follows DeepMD's ``dp show`` (DeepEval with head=0).
That inspection selects a branch to load metadata only; no energy is evaluated.
Scientific calculators always use the user's validated selection.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from . import AdapterError, _clean_head


@lru_cache(maxsize=32)
def _inspect_cached(path: str, mtime_ns: int, size: int, ctime_ns: int) -> dict:
    from deepmd.infer import DeepEval

    try:
        evaluator = DeepEval(path, head=0)
        params = evaluator.get_model_def_script()
    except Exception as exc:
        raise AdapterError(f"Cannot inspect model metadata {path}: {exc}") from exc
    if not isinstance(params, dict):
        raise AdapterError(f"Model metadata is not a mapping: {path}")
    return params


def inspect_model(model: Path) -> dict:
    stat = model.stat()
    return _inspect_cached(str(model), stat.st_mtime_ns, stat.st_size, stat.st_ctime_ns)


def validate_model_head(model: Any, head: Any = None) -> tuple[Path, str | None]:
    if model is None or str(model).strip().lower() in ('', 'none', 'null'):
        raise AdapterError('未配置模型；请显式提供 model / RECNET_DP_MODEL。')
    path = Path(model).expanduser().resolve()
    if not path.is_file():
        raise AdapterError(f'Model file does not exist: {path}')
    params = inspect_model(path)
    if 'model_dict' not in params:
        # A single/frozen model uses the backend's own native invocation.
        return path, None
    from deepmd.utils.model_branch_dict import get_model_dict

    try:
        aliases, _ = get_model_dict(params['model_dict'])
    except Exception as exc:
        raise AdapterError(f'Invalid branch metadata for {path}: {exc}') from exc
    selected = _clean_head(head)
    choices = ', '.join(aliases)
    if selected is None:
        raise AdapterError(f'Multi-head model requires an explicit head (including Default): {path}; choices: {choices}')
    # DeepMD checks exact spelling first, then its case-insensitive fallback.
    matched = selected if selected in aliases else next(
        (name for name in aliases if name.lower() == selected.lower()), None)
    if matched is None:
        raise AdapterError(f'Unknown head {selected!r} for {path}; choices: {choices}')
    return path, matched


def create_calculator(model: Any, head: Any = None, **kwargs):
    path, selected = validate_model_head(model, head)
    from deepmd.calculator import DP

    if selected is not None:
        kwargs['head'] = selected
    return DP(model=str(path), **kwargs)
