"""Recnet's checked DP boundary; never modifies DeepMD's global calculator class."""
from __future__ import annotations

from deepmd.calculator import DP as NativeDP

from . import Resolution, resolve, _resolve_model_path
from .model_selection import validate_model_head

_configured: Resolution | None = None


def configure(resolution: Resolution) -> None:
    global _configured
    _configured = resolution


class DP(NativeDP):
    """Native DP with validated model inputs and process-local pipeline defaults."""

    def __init__(self, model=None, *args, **kwargs):
        explicit_head = 'head' in kwargs
        selected = kwargs.pop('head', None)
        requested = _resolve_model_path(model) if model is not None else None
        if requested is not None:
            requested = requested.resolve()
        if _configured is not None and (model is None or requested == _configured.model):
            requested = requested or _configured.model
            if not explicit_head:
                selected = _configured.head
        else:
            res = resolve(model=model, head=selected if explicit_head and selected is not None else ('none' if explicit_head else None))
            requested = res.model
            selected = res.head
        path, selected = validate_model_head(requested, selected)
        if selected is not None:
            kwargs['head'] = selected
        super().__init__(str(path), *args, **kwargs)
