"""Compatibility import for model metadata and head selection."""
import sys

from recnet_backend import model_selection as _model_selection

sys.modules[__name__] = _model_selection
