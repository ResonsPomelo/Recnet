"""Compatibility import for the shared checked calculator module."""
import sys

from recnet_backend import calculator as _calculator

sys.modules[__name__] = _calculator
