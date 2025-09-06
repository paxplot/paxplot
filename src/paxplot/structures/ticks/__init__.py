"""Tick management for PaxPlot.

This module contains classes for managing axis ticks and labels.
"""

from .base_ticks import BaseTicks
from .numeric_ticks import NumericTicks
from .categorical_ticks import CategoricalTicks

__all__ = [
    "BaseTicks",
    "NumericTicks",
    "CategoricalTicks",
]
