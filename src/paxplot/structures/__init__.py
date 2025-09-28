"""Core data structures for PaxPlot.

This module contains the foundational data structures used by PaxPlot:
- Arrays for storing numerical and categorical data
- ArrayManager for managing collections of arrays
- Tick management for axis customization
- Custom axis limits for display configuration
"""

from .arrays.numerical_array import NumericalArray
from .arrays.categorical_array import CategoricalArray
from .array_manager import ArrayManager
from .limits.custom_axis_limit import CustomAxisLimit
from .tick_manger import TickManager, TickType

__all__ = [
    "NumericalArray",
    "CategoricalArray",
    "ArrayManager",
    "CustomAxisLimit",
    "TickManager",
    "TickType",
]
