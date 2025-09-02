"""Core data structures for PaxPlot.

This module contains the foundational data structures used by PaxPlot:
- Arrays for storing numerical and categorical data
- Matrix for managing collections of arrays
- Tick management for axis customization
- Custom axis limits for display configuration
"""

from .arrays.numerical_array import NumericalArray
from .arrays.categorical_array import CategoricalArray
from .matrix import Matrix
from .limits.custom_axis_limit import CustomAxisLimit

__all__ = [
    "NumericalArray",
    "CategoricalArray",
    "Matrix",
    "CustomAxisLimit",
]
