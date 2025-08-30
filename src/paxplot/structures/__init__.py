"""Core data structures for PaxPlot.

This module contains the foundational data structures used by PaxPlot:
- Arrays for storing numerical and categorical data
- Matrix for managing collections of arrays
- Tick management for axis customization
"""

from .arrays.numerical_array import NumericalArray
from .arrays.categorical_array import CategoricalArray
from .matrix import Matrix

__all__ = [
    "NumericalArray",
    "CategoricalArray",
    "Matrix",
]
