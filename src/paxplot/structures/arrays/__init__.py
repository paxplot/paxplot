"""Array data structures for PaxPlot.

This module contains array classes for storing numerical and categorical data.
"""

from .numerical_array import NumericalArray
from .categorical_array import CategoricalArray

__all__ = [
    "NumericalArray",
    "CategoricalArray",
]
