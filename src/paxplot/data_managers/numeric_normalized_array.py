"""Defines the NumericNormalizedArray class, a specialized normalized array 
for numeric data (floats and integers).

This module extends the abstract BaseNormalizedArray class to support sequences 
of numeric values that are automatically normalized to the range [-1, 1] using 
min-max scaling. It provides methods to append new values and maintain normalization 
state efficiently.

Classes
-------
NumericNormalizedArray
    A concrete implementation of BaseNormalizedArray for numeric (int or float) sequences.
"""

from typing import Sequence

from pydantic import validate_call
import numpy as np

from paxplot.data_managers.base_normalized_array import BaseNormalizedArray
from paxplot.data_managers.array_normalizer import ArrayNormalizer


class NumericNormalizedArray(BaseNormalizedArray):
    """
    A normalized array specifically for numeric (float or int) sequences.

    Automatically normalizes the input sequence to the range [-1, 1]
    using min-max scaling via the internal ArrayNormalizer.

    Attributes
    ----------
    array : Sequence[float | int]
        The raw sequence of numeric values.
    """
    values: Sequence[float | int]

    def _init_normalizer(self) -> ArrayNormalizer:
        arr = np.array(self.values, dtype=np.float64)
        return ArrayNormalizer(array=arr)

    @validate_call
    def append_array(self, new_data: Sequence[float | int]) -> None:
        """
        Appends new numeric data to the raw array and updates the normalized values.

        Parameters
        ----------
        new_data : Sequence[float | int]
            A sequence of numeric values to append.

        Raises
        ------
        TypeError
            If the input is not a sequence of numeric values.
        """
        new_array = np.array(new_data, dtype=np.float64)
        self._normalizer.append_array(new_array)
        self.values = list(self.values) + list(new_data)

    @validate_call
    def update_array(self, new_data: Sequence[float | int]) -> None:
        """
        Replaces the current raw array with new numeric data and updates
        the normalized values accordingly.

        Parameters
        ----------
        new_data : Sequence[float | int]
            A sequence of numeric values to replace the current array.

        Raises
        ------
        TypeError
            If the input is not a sequence of numeric values.
        """
        new_array = np.array(new_data, dtype=np.float64)
        self._normalizer.update_array(new_array)
        self.values = list(new_data)

    @validate_call
    def set_custom_bounds(self, min_val: float | None = None, max_val: float | None = None) -> None:
        """
        Sets custom min and/or max bounds for normalization and updates normalized array.

        Parameters
        ----------
        min_val : float | None
            Custom minimum value for normalization.
        max_val : float | None
            Custom maximum value for normalization.
        """
        self._normalizer.set_custom_bounds(min_val=min_val, max_val=max_val)
