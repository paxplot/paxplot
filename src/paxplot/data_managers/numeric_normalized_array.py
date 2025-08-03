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

from typing import Sequence, Union

from pydantic import validate_call
import numpy as np

from paxplot.data_managers.base_normalized_array import BaseNormalizedArray


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

    array: Sequence[Union[int, float]]

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
        self.array = list(self.array) + list(new_data)
