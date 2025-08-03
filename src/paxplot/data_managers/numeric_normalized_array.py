from typing import Sequence, Union

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
        if not isinstance(new_data, Sequence):
            raise TypeError("Input must be a sequence of floats or ints.")
        new_array = np.array(new_data, dtype=np.float64)
        self._normalizer.append_array(new_array)
        self.array = list(self.array) + list(new_data)
