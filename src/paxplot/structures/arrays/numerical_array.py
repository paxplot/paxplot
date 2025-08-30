"""Numerical array data structure for PaxPlot.

This module defines the NumericalArray class for storing and managing
sequences of numerical values.
"""

import math
from typing import List, Sequence, Union
from .base_array import BaseArray


class NumericalArray(BaseArray[float]):
    """
    A simple array for storing numerical data.

    This class provides basic operations for storing and managing
    sequences of numerical values. All values are converted to float
    for consistency. Supports NaN values which are stored as float('nan').

    Parameters
    ----------
    values : Sequence[Union[float, int]]
        The initial numerical values to store. Can include None, float('nan'),
        or numpy.nan which will be converted to float('nan').

    Attributes
    ----------
    values : List[float]
        The stored numerical values as a list of floats.
    min : float
        The minimum value, ignoring NaN values.
    max : float
        The maximum value, ignoring NaN values.

    Examples
    --------
    >>> array = NumericalArray([1, 2, 3, 4, 5])
    >>> array.values
    [1.0, 2.0, 3.0, 4.0, 5.0]
    >>> array.min
    1.0
    >>> array.max
    5.0
    """

    def _validate_and_convert(
        self, values: Sequence[Union[float, int]]
    ) -> tuple[List[float], bool]:
        """Validate and convert values to a list of floats, also computing NaN state.

        Parameters
        ----------
        values : Sequence[Union[float, int]]
            The values to validate and convert. Can include None, float('nan'),
            or numpy.nan which will be converted to float('nan').

        Returns
        -------
        tuple[List[float], bool]
            The validated values as a list of floats and whether any NaNs were found.

        Raises
        ------
        ValueError
            If any value cannot be converted to float.
        """
        if values is None:
            raise ValueError("Values cannot be None")

        converted_values = []
        has_nan = False

        for i, value in enumerate(values):
            if self._is_nan(value):
                converted_values.append(float("nan"))
                has_nan = True
            else:
                if not isinstance(value, (int, float)):
                    raise ValueError(
                        f"Value at index {i} must be numerical, got {type(value)}: {value}"
                    )
                converted_values.append(float(value))

        return converted_values, has_nan

    def _get_nan_representation(self) -> float:
        """Return the NaN representation for numerical arrays.

        Returns
        -------
        float
            float('nan')
        """
        return float("nan")

    def _is_nan_value(self, value: float) -> bool:
        """Check if a converted value is NaN for numerical arrays.

        Parameters
        ----------
        value : float
            The converted value to check.

        Returns
        -------
        bool
            True if the value is NaN, False otherwise.
        """
        return math.isnan(value)

    @property
    def min(self) -> float:
        """Get the minimum value in the array, ignoring NaN values.

        Returns
        -------
        float
            The minimum value.

        Raises
        ------
        ValueError
            If the array is empty or contains only NaN values.
        """
        if not self._values:
            raise ValueError("Cannot compute min of empty array")

        non_nan_values = self.non_nan_values
        if not non_nan_values:
            raise ValueError(
                "Cannot compute min of array containing only NaN values"
            )

        return min(non_nan_values)

    @property
    def max(self) -> float:
        """Get the maximum value in the array, ignoring NaN values.

        Returns
        -------
        float
            The maximum value.

        Raises
        ------
        ValueError
            If the array is empty or contains only NaN values.
        """
        if not self._values:
            raise ValueError("Cannot compute max of empty array")

        non_nan_values = self.non_nan_values
        if not non_nan_values:
            raise ValueError(
                "Cannot compute max of array containing only NaN values"
            )

        return max(non_nan_values)

    def __repr__(self) -> str:
        """Get a string representation of the array.

        Returns
        -------
        str
            A string representation showing the array length and first few values.
        """
        if len(self._values) <= 10:
            return f"NumericalArray({self._values})"
        return f"NumericalArray({self._values[:5]}...{self._values[-5:]})"
