"""Numerical array data structure for PaxPlot.

This module defines the NumericalArray class for storing and managing
sequences of numerical values.
"""

import math
from typing import List, Sequence, Union


class NumericalArray:
    """
    A simple array for storing numerical data.

    This class provides basic operations for storing and managing
    sequences of numerical values without any normalization logic.
    Supports NaN values which are stored as float('nan').

    Parameters
    ----------
    values : Sequence[Union[float, int]]
        The initial numerical values to store. Can include None, float('nan'),
        or numpy.nan which will be converted to float('nan').

    Attributes
    ----------
    values : List[float]
        The stored numerical values as a list of floats.
    has_nan : bool
        Whether the array contains any NaN values.
    nan_count : int
        The number of NaN values in the array.
    nan_indices : List[int]
        The indices of NaN values in the array.
    non_nan_values : List[float]
        All non-NaN values in the array.
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
    >>> array.append([6, 7])
    >>> array.values
    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    
    >>> array_with_nan = NumericalArray([1.0, None, 3.0, float('nan'), 5.0])
    >>> array_with_nan.has_nan
    True
    >>> array_with_nan.min
    1.0
    >>> array_with_nan.max
    5.0
    """

    def __init__(self, values: Sequence[Union[float, int]]):
        self._values, self._has_nan = self._validate_and_convert(values)

    @property
    def values(self) -> List[float]:
        """Get the stored numerical values.

        Returns
        -------
        List[float]
            The numerical values as a list of floats.
        """
        return self._values.copy()

    @property
    def has_nan(self) -> bool:
        """Check if the array contains any NaN values.

        Returns
        -------
        bool
            True if the array contains any NaN values, False otherwise.
        """
        return self._has_nan

    @property
    def nan_count(self) -> int:
        """Get the number of NaN values in the array.

        Returns
        -------
        int
            The number of NaN values.
        """
        if not self._has_nan:
            return 0
        return sum(1 for value in self._values if self._is_nan(value))

    @property
    def nan_indices(self) -> List[int]:
        """Get the indices of NaN values in the array.

        Returns
        -------
        List[int]
            The indices of NaN values.
        """
        if not self._has_nan:
            return []
        return [i for i, value in enumerate(self._values) if self._is_nan(value)]

    @property
    def non_nan_values(self) -> List[float]:
        """Get all non-NaN values in the array.

        Returns
        -------
        List[float]
            All non-NaN values in the array.
        """
        if not self._has_nan:
            return self._values.copy()
        return [value for value in self._values if not self._is_nan(value)]

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
            raise ValueError("Cannot compute min of array containing only NaN values")
        
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
            raise ValueError("Cannot compute max of array containing only NaN values")
        
        return max(non_nan_values)

    @property
    def length(self) -> int:
        """Get the number of values in the array.

        Returns
        -------
        int
            The number of values.
        """
        return len(self._values)

    def append(self, values: Sequence[Union[float, int]]) -> None:
        """Append new numerical values to the array.

        Parameters
        ----------
        values : Sequence[Union[float, int]]
            The numerical values to append. Can include None, float('nan'),
            or numpy.nan which will be converted to float('nan').

        Raises
        ------
        ValueError
            If any value is not numerical or cannot be converted to float.
        """
        new_values, has_new_nans = self._validate_and_convert(values)
        self._values.extend(new_values)
        # Update NaN state if we don't already have NaNs
        if not self._has_nan:
            self._has_nan = has_new_nans

    def remove(self, indices: Sequence[int]) -> None:
        """Remove values at the specified indices.

        Parameters
        ----------
        indices : Sequence[int]
            The indices of values to remove.

        Raises
        ------
        IndexError
            If any index is out of bounds.
        ValueError
            If indices are not valid integers.
        """
        # Convert to list and sort in reverse order to avoid index shifting
        indices_list = sorted(indices, reverse=True)

        for index in indices_list:
            if not isinstance(index, int):
                raise ValueError(
                    f"Index must be an integer, got {type(index)}"
                )
            if index < 0 or index >= len(self._values):
                raise IndexError(
                    f"Index {index} out of bounds for array of length {len(self._values)}"
                )
            del self._values[index]

        # After removal, we need to recompute NaN state since indices shifted
        self._has_nan = self._compute_has_nan()

    def reset_nan_state(self) -> None:
        """Reset the cached NaN state.
        
        Call this method if you manually modify the underlying _values list
        and need to update the NaN state cache.
        """
        self._has_nan = self._compute_has_nan()

    def _validate_and_convert(self, values: Sequence[Union[float, int]]) -> tuple[List[float], bool]:
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
                converted_values.append(float('nan'))
                has_nan = True
            else:
                if not isinstance(value, (int, float)):
                    raise ValueError(
                        f"Value at index {i} must be numerical, got {type(value)}: {value}"
                    )
                converted_values.append(float(value))

        return converted_values, has_nan

    def _is_nan(self, value) -> bool:
        """Check if a value is NaN.

        Parameters
        ----------
        value : Any
            The value to check.

        Returns
        -------
        bool
            True if the value is NaN, False otherwise.
        """
        if value is None:
            return True
        if isinstance(value, float) and math.isnan(value):
            return True
        return False

    def _compute_has_nan(self) -> bool:
        """Compute whether the array contains any NaN values.

        Returns
        -------
        bool
            True if the array contains any NaN values, False otherwise.
        """
        return any(self._is_nan(value) for value in self._values)

    def __len__(self) -> int:
        """Get the number of values in the array.

        Returns
        -------
        int
            The number of values.
        """
        return len(self._values)

    def __getitem__(self, index: int) -> float:
        """Get a value at the specified index.

        Parameters
        ----------
        index : int
            The index of the value to get.

        Returns
        -------
        float
            The value at the specified index.
        """
        return self._values[index]

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
