"""Categorical array data structure for PaxPlot.

This module defines the CategoricalArray class for storing and managing
sequences of categorical (string) values.
"""

import math
from typing import List, Sequence


class CategoricalArray:
    """
    A simple array for storing categorical data.

    This class provides basic operations for storing and managing
    sequences of categorical (string) values without any normalization logic.
    Supports NaN values which are stored as "<NaN>" string representation.

    Parameters
    ----------
    values : Sequence[str]
        The initial categorical values to store. Can include None, float('nan'),
        or numpy.nan which will be converted to "<NaN>" string.

    Attributes
    ----------
    values : List[str]
        The stored categorical values as a list of strings.
    unique_values : List[str]
        The unique categorical values in order of appearance.
    has_nan : bool
        Whether the array contains any NaN values.
    nan_count : int
        The number of NaN values in the array.
    nan_indices : List[int]
        The indices of NaN values in the array.
    non_nan_values : List[str]
        All non-NaN values in the array.

    Examples
    --------
    >>> array = CategoricalArray(['A', 'B', 'A', 'C'])
    >>> array.values
    ['A', 'B', 'A', 'C']
    >>> array.unique_values
    ['A', 'B', 'C']
    >>> array.append(['B', 'D'])
    >>> array.values
    ['A', 'B', 'A', 'C', 'B', 'D']
    
    >>> array_with_nan = CategoricalArray(['A', None, 'B', float('nan'), 'C'])
    >>> array_with_nan.has_nan
    True
    >>> array_with_nan.nan_count
    2
    >>> array_with_nan.nan_indices
    [1, 3]
    >>> array_with_nan.unique_values
    ['A', '<NaN>', 'B', 'C']
    """

    def __init__(self, values: Sequence[str]):
        self._values, self._has_nan = self._validate_and_convert(values)
        self._update_unique_values()

    @property
    def values(self) -> List[str]:
        """Get the stored categorical values.

        Returns
        -------
        List[str]
            The categorical values as a list of strings.
        """
        return self._values.copy()

    @property
    def unique_values(self) -> List[str]:
        """Get the unique categorical values in order of appearance.

        Returns
        -------
        List[str]
            The unique categorical values.
        """
        return self._unique_values.copy()

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
        return self._values.count("<NaN>")

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
        return [i for i, value in enumerate(self._values) if value == "<NaN>"]

    @property
    def non_nan_values(self) -> List[str]:
        """Get all non-NaN values in the array.

        Returns
        -------
        List[str]
            All non-NaN values in the array.
        """
        if not self._has_nan:
            return self._values.copy()
        return [value for value in self._values if value != "<NaN>"]

    @property
    def length(self) -> int:
        """Get the number of values in the array.

        Returns
        -------
        int
            The number of values.
        """
        return len(self._values)

    def append(self, values: Sequence[str]) -> None:
        """Append new categorical values to the array.

        Parameters
        ----------
        values : Sequence[str]
            The categorical values to append. Can include None, float('nan'),
            or numpy.nan which will be converted to "<NaN>" string.

        Raises
        ------
        ValueError
            If any value is not a string or cannot be converted to string.
        """
        new_values, has_new_nans = self._validate_and_convert(values)
        self._values.extend(new_values)
        # Update NaN state if we don't already have NaNs
        if not self._has_nan:
            self._has_nan = has_new_nans
        self._update_unique_values()

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
        self._update_unique_values()

    def reset_nan_state(self) -> None:
        """Reset the cached NaN state.
        
        Call this method if you manually modify the underlying _values list
        and need to update the NaN state cache.
        """
        self._has_nan = self._compute_has_nan()

    def get_category_indices(self) -> List[int]:
        """Get the indices of each value in the unique_values list.

        This method is useful for future normalization where categories
        need to be mapped to numerical indices.

        Returns
        -------
        List[int]
            The indices of each value in the unique_values list.
        """
        return [self._unique_values.index(value) for value in self._values]

    def _validate_and_convert(self, values: Sequence[str]) -> tuple[List[str], bool]:
        """Validate and convert values to a list of strings, also computing NaN state.

        Parameters
        ----------
        values : Sequence[str]
            The values to validate and convert. Can include None, float('nan'),
            or numpy.nan which will be converted to "<NaN>" string.

        Returns
        -------
        tuple[List[str], bool]
            The validated values as a list of strings and whether any NaNs were found.

        Raises
        ------
        ValueError
            If any value cannot be converted to string.
        """
        if values is None:
            raise ValueError("Values cannot be None")

        converted_values = []
        has_nan = False
        
        for i, value in enumerate(values):
            if self._is_nan(value):
                converted_values.append("<NaN>")
                has_nan = True
            else:
                if not isinstance(value, str):
                    raise ValueError(
                        f"Value at index {i} must be a string, got {type(value)}: {value}"
                    )
                converted_values.append(value)

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
        return "<NaN>" in self._values

    def _update_unique_values(self) -> None:
        """Update the list of unique values in order of appearance."""
        unique_values = []
        for value in self._values:
            if value not in unique_values:
                unique_values.append(value)
        self._unique_values = unique_values

    def __len__(self) -> int:
        """Get the number of values in the array.

        Returns
        -------
        int
            The number of values.
        """
        return len(self._values)

    def __getitem__(self, index: int) -> str:
        """Get a value at the specified index.

        Parameters
        ----------
        index : int
            The index of the value to get.

        Returns
        -------
        str
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
            return f"CategoricalArray({self._values})"
        return f"CategoricalArray({self._values[:5]}...{self._values[-5:]})"
