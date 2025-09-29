"""Base array data structure for PaxPlot.

This module defines the BaseArray abstract class that provides common
functionality for all array types including NaN handling, validation,
and core operations.
"""

import math
from abc import ABC, abstractmethod
from typing import List, Sequence, TypeVar, Generic, Any, Union

T = TypeVar("T")  # Type variable for the stored values


class BaseArray(ABC, Generic[T]):
    """
    Abstract base class for array data structures.

    Provides common functionality for NaN handling, validation,
    and core operations that are shared between NumericalArray
    and CategoricalArray.

    Parameters
    ----------
    values : Sequence
        The initial values to store. Type depends on the concrete implementation.

    Attributes
    ----------
    has_nan : bool
        Whether the array contains any NaN values.
    nan_count : int
        The number of NaN values in the array.
    nan_indices : List[int]
        The indices of NaN values in the array.
    non_nan_values : List[T]
        All non-NaN values in the array.
    length : int
        The number of values in the array.
    """

    def __init__(self, values: Union[Sequence, None] = None):
        """Initialize the array with validated and converted values."""
        # Initialize with empty values first, then use set method
        self._values = []
        self._has_nan = False
        if values is None:
            values = []
        self.set_values(values)

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
        return sum(1 for value in self._values if self._is_nan_value(value))

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
        return [i for i, value in enumerate(self._values) if self._is_nan_value(value)]

    @property
    def non_nan_values(self) -> List[T]:
        """Get all non-NaN values in the array.

        Returns
        -------
        List[T]
            All non-NaN values in the array.
        """
        if not self._has_nan:
            return self._values.copy()
        return [value for value in self._values if not self._is_nan_value(value)]

    @property
    def length(self) -> int:
        """Get the number of values in the array.

        Returns
        -------
        int
            The number of values.
        """
        return len(self._values)

    def append_values(self, values: Sequence) -> None:
        """Append new values to the array.

        Parameters
        ----------
        values : Sequence
            The values to append. Type depends on the concrete implementation.

        Raises
        ------
        ValueError
            If any value is invalid for this array type.
        """
        new_values, has_new_nans = self._validate_and_convert(values)
        self._values.extend(new_values)
        # Update NaN state if we don't already have NaNs
        if not self._has_nan:
            self._has_nan = has_new_nans

    def remove_values(self, indices: Sequence[int]) -> None:
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
                raise ValueError(f"Index must be an integer, got {type(index)}")
            if index < 0 or index >= len(self._values):
                raise IndexError(
                    f"Index {index} out of bounds for array of length "
                    f"{len(self._values)}"
                )
            del self._values[index]

        # After removal, we need to recompute NaN state since indices shifted
        self._has_nan = self._compute_has_nan()

    def set_values(self, values: Union[Sequence, None] = None) -> None:
        """Set new values for the array, replacing all existing values.

        Parameters
        ----------
        values : Sequence, optional
            The new values to set. Type depends on the concrete implementation.
            If None, creates an empty array.

        Raises
        ------
        ValueError
            If any value is invalid for this array type.
        """
        if values is None:
            values = []
        new_values, has_new_nans = self._validate_and_convert(values)
        self._values = new_values
        self._has_nan = has_new_nans

    def get_values(self) -> List[T]:
        """Get all values in the array.

        Returns
        -------
        List[T]
            A copy of all values in the array.
        """
        return self._values.copy()

    def clear_values(self) -> None:
        """Clear all values from the array."""
        self.set_values([])

    def reset_nan_state(self) -> None:
        """Reset the cached NaN state.

        Call this method if you manually modify the underlying _values list
        and need to update the NaN state cache.
        """
        self._has_nan = self._compute_has_nan()

    def _is_nan(self, value: Any) -> bool:
        """Check if a value is NaN (before conversion).

        This method handles all common NaN representations that can be
        passed as input to the array classes.

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
        # For categorical arrays, also recognize the string representation
        if isinstance(value, str) and value == "<NaN>":
            return True
        return False

    @abstractmethod
    def _is_nan_value(self, value: T) -> bool:
        """Check if a converted value is NaN.

        This method should be implemented by subclasses to check
        if a value in the array's internal representation is NaN.

        Parameters
        ----------
        value : T
            The converted value to check.

        Returns
        -------
        bool
            True if the value is NaN, False otherwise.
        """

    def _compute_has_nan(self) -> bool:
        """Compute whether the array contains any NaN values.

        Returns
        -------
        bool
            True if the array contains any NaN values, False otherwise.
        """
        return any(self._is_nan_value(value) for value in self._values)

    @abstractmethod
    def _validate_and_convert(
        self, values: Union[Sequence, None] = None
    ) -> tuple[List[T], bool]:
        """Validate and convert values to the appropriate type.

        Also computes NaN state during validation.

        Parameters
        ----------
        values : Sequence, optional
            The values to validate and convert. If None, returns empty list.

        Returns
        -------
        tuple[List[T], bool]
            The validated values as a list of type T and whether any NaNs were found.

        Raises
        ------
        ValueError
            If any value cannot be converted to the appropriate type.
        """

    @abstractmethod
    def _get_nan_representation(self) -> T:
        """Return the NaN representation for this array type.

        Returns
        -------
        T
            The NaN representation value.
        """

    def __len__(self) -> int:
        """Get the number of values in the array.

        Returns
        -------
        int
            The number of values.
        """
        return len(self._values)

    def __getitem__(self, index: int) -> T:
        """Get a value at the specified index.

        Parameters
        ----------
        index : int
            The index of the value to get.

        Returns
        -------
        T
            The value at the specified index.
        """
        return self._values[index]
