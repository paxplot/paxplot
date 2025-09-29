"""Categorical array data structure for PaxPlot.

This module defines the CategoricalArray class for storing and managing
sequences of categorical (string) values.
"""

from typing import List, Sequence, Union
from .base_array import BaseArray


class CategoricalArray(BaseArray[str]):
    """
    A simple array for storing categorical data.

    This class provides basic operations for storing and managing
    sequences of categorical (string) values. Supports NaN values
    which are stored as "<NaN>" string representation.

    Parameters
    ----------
    values : Sequence[str]
        The initial categorical values to store. Can include None, float('nan'),
        or numpy.nan which will be converted to "<NaN>" string.

    Attributes
    ----------
    unique_values : List[str]
        The unique categorical values in order of appearance.

    Examples
    --------
    >>> array = CategoricalArray(['A', 'B', 'A', 'C'])
    >>> array.get_values()
    ['A', 'B', 'A', 'C']
    >>> array.unique_values
    ['A', 'B', 'C']
    """

    def __init__(self, values: Union[Sequence[str], None] = None):
        """Initialize the categorical array."""
        super().__init__(values)
        self._update_unique_values()

    @property
    def unique_values(self) -> List[str]:
        """Get the unique categorical values in order of appearance.

        Returns
        -------
        List[str]
            The unique categorical values.
        """
        return self._unique_values.copy()

    def append_values(self, values: Sequence[str]) -> None:
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
        super().append_values(values)
        self._update_unique_values()

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
        super().remove_values(indices)
        self._update_unique_values()

    def set_values(self, values: Union[Sequence[str], None] = None) -> None:
        """Set new categorical values, replacing all existing values.

        Parameters
        ----------
        values : Sequence[str], optional
            The new categorical values to set. Can include None, float('nan'),
            or numpy.nan which will be converted to "<NaN>" string. If None,
            creates empty array.

        Raises
        ------
        ValueError
            If any value is not a string or cannot be converted to string.
        """
        super().set_values(values)
        self._update_unique_values()

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

    def _validate_and_convert(
        self, values: Union[Sequence[str], None] = None
    ) -> tuple[List[str], bool]:
        """Validate and convert values to a list of strings, also computing NaN state.

        Parameters
        ----------
        values : Sequence[str], optional
            The values to validate and convert. Can include None, float('nan'), or
            numpy.nan, which will be converted to the "<NaN>" string. If None,
            returns an empty list.

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
            return [], False

        converted_values = []
        has_nan = False

        for i, value in enumerate(values):
            if self._is_nan(value):
                converted_values.append("<NaN>")
                has_nan = True
            else:
                if not isinstance(value, str):
                    raise ValueError(
                        f"Value at index {i} must be a string, "
                        f"got {type(value)}: {value}"
                    )
                converted_values.append(value)

        return converted_values, has_nan

    def _get_nan_representation(self) -> str:
        """Return the NaN representation for categorical arrays.

        Returns
        -------
        str
            "<NaN>"
        """
        return "<NaN>"

    def _is_nan_value(self, value: str) -> bool:
        """Check if a converted value is NaN for categorical arrays.

        Parameters
        ----------
        value : str
            The converted value to check.

        Returns
        -------
        bool
            True if the value is "<NaN>", False otherwise.
        """
        return value == "<NaN>"

    def _update_unique_values(self) -> None:
        """Update the list of unique values in order of appearance."""
        unique_values = []
        for value in self._values:
            if value not in unique_values:
                unique_values.append(value)
        self._unique_values = unique_values

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
