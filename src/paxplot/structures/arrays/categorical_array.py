"""Categorical array data structure for PaxPlot.

This module defines the CategoricalArray class for storing and managing
sequences of categorical (string) values.
"""

from typing import List, Sequence


class CategoricalArray:
    """
    A simple array for storing categorical data.

    This class provides basic operations for storing and managing
    sequences of categorical (string) values without any normalization logic.

    Parameters
    ----------
    values : Sequence[str]
        The initial categorical values to store.

    Attributes
    ----------
    values : List[str]
        The stored categorical values as a list of strings.
    unique_values : List[str]
        The unique categorical values in order of appearance.

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
    """

    def __init__(self, values: Sequence[str]):
        self._values = self._validate_and_convert(values)
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
            The categorical values to append.

        Raises
        ------
        ValueError
            If any value is not a string or is None.
        """
        new_values = self._validate_and_convert(values)
        self._values.extend(new_values)
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

    def _validate_and_convert(self, values: Sequence[str]) -> List[str]:
        """Validate and convert values to a list of strings.

        Parameters
        ----------
        values : Sequence[str]
            The values to validate and convert.

        Returns
        -------
        List[str]
            The validated values as a list of strings.

        Raises
        ------
        ValueError
            If any value is not a string or is None.
        """
        if values is None:
            raise ValueError("Values cannot be None")

        converted_values = []
        for i, value in enumerate(values):
            if value is None:
                raise ValueError(f"Value at index {i} cannot be None")

            if not isinstance(value, str):
                raise ValueError(
                    f"Value at index {i} must be a string, got {type(value)}: {value}"
                )

            converted_values.append(value)

        return converted_values

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
