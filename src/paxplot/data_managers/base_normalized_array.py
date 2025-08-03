# pylint: disable=W2301
"""
Base classes for normalized array types.

This module defines the abstract base class `BaseNormalizedArray`, which provides
a common interface and shared functionality for normalized array data structures.
It manages raw data storage and eager construction of an internal `ArrayNormalizer`
for normalization operations.

The module depends on NumPy for array handling and Pydantic for data validation.
"""

from abc import ABC, abstractmethod
from typing import Any, Sequence
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, PrivateAttr, validate_call
from paxplot.data_managers.array_normalizer import ArrayNormalizer


class BaseNormalizedArray(BaseModel, ABC):
    """
    Abstract base class for normalized arrays.

    This class provides a common interface and partial implementation for
    normalized array types, managing a sequence of raw values and an internal
    ArrayNormalizer instance to perform normalization.

    Subclasses must implement methods for appending data and removing elements by index.

    Parameters
    ----------
    array : Sequence[Any]
        The raw input data sequence to be normalized.

    Attributes
    ----------
    _normalizer : ArrayNormalizer
        Internal instance responsible for normalization operations.
    """

    array: Sequence[Any]
    _normalizer: ArrayNormalizer = PrivateAttr()

    def __init__(self, **data):
        super().__init__(**data)
        arr = np.array(self.array, dtype=np.float64)
        self._normalizer = ArrayNormalizer(array=arr)

    def __len__(self) -> int:
        return len(self.array)

    def __getitem__(self, index: int) -> Any:
        return self.array[index]

    @property
    def array_normalized(self) -> NDArray[np.float64]:
        """
        Returns the normalized NumPy array scaled to the range [-1, 1].

        Returns
        -------
        NDArray[np.float64]
            The normalized array of float64 values.
        """
        return self._normalizer.array_normalized

    @abstractmethod
    def append_array(self, new_data: Sequence[Any]) -> None:
        """
        Appends new data elements to the existing array and updates normalization.

        Subclasses must implement how new data is added and how normalization
        state is updated accordingly.

        Parameters
        ----------
        new_data : Sequence[Any]
            New elements to append to the array.
        """
        ...

    @validate_call
    def remove_indices(self, indices: Sequence[int]) -> None:
        """
        Removes elements at the specified indices from the raw array and updates the internal
        normalization.

        Parameters
        ----------
        indices : Sequence[int]
            A sequence of integer indices indicating which elements to remove.

        Raises
        ------
        TypeError
            If the input is not a sequence of integers.
        IndexError
            If any index is out of bounds.
        """
        if not isinstance(indices, Sequence):
            raise TypeError("Indices must be a sequence of integers.")
        index_array = np.array(indices, dtype=np.int64)
        self._normalizer.remove_indices(index_array)

        # Remove raw entries from `self.array`
        array_np = np.array(self.array)
        mask = np.ones(len(array_np), dtype=bool)
        mask[index_array] = False
        self.array = array_np[mask].tolist()
