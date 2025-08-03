# pylint: disable=W2301
"""
Base classes for normalized array types.

This module defines the abstract base class `BaseNormalizedArray`, which provides
a common interface and shared functionality for normalized array data structures.
It manages raw data storage and lazy construction of an internal `ArrayNormalizer`
for normalization operations.

Subclasses like `NumericNormalizedArray` and `CategoricalNormalizedArray` are
expected to implement specific normalization strategies and data manipulation
methods such as appending new data and removing elements by index.

The module depends on NumPy for array handling and Pydantic for data validation.
"""

from abc import ABC, abstractmethod
from typing import Any, Sequence, Optional
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, PrivateAttr
from paxplot.data_managers.array_normalizer import ArrayNormalizer

class BaseNormalizedArray(BaseModel, ABC):
    """
    Abstract base class for normalized arrays.

    This class provides a common interface and partial implementation for
    normalized array types, managing a sequence of raw values and an internal
    ArrayNormalizer instance to perform normalization.

    Subclasses must implement methods for building the normalizer, appending data,
    and removing elements by index.

    Parameters
    ----------
    array : Sequence[Any]
        The raw input data sequence to be normalized.

    Attributes
    ----------
    _normalizer : Optional[ArrayNormalizer]
        Internal instance responsible for normalization operations.
    """

    array: Sequence[Any]
    _normalizer: Optional[ArrayNormalizer] = PrivateAttr(default=None)

    def __len__(self) -> int:
        return len(self.array)

    def __getitem__(self, index: int) -> Any:
        return self.array[index]

    @property
    def array_normalized(self) -> NDArray[np.float64]:
        """
        Returns the normalized NumPy array scaled to the range [-1, 1].

        This property initializes the internal normalizer if not already created.

        Returns
        -------
        NDArray[np.float64]
            The normalized array of float64 values.
        """
        if self._normalizer is None:
            self._normalizer = self._build_normalizer()
        return self._normalizer.array_normalized

    @abstractmethod
    def _build_normalizer(self) -> ArrayNormalizer:
        pass

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

    @abstractmethod
    def remove_indices(self, indices: Sequence[int]) -> None:
        """
        Removes elements at specified indices from the array and updates normalization.

        Subclasses must implement removal logic and normalization updates.

        Parameters
        ----------
        indices : Sequence[int]
            Indices of elements to remove from the array.
        """
        ...
