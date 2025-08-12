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
from typing import Any, Sequence, Callable, List, TypeVar, Generic
import numpy as np
from numpy.typing import NDArray
from paxplot.data_managers.array_normalizer import ArrayNormalizer

T = TypeVar("T")

class BaseNormalizedArray(ABC, Generic[T]):
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

    values: List[T]
    _normalizer: ArrayNormalizer

    def __init__(self, values: Sequence[T]):
        self.values = list(values)
        self._normalizer = self._init_normalizer()

    @property
    def custom_min_val(self) -> float | None:
        """
        Returns the custom minimum value used for normalization, if any.

        Returns
        -------
        float | None
            The user-specified minimum value, or None if not set.
        """
        return self._normalizer.custom_min_val

    @property
    def custom_max_val(self) -> float | None:
        """
        Returns the custom maximum value used for normalization, if any.

        Returns
        -------
        float | None
            The user-specified maximum value, or None if not set.
        """
        return self._normalizer.custom_max_val

    @property
    def effective_min_val(self) -> float:
        """Returns the effective minimum value, defaults to custom_min_val

        Returns
        -------
        float
            Efective minimum value, defaults to custom_min_val
        """
        return self._normalizer.effective_min_val

    @property
    def effective_max_val(self) -> float:
        """Returns the effective maximum value, defaults to custom_max_val

        Returns
        -------
        float
            Effective maximum value, defaults to custom_max_val
        """
        return self._normalizer.effective_max_val

    @abstractmethod
    def _init_normalizer(self) -> ArrayNormalizer:
        """Subclasses must return a proper normalizer instance.

        Returns
        -------
        ArrayNormalizer
            The normalizer instance for this array type.
        """
        ...

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: int) -> Any:
        return self.values[index]

    @property
    def values_normalized(self) -> NDArray[np.float64]:
        """
        Returns the normalized NumPy values scaled to the range [-1, 1].

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
        array_np = np.array(self.values)
        mask = np.ones(len(array_np), dtype=bool)
        mask[index_array] = False
        self.values = array_np[mask].tolist()

    def register_observer(self, callback: Callable[[], None]) -> None:
        """
        Registers a callback to be called after normalization is recomputed.
        """
        self._normalizer.register_observer(callback)

    def unregister_observer(self, callback: Callable[[], None]) -> None:
        """
        Unregisters a previously registered observer callback.
        """
        self._normalizer.unregister_observer(callback)
