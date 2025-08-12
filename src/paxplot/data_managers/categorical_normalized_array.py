"""Categorical normalized array class for handling and normalizing string data."""

from typing import Sequence, List
import numpy as np
from numpy.typing import NDArray
from pydantic import PrivateAttr, ConfigDict
from paxplot.data_managers.array_normalizer import ArrayNormalizer
from paxplot.data_managers.base_normalized_array import BaseNormalizedArray


class CategoricalNormalizedArray(BaseNormalizedArray):
    """
    A normalized array class for categorical (string) data.

    Stores a list of unique categories in the order they appear.
    Numeric indices correspond to the position of each category in this list.
    """

    values: Sequence[str]
    _categories: List[str] = PrivateAttr(default_factory=list)
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _init_normalizer(self) -> ArrayNormalizer:
        """
        Initializes the ArrayNormalizer by mapping strings to indices via the categories list.

        Returns
        -------
        ArrayNormalizer
            The normalizer initialized with numeric indices derived from the categorical data.
        """
        categories = []
        for category in self.values:
            if category not in categories:
                categories.append(category)
        self._categories = categories

        indices = np.array(
            [categories.index(category) for category in self.values],
            dtype=np.float64
        )
        return ArrayNormalizer(array=indices)

    @property
    def value_indices(self) -> NDArray[np.float64]:
        """
        Returns the numeric index array corresponding to the original categorical values.

        Returns
        -------
        NDArray[np.float64]
            A NumPy array of float64 values representing category indices.
        """
        return self._normalizer.array

    @property
    def categories(self) -> List[str]:
        """
        Returns the list of unique categories in order.

        Returns
        -------
        List[str]
            The list of unique category labels.
        """
        return self._categories

    def append_array(self, new_data: Sequence[str]) -> None:
        """
        Appends new categorical data to the array and updates normalization.

        Parameters
        ----------
        new_data : Sequence[str]
            New categorical string values to append.
        """
        categories = self._categories
        indices = []
        for category in new_data:
            if category not in categories:
                categories.append(category)
            indices.append(categories.index(category))

        new_index_array = np.array(indices, dtype=np.float64)

        self.values = list(self.values) + list(new_data)
        self._normalizer.append_array(new_index_array)

    def update_array(self, new_data: Sequence[str]) -> None:
        """
        Replaces the current array with new data and rebuilds categories and normalization.

        Parameters
        ----------
        new_data : Sequence[str]
            New categorical string values to replace the current array.
        """
        categories = []
        for category in new_data:
            if category not in categories:
                categories.append(category)
        self._categories = categories
        self.values = list(new_data)

        indices = np.array([categories.index(category) for category in new_data], dtype=np.float64)
        self._normalizer.update_array(indices)
