"""Categorical normalized array class for handling and normalizing string data."""

from typing import Sequence
import numpy as np
from numpy.typing import NDArray
from pydantic import PrivateAttr, ConfigDict
from paxplot.data_managers.array_normalizer import ArrayNormalizer
from paxplot.data_managers.base_normalized_array import BaseNormalizedArray


class CategoricalNormalizedArray(BaseNormalizedArray):
    """
    A normalized array class for categorical (string) data.

    This class maps categorical string values to numeric indices and normalizes them
    using an internal `ArrayNormalizer`. The mapping from labels to indices is stored
    internally and used to transform new data as needed.

    Attributes
    ----------
    array : Sequence[str]
        The raw categorical string values provided at initialization.

    _label_to_index : dict[str, int]
        Internal dictionary mapping string labels to numeric indices for normalization.
    """

    array: Sequence[str]
    _label_to_index: dict[str, int] = PrivateAttr(default_factory=dict)
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _init_normalizer(self) -> ArrayNormalizer:
        """
        Initializes the ArrayNormalizer by converting categorical strings to numeric indices.

        This method builds a label-to-index mapping from the raw string array and
        returns an `ArrayNormalizer` initialized with the resulting index array.

        Returns
        -------
        ArrayNormalizer
            The normalizer initialized with numeric indices derived from the categorical data.
        """
        label_to_index = {}
        indices = []
        for label in self.array:
            if label not in label_to_index:
                label_to_index[label] = len(label_to_index)
            indices.append(label_to_index[label])

        self._label_to_index = label_to_index

        index_array = np.array(indices, dtype=np.float64)
        return ArrayNormalizer(array=index_array)

    @property
    def array_indeces(self) -> NDArray[np.float64]:
        """
        Returns the numeric index array corresponding to the original categorical values.

        Returns
        -------
        NDArray[np.float64]
            A NumPy array of float64 values representing label indices.
        """
        return self._normalizer.array

    def append_array(self, new_data: Sequence[str]) -> None:
        """
        Appends new categorical data to the array and updates normalization state.

        This method extends the existing label-to-index mapping, appends new
        labels and their corresponding indices, and updates the internal normalizer.

        Parameters
        ----------
        new_data : Sequence[str]
            New categorical string values to append.
        """
        current_index = len(self._label_to_index)
        indices = []

        for label in new_data:
            if label not in self._label_to_index:
                self._label_to_index[label] = current_index
                current_index += 1
            indices.append(self._label_to_index[label])

        # Convert new labels to index array
        new_index_array = np.array(indices, dtype=np.float64)

        # Append to existing array and update normalizer
        self.array = list(self.array) + list(new_data)
        self._normalizer.append_array(new_index_array)

    def update_array(self, new_data: Sequence[str]) -> None:
        """
        Replaces the current raw array with new categorical data and updates
        the label-to-index mapping and normalized values accordingly.

        This method rebuilds the internal label-to-index mapping from scratch
        based on the new data.

        Parameters
        ----------
        new_data : Sequence[str]
            New categorical string values to replace the current array.
        """
        # Rebuild mapping from scratch
        label_to_index = {}
        indices = []
        for label in new_data:
            if label not in label_to_index:
                label_to_index[label] = len(label_to_index)
            indices.append(label_to_index[label])

        self._label_to_index = label_to_index
        self.array = list(new_data)

        index_array = np.array(indices, dtype=np.float64)
        self._normalizer.update_array(index_array)
