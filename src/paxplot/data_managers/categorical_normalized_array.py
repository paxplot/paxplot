"""Categorical normalized array class for handling and normalizing string data."""

from typing import Sequence, Any
import numpy as np
from numpy.typing import NDArray
from pydantic import PrivateAttr
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

    class Config:
        """Pydantic configuration to allow arbitrary types such as NumPy arrays."""
        arbitrary_types_allowed = True

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

        This method must extend the existing label-to-index mapping and update the
        internal normalizer accordingly.

        Parameters
        ----------
        new_data : Sequence[str]
            New categorical string values to append.

        Raises
        ------
        NotImplementedError
            This method must be implemented in the future to support dynamic updates.
        """
        raise NotImplementedError("Appending categorical data is not yet implemented.")

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the object to a dictionary including both raw and normalized data.

        This method should return all relevant information for full reconstruction.

        Returns
        -------
        dict[str, Any]
            A dictionary representation of the object.

        Raises
        ------
        NotImplementedError
            This method must be implemented for full serialization support.
        """
        raise NotImplementedError("Serialization is not yet implemented.")
