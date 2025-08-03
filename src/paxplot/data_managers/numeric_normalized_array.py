"""Defines the NumericNormalizedArray class, a specialized normalized array 
for numeric data (floats and integers).

This module extends the abstract BaseNormalizedArray class to support sequences 
of numeric values that are automatically normalized to the range [-1, 1] using 
min-max scaling. It provides methods to append new values and maintain normalization 
state efficiently.

Classes
-------
NumericNormalizedArray
    A concrete implementation of BaseNormalizedArray for numeric (int or float) sequences.
"""

from typing import Sequence, Any

from pydantic import validate_call
import numpy as np

from paxplot.data_managers.base_normalized_array import BaseNormalizedArray


class NumericNormalizedArray(BaseNormalizedArray):
    """
    A normalized array specifically for numeric (float or int) sequences.

    Automatically normalizes the input sequence to the range [-1, 1]
    using min-max scaling via the internal ArrayNormalizer.

    Attributes
    ----------
    array : Sequence[float | int]
        The raw sequence of numeric values.
    """

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

    @validate_call
    def append_array(self, new_data: Sequence[float | int]) -> None:
        """
        Appends new numeric data to the raw array and updates the normalized values.

        Parameters
        ----------
        new_data : Sequence[float | int]
            A sequence of numeric values to append.

        Raises
        ------
        TypeError
            If the input is not a sequence of numeric values.
        """
        new_array = np.array(new_data, dtype=np.float64)
        self._normalizer.append_array(new_array)
        self.array = list(self.array) + list(new_data)

    @validate_call
    def set_custom_bounds(self, min_val: float | None = None, max_val: float | None = None) -> None:
        """
        Sets custom min and/or max bounds for normalization and updates normalized array.

        Parameters
        ----------
        min_val : float | None
            Custom minimum value for normalization.
        max_val : float | None
            Custom maximum value for normalization.
        """
        self._normalizer.set_custom_bounds(min_val=min_val, max_val=max_val)

    def to_dict(self) -> dict[str, Any]:
        """
        Converts the object to a dictionary for serialization.
        This method includes the raw array and custom bounds in the output.

        Returns
        -------
        dict
            A dictionary containing the raw array and custom bounds.
        """
        return {
            "array": list(self.array),
            "custom_min_val": self.custom_min_val,
            "custom_max_val": self.custom_max_val,
            "_schema_version": self._schema_version
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NumericNormalizedArray":
        """
        Creates a NumericNormalizedArray instance from a dictionary.
        This method handles schema version checking and reconstructs the object

        Parameters
        ----------
        data : dict
            A dictionary containing the raw array and custom bounds. Must include
            the '_schema_version' key.

        Returns
        -------
        NumericNormalizedArray
            The reconstructed NumericNormalizedArray instance.

        Raises
        ------
        ValueError
            If the schema version in the data is not supported.
        """
        version = data.get("_schema_version", 0)
        if version > cls._schema_version:
            raise ValueError(
                f"Unsupported schema version: {version}. "
                f"Current supported version is {cls._schema_version}."
            )

        instance = cls(array=data["array"])
        instance.set_custom_bounds(
            min_val=data.get("custom_min_val"),
            max_val=data.get("custom_max_val")
        )
        return instance
