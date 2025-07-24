"""The ArrayNormalizer class"""

import json
from typing import Any, Type

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, field_validator, ConfigDict


class ArrayNormalizer(BaseModel):
    """
    Class to normalize arrays
    """

    array: NDArray[np.number]
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("array", mode="before")
    @classmethod
    def validate_numpy_array(cls: Type[Any], v: Any) -> NDArray[np.number]:
        """
        Validates that the input is a 1D NumPy array with a numeric data type.

        Parameters
        ----------
        cls : Type[Any]
            The class this validator belongs to (automatically passed by Pydantic).
        v : Any
            The input value to validate.

        Returns
        -------
        NDArray[np.number]
            The validated 1D numeric NumPy array.

        Raises
        ------
        ValueError
            If the input is not a NumPy array.
        ValueError
            If the array is not one-dimensional.
        ValueError
            If the array's data type is not numeric (e.g., float, int).
        """

        if not isinstance(v, np.ndarray):
            raise ValueError(
                f"`array` must be a NumPy array, got {type(v).__name__}"
            )

        shape: tuple[int, ...] = v.shape
        if len(shape) != 1:
            raise ValueError(f"`array` must be 1D, got shape {shape}")

        dtype: np.dtype = v.dtype
        if not np.issubdtype(dtype, np.number):
            raise ValueError(f"`array` dtype must be numeric, got {dtype}")

        return v

    @staticmethod
    def _normalize_to_minus1_plus1(
        array: NDArray[np.number], min_val: float, max_val: float
    ) -> NDArray[np.number]:
        """
        Normalizes a NumPy array to the range [-1, 1] using the provided min and max values.

        Parameters
        ----------
        array : NDArray[np.number]
            Input array to normalize.
        min_val : float
            Minimum possible value in the original scale.
        max_val : float
            Maximum possible value in the original scale.

        Returns
        -------
        NDArray[np.number]
            Normalized array in the range [-1, 1].
        """
        scale = 2.0 / (max_val - min_val)
        return scale * (array - min_val) - 1.0

    def model_dump_json(self, **kwargs) -> str:
        """
        Serializes the model to a JSON string, converting the NumPy array to a list.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments to pass to `model_dump()` (e.g., `exclude`, `include`).

        Returns
        -------
        str
            A JSON string representation of the model with the array converted to a list.
        """
        raw = self.model_dump(**kwargs)
        raw["array"] = self.array.tolist()
        return json.dumps(raw)
