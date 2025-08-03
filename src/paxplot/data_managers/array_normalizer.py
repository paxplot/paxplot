"""The ArrayNormalizer class"""

from typing import Any, Type, ClassVar

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, field_validator, model_validator, ConfigDict


class ArrayNormalizer(BaseModel):
    """
    Class to normalize arrays
    """

    _schema_version: ClassVar[int] = 1
    array: NDArray[np.float64]
    custom_min_val: float | None = None
    custom_max_val: float | None = None
    _array_normalized: NDArray[np.float64] | None = None
    _effective_min_val: float | None = None
    _effective_max_val: float | None = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def array_normalized(self) -> NDArray[np.float64]:
        """
        Returns the normalized version of the input array, scaled to the range [-1, 1].

        Returns
        -------
        NDArray[np.number]
            A NumPy array where the values have been normalized using min-max scaling.

        Raises
        ------
        AssertionError
            If normalization has not yet been computed.
        """
        if self._array_normalized is None:
            raise ValueError("Normalization has not been computed yet")
        return self._array_normalized

    @property
    def effective_min_val(self) -> float:
        """
        Returns the effective minimum value used for normalization (defaults to custom).

        Raises
        ------
        ValueError
            If normalization has not yet been computed.
        """
        if self._effective_min_val is None:
            raise ValueError("Normalization has not been computed yet")
        return self._effective_min_val

    @property
    def effective_max_val(self) -> float:
        """
        Returns the effective maximum value used for normalization (defaults to custom).

        Raises
        ------
        ValueError
            If normalization has not yet been computed.
        """
        if self._effective_max_val is None:
            raise ValueError("Normalization has not been computed yet")
        return self._effective_max_val

    @field_validator("array", mode="before")
    @classmethod
    def validate_numpy_array(cls: Type[Any], v: Any) -> NDArray[np.float64]:
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
        NDArray[np.float64]
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

        return v.astype(np.float64)

    @model_validator(mode="after")
    def normalize_after_init(self) -> "ArrayNormalizer":
        """
        Recomputes the normalized array and value bounds after model initialization.

        Returns
        -------
        ArrayNormalizer
            The updated model instance with normalization results populated.
        """
        self._recompute_normalization()
        return self

    def _recompute_normalization(self) -> None:
        """
        Computes the normalized array (`_array_normalized`) and stores the
        min and max values used in the normalization.

        If the array has constant values, sets the normalized array to zeros.
        """
        min_val = (
            self.custom_min_val
            if self.custom_min_val is not None
            else float(np.min(self.array))
        )

        max_val = (
            self.custom_max_val
            if self.custom_max_val is not None
            else float(np.max(self.array))
        )

        self._effective_min_val = min_val
        self._effective_max_val = max_val

        if max_val == min_val:
            self._array_normalized = np.zeros_like(self.array, dtype=np.float64)
        else:
            self._array_normalized = self._normalize_to_minus1_plus1(
                self.array, min_val, max_val
            )

    @staticmethod
    def _normalize_to_minus1_plus1(
        array: NDArray[np.number], min_val: float, max_val: float
    ) -> NDArray[np.float64]:
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
        array_float = array.astype(np.float64)
        scale = 2.0 / (max_val - min_val)
        return scale * (array_float - min_val) - 1.0


    def update_array(self, new_array: NDArray[np.number]) -> None:
        """
        Updates the internal array with a new NumPy array and re-applies normalization.

        Parameters
        ----------
        new_array : NDArray[np.number]
            A 1D numeric NumPy array to replace the existing array.
        """
        self.array = self.validate_numpy_array(new_array)
        self._recompute_normalization()

    def to_dict(self, **kwargs) -> dict:
        """
        Serializes the model to a dictionary, converting the NumPy array to a list.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments to pass to `model_dump()`.

        Returns
        -------
        dict
            A dictionary representation of the model with the array converted to a list.
        """
        raw = self.model_dump(**kwargs)
        raw["array"] = self.array.tolist()
        raw["_schema_version"] = self._schema_version
        return raw

    @classmethod
    def from_dict(cls: Type["ArrayNormalizer"], data: dict) -> "ArrayNormalizer":
        """
        Creates an ArrayNormalizer instance from a dictionary, converting
        the array list back into a NumPy array.

        Parameters
        ----------
        data : dict
            Dictionary representation of the serialized ArrayNormalizer model.

        Returns
        -------
        ArrayNormalizer
            A new instance of ArrayNormalizer with the array restored as a NumPy array.
        """
        version = data.get("_schema_version", 0)
        if version > cls._schema_version:
            raise ValueError(
                f"Unsupported schema version: {version}. "
                f"Current supported version is {cls._schema_version}."
            )

        arr = np.array(data["array"])
        return cls(
            array=arr,
            custom_min_val=data.get("custom_min_val"),
            custom_max_val=data.get("custom_max_val"),
        )
