"""Base ticks abstract class for PaxPlot.

This module defines the BaseTicks abstract class that provides common
functionality for all tick types including validation and basic operations.
"""

import math
from abc import ABC
from typing import Sequence, Union

from ..arrays.categorical_array import CategoricalArray
from ..arrays.numerical_array import NumericalArray


class BaseTicks(ABC):
    """
    Abstract base class for tick management.

    Provides a common interface for all tick types and manages the
    relationship between tick labels and locations. Ensures consistency
    across different tick implementations.

    Attributes
    ----------
    labels : CategoricalArray
        CategoricalArray containing tick label text.
    locations : NumericalArray
        NumericalArray containing tick positions on the axis.

    Examples
    --------
    >>> # This is an abstract class - use concrete implementations
    >>> # NumericTicks or CategoricalTicks instead
    """

    def __init__(
        self, labels: Sequence[str], locations: Sequence[Union[float, int]]
    ):
        """
        Initialize BaseTicks with labels and locations.

        Parameters
        ----------
        labels : Sequence[str]
            The tick labels as strings.
        locations : Sequence[Union[float, int]]
            The tick positions on the axis.

        Raises
        ------
        ValueError
            If labels and locations have different lengths or contain invalid values.
        """
        self._labels = CategoricalArray(labels)
        self._locations = NumericalArray(locations)
        self.validate()

    @property
    def labels(self) -> CategoricalArray:
        """
        Get the tick labels.

        Returns
        -------
        CategoricalArray
            The tick labels as a CategoricalArray.
        """
        return self._labels

    @property
    def locations(self) -> NumericalArray:
        """
        Get the tick locations.

        Returns
        -------
        NumericalArray
            The tick locations as a NumericalArray.
        """
        return self._locations

    def validate(self) -> None:
        """
        Validate that labels and locations arrays have matching lengths and valid values.

        Raises
        ------
        ValueError
            If labels and locations have different lengths.
        ValueError
            If any label is empty or any location is infinite/NaN.
        """
        if len(self._labels) != len(self._locations):
            raise ValueError(
                f"Labels and locations must have the same length. "
                f"Got {len(self._labels)} labels and {len(self._locations)} locations."
            )

        # Validate labels
        for i, label in enumerate(self._labels.values):
            if not isinstance(label, str):
                raise ValueError(
                    f"Label at index {i} must be a string, got {type(label)}"
                )
            if label.strip() == "":
                raise ValueError(f"Label at index {i} cannot be empty")

        # Validate locations
        for i, location in enumerate(self._locations.values):
            if not isinstance(location, (int, float)):
                raise ValueError(
                    f"Location at index {i} must be numerical, got {type(location)}"
                )
            if math.isnan(location) or math.isinf(location):
                raise ValueError(
                    f"Location at index {i} must be finite, got {location}"
                )

    def append(
        self, labels: Sequence[str], locations: Sequence[Union[float, int]]
    ) -> bool:
        """
        Append multiple ticks with validation.

        Parameters
        ----------
        labels : Sequence[str]
            The new tick labels to append.
        locations : Sequence[Union[float, int]]
            The new tick locations to append.

        Returns
        -------
        bool
            True if ticks were appended successfully, False otherwise.

        Raises
        ------
        ValueError
            If labels and locations have different lengths or contain invalid values.
        """
        try:
            # Validate new labels
            for i, label in enumerate(labels):
                if not isinstance(label, str):
                    raise ValueError(
                        f"New label at index {i} must be a string, got {type(label)}"
                    )
                if label.strip() == "":
                    raise ValueError(f"New label at index {i} cannot be empty")

            # Validate new locations
            for i, location in enumerate(locations):
                if not isinstance(location, (int, float)):
                    raise ValueError(
                        f"New location at index {i} must be numerical, got {type(location)}"
                    )
                if math.isnan(location) or math.isinf(location):
                    raise ValueError(
                        f"New location at index {i} must be finite, got {location}"
                    )

            if len(labels) != len(locations):
                raise ValueError(
                    f"New labels and locations must have the same length. "
                    f"Got {len(labels)} labels and {len(locations)} locations."
                )

            # Add the new ticks
            self._labels.append(labels)
            self._locations.append(locations)

            return True

        except (ValueError, IndexError):
            # Log the error or handle it as needed
            # For now, we'll just return False to indicate failure
            return False

    def remove(self, indices: Sequence[int]) -> bool:
        """
        Remove ticks at specified indices.

        Parameters
        ----------
        indices : Sequence[int]
            The indices of ticks to remove.

        Returns
        -------
        bool
            True if ticks were removed successfully, False otherwise.

        Raises
        ------
        IndexError
            If any index is out of bounds.
        ValueError
            If indices are not valid integers.
        """
        try:
            # Validate indices
            for index in indices:
                if not isinstance(index, int):
                    raise ValueError(
                        f"Index must be an integer, got {type(index)}"
                    )
                if index < 0 or index >= len(self._labels):
                    raise IndexError(
                        f"Index {index} out of bounds for array of length {len(self._labels)}"
                    )

            # Remove ticks from both arrays
            self._labels.remove(indices)
            self._locations.remove(indices)

            return True

        except (ValueError, IndexError):
            # Log the error or handle it as needed
            # For now, we'll just return False to indicate failure
            return False

    def __len__(self) -> int:
        """
        Get the number of ticks.

        Returns
        -------
        int
            The number of ticks.
        """
        return len(self._labels)

    def __repr__(self) -> str:
        """
        Get a string representation of the ticks.

        Returns
        -------
        str
            A string representation showing the number of ticks and first few values.
        """
        if len(self._labels) == 0:
            return f"{self.__class__.__name__}(empty)"

        labels_preview = self._labels.values[:3]
        locations_preview = self._locations.values[:3]

        if len(self._labels) <= 3:
            return (
                f"{self.__class__.__name__}(labels={labels_preview}, "
                f"locations={locations_preview})"
            )
        return (
            f"{self.__class__.__name__}(labels={labels_preview}..., "
            f"locations={locations_preview}...)"
        )
