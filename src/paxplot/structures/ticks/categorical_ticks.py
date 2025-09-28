"""Categorical ticks implementation for PaxPlot.

This module defines the CategoricalTicks class for managing categorical axis ticks
using category indices for positioning.
"""

from typing import Sequence, Union

from .base_ticks import BaseTicks


class CategoricalTicks(BaseTicks):
    """
    Concrete implementation for categorical axis ticks using category indices.

    Generates categorical ticks from a sequence of category strings and maps
    category labels to their corresponding indices for positioning.

    Attributes
    ----------
    labels : CategoricalArray
        CategoricalArray containing tick label text.
    locations : NumericalArray
        NumericalArray containing tick positions on the axis.

    Examples
    --------
    >>> ticks = CategoricalTicks()
    >>> ticks.set_ticks_from_categories(['Red', 'Blue', 'Green', 'Yellow'])
    >>> print(ticks.labels.get_values())     # ['Red', 'Blue', 'Green', 'Yellow']
    >>> print(ticks.locations.get_values())  # [0, 1, 2, 3]
    """

    def __init__(
        self, 
        labels: Union[Sequence[str], None] = None, 
        locations: Union[Sequence[Union[float, int]], None] = None
    ):
        """
        Initialize CategoricalTicks with optional labels and locations.

        Parameters
        ----------
        labels : Sequence[str], optional
            The tick labels as strings. If None, creates empty labels array.
        locations : Sequence[Union[float, int]], optional
            The tick positions on the axis. If None, creates empty locations array.
        """
        # Initialize with optional labels and locations
        super().__init__(labels, locations)

    def set_ticks_from_categories(self, categories: Sequence[str]) -> None:
        """
        Set ticks from category strings.

        Parameters
        ----------
        categories : Sequence[str]
            The category strings to use for tick generation.

        Raises
        ------
        ValueError
            If categories is empty or contains invalid values.
        """
        if not categories:
            raise ValueError("Categories cannot be empty")

        # Validate categories
        for i, category in enumerate(categories):
            if not isinstance(category, str):
                raise ValueError(
                    f"Category at index {i} must be a string, got {type(category)}"
                )
            if category.strip() == "":
                raise ValueError(f"Category at index {i} cannot be empty")

        # Check for duplicate categories - throw error if found
        seen_categories = set()
        for i, category in enumerate(categories):
            if category in seen_categories:
                raise ValueError(
                    f"Duplicate category '{category}' found at index {i}. "
                    f"Categories must be unique."
                )
            seen_categories.add(category)

        # Use the categories exactly as supplied with sequential indices
        sequential_indices = list(range(len(categories)))

        # Use the base class set_ticks method
        self.set_ticks(categories, sequential_indices)

    def __repr__(self) -> str:
        """
        Get a string representation of the categorical ticks.

        Returns
        -------
        str
            A string representation showing the number of ticks and first few categories.
        """
        if len(self._labels) == 0:
            return "CategoricalTicks(empty)"

        labels_preview = self._labels.get_values()[:3]
        locations_preview = self._locations.get_values()[:3]

        if len(self._labels) <= 3:
            return f"CategoricalTicks(labels={labels_preview}, locations={locations_preview})"
        return f"CategoricalTicks(labels={labels_preview}..., locations={locations_preview}...)"
