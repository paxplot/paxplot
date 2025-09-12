"""Numeric ticks implementation for PaxPlot.

This module defines the NumericTicks class for managing numerical axis ticks
using matplotlib's MaxNLocator for optimal tick generation.
"""

from typing import Union

from matplotlib.ticker import MaxNLocator

from .base_ticks import BaseTicks


class NumericTicks(BaseTicks):
    """
    Concrete implementation for numerical axis ticks using matplotlib's MaxNLocator.

    Generates optimal numerical ticks between min and max values using
    matplotlib's proven tick generation algorithms for optimal spacing and readability.

    Attributes
    ----------
    labels : CategoricalArray
        CategoricalArray containing tick label text.
    locations : NumericalArray
        NumericalArray containing tick positions on the axis.

    Examples
    --------
    >>> ticks = NumericTicks()
    >>> ticks.set_ticks_from_range(0, 100)
    >>> print(ticks.labels.values)  # ['0.0', '25.0', '50.0', '75.0', '100.0']
    >>> print(ticks.locations.values)  # [0.0, 25.0, 50.0, 75.0, 100.0]
    """

    def __init__(self):
        """
        Initialize NumericTicks with empty arrays.
        """
        # Initialize with empty arrays
        super().__init__([], [])

    def set_ticks_from_range(
        self,
        min_value: Union[float, int],
        max_value: Union[float, int],
        max_ticks: int = 5,
        precision: int = 2,
    ) -> None:
        """
        Set ticks from min/max using MaxNLocator.

        Parameters
        ----------
        min_value : Union[float, int]
            The minimum value for the range.
        max_value : Union[float, int]
            The maximum value for the range.
        max_ticks : int, default=5
            Maximum number of ticks to generate.
        precision : int, default=2
            Number of decimal places for tick labels.

        Raises
        ------
        ValueError
            If min_value >= max_value or if max_ticks is not positive.
        """
        if min_value >= max_value:
            raise ValueError(
                f"min_value ({min_value}) must be less than max_value ({max_value})"
            )

        if max_ticks <= 0:
            raise ValueError(f"max_ticks must be positive, got {max_ticks}")

        # Create MaxNLocator instance
        locator = MaxNLocator(nbins=max_ticks)

        # Get optimal tick positions
        tick_positions = locator.tick_values(min_value, max_value)

        # Convert to lists and ensure we have the right number of ticks
        tick_positions_list = list(tick_positions)

        # Filter ticks to be within the specified range
        filtered_positions = [
            pos for pos in tick_positions_list if min_value <= pos <= max_value
        ]

        # If we have too many ticks, take the first max_ticks
        if len(filtered_positions) > max_ticks:
            filtered_positions = filtered_positions[:max_ticks]

        # Generate labels with specified precision
        tick_labels = [f"{pos:.{precision}f}" for pos in filtered_positions]

        # Use the base class set_ticks method
        self.set_ticks(tick_labels, filtered_positions)

    def __repr__(self) -> str:
        """
        Get a string representation of the numeric ticks.

        Returns
        -------
        str
            A string representation showing the number of ticks.
        """
        if len(self._labels) == 0:
            return "NumericTicks(empty)"

        labels_preview = self._labels.values[:3]
        locations_preview = self._locations.values[:3]

        if len(self._labels) <= 3:
            return f"NumericTicks(labels={labels_preview}, locations={locations_preview})"
        return f"NumericTicks(labels={labels_preview}..., locations={locations_preview}...)"
