"""
numeric_axis_tick_manager module.

Provides the NumericAxisTickManager class to manage numeric axis ticks with
normalization to the range [-1, 1]. Tick values are stored internally using the
NumericNormalizedArray class, ensuring consistent normalization and easy access
to both raw and normalized tick values.

The class supports:
- Initialization with a sequence of numeric tick values.
- Generating “nice” tick values for a given numeric range using matplotlib's
  tick locating logic.
- Adding individual tick values.
- Retrieving raw and normalized tick values.
- Convenient class method constructor for creating instances by specifying
  a numeric range and desired tick characteristics.

This module leverages matplotlib.ticker.MaxNLocator to compute aesthetically
pleasing and logically spaced tick values suitable for plotting numeric axes.

Dependencies:
- matplotlib
- numpy
- paxplot.data_managers.numeric_normalized_array.NumericNormalizedArray
"""

from typing import Optional, Sequence, Union, List
import matplotlib.ticker as mticker
from paxplot.data_managers.numeric_normalized_array import NumericNormalizedArray


class NumericAxisTickManager:
    """
    Manages numeric axis ticks normalized to [-1, 1],
    storing ticks only inside the internal NumericNormalizedArray.
    """

    def __init__(self, tick_values: Optional[Sequence[Union[int, float]]] = None):
        """
        Initialize with a sequence of numeric tick values.

        Parameters
        ----------
        tick_values : Sequence[Union[int, float]]
            Initial tick values.
        """
        if tick_values is None:
            tick_values = [0.0]
        self._normalized_array = NumericNormalizedArray(array=tick_values)

    @classmethod
    def _compute_ticks(
        cls,
        min_val: float,
        max_val: float,
        max_ticks: int = 10,
        integer: bool = False,
    ) -> List[float]:
        """
        Internal helper to compute nice ticks using matplotlib logic.

        Parameters
        ----------
        min_val : float
            Minimum axis value.
        max_val : float
            Maximum axis value.
        max_ticks : int
            Maximum number of ticks to generate.
        integer : bool
            Generate integer ticks only if True.

        Returns
        -------
        List[float]
            List of computed ticks within [min_val, max_val].
        """
        if min_val >= max_val:
            raise ValueError("min_val must be less than max_val")

        locator = mticker.MaxNLocator(nbins=max_ticks, integer=integer, prune=None)
        ticks = locator.tick_values(min_val, max_val)
        return [t for t in ticks if min_val <= t <= max_val]

    @classmethod
    def from_range(
        cls,
        min_val: float,
        max_val: float,
        max_ticks: int = 10,
        integer: bool = False,
    ) -> "NumericAxisTickManager":
        """
        Create an instance by generating "nice" ticks in [min_val, max_val]
        using matplotlib logic.

        Parameters
        ----------
        min_val : float
            Minimum value of the axis range.
        max_val : float
            Maximum value of the axis range.
        max_ticks : int, optional
            Maximum number of ticks to generate (default 10).
        integer : bool, optional
            Whether to generate integer ticks only (default False).

        Returns
        -------
        NumericAxisTickManager
            New instance initialized with generated ticks.
        """
        ticks = cls._compute_ticks(min_val, max_val, max_ticks=max_ticks, integer=integer)
        return cls(tick_values=ticks)

    def generate_ticks(
        self,
        min_val: float,
        max_val: float,
        max_ticks: int = 10,
        integer: bool = False,
    ) -> None:
        """
        Generate "nice" tick values within [min_val, max_val] using matplotlib logic,
        then set those ticks internally.

        Parameters
        ----------
        min_val : float
            Minimum axis value.
        max_val : float
            Maximum axis value.
        max_ticks : int, optional
            Maximum number of ticks to generate (default 10).
        integer : bool, optional
            If True, generate integer ticks only (default False).
        """
        ticks = self._compute_ticks(min_val, max_val, max_ticks=max_ticks, integer=integer)
        self.set_ticks(ticks)

    def set_ticks(self, tick_values: Sequence[Union[int, float]]) -> None:
        """
        Replace current ticks with a new sequence.

        Parameters
        ----------
        tick_values : Sequence[Union[int, float]]
            New tick values.
        """
        self._normalized_array = NumericNormalizedArray(array=tick_values)

    def add_tick(self, tick_value: Union[int, float]) -> None:
        """
        Add a single numeric tick value.

        Parameters
        ----------
        tick_value : Union[int, float]
            Tick value to add.
        """
        self._normalized_array.append_array([tick_value])

    def get_raw_values(self) -> List[Union[int, float]]:
        """
        Get the raw tick values stored in the normalized array.

        Returns
        -------
        List[Union[int, float]]
            Raw tick values.
        """
        return list(self._normalized_array.array)

    def get_normalized_values(self) -> List[float]:
        """
        Get normalized tick values in [-1, 1].

        Returns
        -------
        List[float]
            Normalized tick values.
        """
        return self._normalized_array.array_normalized.tolist()
