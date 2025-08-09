from typing import Sequence, Union, List
from paxplot.data_managers.numeric_normalized_array import NumericNormalizedArray

class NumericAxisTickManager:
    """
    Manages numeric axis ticks normalized to [-1, 1],
    storing ticks only inside the internal NumericNormalizedArray.
    """

    def __init__(self, tick_values: Sequence[Union[int, float]]):
        """
        Initialize with a sequence of numeric tick values.

        Parameters
        ----------
        tick_values : Sequence[Union[int, float]]
            Initial tick values.
        """
        self._normalized_array = NumericNormalizedArray(array=tick_values)

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
