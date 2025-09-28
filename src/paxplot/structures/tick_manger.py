"""Tick manager for PaxPlot.

This module defines the TickManager class that provides a unified interface
for managing ticks (NumericTicks and CategoricalTicks).
"""

from enum import Enum
from typing import List, Sequence, Union

from .ticks.categorical_ticks import CategoricalTicks
from .ticks.numeric_ticks import NumericTicks


class TickType(str, Enum):
    """Represents the type of a tick in the manager.

    Parameters
    ----------
    NUMERIC : str
        Tick contains NumericTicks.
    CATEGORICAL : str
        Tick contains CategoricalTicks.
    """

    NUMERIC = "numeric"
    CATEGORICAL = "categorical"


class TickManager:
    """
    A manager for tick objects.

    This class provides a unified interface for managing multiple ticks,
    where each tick can be either NumericTicks or CategoricalTicks.
    The manager ensures consistency across ticks and provides
    convenient access methods.

    Parameters
    ----------
    ticks : Sequence[Union[NumericTicks, CategoricalTicks]], optional
        Pre-existing ticks to use. If None, creates empty manager.

    Attributes
    ----------
    ticks : List[Union[NumericTicks, CategoricalTicks]]
        The stored ticks.
    num_ticks : int
        The number of ticks managed.

    Examples
    --------
    >>> # Initialize with existing ticks
    >>> numeric_ticks = NumericTicks()
    >>> categorical_ticks = CategoricalTicks()
    >>> manager = TickManager([numeric_ticks, categorical_ticks])
    >>> print(manager.num_ticks)  # 2
    >>>
    >>> # Or initialize empty and set from types
    >>> manager = TickManager()  # Empty
    >>> manager.set_ticks_from_types([TickType.NUMERIC, TickType.CATEGORICAL])
    >>>
    >>> # Configure the ticks
    >>> numeric_ticks = manager.get_numeric_ticks(0)
    >>> numeric_ticks.set_ticks_from_range(0, 100)
    >>>
    >>> categorical_ticks = manager.get_categorical_ticks(1)
    >>> categorical_ticks.set_ticks_from_categories(['A', 'B', 'C'])
    >>>
    >>> print(manager.get_ticks(0).labels.get_values())  # ['0.0', '25.0', ...]
    """

    def __init__(
        self, 
        ticks: Union[Sequence[Union[NumericTicks, CategoricalTicks]], None] = None
    ):
        """
        Initialize TickManager with ticks.

        Parameters
        ----------
        ticks : Sequence[Union[NumericTicks, CategoricalTicks]], optional
            Pre-existing ticks to use. If None, creates empty manager.

        Raises
        ------
        TypeError
            If any tick is not a valid tick type.
        """
        # Initialize with empty ticks first, then use set_ticks method
        self._ticks: List[Union[NumericTicks, CategoricalTicks]] = []
        self.set_ticks(ticks)


    def get_ticks(
        self, index: int
    ) -> Union[NumericTicks, CategoricalTicks]:
        """Get a tick at the specified index.

        Parameters
        ----------
        index : int
            The index of the tick to get.

        Returns
        -------
        Union[NumericTicks, CategoricalTicks]
            The tick at the specified index.

        Raises
        ------
        IndexError
            If the index is out of bounds.
        """
        if index < 0 or index >= len(self._ticks):
            raise IndexError(
                f"Index {index} out of bounds for manager with "
                f"{len(self._ticks)} ticks"
            )

        return self._ticks[index]

    def get_numeric_ticks(self, index: int) -> NumericTicks:
        """Get a NumericTicks at the specified index.

        Parameters
        ----------
        index : int
            The index of the tick to get.

        Returns
        -------
        NumericTicks
            The NumericTicks at the specified index.

        Raises
        ------
        IndexError
            If the index is out of bounds.
        TypeError
            If the tick at the specified index is not NumericTicks.
        """
        if self.get_tick_type(index) != TickType.NUMERIC:
            tick_type = type(self._ticks[index]).__name__
            raise TypeError(
                f"Tick {index} is not numeric, it is {tick_type}"
            )
        tick = self.get_ticks(index)
        assert isinstance(tick, NumericTicks)
        return tick

    def get_categorical_ticks(self, index: int) -> CategoricalTicks:
        """Get a CategoricalTicks at the specified index.

        Parameters
        ----------
        index : int
            The index of the tick to get.

        Returns
        -------
        CategoricalTicks
            The CategoricalTicks at the specified index.

        Raises
        ------
        IndexError
            If the index is out of bounds.
        TypeError
            If the tick at the specified index is not CategoricalTicks.
        """
        if self.get_tick_type(index) != TickType.CATEGORICAL:
            tick_type = type(self._ticks[index]).__name__
            raise TypeError(
                f"Tick {index} is not categorical, it is {tick_type}"
            )
        tick = self.get_ticks(index)
        assert isinstance(tick, CategoricalTicks)
        return tick

    def get_tick_type(self, index: int) -> TickType:
        """Get the type of a tick at the specified index.

        Parameters
        ----------
        index : int
            The index of the tick to get.

        Returns
        -------
        TickType
            The type of the tick.

        Raises
        ------
        IndexError
            If the index is out of bounds.
        """
        tick = self.get_ticks(index)

        if isinstance(tick, NumericTicks):
            return TickType.NUMERIC
        if isinstance(tick, CategoricalTicks):
            return TickType.CATEGORICAL
        raise TypeError(f"Unknown tick type: {type(tick)}")

    def __len__(self) -> int:
        """Get the number of ticks.

        Returns
        -------
        int
            The number of ticks.
        """
        return len(self._ticks)

    def __getitem__(self, index: int) -> Union[NumericTicks, CategoricalTicks]:
        """Get a tick at the specified index.

        Parameters
        ----------
        index : int
            The index of the tick to get.

        Returns
        -------
        Union[NumericTicks, CategoricalTicks]
            The tick at the specified index.
        """
        return self.get_ticks(index)

    def __repr__(self) -> str:
        """Get a string representation of the tick manager.

        Returns
        -------
        str
            A string representation showing the number of ticks and their types.
        """
        if len(self._ticks) == 0:
            return "TickManager(empty)"

        type_info = []
        for i, tick in enumerate(self._ticks):
            if isinstance(tick, NumericTicks):
                type_info.append(f"{i}:NUMERIC")
            elif isinstance(tick, CategoricalTicks):
                type_info.append(f"{i}:CATEGORICAL")
            else:
                type_info.append(f"{i}:UNKNOWN")

        return f"TickManager({len(self._ticks)} ticks: {', '.join(type_info)})"

    def set_ticks(
        self, 
        ticks: Union[Sequence[Union[NumericTicks, CategoricalTicks]], None] = None
    ) -> None:
        """Set new ticks, replacing all existing ticks.

        Parameters
        ----------
        ticks : Sequence[Union[NumericTicks, CategoricalTicks]], optional
            The new ticks to set. If None, creates empty manager.

        Raises
        ------
        TypeError
            If any tick is not a valid tick type.
        """
        if ticks is None:
            ticks = []
        
        # Validate ticks
        for i, tick in enumerate(ticks):
            if not isinstance(tick, (NumericTicks, CategoricalTicks)):
                raise TypeError(
                    f"Tick at index {i} must be NumericTicks or CategoricalTicks, "
                    f"got {type(tick)}"
                )
        
        self._ticks = list(ticks)

    def set_ticks_from_types(
        self, 
        tick_types: Sequence[TickType]
    ) -> None:
        """Set new ticks from tick types, creating empty ticks.

        Parameters
        ----------
        tick_types : Sequence[TickType]
            The types of ticks to create.

        Raises
        ------
        ValueError
            If tick_types is empty.
        TypeError
            If any element in tick_types is not a TickType.
        """
        if not tick_types:
            raise ValueError("tick_types cannot be empty")
        
        # Validate tick types
        for i, tick_type in enumerate(tick_types):
            if not isinstance(tick_type, TickType):
                raise TypeError(
                    f"tick_type at index {i} must be a TickType, got {type(tick_type)}"
                )
        
        # Create empty ticks based on types
        new_ticks = []
        for tick_type in tick_types:
            if tick_type == TickType.NUMERIC:
                new_ticks.append(NumericTicks())
            elif tick_type == TickType.CATEGORICAL:
                new_ticks.append(CategoricalTicks())
            else:
                raise ValueError(f"Unknown tick type: {tick_type}")
        
        self._ticks = new_ticks

    def append_ticks(
        self, 
        tick: Union[NumericTicks, CategoricalTicks]
    ) -> None:
        """Append a new tick to the manager.

        Parameters
        ----------
        tick : Union[NumericTicks, CategoricalTicks]
            The tick to append.

        Raises
        ------
        TypeError
            If the tick is not a valid tick type.
        """
        if not isinstance(tick, (NumericTicks, CategoricalTicks)):
            raise TypeError(
                f"Tick must be NumericTicks or CategoricalTicks, "
                f"got {type(tick)}"
            )
        
        self._ticks.append(tick)

    def remove_ticks(self, indices: Sequence[int]) -> None:
        """Remove ticks at the specified indices.

        Parameters
        ----------
        indices : Sequence[int]
            The indices of ticks to remove.

        Raises
        ------
        IndexError
            If any index is out of bounds.
        ValueError
            If indices are not valid integers.
        """
        # Convert to list and sort in reverse order to avoid index shifting
        indices_list = sorted(indices, reverse=True)
        
        for index in indices_list:
            if not isinstance(index, int):
                raise ValueError(
                    f"Index must be an integer, got {type(index)}"
                )
            if index < 0 or index >= len(self._ticks):
                raise IndexError(
                    f"Index {index} out of bounds for manager with "
                    f"{len(self._ticks)} ticks"
                )
            del self._ticks[index]

    def clear_ticks(self) -> None:
        """Clear all ticks from the manager."""
        self.set_ticks([])

    def infer_tick_type(
        self, 
        tick: Union[NumericTicks, CategoricalTicks]
    ) -> TickType:
        """Infer the type of a tick.

        Parameters
        ----------
        tick : Union[NumericTicks, CategoricalTicks]
            The tick to infer the type of.

        Returns
        -------
        TickType
            The inferred tick type.
        """
        if isinstance(tick, NumericTicks):
            return TickType.NUMERIC
        elif isinstance(tick, CategoricalTicks):
            return TickType.CATEGORICAL
        else:
            raise TypeError(
                f"Tick must be NumericTicks or CategoricalTicks, "
                f"got {type(tick)}"
            )

    @property
    def ticks(self) -> List[Union[NumericTicks, CategoricalTicks]]:
        """Get the ticks as a list.

        Returns
        -------
        List[Union[NumericTicks, CategoricalTicks]]
            A copy of the ticks list.
        """
        return self._ticks.copy()

    @property
    def num_ticks(self) -> int:
        """Get the number of ticks.

        Returns
        -------
        int
            The number of ticks.
        """
        return len(self._ticks)
