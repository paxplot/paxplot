"""Tick manager for PaxPlot.

This module defines the TickManager class that provides a unified interface
for managing collections of tick objects (NumericTicks and CategoricalTicks).
"""

from enum import Enum
from typing import List, Sequence, Union

from .ticks.categorical_ticks import CategoricalTicks
from .ticks.numeric_ticks import NumericTicks


class TickType(str, Enum):
    """Represents the type of a tick collection in the manager.

    Parameters
    ----------
    NUMERIC : str
        Collection contains NumericTicks.
    CATEGORICAL : str
        Collection contains CategoricalTicks.
    """

    NUMERIC = "numeric"
    CATEGORICAL = "categorical"


class TickManager:
    """
    A manager for collections of tick objects.

    This class provides a unified interface for managing multiple tick collections,
    where each collection can be either NumericTicks or CategoricalTicks.
    The manager ensures consistency across tick collections and provides
    convenient access methods.

    Parameters
    ----------
    tick_types : Sequence[TickType]
        The types of tick collections to create and manage.

    Attributes
    ----------
    tick_collections : List[Union[NumericTicks, CategoricalTicks]]
        The stored tick collections.
    num_collections : int
        The number of tick collections managed.

    Examples
    --------
    >>> # Initialize with tick types
    >>> manager = TickManager([
    ...     TickType.NUMERIC,
    ...     TickType.CATEGORICAL
    ... ])
    >>> print(manager.num_collections)  # 2
    >>>
    >>> # Get and configure tick collections
    >>> numeric_ticks = manager.get_numeric_ticks(0)
    >>> numeric_ticks.set_ticks_from_range(0, 100)
    >>>
    >>> categorical_ticks = manager.get_categorical_ticks(1)
    >>> categorical_ticks.set_ticks_from_categories(['A', 'B', 'C'])
    >>>
    >>> print(manager.get_tick_collection(0).labels.values)  # ['0.0', '25.0', ...]
    """

    def __init__(self, tick_types: Sequence[TickType]):
        """
        Initialize TickManager with tick collections based on provided types.

        Parameters
        ----------
        tick_types : Sequence[TickType]
            The types of tick collections to create and manage.

        Raises
        ------
        ValueError
            If tick_types is empty or contains invalid values.
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

        # Create tick collections based on types
        self._tick_collections: List[Union[NumericTicks, CategoricalTicks]] = (
            []
        )
        for tick_type in tick_types:
            if tick_type == TickType.NUMERIC:
                self._tick_collections.append(NumericTicks())
            elif tick_type == TickType.CATEGORICAL:
                self._tick_collections.append(CategoricalTicks())
            else:
                raise ValueError(f"Unknown tick type: {tick_type}")

    def remove_tick_collection(self, index: int) -> None:
        """Remove a tick collection at the specified index.

        Parameters
        ----------
        index : int
            The index of the collection to remove.

        Raises
        ------
        IndexError
            If the index is out of bounds.
        """
        if index < 0 or index >= len(self._tick_collections):
            raise IndexError(
                f"Index {index} out of bounds for manager with "
                f"{len(self._tick_collections)} collections"
            )

        del self._tick_collections[index]

    def get_tick_collection(
        self, index: int
    ) -> Union[NumericTicks, CategoricalTicks]:
        """Get a tick collection at the specified index.

        Parameters
        ----------
        index : int
            The index of the collection to get.

        Returns
        -------
        Union[NumericTicks, CategoricalTicks]
            The tick collection at the specified index.

        Raises
        ------
        IndexError
            If the index is out of bounds.
        """
        if index < 0 or index >= len(self._tick_collections):
            raise IndexError(
                f"Index {index} out of bounds for manager with "
                f"{len(self._tick_collections)} collections"
            )

        return self._tick_collections[index]

    def get_numeric_ticks(self, index: int) -> NumericTicks:
        """Get a NumericTicks collection at the specified index.

        Parameters
        ----------
        index : int
            The index of the collection to get.

        Returns
        -------
        NumericTicks
            The NumericTicks collection at the specified index.

        Raises
        ------
        IndexError
            If the index is out of bounds.
        TypeError
            If the collection at the specified index is not NumericTicks.
        """
        if self.get_tick_type(index) != TickType.NUMERIC:
            collection_type = type(self._tick_collections[index]).__name__
            raise TypeError(
                f"Collection {index} is not numeric, it is {collection_type}"
            )
        collection = self.get_tick_collection(index)
        assert isinstance(collection, NumericTicks)
        return collection

    def get_categorical_ticks(self, index: int) -> CategoricalTicks:
        """Get a CategoricalTicks collection at the specified index.

        Parameters
        ----------
        index : int
            The index of the collection to get.

        Returns
        -------
        CategoricalTicks
            The CategoricalTicks collection at the specified index.

        Raises
        ------
        IndexError
            If the index is out of bounds.
        TypeError
            If the collection at the specified index is not CategoricalTicks.
        """
        if self.get_tick_type(index) != TickType.CATEGORICAL:
            collection_type = type(self._tick_collections[index]).__name__
            raise TypeError(
                f"Collection {index} is not categorical, it is {collection_type}"
            )
        collection = self.get_tick_collection(index)
        assert isinstance(collection, CategoricalTicks)
        return collection

    def get_tick_type(self, index: int) -> TickType:
        """Get the type of a tick collection at the specified index.

        Parameters
        ----------
        index : int
            The index of the collection to get.

        Returns
        -------
        TickType
            The type of the tick collection.

        Raises
        ------
        IndexError
            If the index is out of bounds.
        """
        collection = self.get_tick_collection(index)

        if isinstance(collection, NumericTicks):
            return TickType.NUMERIC
        if isinstance(collection, CategoricalTicks):
            return TickType.CATEGORICAL
        raise TypeError(f"Unknown collection type: {type(collection)}")

    def __len__(self) -> int:
        """Get the number of tick collections.

        Returns
        -------
        int
            The number of tick collections.
        """
        return len(self._tick_collections)

    def __getitem__(self, index: int) -> Union[NumericTicks, CategoricalTicks]:
        """Get a tick collection at the specified index.

        Parameters
        ----------
        index : int
            The index of the collection to get.

        Returns
        -------
        Union[NumericTicks, CategoricalTicks]
            The tick collection at the specified index.
        """
        return self.get_tick_collection(index)

    def __repr__(self) -> str:
        """Get a string representation of the tick manager.

        Returns
        -------
        str
            A string representation showing the number of collections and their types.
        """
        if len(self._tick_collections) == 0:
            return "TickManager(empty)"

        type_info = []
        for i, collection in enumerate(self._tick_collections):
            if isinstance(collection, NumericTicks):
                type_info.append(f"{i}:NUMERIC")
            elif isinstance(collection, CategoricalTicks):
                type_info.append(f"{i}:CATEGORICAL")
            else:
                type_info.append(f"{i}:UNKNOWN")

        return f"TickManager({len(self._tick_collections)} collections: {', '.join(type_info)})"

    @property
    def tick_collections(self) -> List[Union[NumericTicks, CategoricalTicks]]:
        """Get the tick collections as a list.

        Returns
        -------
        List[Union[NumericTicks, CategoricalTicks]]
            A copy of the tick collections list.
        """
        return self._tick_collections.copy()

    @property
    def num_collections(self) -> int:
        """Get the number of tick collections.

        Returns
        -------
        int
            The number of tick collections.
        """
        return len(self._tick_collections)
