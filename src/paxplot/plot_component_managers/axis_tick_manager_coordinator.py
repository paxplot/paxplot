"""
Axis Tick Manager Coordinator for PaxPlot

This module provides the AxisTickManagerCoordinator class which manages multiple
axis tick managers for a NormalizedMatrix, automatically creating and updating tick
managers when data changes.

The coordinator maintains a collection of NumericAxisTickManager and
CategoricalAxisTickManager instances, one for each column in the NormalizedMatrix.
It automatically generates appropriate ticks when data is updated and provides
a unified interface for tick management across all axes.
"""

from typing import List, Optional, Union
from paxplot.plot_component_managers.numeric_axis_tick_manager import (
    NumericAxisTickManager,
)
from paxplot.plot_component_managers.categorical_axis_tick_manager import (
    CategoricalAxisTickManager,
)
from paxplot.data_managers.normalized_matrix import (
    NormalizedMatrix,
    ColumnType,
)


class AxisTickManagerCoordinator:
    """
    Coordinates multiple axis tick managers for a NormalizedMatrix.

    This class manages the relationship between NormalizedMatrix columns and their
    corresponding axis tick managers, providing a unified interface for
    tick management across all axes. It automatically creates and updates
    tick managers when data changes.
    """

    def __init__(self, matrix: NormalizedMatrix):
        """
        Initialize the coordinator with a NormalizedMatrix.

        Parameters
        ----------
        matrix : NormalizedMatrix
            The normalized matrix to coordinate tick managers for.
        """
        self._matrix = matrix
        self._tick_managers: List[
            Optional[Union[NumericAxisTickManager, CategoricalAxisTickManager]]
        ] = []
        self._initialize_tick_managers()

    def _initialize_tick_managers(self) -> None:
        """Initialize tick managers for each column based on data type."""
        self._tick_managers.clear()

        for col_idx in range(self._matrix.num_columns):
            tick_manager = self._create_tick_manager_for_column(col_idx)
            self._tick_managers.append(tick_manager)

    def _create_tick_manager_for_column(
        self, column_index: int
    ) -> Optional[Union[NumericAxisTickManager, CategoricalAxisTickManager]]:
        """
        Create an appropriate tick manager for a specific column.

        Parameters
        ----------
        column_index : int
            Index of the column to create a tick manager for.

        Returns
        -------
        Union[NumericAxisTickManager, CategoricalAxisTickManager] or None
            The created tick manager, or None if column is empty.
        """
        column_type = self._matrix.get_column_type(column_index)

        if column_type == ColumnType.NUMERIC:
            numeric_column = self._matrix.get_numeric_column(column_index)
            
            # Generate ticks if data exists and min != max
            if (
                numeric_column.effective_min_val is not None
                and numeric_column.effective_min_val
                != numeric_column.effective_max_val
            ):
                tick_manager = NumericAxisTickManager.from_range(
                    min_val=numeric_column.effective_min_val,
                    max_val=numeric_column.effective_max_val,
                    axis_data=numeric_column,
                )
                return tick_manager
            else:
                # Create tick manager without ticks for single values
                tick_manager = NumericAxisTickManager(axis_data=numeric_column)
                return tick_manager

        elif column_type == ColumnType.CATEGORICAL:
            categorical_column = self._matrix.get_categorical_column(
                column_index
            )

            # Only create tick manager if categories exist
            if categorical_column.categories:
                tick_manager = CategoricalAxisTickManager.from_categories(
                    categories=categorical_column.categories,
                    axis_data=categorical_column,
                )
                return tick_manager
            else:
                # No categories, return None
                return None

        # Unknown column type, return None
        return None

    def get_tick_manager(
        self, column_index: int
    ) -> Optional[Union[NumericAxisTickManager, CategoricalAxisTickManager]]:
        """
        Get the tick manager for a specific column.

        Parameters
        ----------
        column_index : int
            Index of the column.

        Returns
        -------
        Union[NumericAxisTickManager, CategoricalAxisTickManager] or None
            The tick manager for the column, or None if not available.
        """
        if 0 <= column_index < len(self._tick_managers):
            return self._tick_managers[column_index]
        return None

    def get_numeric_tick_manager(
        self, column_index: int
    ) -> NumericAxisTickManager:
        """
        Get a tick manager as a NumericAxisTickManager. Raises TypeError if not numeric.

        Parameters
        ----------
        column_index : int
            Index of the column.

        Returns
        -------
        NumericAxisTickManager
            The numeric tick manager for the column.

        Raises
        ------
        TypeError
            If the column is not numeric.
        """
        tick_manager = self.get_tick_manager(column_index)
        if not isinstance(tick_manager, NumericAxisTickManager):
            raise TypeError(f"Column {column_index} is not numeric.")
        return tick_manager

    def get_categorical_tick_manager(
        self, column_index: int
    ) -> CategoricalAxisTickManager:
        """
        Get a tick manager as a CategoricalAxisTickManager. Raises TypeError if not categorical.

        Parameters
        ----------
        column_index : int
            Index of the column.

        Returns
        -------
        CategoricalAxisTickManager
            The categorical tick manager for the column.

        Raises
        ------
        TypeError
            If the column is not categorical.
        """
        tick_manager = self.get_tick_manager(column_index)
        if not isinstance(tick_manager, CategoricalAxisTickManager):
            raise TypeError(f"Column {column_index} is not categorical.")
        return tick_manager
