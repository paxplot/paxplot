import pytest
from paxplot.data_managers.categorical_normalized_array import CategoricalNormalizedArray
from paxplot.plot_component_managers.categorical_axis_tick_manager import CategoricalAxisTickManager


def test_from_categories_creates_with_all_categories():
    axis_data = CategoricalNormalizedArray(array=["X", "Y", "Z"])
    manager = CategoricalAxisTickManager.from_categories(
        categories=["X", "Y", "Z"],
        axis_data=axis_data
    )

    assert manager.get_raw_values() == ["X", "Y", "Z"]
    assert manager.get_normalized_values() == [-1.0, 0.0, 1.0]
    assert manager._axis_data.array == ["X", "Y", "Z"]
    assert manager._axis_data.array_normalized.tolist() == [-1.0, 0.0, 1.0]

def test_set_ticks_replaces_existing():
    axis_data = CategoricalNormalizedArray(array=["X", "Y", "Z"])
    manager = CategoricalAxisTickManager.from_categories(
        categories=["X", "Y", "Z"],
        axis_data=axis_data
    )

    manager.set_ticks(["FOO", "BAR", "BAT"])

    assert manager.get_raw_values() == ["FOO", "BAR", "BAT"]
    assert manager.get_normalized_values() == [-1.0, 0.0, 1.0]
    assert manager._axis_data.array == ["X", "Y", "Z"]
    assert manager._axis_data.array_normalized.tolist() == [-1.0, 0.0, 1.0]