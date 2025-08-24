# pylint: disable=C0116, W0212
"""Tests for the NumericTicks"""

import pytest
from numpy.testing import assert_almost_equal

from paxplot.plot_component_managers.numeric_ticks import NumericTicks
from paxplot.data_managers.numeric_normalized_array import NumericNormalizedArray


def test_initialization():
    axis_data = NumericNormalizedArray(values=[10, 20, 30])
    ticks = [10, 20, 30]
    manager = NumericTicks(axis_data=axis_data, tick_values=ticks)
    assert manager.get_raw_values() == ticks

    normalized = manager.get_normalized_values()
    assert all(-1.0 <= v <= 1.0 for v in normalized)
    assert abs(normalized[0] + 1) < 1e-6
    assert abs(normalized[-1] - 1) < 1e-6


def test_set_ticks():
    axis_data = NumericNormalizedArray(values=[-10.0, 10.0])
    manager = NumericTicks(axis_data=axis_data, tick_values=[1, 2])

    assert manager.get_raw_values() == [1, 2]
    assert_almost_equal(manager.get_normalized_values(), [0.1, 0.2])

    manager.set_ticks([3, 4])

    assert manager.get_raw_values() == [3, 4]
    assert_almost_equal(manager.get_normalized_values(), [0.3, 0.4])


def test_add_tick():
    axis_data = NumericNormalizedArray(values=[1, 2, 3, 4])
    manager = NumericTicks(axis_data=axis_data, tick_values=[1, 2, 3])
    manager.add_tick(4)
    assert manager.get_raw_values() == [1, 2, 3, 4]

    normalized = manager.get_normalized_values()
    assert all(-1.0 <= v <= 1.0 for v in normalized)
    assert abs(normalized[-1] - 1) < 1e-6


def test_single_value_behavior():
    axis_data = NumericNormalizedArray(values=[42])
    manager = NumericTicks(axis_data=axis_data, tick_values=[42])
    assert manager.get_raw_values() == [42]

    normalized = manager.get_normalized_values()
    assert len(normalized) == 1
    assert -1.0 <= normalized[0] <= 1.0


def test_add_after_set():
    axis_data = NumericNormalizedArray(values=[-10.0, 10.0])
    manager = NumericTicks(axis_data=axis_data, tick_values=[1, 2])

    # Replace ticks
    manager.set_ticks([3, 4])
    assert manager.get_raw_values() == [3, 4]
    assert_almost_equal(manager.get_normalized_values(), [0.3, 0.4])

    # Add a tick and verify updated state
    manager.add_tick(5)
    assert manager.get_raw_values() == [3, 4, 5]
    assert_almost_equal(manager.get_normalized_values(), [0.3, 0.4, 0.5])


def test_generate_ticks_basic():
    axis_data = NumericNormalizedArray(values=[0, 100])
    manager = NumericTicks(axis_data=axis_data)
    manager.generate_ticks(0, 100, max_ticks=5)
    ticks = manager.get_raw_values()
    assert ticks[0] >= 0
    assert ticks[-1] <= 100
    assert len(ticks) <= 6

    normalized = manager.get_normalized_values()
    assert len(normalized) == len(ticks)
    assert all(-1.0 <= v <= 1.0 for v in normalized)


def test_generate_ticks_integer_only():
    axis_data = NumericNormalizedArray(values=[3, 17])
    manager = NumericTicks(axis_data=axis_data)
    manager.generate_ticks(3, 17, max_ticks=4, integer=True)
    ticks = manager.get_raw_values()
    assert all(isinstance(t, int) or t.is_integer() for t in ticks)
    assert ticks[0] >= 3
    assert ticks[-1] <= 17
    assert len(ticks) <= 5


def test_generate_ticks_min_greater_than_max():
    axis_data = NumericNormalizedArray(values=[10, 5])
    manager = NumericTicks(axis_data=axis_data)
    with pytest.raises(ValueError):
        manager.generate_ticks(10, 5)


def test_generate_ticks_single_tick():
    axis_data = NumericNormalizedArray(values=[0, 0.0001])
    manager = NumericTicks(axis_data=axis_data)
    manager.generate_ticks(0, 0.0001, max_ticks=3)
    ticks = manager.get_raw_values()
    assert len(ticks) >= 1
    normalized = manager.get_normalized_values()
    assert len(normalized) == len(ticks)


def test_generate_ticks_with_existing_ticks_replacement():
    axis_data = NumericNormalizedArray(values=[10, 20, 30])
    manager = NumericTicks(axis_data=axis_data, tick_values=[10, 20, 30])
    manager.generate_ticks(50, 100, max_ticks=4)
    ticks = manager.get_raw_values()
    assert all(50 <= t <= 100 for t in ticks)
    assert len(ticks) <= 5


def test_axis_data_observer_triggers():
    axis_data = NumericNormalizedArray(values=[1, 2, 3, 4, 5])
    tick_manager = NumericTicks(tick_values=[1, 3, 5], axis_data=axis_data)

    assert axis_data.effective_min_val == 1
    assert axis_data.effective_max_val == 5
    assert tick_manager._ticks.effective_min_val == 1
    assert tick_manager._ticks.effective_max_val == 5

    axis_data.append_array([10])
    assert axis_data.effective_min_val == 1
    assert axis_data.effective_max_val == 10
    assert tick_manager._ticks.effective_min_val == 1
    assert tick_manager._ticks.effective_max_val == 10
