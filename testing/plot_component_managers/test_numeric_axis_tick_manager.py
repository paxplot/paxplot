# pylint: disable=C0116, W0212
"""Tests for the NumericAxisTickManager"""

import pytest

from paxplot.plot_component_managers.numeric_axis_tick_manager import NumericAxisTickManager
from paxplot.data_managers.numeric_normalized_array import NumericNormalizedArray


def test_initialization():
    ticks = [10, 20, 30]
    manager = NumericAxisTickManager(ticks)
    assert manager.get_raw_values() == ticks

    normalized = manager.get_normalized_values()
    # normalized values should be in [-1, 1]
    assert all(-1.0 <= v <= 1.0 for v in normalized)
    # The first should be -1, last should be 1 for linear scaling
    assert abs(normalized[0] + 1) < 1e-6
    assert abs(normalized[-1] - 1) < 1e-6


def test_set_ticks():
    manager = NumericAxisTickManager([1, 2])
    manager.set_ticks([5, 10, 15])
    assert manager.get_raw_values() == [5, 10, 15]

    normalized = manager.get_normalized_values()
    assert all(-1.0 <= v <= 1.0 for v in normalized)
    assert abs(normalized[0] + 1) < 1e-6
    assert abs(normalized[-1] - 1) < 1e-6


def test_add_tick():
    manager = NumericAxisTickManager([1, 2, 3])
    manager.add_tick(4)
    assert manager.get_raw_values() == [1, 2, 3, 4]

    normalized = manager.get_normalized_values()
    assert all(-1.0 <= v <= 1.0 for v in normalized)
    # Adding 4 extends max value, so last normalized should be 1
    assert abs(normalized[-1] - 1) < 1e-6


def test_single_value_behavior():
    # If only one tick value, normalized should be zero (or handled gracefully)
    manager = NumericAxisTickManager([42])
    assert manager.get_raw_values() == [42]

    normalized = manager.get_normalized_values()
    assert len(normalized) == 1
    # Normalization might be zero or some constant, depending on implementation
    assert -1.0 <= normalized[0] <= 1.0


def test_add_after_set():
    manager = NumericAxisTickManager([10, 20])
    manager.set_ticks([100, 200])
    manager.add_tick(300)
    assert manager.get_raw_values() == [100, 200, 300]

    normalized = manager.get_normalized_values()
    assert all(-1.0 <= v <= 1.0 for v in normalized)
    assert abs(normalized[0] + 1) < 1e-6
    assert abs(normalized[-1] - 1) < 1e-6

def test_generate_ticks_basic():
    manager = NumericAxisTickManager()
    manager.generate_ticks(0, 100, max_ticks=5)
    ticks = manager.get_raw_values()
    # Should generate ~5 ticks within the range
    assert ticks[0] >= 0
    assert ticks[-1] <= 100
    assert len(ticks) <= 6  # MaxNLocator can generate up to max_ticks + 1

    normalized = manager.get_normalized_values()
    assert len(normalized) == len(ticks)
    assert all(-1.0 <= v <= 1.0 for v in normalized)


def test_generate_ticks_integer_only():
    manager = NumericAxisTickManager()
    manager.generate_ticks(3, 17, max_ticks=4, integer=True)
    ticks = manager.get_raw_values()
    assert all(isinstance(t, int) or t.is_integer() for t in ticks)
    # Check the range and tick count
    assert ticks[0] >= 3
    assert ticks[-1] <= 17
    assert len(ticks) <= 5


def test_generate_ticks_min_greater_than_max():
    manager = NumericAxisTickManager()
    with pytest.raises(ValueError):
        manager.generate_ticks(10, 5)


def test_generate_ticks_single_tick():
    manager = NumericAxisTickManager()
    manager.generate_ticks(0, 0.0001, max_ticks=3)
    ticks = manager.get_raw_values()
    assert len(ticks) >= 1
    normalized = manager.get_normalized_values()
    assert len(normalized) == len(ticks)


def test_generate_ticks_with_existing_ticks_replacement():
    manager = NumericAxisTickManager([10, 20, 30])
    manager.generate_ticks(50, 100, max_ticks=4)
    ticks = manager.get_raw_values()
    assert all(50 <= t <= 100 for t in ticks)
    assert len(ticks) <= 5  # max_ticks + 1

def test_axis_data_observer_triggers():
    axis_data = NumericNormalizedArray(array=[1, 2, 3, 4, 5])
    tick_manager = NumericAxisTickManager([1, 3, 5], axis_data=axis_data)

    assert axis_data.normalizer.effective_min_val == 1
    assert axis_data.normalizer.effective_max_val == 5
    assert tick_manager._ticks.normalizer.effective_min_val == 1
    assert tick_manager._ticks.normalizer.effective_max_val == 5

    axis_data.append_array([10])
    assert axis_data.normalizer.effective_min_val == 1
    assert axis_data.normalizer.effective_max_val == 10
    assert tick_manager._ticks.normalizer.effective_min_val == 1
    assert tick_manager._ticks.normalizer.effective_max_val == 10
