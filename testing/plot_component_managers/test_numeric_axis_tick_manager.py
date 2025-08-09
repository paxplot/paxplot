# pylint: disable=C0116, W0212
"""Tests for the NumericAxisTickManager"""


from paxplot.plot_component_managers.numeric_axis_tick_manager import NumericAxisTickManager


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
