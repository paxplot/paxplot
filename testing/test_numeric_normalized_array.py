# pylint: disable=C0116, W0212
"""Tests for the NumericNormalizedArray"""

import numpy as np
import pytest
from pydantic import ValidationError
from paxplot.data_managers.numeric_normalized_array import NumericNormalizedArray

def test_numeric_normalized_array():
    raw = [1, 2, 3]
    obj = NumericNormalizedArray(array=raw)

    expected = np.array([-1.0, 0.0, 1.0])
    result = obj.array_normalized

    assert isinstance(result, np.ndarray)
    np.testing.assert_array_almost_equal(result, expected)
    assert len(obj) == 3
    assert obj[0] == 1


def test_append_array():
    obj = NumericNormalizedArray(array=[1, 2])
    obj.append_array([3, 4])

    # Should now contain [1, 2, 3, 4]
    assert obj.array == [1, 2, 3, 4]
    np.testing.assert_array_almost_equal(
        obj.array_normalized, np.array([-1.0, -0.3333, 0.3333, 1.0]), decimal=4
    )


def test_remove_indices():
    obj = NumericNormalizedArray(array=[10, 20, 30, 40])
    obj.remove_indices([1, 3])

    # Should now contain [10, 30]
    assert obj.array == [10, 30]
    np.testing.assert_array_almost_equal(
        obj.array_normalized, np.array([-1.0, 1.0])
    )


def test_append_triggers_renormalization():
    obj = NumericNormalizedArray(array=[1, 2, 3])
    obj.append_array([10])  # out-of-bounds

    assert obj.array == [1, 2, 3, 10]
    expected = np.array([-1.0, -1.0+1/9*(1-(-1.0)), -1.0+2/9*(1-(-1.0)), 1.0])
    np.testing.assert_array_almost_equal(obj.array_normalized, expected)

def test_array_invalid_type():
    with pytest.raises(ValidationError):
        NumericNormalizedArray(array="not a sequence") # type: ignore

def test_append_array_invalid_type():
    obj = NumericNormalizedArray(array=[1, 2, 3])
    with pytest.raises(ValidationError):
        obj.append_array("not a sequence") # type: ignore


def test_remove_indices_invalid_type():
    obj = NumericNormalizedArray(array=[1, 2, 3])
    with pytest.raises(ValidationError):
        obj.remove_indices("not a sequence") # type: ignore


def test_remove_indices_out_of_bounds():
    obj = NumericNormalizedArray(array=[1, 2, 3])
    with pytest.raises(IndexError):
        obj.remove_indices([10])


def test_constant_array_normalization():
    obj = NumericNormalizedArray(array=[7, 7, 7])
    result = obj.array_normalized
    np.testing.assert_array_almost_equal(result, np.zeros(3))

def test_set_custom_bounds():
    arr = [1.0, 2.0, 3.0]
    obj = NumericNormalizedArray(array=arr)
    obj.set_custom_bounds(min_val=0.0, max_val=10.0)
    assert np.allclose(obj.array_normalized, [-0.8, -0.6, -0.4])

def test_update_array():
    obj = NumericNormalizedArray(array=[1, 2, 3])

    # Replace with a completely different array
    obj.update_array([10, 20, 30])

    # The normalized array should reflect the new range
    assert obj.array == [10, 20, 30]
    np.testing.assert_array_almost_equal(
        obj.array_normalized, np.array([-1.0, 0.0, 1.0])
    )