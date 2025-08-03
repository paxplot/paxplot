# pylint: disable=C0116, W0212
"""Tests for the NumericNormalizedArray"""

import numpy as np
import pytest
from pydantic import ValidationError
from numpy.testing import assert_allclose
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

def test_to_dict_basic():
    nna = NumericNormalizedArray(array=[1.0, 2.0, 3.0])
    data = nna.to_dict()

    assert data["array"] == [1.0, 2.0, 3.0]
    assert data["custom_min_val"] is None
    assert data["custom_max_val"] is None
    assert data["_schema_version"] == nna._schema_version

def test_from_dict_basic():
    data = {
        "array": [10.0, 20.0, 30.0],
        "custom_min_val": None,
        "custom_max_val": None,
        "_schema_version": 1
    }

    nna = NumericNormalizedArray.from_dict(data)

    assert nna.array == [10.0, 20.0, 30.0]
    assert nna.custom_min_val is None
    assert nna.custom_max_val is None
    assert isinstance(nna.array_normalized, np.ndarray)

def test_round_trip_serialization():
    original = NumericNormalizedArray(array=[100.0, 200.0, 300.0])
    original.set_custom_bounds(min_val=0.0, max_val=400.0)

    data = original.to_dict()
    restored = NumericNormalizedArray.from_dict(data)

    assert np.allclose(original.array_normalized, restored.array_normalized)
    assert restored.custom_min_val == 0.0
    assert restored.custom_max_val == 400.0
    assert restored.array == [100.0, 200.0, 300.0]

def test_schema_version_check_raises():
    data = {
        "array": [1.0, 2.0],
        "custom_min_val": 0.0,
        "custom_max_val": 5.0,
        "_schema_version": 999  # intentionally too new
    }

    with pytest.raises(ValueError, match="Unsupported schema version"):
        NumericNormalizedArray.from_dict(data)

def test_from_dict_restores_normalized_array():
    original = NumericNormalizedArray(array=[100.0, 200.0, 300.0])
    original.set_custom_bounds(min_val=0.0, max_val=400.0)
    expected_normalized = original.array_normalized.copy()

    data = original.to_dict()
    restored = NumericNormalizedArray.from_dict(data)

    assert restored.array == [100.0, 200.0, 300.0]
    assert restored.custom_min_val == 0.0
    assert restored.custom_max_val == 400.0
    assert_allclose(restored.array_normalized, expected_normalized)
