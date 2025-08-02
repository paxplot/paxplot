# pylint: disable=C0116, W0212
"""Tests for the ArrayNormalizer"""

import pytest
import numpy as np
from pydantic import ValidationError
from paxplot.data_managers.array_normalizer import ArrayNormalizer


def test_accepts_numpy_array_float64():
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    model = ArrayNormalizer(array=arr)

    assert isinstance(model.array, np.ndarray)
    assert model.min_val == 1.0
    assert model.max_val == 3.0

    expected = (2.0 / (3.0 - 1.0)) * (arr - 1.0) - 1.0
    np.testing.assert_array_almost_equal(model.array, expected)


def test_accepts_numpy_array_int():
    arr = np.array([1, 2, 3], dtype=int)
    model = ArrayNormalizer(array=arr)

    assert isinstance(model.array, np.ndarray)
    assert model.min_val == 1.0
    assert model.max_val == 3.0

    expected = (2.0 / (3.0 - 1.0)) * (arr - 1.0) - 1.0
    np.testing.assert_array_almost_equal(model.array, expected)


def test_accepts_float32_array():
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    model = ArrayNormalizer(array=arr)

    assert isinstance(model.array, np.ndarray)
    assert model.min_val == 1.0
    assert model.max_val == 3.0
    assert model.array.dtype == np.float32 or model.array.dtype == np.float64

    expected = (2.0 / (3.0 - 1.0)) * (arr - 1.0) - 1.0
    np.testing.assert_array_almost_equal(model.array, expected.astype(model.array.dtype))

def test_already_normalized_range():
    arr = np.array([-1.0, 0.0, 1.0])
    model = ArrayNormalizer(array=arr)
    expected = arr  # Should be unchanged after normalization
    np.testing.assert_array_almost_equal(model.array, expected)
    assert model.min_val == -1.0
    assert model.max_val == 1.0

def test_all_identical_values():
    arr = np.array([5.0, 5.0, 5.0])
    model = ArrayNormalizer(array=arr)
    expected = np.array([0.0, 0.0, 0.0])
    np.testing.assert_array_almost_equal(model.array, expected)
    assert model.min_val == 5.0
    assert model.max_val == 5.0

def test_negative_values():
    arr = np.array([-3, -2, -1])
    model = ArrayNormalizer(array=arr)
    expected = np.array([-1.0, 0.0, 1.0])
    np.testing.assert_array_almost_equal(model.array, expected)
    assert model.min_val == -3.0
    assert model.max_val == -1.0

def test_mixed_negative_and_positive():
    arr = np.array([-2, 0, 2])
    model = ArrayNormalizer(array=arr)
    expected = np.array([-1.0, 0.0, 1.0])
    np.testing.assert_array_almost_equal(model.array, expected)
    assert model.min_val == -2.0
    assert model.max_val == 2.0

def test_single_element():
    arr = np.array([7.0])
    model = ArrayNormalizer(array=arr)
    expected = np.array([0.0])
    np.testing.assert_array_almost_equal(model.array, expected)
    assert model.min_val == 7.0
    assert model.max_val == 7.0

def test_large_integer_range():
    arr = np.array([0, 2**31 - 1], dtype=np.int64)
    model = ArrayNormalizer(array=arr)
    expected = np.array([-1.0, 1.0])
    np.testing.assert_array_almost_equal(model.array, expected)
    assert model.min_val == 0.0
    assert model.max_val == float(2**31 - 1)

def test_float16_array():
    arr = np.array([100.0, 200.0], dtype=np.float16)
    model = ArrayNormalizer(array=arr)
    expected = np.array([-1.0, 1.0], dtype=np.float16)
    np.testing.assert_array_almost_equal(model.array, expected.astype(np.float16), decimal=2)

def test_rejects_list_input():
    with pytest.raises(ValidationError):
        ArrayNormalizer(array=[1.0, 2.0, 3.0])  # type: ignore

def test_rejects_invalid_type():
    with pytest.raises(ValidationError):
        ArrayNormalizer(array="not an array")  # type: ignore

def test_rejects_multidimensional_array():
    arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValidationError):
        ArrayNormalizer(array=arr)

def test_rejects_non_numeric_dtype():
    arr = np.array(['a', 'b', 'c'])
    with pytest.raises(ValidationError):
        ArrayNormalizer(array=arr)


def test_to_dict():
    arr = np.array([1, 2, 3])
    model = ArrayNormalizer(array=arr)

    # Serialize to dict
    data = model.to_dict()

    expected_array = [-1.0, 0.0, 1.0]
    assert data["array"] == expected_array
    assert data["min_val"] == 1.0
    assert data["max_val"] == 3.0
    assert data["_schema_version"] == model._schema_version


def test_from_dict():
    data = {
        "array": [-1.0, 0.0, 1.0],
        "min_val": 1.0,
        "max_val": 3.0,
        "_schema_version": 1
    }

    model = ArrayNormalizer.from_dict(data)

    np.testing.assert_array_equal(model.array, np.array([-1.0, 0.0, 1.0]))
    assert model.min_val == 1.0
    assert model.max_val == 3.0


def test_from_dict_without_schema_version():
    """Ensure backward compatibility if `_schema_version` is missing"""
    data = {
        "array": [-1.0, 0.0, 1.0],
        "min_val": 1.0,
        "max_val": 3.0
    }

    model = ArrayNormalizer.from_dict(data)

    np.testing.assert_array_equal(model.array, np.array([-1.0, 0.0, 1.0]))
    assert model.min_val == 1.0
    assert model.max_val == 3.0


def test_from_dict_future_schema_version_raises():
    """Ensure it raises for unsupported newer versions"""
    data = {
        "array": [-1.0, 0.0, 1.0],
        "min_val": 1.0,
        "max_val": 3.0,
        "_schema_version": 999
    }

    with pytest.raises(ValueError, match="Unsupported schema version"):
        ArrayNormalizer.from_dict(data)

def test_normalize_to_minus1_plus1_basic():
    array = np.array([10.0, 15.0, 20.0])
    expected = np.array([-1.0, 0.0, 1.0])
    model = ArrayNormalizer(array=array)
    result = model._normalize_to_minus1_plus1(array, min_val=10.0, max_val=20.0)
    np.testing.assert_allclose(result, expected, rtol=1e-6)

def test_all_same_values():
    array = np.array([5.0, 5.0, 5.0])
    model = ArrayNormalizer(array=array)
    with pytest.raises(ZeroDivisionError):
        model._normalize_to_minus1_plus1(array, min_val=5.0, max_val=5.0)

def test_negative_range():
    array = np.array([-5.0, 0.0, 5.0])
    expected = np.array([-1.0, 0.0, 1.0])
    model = ArrayNormalizer(array=array)
    result = model._normalize_to_minus1_plus1(array, min_val=-5.0, max_val=5.0)
    np.testing.assert_allclose(result, expected, rtol=1e-6)

def test_out_of_bounds_values():
    array = np.array([0.0, 10.0, 20.0, 30.0])
    model = ArrayNormalizer(array=array)
    result = model._normalize_to_minus1_plus1(array, min_val=10.0, max_val=20.0)
    expected = np.array([-3.0, -1.0, 1.0, 3.0])  # Values outside the range are linearly projected
    np.testing.assert_allclose(result, expected, rtol=1e-6)
