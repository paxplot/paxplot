# pylint: disable=C0116
"""Tests for the ArrayNormalizer"""

import pytest
import numpy as np
from pydantic import ValidationError
from paxplot.data_managers.array_normalizer import ArrayNormalizer

def test_accepts_numpy_array_float64():
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    model = ArrayNormalizer(array=arr)
    assert isinstance(model.array, np.ndarray)
    np.testing.assert_array_equal(model.array, arr)

def test_accepts_numpy_array_int():
    arr = np.array([1, 2, 3], dtype=int)
    model = ArrayNormalizer(array=arr)
    assert isinstance(model.array, np.ndarray)
    np.testing.assert_array_equal(model.array, arr)

def test_accepts_float32_array():
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    model = ArrayNormalizer(array=arr)
    assert isinstance(model.array, np.ndarray)
    np.testing.assert_array_equal(model.array, arr)
    assert model.array.dtype == np.float32

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

def test_json_serialization():
    arr = np.array([1, 2, 3])
    model = ArrayNormalizer(array=arr)
    json_str = model.model_dump_json()
    assert json_str == '{"array": [1, 2, 3]}'
