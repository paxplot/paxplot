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
    assert model.effective_min_val == 1.0
    assert model.effective_max_val == 3.0

    expected = (2.0 / (3.0 - 1.0)) * (arr - 1.0) - 1.0
    np.testing.assert_array_almost_equal(model.array_normalized, expected)


def test_accepts_numpy_array_int():
    arr = np.array([1, 2, 3], dtype=int)
    model = ArrayNormalizer(array=arr)

    assert isinstance(model.array, np.ndarray)
    assert model.effective_min_val == 1.0
    assert model.effective_max_val == 3.0

    expected = (2.0 / (3.0 - 1.0)) * (arr - 1.0) - 1.0
    np.testing.assert_array_almost_equal(model.array_normalized, expected)


def test_accepts_float32_array():
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    model = ArrayNormalizer(array=arr) # type: ignore

    assert isinstance(model.array, np.ndarray)
    assert model.effective_min_val == 1.0
    assert model.effective_max_val == 3.0
    assert model.array.dtype == np.float32 or model.array.dtype == np.float64

    expected = (2.0 / (3.0 - 1.0)) * (arr - 1.0) - 1.0
    np.testing.assert_array_almost_equal(model.array_normalized, expected.astype(model.array.dtype))


def test_already_normalized_range():
    arr = np.array([-1.0, 0.0, 1.0])
    model = ArrayNormalizer(array=arr)
    expected = arr  # Should be unchanged after normalization
    np.testing.assert_array_almost_equal(model.array_normalized, expected)
    assert model.effective_min_val == -1.0
    assert model.effective_max_val == 1.0


def test_all_identical_values():
    arr = np.array([5.0, 5.0, 5.0])
    model = ArrayNormalizer(array=arr)
    expected = np.array([0.0, 0.0, 0.0])
    np.testing.assert_array_almost_equal(model.array_normalized, expected)
    assert model.effective_min_val == 5.0
    assert model.effective_max_val == 5.0


def test_negative_values():
    arr = np.array([-3, -2, -1])
    model = ArrayNormalizer(array=arr)
    expected = np.array([-1.0, 0.0, 1.0])
    np.testing.assert_array_almost_equal(model.array_normalized, expected)
    assert model.effective_min_val == -3.0
    assert model.effective_max_val == -1.0


def test_mixed_negative_and_positive():
    arr = np.array([-2, 0, 2])
    model = ArrayNormalizer(array=arr)
    expected = np.array([-1.0, 0.0, 1.0])
    np.testing.assert_array_almost_equal(model.array_normalized, expected)
    assert model.effective_min_val == -2.0
    assert model.effective_max_val == 2.0


def test_single_element():
    arr = np.array([7.0])
    model = ArrayNormalizer(array=arr)
    expected = np.array([0.0])
    np.testing.assert_array_almost_equal(model.array_normalized, expected)
    assert model.effective_min_val == 7.0
    assert model.effective_max_val == 7.0


def test_large_integer_range():
    arr = np.array([0, 2**31 - 1], dtype=np.int64)
    model = ArrayNormalizer(array=arr) # type: ignore
    expected = np.array([-1.0, 1.0])
    np.testing.assert_array_almost_equal(model.array_normalized, expected)
    assert model.effective_min_val == 0.0
    assert model.effective_max_val == float(2**31 - 1)


def test_float16_array():
    arr = np.array([100.0, 200.0], dtype=np.float16)
    model = ArrayNormalizer(array=arr) # type: ignore
    expected = np.array([-1.0, 1.0], dtype=np.float16)
    np.testing.assert_array_almost_equal(
        model.array_normalized,
        expected.astype(np.float16),
        decimal=2
    )


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

def test_normalize_to_minus1_plus1_basic():
    array = np.array([10.0, 15.0, 20.0])
    expected = np.array([-1.0, 0.0, 1.0])
    model = ArrayNormalizer(array=array)
    result = model._normalize_to_minus1_plus1(
        array,
        min_val=10.0,
        max_val=20.0
    )
    np.testing.assert_allclose(result, expected, rtol=1e-6)

def test_normalize_raises_when_min_equals_max():
    arr = np.array([5.0, 5.0, 5.0])
    with pytest.raises(ZeroDivisionError):
        ArrayNormalizer._normalize_to_minus1_plus1(
            arr,
            min_val=5.0,
            max_val=5.0
        )

def test_all_same_values():
    array = np.array([5.0, 5.0, 5.0])
    model = ArrayNormalizer(array=array)
    with pytest.raises(ZeroDivisionError):
        model._normalize_to_minus1_plus1(
            array,
            min_val=5.0,
            max_val=5.0
        )


def test_negative_range():
    array = np.array([-5.0, 0.0, 5.0])
    expected = np.array([-1.0, 0.0, 1.0])
    model = ArrayNormalizer(array=array)
    result = model._normalize_to_minus1_plus1(
        array,
        min_val=-5.0,
        max_val=5.0
    )
    np.testing.assert_allclose(result, expected, rtol=1e-6)


def test_out_of_bounds_values():
    array = np.array([0.0, 10.0, 20.0, 30.0])
    model = ArrayNormalizer(array=array)
    result = model._normalize_to_minus1_plus1(
        array,
        min_val=10.0,
        max_val=20.0
    )
    expected = np.array([-3.0, -1.0, 1.0, 3.0])  # Values outside the range are linearly projected
    np.testing.assert_allclose(result, expected, rtol=1e-6)

def test_update_array_recomputes_normalization():
    arr = np.array([1.0, 2.0, 3.0])
    model = ArrayNormalizer(array=arr)

    new_arr = np.array([10.0, 20.0, 30.0])
    model.update_array(new_arr)

    assert np.allclose(model.array, new_arr)
    expected = (2.0 / (30.0 - 10.0)) * (new_arr - 10.0) - 1.0
    np.testing.assert_array_almost_equal(model.array_normalized, expected)

def test_update_array_with_custom_bounds():
    arr = np.array([1.0, 2.0, 3.0])
    model = ArrayNormalizer(array=arr, custom_min_val=0.0, custom_max_val=4.0)

    new_arr = np.array([5.0, 6.0, 7.0])
    model.update_array(new_arr)

    # After update, custom bounds remain the same
    assert model.custom_min_val == 0.0
    assert model.custom_max_val == 4.0

    expected = (2.0 / (4.0 - 0.0)) * (new_arr - 0.0) - 1.0
    np.testing.assert_array_almost_equal(model.array_normalized, expected)

def test_custom_min_val_only():
    arr = np.array([1.0, 2.0, 3.0])
    custom_min = 0.0
    model = ArrayNormalizer(array=arr, custom_min_val=custom_min)

    # The effective min should be the custom_min_val, max should be array max
    assert model.effective_min_val == custom_min
    assert model.effective_max_val == 3.0

    # Expected normalization: scale using custom_min=0, max=3
    expected = (2.0 / (3.0 - 0.0)) * (arr - 0.0) - 1.0
    np.testing.assert_array_almost_equal(model.array_normalized, expected)

def test_custom_max_val_only():
    arr = np.array([1.0, 2.0, 3.0])
    custom_max = 4.0
    model = ArrayNormalizer(array=arr, custom_max_val=custom_max)

    # The effective max should be custom_max_val, min should be array min
    assert model.effective_max_val == custom_max
    assert model.effective_min_val == 1.0

    # Expected normalization: scale using min=1, custom_max=4
    expected = (2.0 / (4.0 - 1.0)) * (arr - 1.0) - 1.0
    np.testing.assert_array_almost_equal(model.array_normalized, expected)

def test_custom_min_max_vals():
    arr = np.array([1.0, 2.0, 3.0])
    custom_min = 0.0
    custom_max = 4.0
    model = ArrayNormalizer(array=arr, custom_min_val=custom_min, custom_max_val=custom_max)

    assert model.effective_min_val == custom_min
    assert model.effective_max_val == custom_max

    expected = (2.0 / (4.0 - 0.0)) * (arr - 0.0) - 1.0
    np.testing.assert_array_almost_equal(model.array_normalized, expected)

def test_custom_min_equals_max_raises():
    arr = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ZeroDivisionError):
        # Force custom min and max equal
        ArrayNormalizer._normalize_to_minus1_plus1(arr, min_val=5.0, max_val=5.0)

def test_set_custom_min_val_only():
    arr = np.array([1.0, 2.0, 3.0])
    model = ArrayNormalizer(array=arr)

    model.set_custom_bounds(min_val=0.0)

    assert model.custom_min_val == 0.0
    assert model.custom_max_val is None
    assert model.effective_min_val == 0.0
    assert model.effective_max_val == 3.0

    expected = (2.0 / (3.0 - 0.0)) * (arr - 0.0) - 1.0
    np.testing.assert_array_almost_equal(model.array_normalized, expected)


def test_set_custom_max_val_only():
    arr = np.array([1.0, 2.0, 3.0])
    model = ArrayNormalizer(array=arr)

    model.set_custom_bounds(max_val=4.0)

    assert model.custom_min_val is None
    assert model.custom_max_val == 4.0
    assert model.effective_min_val == 1.0
    assert model.effective_max_val == 4.0

    expected = (2.0 / (4.0 - 1.0)) * (arr - 1.0) - 1.0
    np.testing.assert_array_almost_equal(model.array_normalized, expected)


def test_set_both_custom_bounds():
    arr = np.array([1.0, 2.0, 3.0])
    model = ArrayNormalizer(array=arr)

    model.set_custom_bounds(min_val=0.0, max_val=4.0)

    assert model.custom_min_val == 0.0
    assert model.custom_max_val == 4.0
    assert model.effective_min_val == 0.0
    assert model.effective_max_val == 4.0

    expected = (2.0 / (4.0 - 0.0)) * (arr - 0.0) - 1.0
    np.testing.assert_array_almost_equal(model.array_normalized, expected)


def test_reset_custom_bounds():
    arr = np.array([1.0, 2.0, 3.0])
    model = ArrayNormalizer(array=arr)

    model.set_custom_bounds(min_val=0.0, max_val=4.0)

    # Reset to defaults
    model.set_custom_bounds(min_val=None, max_val=None)

    assert model.custom_min_val is None
    assert model.custom_max_val is None
    assert model.effective_min_val == 1.0
    assert model.effective_max_val == 3.0

    expected = (2.0 / (3.0 - 1.0)) * (arr - 1.0) - 1.0
    np.testing.assert_array_almost_equal(model.array_normalized, expected)

def test_append_within_bounds_appends_and_normalizes_partial():
    arr = np.array([1.0, 2.0, 3.0])
    model = ArrayNormalizer(array=arr)

    # Append data fully within current normalization bounds (1.0 to 3.0)
    new_data = np.array([1.5, 2.5])
    model.append_array(new_data)

    # Check that the array length increased correctly
    assert model.array.shape[0] == 5

    # Check that the newly appended values are normalized correctly using existing bounds
    expected_new_norm = (2.0 / (3.0 - 1.0)) * (new_data - 1.0) - 1.0
    np.testing.assert_array_almost_equal(model.array_normalized[-2:], expected_new_norm)


def test_append_outside_bounds_triggers_renormalization():
    arr = np.array([1.0, 2.0, 3.0])
    model = ArrayNormalizer(array=arr)

    # Append data outside current normalization bounds (e.g., new min 0.0)
    new_data = np.array([0.0, 4.0])
    model.append_array(new_data)

    # Check that array length increased
    assert model.array.shape[0] == 5

    # Check that effective min and max updated
    assert model.effective_min_val == 0.0
    assert model.effective_max_val == 4.0

    # Check normalized array has length 5
    assert model.array_normalized.shape[0] == 5

    # Check normalization is correct on the entire array
    expected_norm = (2.0 / (4.0 - 0.0)) * (model.array - 0.0) - 1.0
    np.testing.assert_array_almost_equal(model.array_normalized, expected_norm)


def test_append_raises_if_normalization_not_computed():
    arr = np.array([1.0, 2.0, 3.0])
    model = ArrayNormalizer(array=arr)
    # Manually clear normalized array to simulate uninitialized state
    model._array_normalized = None

    new_data = np.array([1.5])
    with pytest.raises(ValueError, match="Normalization must be computed before appending"):
        model.append_array(new_data)


def test_append_rejects_invalid_input_type():
    arr = np.array([1.0, 2.0, 3.0])
    model = ArrayNormalizer(array=arr)

    # Passing a list instead of a NumPy array should raise ValueError
    with pytest.raises(ValueError):
        model.append_array([4, 5]) # type: ignore


def test_append_rejects_multidimensional_array():
    arr = np.array([1.0, 2.0, 3.0])
    model = ArrayNormalizer(array=arr)

    multi_dim_data = np.array([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError):
        model.append_array(multi_dim_data)

def test_remove_indices_basic():
    arr = np.array([1.0, 2.0, 3.0])
    normalizer = ArrayNormalizer(array=arr)

    # Remove middle element (index 1)
    normalizer.remove_indices(np.array([1]))

    np.testing.assert_array_equal(normalizer.array, np.array([1.0, 3.0]))
    expected_norm = (2.0 / (3.0 - 1.0)) * (normalizer.array - 1.0) - 1.0
    np.testing.assert_array_almost_equal(normalizer.array_normalized, expected_norm)

def test_remove_indices_recompute_normalization():
    arr = np.array([1.0, 2.0, 3.0])
    normalizer = ArrayNormalizer(array=arr)

    # Remove element at max value index (2)
    normalizer.remove_indices(np.array([2]))

    np.testing.assert_array_equal(normalizer.array, np.array([1.0, 2.0]))
    expected_norm = (2.0 / (2.0 - 1.0)) * (normalizer.array - 1.0) - 1.0
    np.testing.assert_array_almost_equal(normalizer.array_normalized, expected_norm)
    assert normalizer.effective_max_val == 2.0

def test_remove_indices_recompute_min_and_max():
    arr = np.array([1.0, 2.0, 3.0, 0.0])
    normalizer = ArrayNormalizer(array=arr)

    # Remove elements at indices 3 and 0 (min and max)
    normalizer.remove_indices(np.array([0, 3]))

    np.testing.assert_array_equal(normalizer.array, np.array([2.0, 3.0]))
    expected_norm = (2.0 / (3.0 - 2.0)) * (normalizer.array - 2.0) - 1.0
    np.testing.assert_array_almost_equal(normalizer.array_normalized, expected_norm)
    assert normalizer.effective_min_val == 2.0
    assert normalizer.effective_max_val == 3.0

def test_remove_indices_out_of_bounds():
    arr = np.array([1.0, 2.0, 3.0])
    normalizer = ArrayNormalizer(array=arr)

    with pytest.raises(IndexError):
        normalizer.remove_indices(np.array([5]))

def test_remove_indices_without_normalization():
    normalizer = ArrayNormalizer(array=np.array([1.0, 2.0, 3.0]))
    # Force _array_normalized to None to simulate not normalized
    normalizer._array_normalized = None

    with pytest.raises(ValueError):
        normalizer.remove_indices(np.array([1]))

def test_remove_indices_duplicates():
    arr = np.array([1.0, 2.0, 3.0, 4.0])
    normalizer = ArrayNormalizer(array=arr)

    # Remove indices with duplicates
    normalizer.remove_indices(np.array([1, 1, 2]))

    np.testing.assert_array_equal(normalizer.array, np.array([1.0, 4.0]))
