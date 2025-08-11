# pylint: disable=C0116, W0212
"""Tests for the CategoricalNormalizedArray"""

import numpy as np

from paxplot.data_managers.categorical_normalized_array import CategoricalNormalizedArray


def test_basic_construction():
    arr = CategoricalNormalizedArray(array=["apple", "banana", "apple", "cherry"])

    np.testing.assert_array_equal(
        arr.array,
        np.array(["apple", "banana", "apple", "cherry"], dtype=np.str_)
    )
    expected_label_to_index = {"apple": 0, "banana": 1, "cherry": 2}
    assert arr._label_to_index == expected_label_to_index
    np.testing.assert_array_equal(arr.array_indeces, np.array([0, 1, 0, 2], dtype=np.float64))
    np.testing.assert_array_equal(arr.array_normalized, np.array([-1.0, 0, -1.0, 1.0]))

def test_append_array():
    arr = CategoricalNormalizedArray(array=["apple", "banana"])
    arr.append_array(["cherry", "banana", "date"])

    # Expected label map
    expected_label_to_index = {"apple": 0, "banana": 1, "cherry": 2, "date": 3}
    assert arr._label_to_index == expected_label_to_index

    # Combined categorical array
    np.testing.assert_array_equal(
        arr.array,
        np.array(["apple", "banana", "cherry", "banana", "date"], dtype=np.str_)
    )

    # Corresponding numeric indices
    expected_indices = np.array([0, 1, 2, 1, 3], dtype=np.float64)
    np.testing.assert_array_equal(arr.array_indeces, expected_indices)

    # Normalized values from indices
    expected_normalized = np.array([-1.0, -0.33333333, 0.33333333, -0.33333333, 1.0])
    np.testing.assert_allclose(arr.array_normalized, expected_normalized, rtol=1e-6)

def test_append_array_with_existing_labels_only():
    arr = CategoricalNormalizedArray(array=["apple", "banana", "cherry"])
    arr.append_array(["banana", "cherry", "apple"])

    # Label map shouldn't change
    expected_label_to_index = {"apple": 0, "banana": 1, "cherry": 2}
    assert arr._label_to_index == expected_label_to_index

    # Combined categorical array
    np.testing.assert_array_equal(
        arr.array,
        np.array(["apple", "banana", "cherry", "banana", "cherry", "apple"], dtype=np.str_)
    )

    # Corresponding numeric indices
    expected_indices = np.array([0, 1, 2, 1, 2, 0], dtype=np.float64)
    np.testing.assert_array_equal(arr.array_indeces, expected_indices)

    # Normalized values from indices
    expected_normalized = np.array([-1.0, 0.0, 1.0, 0.0, 1.0, -1.0])
    np.testing.assert_allclose(arr.array_normalized, expected_normalized, rtol=1e-6)

def test_append_array_extends_normalization_range():
    arr = CategoricalNormalizedArray(array=["one", "two"])
    arr.append_array(["three", "four", "five"])

    # Expect new labels added and normalization updated
    expected_label_to_index = {"one": 0, "two": 1, "three": 2, "four": 3, "five": 4}
    assert arr._label_to_index == expected_label_to_index

    expected_indices = np.array([0, 1, 2, 3, 4], dtype=np.float64)
    np.testing.assert_array_equal(arr.array_indeces, expected_indices)

    # Normalized range [-1, 1]
    expected_normalized = np.linspace(-1, 1, 5)
    np.testing.assert_allclose(arr.array_normalized, expected_normalized, rtol=1e-6)

def test_update_array():
    arr = CategoricalNormalizedArray(array=["apple", "banana", "cherry"])

    # Replace the array entirely
    arr.update_array(["date", "elderberry", "fig", "date"])

    # The internal label-to-index mapping should be rebuilt
    expected_label_to_index = {"date": 0, "elderberry": 1, "fig": 2}
    assert arr._label_to_index == expected_label_to_index

    # Raw array should be updated
    np.testing.assert_array_equal(
        arr.array,
        np.array(["date", "elderberry", "fig", "date"], dtype=np.str_)
    )

    # Numeric indices should correspond to new mapping
    expected_indices = np.array([0, 1, 2, 0], dtype=np.float64)
    np.testing.assert_array_equal(arr.array_indeces, expected_indices)

    # Normalized values should be scaled between -1 and 1
    expected_normalized = np.array([-1.0, 0.0, 1.0, -1.0])
    np.testing.assert_allclose(arr.array_normalized, expected_normalized, rtol=1e-6)
