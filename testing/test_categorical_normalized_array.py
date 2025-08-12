# pylint: disable=C0116, W0212
"""Tests for the CategoricalNormalizedArray"""

import numpy as np
from paxplot.data_managers.categorical_normalized_array import CategoricalNormalizedArray


def test_basic_construction():
    arr = CategoricalNormalizedArray(values=["apple", "banana", "apple", "cherry"])

    np.testing.assert_array_equal(
        arr.values,
        np.array(["apple", "banana", "apple", "cherry"], dtype=np.str_)
    )
    expected_categories = ["apple", "banana", "cherry"]
    assert arr.categories == expected_categories
    np.testing.assert_array_equal(arr.value_indices, np.array([0, 1, 0, 2], dtype=np.float64))
    np.testing.assert_array_equal(arr.values_normalized, np.array([-1.0, 0, -1.0, 1.0]))


def test_append_array():
    arr = CategoricalNormalizedArray(values=["apple", "banana"])
    arr.append_array(["cherry", "banana", "date"])

    expected_categories = ["apple", "banana", "cherry", "date"]
    assert arr.categories == expected_categories

    np.testing.assert_array_equal(
        arr.values,
        np.array(["apple", "banana", "cherry", "banana", "date"], dtype=np.str_)
    )

    expected_indices = np.array([0, 1, 2, 1, 3], dtype=np.float64)
    np.testing.assert_array_equal(arr.value_indices, expected_indices)

    expected_normalized = np.array([-1.0, -0.33333333, 0.33333333, -0.33333333, 1.0])
    np.testing.assert_allclose(arr.values_normalized, expected_normalized, rtol=1e-6)


def test_append_array_with_existing_categories_only():
    arr = CategoricalNormalizedArray(values=["apple", "banana", "cherry"])
    arr.append_array(["banana", "cherry", "apple"])

    expected_categories = ["apple", "banana", "cherry"]
    assert arr.categories == expected_categories

    np.testing.assert_array_equal(
        arr.values,
        np.array(["apple", "banana", "cherry", "banana", "cherry", "apple"], dtype=np.str_)
    )

    expected_indices = np.array([0, 1, 2, 1, 2, 0], dtype=np.float64)
    np.testing.assert_array_equal(arr.value_indices, expected_indices)

    expected_normalized = np.array([-1.0, 0.0, 1.0, 0.0, 1.0, -1.0])
    np.testing.assert_allclose(arr.values_normalized, expected_normalized, rtol=1e-6)


def test_append_array_extends_normalization_range():
    arr = CategoricalNormalizedArray(values=["one", "two"])
    arr.append_array(["three", "four", "five"])

    expected_categories = ["one", "two", "three", "four", "five"]
    assert arr.categories == expected_categories

    expected_indices = np.array([0, 1, 2, 3, 4], dtype=np.float64)
    np.testing.assert_array_equal(arr.value_indices, expected_indices)

    expected_normalized = np.linspace(-1, 1, 5)
    np.testing.assert_allclose(arr.values_normalized, expected_normalized, rtol=1e-6)


def test_update_array():
    arr = CategoricalNormalizedArray(values=["apple", "banana", "cherry"])

    arr.update_array(["date", "elderberry", "fig", "date"])

    expected_categories = ["date", "elderberry", "fig"]
    assert arr.categories == expected_categories

    np.testing.assert_array_equal(
        arr.values,
        np.array(["date", "elderberry", "fig", "date"], dtype=np.str_)
    )

    expected_indices = np.array([0, 1, 2, 0], dtype=np.float64)
    np.testing.assert_array_equal(arr.value_indices, expected_indices)

    expected_normalized = np.array([-1.0, 0.0, 1.0, -1.0])
    np.testing.assert_allclose(arr.values_normalized, expected_normalized, rtol=1e-6)
