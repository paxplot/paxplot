
import numpy as np

from paxplot.data_managers.categorical_normalized_array import CategoricalNormalizedArray


def test_basic_construction():
    arr = CategoricalNormalizedArray(array_categorical=["apple", "banana", "apple", "cherry"])
    
    np.testing.assert_array_equal(
        arr.array_categorical,
        np.array(["apple", "banana", "apple", "cherry"], dtype=np.str_)
    )
    expected_label_to_index = {"apple": 0, "banana": 1, "cherry": 2}
    assert arr._label_to_index == expected_label_to_index
    expected_indices = np.array([0, 1, 0, 2], dtype=np.float64)
    np.testing.assert_array_equal(np.array(arr.array, dtype=np.float64), expected_indices)
    np.testing.assert_array_equal(arr.array_normalized, np.array([-1.0, 0, -1.0, 1.0]))
