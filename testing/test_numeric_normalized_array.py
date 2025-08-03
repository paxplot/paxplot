# pylint: disable=C0116, W0212
"""Tests for the NumericNormalizedArray"""

import numpy as np
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
