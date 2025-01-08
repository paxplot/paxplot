"""Tests for NormalizedDataManager"""

import unittest
import paxplot


class NormalizedDataManagerTests(unittest.TestCase):
    """Tests for NormalizedDataManager

    Args:
        unittest (_type_): _description_
    """

    def test_append_success(self):
        """
        Basic appending data
        """
        # Arrange
        data = [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0]
        ]
        paxdataset = paxplot.data_managers.NormalizedDataManager()

        # Act
        paxdataset.append(data)

        # Assert
        self.assertEqual(paxdataset.true_data[0][0], 0.0)
