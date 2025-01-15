"""Tests for DataFrameNormalizer"""

import unittest
import pandas as pd
import paxplot


class DataFrameNormalizerTests(unittest.TestCase):
    """Tests for DataFrameNormalizer

    Args:
        unittest (_type_): _description_
    """

    def test_append_multiple_success(self):
        """
        Appending multiple data
        """
        # Arrange
        reference_data = {'A': [-1, 1], 'B': [-2, 2]}
        reference_df = pd.DataFrame(reference_data)
        incoming_data = {'A': [-2, 2], 'B': [0, 0]}
        incoming_df = pd.DataFrame(incoming_data)

        # Act
        normalizer = paxplot.data_managers.DataFrameNormalizer(reference_df)
        normalizer.append(incoming_df)

        # Assert
        self.assertEqual(normalizer.reference_df.shape[0], 4)
        self.assertEqual(normalizer.reference_df.shape[1], 2)
        self.assertEqual(normalizer.normalized_df.shape[0], 4)
        self.assertEqual(normalizer.normalized_df.shape[1], 2)
        self.assertEqual(normalizer.normalized_df.iloc[0, 0], 0.25)
        self.assertEqual(normalizer.normalized_df.iloc[1, 0], 0.75)
        self.assertEqual(normalizer.normalized_df.iloc[2, 0], 0.0)
        self.assertEqual(normalizer.normalized_df.iloc[3, 0], 1.0)
        self.assertEqual(normalizer.normalized_df.iloc[0, 1], 0.0)
        self.assertEqual(normalizer.normalized_df.iloc[1, 1], 1.0)
        self.assertEqual(normalizer.normalized_df.iloc[2, 1], 0.5)
        self.assertEqual(normalizer.normalized_df.iloc[3, 1], 0.5)
        self.assertEqual(normalizer.reference_min.iloc[0], -2)
        self.assertEqual(normalizer.reference_min.iloc[1], -2)
        self.assertEqual(normalizer.reference_max.iloc[0], 2)
        self.assertEqual(normalizer.reference_max.iloc[1], 2)
