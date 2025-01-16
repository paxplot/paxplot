"""Tests for DataFrameNormalizer"""

import unittest
import pandas as pd
import paxplot


class DataFrameNormalizerTests(unittest.TestCase):
    """Tests for DataFrameNormalizer

    Args:
        unittest (_type_): _description_
    """


    def test_append_success(self):
        """
        Appending multiple data
        """
        # Arrange
        a_data = {'A': [-1, 0, 1], 'B': [-2, 0, 2]}
        df_a = pd.DataFrame(a_data)

        # Act
        normalizer = paxplot.data_managers.DataFrameNormalizer()
        normalizer.append(df_a)

        # Assert
        self.assertEqual(normalizer.true_data.shape[0], 3)
        self.assertEqual(normalizer.true_data.shape[1], 2)
        self.assertEqual(normalizer.scaled_data.shape[0], 3)
        self.assertEqual(normalizer.scaled_data.shape[1], 2)
        self.assertEqual(normalizer.scaled_data.iloc[0, 0], 0.0)
        self.assertEqual(normalizer.scaled_data.iloc[1, 0], 0.5)
        self.assertEqual(normalizer.scaled_data.iloc[2, 0], 1.0)
        self.assertEqual(normalizer.scaled_data.iloc[0, 1], 0.0)
        self.assertEqual(normalizer.scaled_data.iloc[1, 1], 0.5)
        self.assertEqual(normalizer.scaled_data.iloc[2, 1], 1.0)
        self.assertEqual(normalizer.bottoms.iloc[0], -1)
        self.assertEqual(normalizer.bottoms.iloc[1], -2)
        self.assertEqual(normalizer.tops.iloc[0], 1)
        self.assertEqual(normalizer.tops.iloc[1], 2)


    def test_append_multiple_success(self):
        """
        Appending multiple data
        """
        # Arrange
        a_data = {'A': [-1, 1], 'B': [-2, 2]}
        df_a = pd.DataFrame(a_data)
        b_data = {'A': [-2, 2], 'B': [0, 0]}
        df_b = pd.DataFrame(b_data)

        # Act
        normalizer = paxplot.data_managers.DataFrameNormalizer()
        normalizer.append(df_a)
        normalizer.append(df_b)

        # Assert
        self.assertEqual(normalizer.true_data.shape[0], 4)
        self.assertEqual(normalizer.true_data.shape[1], 2)
        self.assertEqual(normalizer.scaled_data.shape[0], 4)
        self.assertEqual(normalizer.scaled_data.shape[1], 2)
        self.assertEqual(normalizer.scaled_data.iloc[0, 0], 0.25)
        self.assertEqual(normalizer.scaled_data.iloc[1, 0], 0.75)
        self.assertEqual(normalizer.scaled_data.iloc[2, 0], 0.0)
        self.assertEqual(normalizer.scaled_data.iloc[3, 0], 1.0)
        self.assertEqual(normalizer.scaled_data.iloc[0, 1], 0.0)
        self.assertEqual(normalizer.scaled_data.iloc[1, 1], 1.0)
        self.assertEqual(normalizer.scaled_data.iloc[2, 1], 0.5)
        self.assertEqual(normalizer.scaled_data.iloc[3, 1], 0.5)
        self.assertEqual(normalizer.bottoms.iloc[0], -2)
        self.assertEqual(normalizer.bottoms.iloc[1], -2)
        self.assertEqual(normalizer.tops.iloc[0], 2)
        self.assertEqual(normalizer.tops.iloc[1], 2)
