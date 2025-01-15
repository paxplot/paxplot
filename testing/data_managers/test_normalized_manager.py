"""Tests for NormalizedDataManager"""

import unittest
import paxplot


class NormalizedDataManagerTests(unittest.TestCase):
    """Tests for NormalizedDataManager

    Args:
        unittest (_type_): _description_
    """

    def test_append_true_data_appended(self):
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
        self.assertEqual(paxdataset.true_data.iloc[0, 0], 0.0)
        self.assertEqual(paxdataset.true_data.iloc[0, 1], 0.0)
        self.assertEqual(paxdataset.true_data.iloc[1, 0], 1.0)
        self.assertEqual(paxdataset.true_data.iloc[1, 1], 1.0)
        self.assertEqual(paxdataset.true_data.iloc[2, 0], 2.0)
        self.assertEqual(paxdataset.true_data.iloc[2, 1], 2.0)
        self.assertEqual(paxdataset.normalized_data.iloc[0, 0], 0.0)
        self.assertEqual(paxdataset.normalized_data.iloc[0, 1], 0.0)
        self.assertEqual(paxdataset.normalized_data.iloc[1, 0], 0.5)
        self.assertEqual(paxdataset.normalized_data.iloc[1, 1], 0.5)
        self.assertEqual(paxdataset.normalized_data.iloc[2, 0], 1.0)
        self.assertEqual(paxdataset.normalized_data.iloc[2, 1], 1.0)
        self.assertEqual(paxdataset.empty, False)
        self.assertEqual(paxdataset.column_datatypes[0], float)
        self.assertEqual(paxdataset.column_datatypes[1], float)
        self.assertEqual(paxdataset.row_uuids.shape[0], 3)
        self.assertEqual(paxdataset.column_uuids.shape[0], 2)

    def test_append_multiple(self):
        """
        Multiple appending data
        """
        # Arrange
        data = [
            [-1.0, -1.0],
            [1.0, 1.0],
        ]
        out_of_range_data = [
            [-2.0, -2.0],
            [2.0, 2.0]
        ]
        paxdataset = paxplot.data_managers.NormalizedDataManager()
        paxdataset.append(data)

        # Act
        paxdataset.append(out_of_range_data) # TODO need to use DataFrameNormalizer

        # Assert
        self.assertEqual(paxdataset.true_data.iloc[0, 0], 0.0)

    def test_append_named_data(self):
        """
        Basic naming of data
        """
        # Arrange
        data = [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0]
        ]
        row_names = ["row_a", "row_b", "row_c"]
        column_names = ["col_a", "col_b"]
        paxdataset = paxplot.data_managers.NormalizedDataManager()

        # Act
        paxdataset.append(data, column_names, row_names)

        # Assert
        self.assertEqual(paxdataset.column_uuids.iloc[0], "col_a")
        self.assertEqual(paxdataset.column_uuids.iloc[1], "col_b")
        self.assertEqual(paxdataset.row_uuids.iloc[0], "row_a")
        self.assertEqual(paxdataset.row_uuids.iloc[1], "row_b")
        self.assertEqual(paxdataset.row_uuids.iloc[2], "row_c")

    def test_drop_rows_by_uuid_success(self):
        """
        Basic dropping of row data by uuids
        """
        # Arrange
        data = [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0]
        ]
        row_names = ["row_a", "row_b", "row_c"]
        column_names = ["col_a", "col_b"]
        paxdataset = paxplot.data_managers.NormalizedDataManager()
        paxdataset.append(data, column_names, row_names)
        data_to_drop = paxdataset.get_row_uuids(["row_b"])

        # Act
        paxdataset.drop_rows_by_uuid(data_to_drop)

        # Assert
        self.assertEqual(paxdataset.true_data.iloc[0, 0], 0.0)
        self.assertEqual(paxdataset.true_data.iloc[0, 1], 0.0)
        self.assertEqual(paxdataset.true_data.iloc[1, 0], 2.0)
        self.assertEqual(paxdataset.true_data.iloc[1, 1], 2.0)
        self.assertEqual(paxdataset.normalized_data.iloc[0, 0], 0.0)
        self.assertEqual(paxdataset.normalized_data.iloc[0, 1], 0.0)
        self.assertEqual(paxdataset.normalized_data.iloc[1, 0], 1.0)
        self.assertEqual(paxdataset.normalized_data.iloc[1, 1], 1.0)
        self.assertEqual(paxdataset.row_uuids.shape[0], 2)

    def test_drop_rows_by_row_names_success(self):
        """
        Basic dropping of by row names
        """
        # Arrange
        data = [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0]
        ]
        row_names = ["row_a", "row_b", "row_c"]
        column_names = ["col_a", "col_b"]
        paxdataset = paxplot.data_managers.NormalizedDataManager()
        paxdataset.append(data, column_names, row_names)

        # Act
        paxdataset.drop_rows_by_names(["row_a", "row_b"])

        # Assert
        self.assertEqual(paxdataset.true_data.iloc[0, 0], 2.0)
        self.assertEqual(paxdataset.true_data.iloc[0, 1], 2.0)
        self.assertEqual(paxdataset.row_uuids.shape[0], 1)
