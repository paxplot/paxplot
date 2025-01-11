"""Tests for DataValidator"""

import unittest
import paxplot


class DataValidatorTests(unittest.TestCase):
    """Tests for DataValidator"""

    def test_self_validate_success(self):
        """
        Test case for valid data that passes self_validate.
        """
        # Arrange
        data = [
            [0.0, 0.0, "A"],
            [1.0, 1.0, "B"],
            [2.0, 2.0, "C"]
        ]
        datavalidator = paxplot.data_managers.DataValidator()

        # Act
        result = datavalidator.self_validate(data)

        # Assert
        self.assertEqual(result, True)

    def test_self_validate_nonrectangular(self):
        """
        Test case for invalid data (non-rectangular).
        """
        # Arrange
        data = [
            [0.0, 0.0, "A"],
            [1.0, 1.0],
            [2.0, 2.0]
        ]
        datavalidator = paxplot.data_managers.DataValidator()

        # Act and Assert
        with self.assertRaises(paxplot.data_managers.DataValidationError) as cm:
            datavalidator.self_validate(data)

        # Assert: Check that the exception message contains the expected error
        self.assertEqual(
            str(cm.exception),
            "Row 2 does not have the correct length (2 instead of 3)."
        )

    def test_can_append_success(self):
        """
        Test case for appending data successfully.
        """
        # Arrange
        new_data = [
            [2.0, 2.0, "C"],
            [3.0, 3.0, "D"]
        ]
        expected_types = [float, float, str]

        datavalidator = paxplot.data_managers.DataValidator()

        # Act: Check if new data can be appended
        result = datavalidator.can_append(new_data, expected_types)

        # Assert: The result should be True since data matches the existing data's shape and types
        self.assertEqual(result, True)

    def test_can_append_invalid_data(self):
        """
        Test case for invalid data that cannot be appended.
        """
        # Arrange
        new_data = [
            [2.0, "string", 5]  # Invalid data (types mismatch)
        ]
        expected_types = [float, float, str]
        datavalidator = paxplot.data_managers.DataValidator()

        # Act and Assert: Check that can_append raises an error due to type mismatch
        with self.assertRaises(paxplot.data_managers.DataValidationError) as cm:
            datavalidator.can_append(new_data, expected_types)

        # Assert: Check the error message for type mismatch
        self.assertEqual(
            str(cm.exception),
            "Column 2 has incorrect type. Expected <class 'float'>, but got <class 'str'>."
        )
