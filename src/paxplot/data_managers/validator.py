"""Class for DataValidator"""

class DataValidationError(Exception):
    """Custom exception for data validation errors."""


class DataValidator:
    """
    Class for validating data in the form of a list of lists.

    This class provides a method to check that:
    - The data is not empty.
    - The data is a list of lists.
    - All rows have the same length.
    - All rows have consistent data types across their elements.

    Attributes:
        None
    """

    def self_validate(self, data: list) -> bool:
        """
        Validates if the given data is rectangular and has consistent types.

        Args:
            data (list): A list of lists to validate. Each inner list represents a row,
                         and all rows should have the same number of elements and consistent
                         data types.

        Raises:
            DataValidationError: If the data is empty, not a list of lists, has rows of
                                 inconsistent lengths, or has rows with inconsistent data types.

        Returns:
            bool: True if the data is valid, otherwise raises a DataValidationError.

        Example:
            data = [
                [0.0, 0.0, "A"],
                [1.0, 1.0, "B"],
                [2.0, 2.0, "C"]
            ]
            validator = DataValidator()
            validator.self_validate(data)  # Returns True if data is valid
        """
        if not data:
            raise DataValidationError("The data is empty.")

        if not all(isinstance(row, list) for row in data):
            raise DataValidationError("The data must be a list of lists.")

        first_row_length = len(data[0])
        first_row_types = tuple(type(item) for item in data[0])

        for index, row in enumerate(data):
            if len(row) != first_row_length:
                raise DataValidationError(
                    f"Row {index + 1} does not have the correct length ({len(row)} instead of "
                    f"{first_row_length})."
                )
            if tuple(type(item) for item in row) != first_row_types:
                raise DataValidationError(
                    f"Row {index + 1} has inconsistent types. Expected {first_row_types}, "
                    f"but got {tuple(type(item) for item in row)}."
                )

        return True

    def can_append(self, data: list, expected_types: list) -> bool:
        """
        Checks if a validated dataset can be appended with new data, ensuring that the new data 
        matches the expected column types and the number of columns.

        **Note**: This method assumes that the provided `data` has already been validated 
        (i.e., it has passed `self_validate`).

        Args:
            data (list): A dataset that has already been validated, represented as a list of lists.
                         Each row represents a data entry, and the columns should match the 
                         expected types and shape of the existing data.
            expected_types (list): A list of expected types for each column in the data. 
                                    This list is used to ensure the new data matches the column
                                    types.

        Raises:
            DataValidationError: If the new data doesn't match the expected column types or 
                                 if the number of columns does not match.

        Returns:
            bool: True if the new data can be appended, otherwise raises a DataValidationError.

        Example:
            data = [
                [0.0, 0.0, "A"],
                [1.0, 1.0, "B"]
            ]
            expected_types = [float, float, str]
            validator = DataValidator()
            validator.can_append(data, expected_types)  # Returns True if append is valid
        """
        # Check that the number of columns matches the expected number of types
        if len(data[0]) != len(expected_types):
            raise DataValidationError(
                f"Number of columns in data ({len(data[0])}) does not match the number of expected "
                f" types ({len(expected_types)})."
            )

        # Check that each column in the new data matches the expected type
        for col_index, item in enumerate(data[0]):
            if not isinstance(item, expected_types[col_index]):
                raise DataValidationError(
                    f"Column {col_index + 1} has incorrect type. Expected "
                    f"{expected_types[col_index]}, but got {type(item)}."
                )

        return True
