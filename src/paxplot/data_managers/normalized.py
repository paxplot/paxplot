"""Class for NormalizedDataManager"""

import uuid
import pandas as pd
import paxplot

class NormalizedDataManager():
    """
    Manages data and a normalized (between 0 and 1) representation of that same data 
    """

    def __init__(self):
        self.true_data = pd.DataFrame()
        self.normalized_data = pd.DataFrame()
        self.uuid_name = "uuid"
        self.column_name = "column"
        self.row_name = "row"
        self.column_uuids = pd.Series(name=self.column_name)
        self.row_uuids = pd.Series(name=self.row_name)
        self.column_datatypes = []
        self._data_validator = paxplot.data_managers.DataValidator()
        self.empty = True


    def append(self, data: list, column_names: list=None, row_names: list=None):
        """Append data to manager. Updates both the true and normalized data.

        Args:
            data (list): list of list (must rectangular) to append to data
            column_names (list, optional): Column names. Defaults to None.
            row_names (list, optional): Row names. Defaults to None.
        """
        # Check data validity
        self._data_validator.self_validate(data)
        if self.empty is False:
            self._data_validator.can_append(data, self.column_datatypes)

        # Generate UUIDs
        n_rows = len(data)
        n_cols = len(data[0])
        series_row_uuids = self._generate_uuid_series(n_rows, self.row_name, row_names)
        series_column_uuids = self._generate_uuid_series(n_cols, self.column_name, column_names)

        # Create a true DataFrame from the new data
        df_true = pd.DataFrame(
            data=data,
            columns=series_column_uuids.index,
            index=series_row_uuids.index
        )

        # Create a normalized DataFrame from the new data
        df_normalized = self._normalize_dataframe(df_true)

        # Store new data
        self.empty = False
        self.true_data = pd.concat([self.true_data, df_true])
        self.normalized_data = pd.concat([self.normalized_data, df_normalized])

        # Store column data types
        self.column_datatypes = [type(i) for i in data[0]]

        # Store UUID information
        self.row_uuids = pd.concat([self.row_uuids, series_row_uuids])
        self.column_uuids = pd.concat([self.column_uuids, series_column_uuids])

    def _generate_uuid_series(self, n_objects: int, object_name: str, names: list=None):
        """Generate series of UUID objects (and names, if supplied)

        Args:
            n_objects (int): Number of objects with UUID to generate
            object_name (str): Name of object to generate UUIDs
            names (list, optional): Names of objects. Defaults to None.

        Returns:
            pandas.Series: UUID series
        """
        # Generate UUIDs
        uuids = [str(uuid.uuid4()) for _ in range(n_objects)]

        if names is None:
            names = uuids

        # Create series
        series_uuid = pd.Series(
            names,
            name=object_name,
            index=uuids
        )

        return series_uuid

    def _normalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize all columns to be between 0 and 1

        Args:
            df (pd.DataFrame): DataFrame to normalize

        Returns:
            pd.DataFrame: Normalized DataFrame
        """
        # Compute the minimum and maximum for all columns at once
        min_vals = df.min()
        max_vals = df.max()
        # Compute the range, avoiding division by zero
        ranges = max_vals - min_vals
        ranges[ranges == 0] = 1  # Prevent division by zero for constant columns
        # Normalize using vectorized operations
        df_normalize = (df - min_vals) / ranges
        return df_normalize

    def get_row_uuids(self, row_names: list[str]) -> list:
        """
        Retrieve the UUIDs corresponding to the provided row names.

        This method checks the `row_uuids` DataFrame and returns a list of UUIDs 
        that correspond to the given list of `row_names`.

        Args:
            row_names (list[str]): A list of row names for which UUIDs are to be retrieved.

        Returns:
            list: A list of UUIDs corresponding to the provided row names.
            If no matching row names are found, an empty list will be returned.
        """
        # Filter the row_uuids DataFrame to include only the rows with matching row names
        # and return the index (UUIDs) as a list.
        return self.row_uuids[self.row_uuids.isin(row_names)].index.tolist()

    def drop_rows_by_uuid(self, uuids: list[str]) -> None:
        """
        Drop rows from both the true_data and row_uuids DataFrames based on the provided UUIDs.

        This method checks if the UUIDs exist in the `row_uuids` DataFrame. If any 
        UUIDs do not exist, a KeyError is raised. After validation, it drops the 
        corresponding rows from both `true_data` and `row_uuids`.

        Args:
            uuids (list[str]): A list of UUIDs to be dropped from both DataFrames.

        Raises:
            KeyError: If any of the provided UUIDs do not exist in the `row_uuids` DataFrame.
        """
        # Check if all provided UUIDs exist in the row_uuids DataFrame
        missing_uuids = set(uuids) - set(self.row_uuids.index)

        if missing_uuids:
            # If any UUIDs are missing, raise a KeyError with a descriptive message
            raise KeyError(f"The following UUIDs do not exist in row_uuids: {missing_uuids}")

        # Drop the rows from true_data and row_uuids DataFrames using the provided UUIDs
        # The inplace=True ensures the changes are applied directly to the DataFrames
        self.true_data.drop(index=uuids, inplace=True)
        self.normalized_data.drop(index=uuids, inplace=True)
        self.row_uuids.drop(index=uuids, inplace=True)

    def drop_rows_by_names(self, row_names: list[str]) -> None:
        """
        Drop rows from both the true_data and row_uuids DataFrames based on the provided row names.

        Args:
            row_names (list[str]): A list of row names to be dropped from both DataFrames.
        """
        row_uuids = self.get_row_uuids(row_names)
        self.drop_rows_by_uuid(row_uuids)
