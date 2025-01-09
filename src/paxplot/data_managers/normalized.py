"""Class for NormalizedDataManager"""

import uuid
import pandas as pd



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
        self.column_uuids = pd.DataFrame(columns=[self.uuid_name, self.column_name])
        self.row_uuids = pd.DataFrame(columns=[self.uuid_name, self.row_name])


    def append(self, data: list, column_names: list=None, row_names: list=None):
        """Append data to manager. Updates both the true and normalized data.

        Args:
            data (list): list of list (must rectangular) to append to data
            column_names (list, optional): Column names. Defaults to None.
            row_names (list, optional): Row names. Defaults to None.
        """
        # Generate UUIDs
        n_rows = len(data)
        n_cols = len(data[0])
        df_row_uuids = self._generate_uuid_df(n_rows, self.row_name, row_names)
        df_column_uuids = self._generate_uuid_df(n_cols, self.column_name, column_names)

        # Create a DataFrame from the new data
        df_true = pd.DataFrame(
            data=data,
            columns=df_column_uuids[self.uuid_name],
            index=df_row_uuids[self.uuid_name]
        )

        # Append new data to the true data
        self.true_data = pd.concat([self.true_data, df_true])

        # Update UUID information
        self.row_uuids = pd.concat([self.row_uuids, df_row_uuids])
        self.column_uuids = pd.concat([self.column_uuids, df_column_uuids])

    def _generate_uuid_df(self, n_objects: int, object_name: str, names: list=None):
        """Generate dataframe of UUID objects (and names, if supplied)

        Args:
            n_objects (int): Number of objects with UUID to generate
            object_name (str): Name of object to generate UUIDs
            names (list, optional): Names of objects. Defaults to None.

        Returns:
            pandas.DataFrame: UUID dataframe
        """
        # Generate UUIDs
        uuids = [str(uuid.uuid4()) for _ in range(n_objects)]

        if names is None:
            names = uuids

        # Create dataframe
        df_uuid = pd.DataFrame(
            {
                self.uuid_name: uuids,
                object_name: names
            }
        )

        return df_uuid
