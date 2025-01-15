"""Class for DataFrameNormalizer"""

import pandas as pd


class DataFrameNormalizer:
    """
    A class to normalize the columns of a DataFrame and dynamically update normalization 
    when appending new data.

    """

    # TODO update implementation to allow for blank dataframe at begining and use append
    def __init__(self, reference_df):
        """
        Initialize the DataFrameNormalizer with a reference DataFrame.

        Args:
            reference_df (pd.DataFrame): A pandas DataFrame containing the reference data.
        """
        self.reference_df = reference_df
        self.reference_min = reference_df.min()
        self.reference_max = reference_df.max()
        self.normalized_df = self._normalize_dataframe(reference_df)

    def _normalize_dataframe(self, df):
        """
        Normalize a DataFrame using the current reference min and max values.

        Args:
            df (pd.DataFrame): The DataFrame to normalize.

        Returns:
            pd.DataFrame: A DataFrame of normalized values.
        """
        return (df - self.reference_min) / (self.reference_max - self.reference_min)

    def normalize_column(self, column, min_value, max_value):
        """
        Normalize a single column between 0 and 1 based on its min and max values.

        Args:
            column (pd.Series): The column to normalize.
            min_value (float): The minimum value of the column.
            max_value (float): The maximum value of the column.

        Returns:
            pd.Series: The normalized column.
        """
        return (column - min_value) / (max_value - min_value)

    def append(self, incoming_df):
        """
        Append an incoming DataFrame to the reference DataFrame, normalizing its columns 
        as needed, and updating the normalized DataFrame.

        If the incoming column is within the reference column's min/max range, it is normalized 
        and added. If it is out of range, the min/max are updated, and the normalized DataFrame 
        is recomputed for the affected columns.

        Args:
            incoming_df (pd.DataFrame): The incoming DataFrame to append.
        """
        for column in incoming_df.columns:
            if column in self.reference_df.columns:
                # Get the incoming column's min and max
                incoming_min = incoming_df[column].min()
                incoming_max = incoming_df[column].max()

                # Update reference min/max if incoming values are out of range
                if (
                    incoming_min < self.reference_min[column]
                    or incoming_max > self.reference_max[column]
                ):
                    self.reference_min[column] = min(self.reference_min[column], incoming_min)
                    self.reference_max[column] = max(self.reference_max[column], incoming_max)

                    # Re-normalize the reference column
                    self.normalized_df[column] = self.normalize_column(
                        self.reference_df[column],
                        self.reference_min[column],
                        self.reference_max[column]
                    )

                # Normalize the incoming column using the updated min/max
                incoming_df[column] = self.normalize_column(
                    incoming_df[column],
                    self.reference_min[column],
                    self.reference_max[column]
                )
            else:
                # Handle new columns in the incoming DataFrame
                incoming_min = incoming_df[column].min()
                incoming_max = incoming_df[column].max()

                # Normalize the incoming column
                incoming_df[column] = self.normalize_column(
                    incoming_df[column],
                    incoming_min,
                    incoming_max
                )

                # Add new columns to reference and normalized DataFrames
                self.reference_min[column] = incoming_min
                self.reference_max[column] = incoming_max
                self.normalized_df[column] = incoming_df[column]

        # Append the incoming DataFrame to the reference DataFrame
        self.reference_df = pd.concat([self.reference_df, incoming_df])

        # Update normalized DataFrame
        self.normalized_df = pd.concat([self.normalized_df, incoming_df])
