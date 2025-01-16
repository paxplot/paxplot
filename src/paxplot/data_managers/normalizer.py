"""Module for normalizing and scaling pandas DataFrame columns."""

import pandas as pd


class DataFrameNormalizer:
    """
    A class for normalizing and scaling data in a pandas DataFrame. 
    This allows dynamic addition of data and scales the new data based 
    on existing ranges or updates the range if new data exceeds existing bounds.
    """

    def __init__(self):
        """
        Initializes a DataFrameNormalizer instance with empty dataframes 
        for true data and scaled data, and series for tracking column ranges.
        """
        self.true_data: pd.DataFrame = pd.DataFrame()
        self.scaled_data: pd.DataFrame = pd.DataFrame()
        self.bottoms: pd.Series = pd.Series(dtype=float)
        self.tops: pd.Series = pd.Series(dtype=float)

    def scale_column(self, column: pd.Series, lower: float, upper: float) -> pd.Series:
        """
        Scales a pandas Series column to a 0-1 range based on specified bounds.

        Args:
            column (pd.Series): The column to scale.
            lower (float): The lower bound for scaling.
            upper (float): The upper bound for scaling.

        Returns:
            pd.Series: The scaled column with values between 0 and 1.
        """
        return (column - lower) / (upper - lower)

    def append(self, incoming_df: pd.DataFrame) -> None:
        """
        Appends new data to the true data DataFrame and updates the scaled 
        data DataFrame accordingly. Adjusts scaling bounds dynamically if 
        the new data exceeds existing ranges.

        Args:
            incoming_df (pd.DataFrame): The incoming DataFrame to append.
        """
        incoming_scaled_df = pd.DataFrame()
        for column in incoming_df.columns:
            # Get the incoming column's min and max
            incoming_min = incoming_df[column].min()
            incoming_max = incoming_df[column].max()

            # If new column
            if column not in self.true_data.columns:
                # Store bottoms/tops
                self.bottoms[column] = incoming_min
                self.tops[column] = incoming_max

            # If incoming data out of range
            elif (incoming_min < self.bottoms[column] or incoming_max > self.tops[column]):
                # Store new bottoms/tops
                self.bottoms[column] = min(self.bottoms[column], incoming_min)
                self.tops[column] = max(self.tops[column], incoming_max)

                # Scale the existing true data
                self.scaled_data[column] = self.scale_column(
                    self.true_data[column],
                    self.bottoms[column],
                    self.tops[column]
                )

            # Scale the incoming column
            incoming_scaled_df[column] = self.scale_column(
                incoming_df[column],
                self.bottoms[column],
                self.tops[column]
            )

        # Append data
        self.true_data = pd.concat([self.true_data, incoming_df])
        self.scaled_data = pd.concat([self.scaled_data, incoming_scaled_df])
