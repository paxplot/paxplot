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


    def append(self, data: list):
        """Append data to manager. Updates both the true and normalized data.

        Args:
            data (list): list of list (must rectangular) to append to data
        """
        # Generate unique row indices
        uuid_row = [str(uuid.uuid4()) for _ in data]

        # Generate unique column indices
        uuid_column = [str(uuid.uuid4()) for _ in data[0]]

        # Create a DataFrame from the new data
        df_true = pd.DataFrame(data=data, columns=uuid_column, index=uuid_row)

        # Append new data to the true data
        self.true_data = pd.concat([self.true_data, df_true])
