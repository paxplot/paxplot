"""Class for NormalizedDataManager"""


class NormalizedDataManager():
    """
    Manages data and a normalized (between 0 and 1) representation of that same data 
    """

    def __init__(self):
        self.true_data = []
        self.normalized_data = []


    def append(self, data=list):
        """Append data to manager. Updates both the true and normalized data.

        Args:
            data (_type_, optional): _description_. Defaults to list.
        """
        self.true_data = data
        self.normalized_data = data
