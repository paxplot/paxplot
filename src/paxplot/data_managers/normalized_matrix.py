from typing import Sequence, Union
from pydantic import BaseModel, field_validator
import numpy as np

from paxplot.data_managers.numeric_normalized_array import NumericNormalizedArray
from paxplot.data_managers.categorical_normalized_array import CategoricalNormalizedArray
from paxplot.data_managers.base_normalized_array import BaseNormalizedArray


class NormalizedMatrix(BaseModel):
    """
    A typed, column-oriented matrix that normalizes each column independently.

    Each column is either numeric or categorical and stored as a normalized array.
    Columns are accessed by integer index.
    """

    data: Sequence[Sequence[Union[str, int, float]]]
    _columns: list[BaseNormalizedArray] = []

    @field_validator("data", mode="before")
    @classmethod
    def validate_input(cls, v: Sequence[Sequence[Union[str, int, float]]]) -> Sequence[Sequence[Union[str, int, float]]]:
        # Just sanity check for now; real parsing goes in __init__
        arr = np.array(v, dtype=object)
        if arr.ndim != 2:
            raise ValueError("Input must be a 2D array-like structure")
        return v

    def __init__(self, **data):
        super().__init__(**data)

        # Convert array to columns and store them
        data_array = np.array(self.data, dtype=object)
        n_cols = data_array.shape[1]

        self._columns = []
        for col_index in range(n_cols):
            col_data = data_array[:, col_index].tolist()

            if all(isinstance(x, (int, float)) or x is None for x in col_data):
                col = NumericNormalizedArray(array=col_data)
            elif all(isinstance(x, str) or x is None for x in col_data):
                col = CategoricalNormalizedArray(array=col_data)
            else:
                raise TypeError(
                    f"Column {col_index} contains mixed or unsupported types: {set(type(x) for x in col_data)}"
                )

            self._columns.append(col)

    def get_column(self, index: int) -> BaseNormalizedArray:
        """Return the normalized array at the given column index."""
        return self._columns[index]

    def num_columns(self) -> int:
        return len(self._columns)

    def num_rows(self) -> int:
        if not self._columns:
            return 0
        return len(self._columns[0].array)

    class Config:
        arbitrary_types_allowed = True
        extra = "forbid"
