from typing import Sequence, Union, Literal, Tuple, List
from pydantic import BaseModel, field_validator
import numpy as np

from paxplot.data_managers.numeric_normalized_array import NumericNormalizedArray
from paxplot.data_managers.categorical_normalized_array import CategoricalNormalizedArray
from paxplot.data_managers.base_normalized_array import BaseNormalizedArray


ColumnType = Literal["numeric", "categorical"]


class NormalizedMatrix(BaseModel):
    """
    A typed, column-oriented matrix that normalizes each column independently.

    Each column is either numeric or categorical and stored as a normalized array.
    Columns are accessed by integer index.
    """

    data: Sequence[Sequence[Union[str, int, float, None]]]
    _columns: List[BaseNormalizedArray] = []

    @field_validator("data", mode="before")
    @classmethod
    def validate_input(cls, v: Sequence[Sequence[Union[str, int, float, None]]]) -> Sequence[Sequence[Union[str, int, float, None]]]:
        """
        Validate the input data structure.

        Parameters
        ----------
        v : Sequence[Sequence[Union[str, int, float, None]]]
            The input data to validate.

        Returns
        -------
        Sequence[Sequence[Union[str, int, float, None]]]
            The validated input data.

        Raises
        ------
        ValueError
            If the input is not a 2D array-like structure.
        """
        arr = np.array(v, dtype=object)
        if arr.ndim != 2:
            raise ValueError("Input must be a 2D array-like structure")
        return v

    def __init__(self, **data):
        super().__init__(**data)
        arr, col_types = self.validate_and_infer_column_types(self.data)

        self._columns = []
        for i, col_type in enumerate(col_types):
            col_data = arr[:, i].tolist()
            if col_type == "numeric":
                self._columns.append(NumericNormalizedArray(array=col_data))
            else:
                self._columns.append(CategoricalNormalizedArray(array=col_data))

    @staticmethod
    def validate_and_infer_column_types(
        rows: Sequence[Sequence[Union[str, int, float, None]]]
    ) -> Tuple[np.ndarray, List[ColumnType]]:
        """
        Validates the input rows and infers column types.

        Parameters
        ----------
        rows : Sequence[Sequence[Union[str, int, float, None]]]
            The input tabular data in row-major form.

        Returns
        -------
        tuple
            - A NumPy ndarray (2D) with dtype=object.
            - A list of inferred column types: 'numeric' or 'categorical'.

        Raises
        ------
        ValueError
            If the input is not 2D or contains unsupported/mixed column types.
        """
        arr = np.array(rows, dtype=object)

        if arr.ndim != 2:
            raise ValueError("Input must be a 2D array-like structure")

        n_cols = arr.shape[1]
        column_types: List[ColumnType] = []

        for col_index in range(n_cols):
            col_data = arr[:, col_index]
            types = {type(x) for x in col_data if x is not None}

            if types.issubset({int, float}):
                column_types.append("numeric")
            elif types.issubset({str}):
                column_types.append("categorical")
            else:
                raise ValueError(
                    f"Column {col_index} contains mixed or unsupported types: {types}"
                )

        return arr, column_types

    @property
    def num_columns(self) -> int:
        """
        Returns the number of columns in the matrix.

        Returns
        -------
        int
            The number of columns.
        """
        return len(self._columns)

    @property
    def num_rows(self) -> int:
        """
        Returns the number of rows in the matrix.

        Returns
        -------
        int
            The number of rows.
        """
        if not self._columns:
            return 0
        return len(self._columns[0].array)

    class Config:
        arbitrary_types_allowed = True
        extra = "forbid"
