from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from paxplot.data_managers.array_normalizer import ArrayNormalizer
from paxplot.data_managers.base_normalized_array import BaseNormalizedArray

class NumericNormalizedArray(BaseNormalizedArray):
    array: Sequence[float | int]

    def _build_normalizer(self) -> ArrayNormalizer:
        arr: NDArray[np.number] = np.array(self.array, dtype=np.float64)
        return ArrayNormalizer(array=arr)

    def append_array(self, new_data: Sequence[float | int]) -> None:
        raise NotImplementedError

    def remove_indices(self, indices: Sequence[int]) -> None:
        raise NotImplementedError

