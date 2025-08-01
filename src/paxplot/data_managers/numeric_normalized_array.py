from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from paxplot.data_managers.array_normalizer import ArrayNormalizer
from paxplot.data_managers.base_normalized_array import BaseNormalizedArray

class NumericNormalizedArray(BaseNormalizedArray):
    raw_values: Sequence[float | int]

    def _build_normalizer(self) -> ArrayNormalizer:
        arr: NDArray[np.number] = np.array(self.raw_values, dtype=float)
        return ArrayNormalizer(array=arr)
