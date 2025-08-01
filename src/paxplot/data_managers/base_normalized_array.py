from abc import ABC, abstractmethod
from typing import Any, Sequence, Optional
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, PrivateAttr
from paxplot.data_managers.array_normalizer import ArrayNormalizer

class BaseNormalizedArray(BaseModel, ABC):
    raw_values: Sequence[Any]
    _normalizer: Optional[ArrayNormalizer] = PrivateAttr(default=None)

    def __len__(self) -> int:
        return len(self.raw_values)

    def __getitem__(self, index: int) -> Any:
        return self.raw_values[index]

    @property
    def normalized_values(self) -> NDArray[np.float64]:
        if self._normalizer is None:
            self._normalizer = self._build_normalizer()
        return self._normalizer.array_normalized

    @abstractmethod
    def _build_normalizer(self) -> ArrayNormalizer:
        pass
