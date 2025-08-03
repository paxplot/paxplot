from typing import Sequence, Any
import numpy as np
from numpy.typing import NDArray
from pydantic import PrivateAttr, model_validator
from paxplot.data_managers.base_normalized_array import BaseNormalizedArray


class CategoricalNormalizedArray(BaseNormalizedArray):
    array_categorical: NDArray[np.str_]
    _label_to_index: dict[str, int] = PrivateAttr(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True

    @model_validator(mode="before")
    @classmethod
    def convert_categorical_to_array(cls, values):
        cat = values.get("array_categorical")
        if cat is not None:
            cat = np.array(cat, dtype=np.str_)
            label_to_index = {}
            indices = []
            for label in cat:
                if label not in label_to_index:
                    label_to_index[label] = len(label_to_index)
                indices.append(label_to_index[label])
            values["array"] = indices
            values["array_categorical"] = cat
        return values

    def __init__(self, **data):
        super().__init__(**data)
        # Rebuild label_to_index dict for use later
        self._label_to_index = {label: i for i, label in enumerate(np.unique(self.array_categorical))}


    def _map_array_to_indices(self, array: Sequence[str]) -> np.ndarray:
        index_list = []
        for label in array:
            if label not in self._label_to_index:
                self._label_to_index[label] = len(self._label_to_index)
            index_list.append(self._label_to_index[label])
        return np.array(index_list, dtype=np.float64)

    def append_array(self, new_data: Sequence[str]) -> None:
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError
