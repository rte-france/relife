from typing import Optional

import numpy as np
from numpy.typing import NDArray



class CostStructure(dict):
    _allowed_keys = ("cp", "cf", "cr")

    def __init__(
        self,
        mapping: Optional[dict[str, float | NDArray[np.float64]]] = None,
        /,
        **kwargs: float | NDArray[np.float64],
    ):
        if mapping is None:
            mapping = {}
        mapping.update(kwargs)
        if not set(mapping.keys()).issubset(self._allowed_keys):
            raise ValueError(f"Only {self._allowed_keys} parameters are allowed")
        mapping = {k : np.asarray(v, dtype=np.float64).reshape(-1, 1) for k, v in mapping.items()}
        super().__init__(mapping)

    def __setitem__(self, key, val):
        raise AttributeError("Can't set item")

    def update(self, *args, **kwargs):
        raise AttributeError("Can't update items")
