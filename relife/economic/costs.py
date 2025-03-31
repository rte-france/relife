from typing import NewType

import numpy as np
from numpy.typing import NDArray

Cost = NewType("Cost", NDArray[np.floating] | NDArray[np.integer] | float | int)


def _reshape_mapping(mapping: dict[str, Cost]):
    nb_assets = 1  # minimum value
    for k, v in mapping.items():
        arr = np.asarray(v)
        ndim = arr.ndim
        if ndim > 2:
            raise ValueError(
                f"Number of dimension can't be higher than 2. Got {ndim} for {k}"
            )
        if arr.size == 1:
            mapping[k] = arr
        else:
            arr = arr.reshape(-1, 1)
            if nb_assets != 1 and arr.shape[0] != nb_assets:
                raise ValueError("Different number of assets are given in model args")
            else:  # update nb_assets
                nb_assets = arr.shape[0]
            mapping[k] = arr
    return mapping


class CostStructure(dict):
    _allowed_keys = ("cp", "cf", "cr")

    def __init__(self, mapping=None, /, **kwargs: Cost):
        if mapping is None:
            mapping = {}
        mapping.update(kwargs)
        if not set(self.keys()).issubset(self._allowed_keys):
            raise ValueError(f"Only {self._allowed_keys} parameters are allowed")
        mapping = _reshape_mapping(mapping)
        super().__init__(mapping)

    @property
    def nb_assets(self):
        return max(
            map(lambda x: x.shape[0] if x.ndim >= 1 else 1, self.values()), default=1
        )

    def __setitem__(self, key, val):
        raise AttributeError("Can't set item")

    def update(self, *args, **kwargs):
        raise AttributeError("Can't update items")
