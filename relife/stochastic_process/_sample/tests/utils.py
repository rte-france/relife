from typing import Optional
import numpy as np
from numpy.typing import NDArray

from relife.stochastic_process._sample._data import IntArrayLike


def select_from_struct(
    struct_array: NDArray,
    sample_id: Optional[IntArrayLike] = None,
    asset_id: Optional[IntArrayLike] = None,
):
    """
    Method used for dev and tests
    """
    mask: NDArray[np.bool_] = np.ones_like(struct_array, dtype=np.bool_)
    if sample_id is not None:
        mask = mask & np.isin(struct_array["sample_id"], sample_id)
    if asset_id is not None:
        mask = mask & np.isin(struct_array["asset_id"], asset_id)
    return struct_array[mask].copy()
