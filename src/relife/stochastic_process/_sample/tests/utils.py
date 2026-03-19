import numpy as np
from numpy.typing import NDArray


def select_from_struct(
    struct_array,
    sample_id=None,
    asset_id=None,
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
