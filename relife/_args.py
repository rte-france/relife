from __future__ import annotations

from typing import TYPE_CHECKING, Iterator, Union

import numpy as np
from numpy.typing import NDArray

from relife.lifetime_model import FrozenLifetimeRegression

if TYPE_CHECKING:
    from relife import ParametricModel
    from relife.economic import CostStructure

"""
arrays cannot have more than 2 dim

if input is -> output is :
* float -> float
* 1darray, size 1 -> float
* 1darray, size m -> 2darray shape (m, 1) or (1, m)
* 2darray, size 1 -> float
* 2darray, size m*n -> 2darray shape (n, m)
"""




def get_nb_assets(*args: float | NDArray[np.float64]) -> int:
    def get_nb_asset(x: float | NDArray[np.float64]):
        if isinstance(x, np.ndarray):
            return x.shape[0]
        else:
            return 1

    return max(map(lambda x: get_nb_asset(x), args), default=1)



