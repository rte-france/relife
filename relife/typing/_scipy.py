from typing import NotRequired, TypedDict

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import Bounds

__all__ = ["ScipyMinimizeOptions"]


class ScipyMinimizeOptions(TypedDict):

    x0: NotRequired[NDArray[np.float64]]
    method: NotRequired[str]
    bounds: NotRequired[Bounds | None]
