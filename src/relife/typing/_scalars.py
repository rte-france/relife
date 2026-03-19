from typing import (
    TypeAlias,
)

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "NumpyFloat",
    "NumpyBool",
    "AnyFloat",
]


# np.float64 is a subtype of float. See np.float64.mro()
AnyFloat: TypeAlias = float | NDArray[np.float64]
NumpyFloat: TypeAlias = np.float64 | NDArray[np.float64]
NumpyBool: TypeAlias = np.bool | NDArray[np.bool_]
