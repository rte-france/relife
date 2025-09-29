from typing import (
    TypeAlias,
    TypeVarTuple,
    Union,
)

import numpy as np
from numpy.typing import NDArray

_IntOrFloatValues: TypeAlias = int | float | NDArray[np.int64] | NDArray[np.float64]
_AdditionalIntOrFloatValues = TypeVarTuple("_AdditionalIntOrFloatValues")
_NumpyFloatValues: TypeAlias = np.float64 | NDArray[np.float64]
_BooleanValues: TypeAlias = Union[NDArray[np.bool_], np.bool_]
