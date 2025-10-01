from typing import (
    TypeAlias,
    TypeVarTuple,
    Union,
)

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "_NumpyArray_OfNumber",
    "_NumpyArray_OfBool",
    "_Any_Numpy_Number",
    "_Any_Numpy_Bool",
    "_Any_Integer",
    "_Any_Float",
    "_Any_Number",
    "_Any_Number_Ts",
]

_NumpyArray_OfNumber: TypeAlias = Union[NDArray[np.int64], NDArray[np.float64]]
_NumpyArray_OfBool: TypeAlias = NDArray[np.bool_]

_Any_Numpy_Number: TypeAlias = Union[np.int64, np.float64, _NumpyArray_OfNumber]
_Any_Numpy_Bool: TypeAlias = Union[np.bool, _NumpyArray_OfBool]

_Any_Integer: TypeAlias = Union[int, np.int64, NDArray[np.int64]]
_Any_Float: TypeAlias = Union[float, np.float64, NDArray[np.float64]]
_Any_Number: TypeAlias = Union[_Any_Integer, _Any_Float]

_Any_Number_Ts = TypeVarTuple("_Any_Number_Ts")
