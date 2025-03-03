from typing import TypeVarTuple, Union, NewType, Tuple
import numpy as np
from numpy.typing import NDArray

VariadicArgs = TypeVarTuple("VariadicArgs")


# NDArray[np.float64] = np.ndarray[tuple[int, ...], numpy.dtype[numpy.float64]]
Arg = NewType("Args", Union[NDArray[np.float64], float, int])
# tuple consisting of zero or more args
