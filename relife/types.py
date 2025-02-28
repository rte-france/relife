from typing import Optional, TypeVarTuple, Protocol, Callable, Concatenate
import numpy as np
from numpy.typing import NDArray

VariadicArgs = TypeVarTuple("VariadicArgs")

# tuple consisting of zero or more NDArray[np.float64]
TupleArrays = tuple[Optional[NDArray[np.float64]], ...]
