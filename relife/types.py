from typing import TypeVarTuple, Union, NewType, Tuple
import numpy as np
from numpy.typing import NDArray

VariadicArgs = TypeVarTuple("VariadicArgs")


# NDArray[np.float64] = np.ndarray[tuple[int, ...], numpy.dtype[numpy.float64]]
# == np.ndarray of variable shape (tuple[int, ...]) and whose values are of type np.float64


# float >: int
# float = np.float64
# bool = np.bool_
# see : https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
Args = NewType(
    "Args", Union[NDArray[np.floating], NDArray[np.integer], NDArray[np.bool], float]
)
# tuple consisting of zero or more args
