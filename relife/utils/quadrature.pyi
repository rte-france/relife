from __future__ import annotations

from typing import (
    Callable,
    Optional,
    Union,
    overload,
)

import numpy as np
from numpy.typing import NDArray

__all__ = ["legendre_quadrature", "laguerre_quadrature", "unweighted_laguerre_quadrature", "broadcast_bounds"]

@overload
def broadcast_bounds(
    a: float | NDArray[np.float64],
) -> NDArray[np.float64]: ...
@overload
def broadcast_bounds(
    a: float | NDArray[np.float64],
    b: float | NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
def broadcast_bounds(
    a: float | NDArray[np.float64],
    b: Optional[float | NDArray[np.float64]] = None,
) -> Union[NDArray[np.float64], tuple[NDArray[np.float64], NDArray[np.float64]]]: ...
def _control_shape(bound: float | NDArray[np.float64]) -> NDArray[np.float64]: ...
def legendre_quadrature(
    func: Callable[[float | NDArray[np.float64]], np.float64 | NDArray[np.float64]],
    a: float | NDArray[np.float64],
    b: float | NDArray[np.float64],
    deg: int = 10,
) -> np.float64 | NDArray[np.float64]: ...
def laguerre_quadrature(
    func: Callable[[float | NDArray[np.float64]], np.float64 | NDArray[np.float64]],
    a: float | NDArray[np.float64] = 0.0,
    deg: int = 10,
) -> np.float64 | NDArray[np.float64]: ...
def unweighted_laguerre_quadrature(
    func: Callable[[float | NDArray[np.float64]], np.float64 | NDArray[np.float64]],
    a: float | NDArray[np.float64] = 0.0,
    deg: int = 10,
) -> np.float64 | NDArray[np.float64]: ...
