from typing import Callable, Literal, NotRequired, TypeAlias, TypedDict

import numpy as np
from optype.numpy import Array1D, ToFloat, ToFloat1D, ToFloat2D
from scipy.optimize import Bounds

__all__ = ["ScipyMinimizeOptions", "MethodMinimize"]

_1DArray: TypeAlias = np.ndarray[tuple[int], np.dtype[np.float64]]
_2DArray: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.float64]]

# ref : scipy-stubs/optimize/_typing.pyi
MethodMinimize: TypeAlias = Literal[
    "Nelder-Mead",
    "nelder-mead",
    "Powell",
    "powell",
    "CG",
    "cg",
    "BFGS",
    "bfgs",
    "Newton-CG",
    "newton-cg",
    "L-BFGS-B",
    "l-bfgs-b",
    "TNC",
    "tnc",
    "COBYLA",
    "cobyla",
    "COBYQA",
    "cobyqa",
    "SLSQP",
    "slsqp",
    "Trust-Constr",
    "trust-constr",
    "Dogleg",
    "dogleg",
    "Trust-NCG",
    "trust-ncg",
    "Trust-Exact",
    "trust-exact",
    "Trust-Krylov",
    "trust-krylov",
]


class ScipyMinimizeOptions(TypedDict):
    x0: NotRequired[ToFloat | ToFloat1D]
    method: NotRequired[MethodMinimize]
    bounds: NotRequired[Bounds | None]
    jac: NotRequired[Callable[[Array1D[np.float64]], ToFloat1D] | None]
    hess: NotRequired[Callable[[Array1D[np.float64]], ToFloat2D] | None]
