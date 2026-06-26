from __future__ import annotations

from typing import TypeAlias, TypeVarTuple

import numpy as np
from optype.numpy import ArrayND

ST: TypeAlias = int | float
NumpyST: TypeAlias = np.floating | np.uint

Ts = TypeVarTuple("Ts")
VT: TypeAlias = ST | NumpyST | ArrayND[NumpyST]

# class FittableParametricLifetimeModel(Protocol):
#     """
#     A structural type for any ParametricLifetimeModel having a fit method.
#     """
#
#     def fit(
#         self,
#         time: Array1D[np.float64],
#         *args: Any,
#         event: Array1D[np.bool_] | None = None,
#         entry: Array1D[np.float64] | None = None,
#         **kwargs: Any,
#     ) -> Self: ...
#
#     def dhf(
#         self, time: ST | NumpyST | ArrayND[NumpyST], *args: Any
#     ) -> ArrayND[np.float64]: ...
#
#     def jac_chf(
#         self, time: ST | NumpyST | ArrayND[NumpyST], *args: Any
#     ) -> ArrayND[np.float64]: ...
#
#     def jac_hf(
#         self, time: ST | NumpyST | ArrayND[NumpyST], *args: Any
#     ) -> ArrayND[np.float64]: ...
#
#     def jac_sf(
#         self, time: ST | NumpyST | ArrayND[NumpyST], *args: Any
#     ) -> ArrayND[np.float64]: ...
#
#     def jac_cdf(
#         self, time: ST | NumpyST | ArrayND[NumpyST], *args: Any
#     ) -> ArrayND[np.float64]: ...
#
#     def jac_pdf(
#         self, time: ST | NumpyST | ArrayND[NumpyST], *args: Any
#     ) -> ArrayND[np.float64]: ...
