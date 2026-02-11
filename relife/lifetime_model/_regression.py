"""Lifetime regression

Notes
-----
This module contains two parametric lifetime regressions.
ProportionalHazard is not Cox regression (Cox is semiparametric).
"""

from __future__ import annotations

from typing import Literal, final

import numpy as np
from typing_extensions import overload

from relife.base import ParametricModel
from relife.typing import (
    AnyFloat,
    NumpyFloat,
)

__all__: list[str] = []


def _broadcast_time_covar(time: AnyFloat, covar: AnyFloat) -> tuple[NumpyFloat, NumpyFloat]:
    time = np.atleast_2d(np.asarray(time))  #  (m, n)
    covar = np.atleast_2d(np.asarray(covar))  #  (m, nb_coef)
    match (time.shape[0], covar.shape[0]):
        case (1, _):
            time = np.repeat(time, covar.shape[0], axis=0)
        case (_, 1):
            covar = np.repeat(covar, time.shape[0], axis=0)
        case (m1, m2) if m1 != m2:
            raise ValueError(f"Incompatible time and covar. time has {m1} nb_assets but covar has {m2} nb_assets")
        case _:
            pass
    return time, covar


def _broadcast_time_covar_shapes(time_shape: tuple[int, ...], covar_shape: tuple[int, ...]) -> tuple[int, ...]:
    """
    time_shape : (), (n,) or (m, n)
    covar_shape : (), (nb_coef,) or (m, nb_coef)
    """
    match [time_shape, covar_shape]:
        case [(), ()] | [(), (_,)]:
            return ()
        case [(), (m, _)]:
            return m, 1
        case [(n,), ()] | [(n,), (_,)]:
            return (n,)
        case [(n,), (m, _)] | [(m, n), ()] | [(m, n), (_,)]:
            return m, n
        case [(mt, n), (mc, _)] if mt != mc:
            if mt != 1 and mc != 1:
                raise ValueError(f"Invalid time and covar : time got {mt} nb assets but covar got {mc} nb assets")
            return max(mt, mc), n
        case [(mt, n), (mc, _)] if mt == mc:
            return mt, n
        case _:
            raise ValueError(f"Invalid time or covar shape. Got {time_shape} and {covar_shape}")


@final
class LinearCovarEffect(ParametricModel):
    """
    Covariates effect.

    Parameters
    ----------
    *coefficients : float
        Coefficients of the covariates effect.
    """

    def __init__(self, coefficients: tuple[float | None, ...] = (None,)):
        super().__init__(**{f"coef_{i + 1}": v for i, v in enumerate(coefficients)})

    @property
    def nb_coef(self) -> int:
        """
        The number of coefficients

        Returns
        -------
        int
        """
        return self.nb_params

    def g(self, covar: AnyFloat) -> NumpyFloat:
        """
        Compute the covariates effect.
        If covar.shape : () or (nb_coef,) => out.shape : (), float
        If covar.shape : (m, nb_coef) => out.shape : (m, 1)
        """
        arr_covar = np.asarray(covar)  # (), (nb_coef,) or (m, nb_coef)
        if arr_covar.ndim > 2:
            raise ValueError(f"Invalid covar shape. Expected (nb_coef,) or (m, nb_coef) but got {arr_covar.shape}")
        covar_nb_coef = arr_covar.size if arr_covar.ndim <= 1 else arr_covar.shape[-1]
        if covar_nb_coef != self.nb_coef:
            raise ValueError(
                f"Invalid covar. Number of covar does not match number of coefficients. Got {self.nb_coef} nb_coef but covar shape is {arr_covar.shape}"
            )
        g = np.exp(np.sum(self.params * arr_covar, axis=-1, keepdims=True))  # (m, 1)
        if arr_covar.ndim <= 1:
            return np.float64(g.item())
        return g

    @overload
    def jac_g(self, covar: AnyFloat, asarray: Literal[True]) -> tuple[NumpyFloat, ...]: ...
    @overload
    def jac_g(self, covar: AnyFloat, asarray: Literal[False]) -> NumpyFloat: ...
    @overload
    def jac_g(self, covar: AnyFloat, asarray: bool = False) -> tuple[NumpyFloat, ...] | NumpyFloat: ...
    def jac_g(self, covar: AnyFloat, asarray: bool = False) -> tuple[NumpyFloat, ...] | NumpyFloat:
        """
        Compute the Jacobian of the covariates effect.
        If covar.shape : () or (nb_coef,) => out.shape : (nb_coef,)
        If covar.shape : (m, nb_coef) => out.shape : (nb_coef, m, 1)
        """
        arr_covar = np.asarray(covar)  # (), (nb_coef,) or (m, nb_coef)
        g = self.g(arr_covar)  # () or (m, 1)
        jac = arr_covar.T.reshape(self.nb_coef, -1, 1) * g  # (nb_coef, m, 1)
        if arr_covar.ndim <= 1:
            jac = jac.reshape(self.nb_coef)  # (nb_coef,) or (nb_coef, m, 1)
        if not asarray:
            return np.unstack(jac, axis=0)  # tuple
        return jac  # (nb_coef, m, 1)


