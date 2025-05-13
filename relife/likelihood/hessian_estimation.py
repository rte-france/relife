from __future__ import annotations

from typing import TYPE_CHECKING, Union

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import approx_fprime

if TYPE_CHECKING:
    from relife.lifetime_model import FittableParametricLifetimeModel
    from relife.likelihood import Likelihood


def hessian_cs(
    likelihood: Likelihood,
    eps: float = 1e-6,
) -> Union[NDArray[np.float64], None]:
    """

    Args:
        likelihood ():
        eps ():

    Returns:

    """
    if likelihood.hasjac:
        size = likelihood.params.size
        hess = np.empty((size, size))
        u = eps * 1j * np.eye(size)
        # params = likelihood.params.copy()
        params = np.copy(likelihood.params).astype(np.complex64)
        for i in range(size):
            for j in range(i, size):
                hess[i, j] = np.imag(likelihood.jac_negative_log(params + u[i])[j]) / eps
                if i != j:
                    hess[j, i] = hess[i, j]
        return hess
    return None


def hessian_2point(
    likelihood: Likelihood,
    eps: float = 1e-6,
) -> Union[NDArray[np.float64], None]:
    """

    Args:
        likelihood ():
        eps ():

    Returns:

    """
    if likelihood.hasjac:
        size = likelihood.params.size
        params = likelihood.params.copy()
        hess = np.empty((size, size))
        for i in range(size):
            hess[i] = approx_fprime(
                params,
                lambda x: likelihood.jac_negative_log(x)[i],
                eps,
            )
        return hess
    return None


def _hessian_scheme(model: FittableParametricLifetimeModel):
    from relife.lifetime_model import Gamma, LifetimeRegression

    if isinstance(model, LifetimeRegression):
        return _hessian_scheme(model.baseline)
    if isinstance(model, Gamma):
        return hessian_2point
    return hessian_cs
