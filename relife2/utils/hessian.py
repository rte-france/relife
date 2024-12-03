"""
This module defines functions used for estimation of hessian matrix given likelihood

Copyright (c) 2022, RTE (https://www.rte-france.com)
See AUTHORS.txt
SPDX-License-Identifier: Apache-2.0 (see LICENSE.txt)
"""

from typing import Union

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import approx_fprime

from relife2.fiability import Likelihood

FloatArray = NDArray[np.float64]


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
        init_params = likelihood.params.copy()
        params = likelihood.params.copy()
        for i in range(size):
            for j in range(i, size):
                params += u[i]
                hess[i, j] = np.imag(likelihood.jac_negative_log(params)[j]) / eps
                params = init_params
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
                likelihood.jac_negative_log(params)[i],
                eps,
            )
        return hess
    return None


def hessian_from_likelihood(method: str):
    match method:
        case "2-point":
            return hessian_2point
        case "cs":
            return hessian_cs
        case _:
            return hessian_2point
