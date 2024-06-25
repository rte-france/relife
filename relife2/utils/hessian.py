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

from relife2.likelihoods import Likelihood

FloatArray = NDArray[np.float64]


def hessian_cs(
    likelihood: Likelihood,
    eps: float = 1e-6,
) -> Union[FloatArray, None]:
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
        for i in range(size):
            for j in range(i, size):
                likelihood.params = likelihood.params.values + u[i]
                hess[i, j] = np.imag(likelihood.jac_negative_log_likelihood()[j]) / eps
                likelihood.params = init_params
                if i != j:
                    hess[j, i] = hess[i, j]

        return hess
    return None


def hessian_2point(
    likelihood: Likelihood,
    eps: float = 1e-6,
) -> Union[FloatArray, None]:
    """

    Args:
        likelihood ():
        eps ():

    Returns:

    """
    if likelihood.hasjac:
        size = likelihood.params.size
        hess = np.empty((size, size))
        for i in range(size):
            hess[i] = approx_fprime(
                likelihood.params.values,
                likelihood.jac_negative_log_likelihood()[i],
                eps,
            )
        return hess
    return None
