"""
This module defines optimizers used for estimations of parameters

Copyright (c) 2022, RTE (https://www.rte-france.com)
See AUTHORS.txt
SPDX-License-Identifier: Apache-2.0 (see LICENSE.txt)
"""

import copy
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import Bounds, approx_fprime, minimize

from relife2.survival.parameters import Parameters

FloatArray = NDArray[np.float64]


class LikelihoodOptimizer:
    """BLABLABLA"""

    def __init__(
        self,
        likelihood,
        param0: Optional[FloatArray] = None,
        bounds: Optional[Bounds] = None,
        method: str = "L-BFGS-B",
    ):
        """

        Args:
            likelihood ():
        """
        self.likelihood = likelihood
        if param0 is not None:
            if not param0.shape == self.likelihood.params.values.shape:
                raise ValueError("incompatible param0 shape")
        else:
            param0 = self.likelihood.initial_params()
        if bounds is not None:
            if not isinstance(bounds, Bounds):
                raise ValueError("bounds must be scipy.optimize.Bounds instance")
        else:
            bounds = self.likelihood.functions.params_bounds
        if method is not None:
            method = method
        self.param0 = param0
        self.bounds = bounds
        self.method = method

    def func(self, x: FloatArray) -> float:
        """

        Args:
            x ():

        Returns:

        """
        self.likelihood.params = x
        return self.likelihood.negative_log_likelihood()

    def jac(self, x: FloatArray) -> FloatArray:
        """

        Args:
            x ():

        Returns:

        """
        self.likelihood.params = x
        return self.likelihood.jac_negative_log_likelihood()

    def fit(
        self,
        **kwargs,
    ) -> Parameters:
        """

        Args:
            **kwargs ():

        Returns:

        """
        opt = minimize(
            self.func,
            self.param0,
            method=self.method,
            jac=self.jac,
            bounds=self.bounds,
            **kwargs,
        )
        self.likelihood.params = opt.x

        return copy.deepcopy(self.likelihood.functions.params)


def hess_negative_log_likelihood(
    likelihood,
    eps: float = 1e-6,
    scheme="cs",
) -> FloatArray:
    """

    Args:
        likelihood ():
        eps ():
        scheme ():

    Returns:

    """

    size = likelihood.functions.params.size
    # print(size)
    hess = np.empty((size, size))
    params_values = likelihood.functions.params.values

    if scheme == "cs":
        u = eps * 1j * np.eye(size)
        for i in range(size):
            for j in range(i, size):
                # print(type(u[i]))
                # print(u[i])
                # print(self.functions.params.values)
                # print(self.functions.params.values + u[i])
                likelihood.functions.params.values = (
                    likelihood.functions.params.values + u[i]
                )
                # print(self.jac_negative_log_likelihood(self.functions))

                hess[i, j] = np.imag(likelihood.jac_negative_log_likelihood()[j]) / eps
                likelihood.functions.params.values = params_values
                if i != j:
                    hess[j, i] = hess[i, j]

    elif scheme == "2-point":

        for i in range(size):
            hess[i] = approx_fprime(
                likelihood.functions.params.values,
                likelihood.jac_negative_log_likelihood()[i],
                eps,
            )
    else:
        raise ValueError("scheme argument must be 'cs' or '2-point'")

    return hess
