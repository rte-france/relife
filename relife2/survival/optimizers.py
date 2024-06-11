"""
This module defines optimizers used for estimations of parameters

Copyright (c) 2022, RTE (https://www.rte-france.com)
See AUTHORS.txt
SPDX-License-Identifier: Apache-2.0 (see LICENSE.txt)
"""

import copy
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import Bounds, approx_fprime, minimize

from relife2.survival.parameters import Parameters

FloatArray = NDArray[np.float64]


class LikelihoodOptimizer(ABC):
    """BLABLABLA"""

    method: str = "L-BFGS-B"

    def __init__(self, likelihood):
        """

        Args:
            likelihood ():
        """
        self.likelihood = likelihood
        self.param0 = self.init_params()
        self.bounds = self.get_params_bounds()

    @abstractmethod
    def update_params(self, new_values: FloatArray) -> None:
        """BLABLABLA"""

    def init_params(self) -> FloatArray:
        """

        Returns:

        """
        nb_params = self.likelihood.functions.params.size
        param0 = np.ones(nb_params)
        param0[-1] = 1 / np.median(self.likelihood.observed_lifetimes.rlc.values)
        return param0

    def get_params_bounds(self) -> Bounds:
        """

        Returns:

        """
        return Bounds(
            np.full(self.likelihood.functions.params.size, np.finfo(float).resolution),
            np.full(self.likelihood.functions.params.size, np.inf),
        )

    def func(self, x: FloatArray) -> float:
        """

        Args:
            x ():

        Returns:

        """
        self.update_params(x)
        return self.likelihood.negative_log_likelihood()

    def jac(self, x: FloatArray) -> FloatArray:
        """

        Args:
            x ():

        Returns:

        """
        self.update_params(x)
        return self.likelihood.jac_negative_log_likelihood()

    def fit(
        self,
        param0: Optional[FloatArray] = None,
        bounds=None,
        method: Optional[str] = None,
        **kwargs,
    ) -> Parameters:
        """

        Args:
            param0 ():
            bounds ():
            method ():
            **kwargs ():

        Returns:

        """

        if param0 is not None:
            param0 = np.asanyarray(param0, float)
            if param0 != self.param0.size:
                raise ValueError(
                    "Wrong dimension for param0, expected"
                    f" {self.likelihood.functions.params.size} but got {param0.size}"
                )
            self.param0 = param0
        if bounds is not None:
            if not isinstance(bounds, Bounds):
                raise ValueError("bounds must be scipy.optimize.Bounds instance")
            self.bounds = bounds
        if method is not None:
            self.method = method

        opt = minimize(
            self.func,
            self.param0,
            method=self.method,
            jac=self.jac,
            bounds=self.bounds,
            **kwargs,
        )
        self.update_params(opt.x)

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
