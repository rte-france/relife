"""
This module defines optimizers used for estimations of parameters

Copyright (c) 2022, RTE (https://www.rte-france.com)
See AUTHORS.txt
SPDX-License-Identifier: Apache-2.0 (see LICENSE.txt)
"""

from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import Bounds, minimize

from relife2.survival.types import Likelihood, Parameters

FloatArray = NDArray[np.float64]


class LikelihoodOptimizer:
    """BLABLABLA"""

    def __init__(
        self,
        likelihood: Likelihood,
        param0: Optional[FloatArray] = None,
        bounds: Optional[Bounds] = None,
        method: str = "L-BFGS-B",
    ):
        """

        Args:
            likelihood ():
        """
        self.likelihood = likelihood
        if param0 is None:
            param0 = np.random.random(self.likelihood.params.size)
        else:
            if param0.shape != self.likelihood.params.values.shape:
                raise ValueError("incompatible param0 shape")
        if bounds is None:
            bounds = self.likelihood.functions.params_bounds
        else:
            if not isinstance(bounds, Bounds):
                raise ValueError("bounds must be scipy.optimize.Bounds instance")

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

        return self.likelihood.params.copy()
