"""
This module defines optimizers of likelihoods used in regressions

Copyright (c) 2022, RTE (https://www.rte-france.com)
See AUTHORS.txt
SPDX-License-Identifier: Apache-2.0 (see LICENSE.txt)
"""

import copy
from typing import Optional

import numpy as np
from scipy.optimize import Bounds, minimize

from relife2.survival.parameters import Parameters
from relife2.survival.regressions.types import FloatArray, RegressionOptimizer


class GenericRegressionOptimizer(RegressionOptimizer):
    """BLABLABLA"""

    def init_params(self) -> FloatArray:
        nb_params = self.likelihood.functions.params.size
        param0 = np.ones(nb_params)
        param0[-1] = 1 / np.median(
            np.concatenate(
                [
                    self.likelihood.observed_lifetimes.complete.values,
                    self.likelihood.observed_lifetimes.left_censored.values,
                    self.likelihood.observed_lifetimes.right_censored.values,
                ]
            )
        )
        return param0

    def get_params_bounds(self) -> Bounds:
        return Bounds(
            np.full(self.likelihood.functions.params.size, np.finfo(float).resolution),
            np.full(self.likelihood.functions.params.size, np.inf),
        )

    def _func(
        self,
        x: FloatArray,
    ) -> float:

        self.likelihood.functions.params.values = x
        self.likelihood.functions.baseline.params.values = x[
            : self.likelihood.functions.baseline.params.size
        ]
        self.likelihood.functions.covar_effect.params.values = x[
            self.likelihood.functions.covar_effect.params.size - 1 :
        ]
        return self.likelihood.negative_log_likelihood()

    def _jac(
        self,
        x: FloatArray,
    ) -> FloatArray:
        self.likelihood.functions.params.values = x
        self.likelihood.functions.baseline.params.values = x[
            : self.likelihood.functions.baseline.params.size
        ]
        self.likelihood.functions.covar_effect.params.values = x[
            self.likelihood.functions.covar_effect.params.size - 1 :
        ]
        return self.likelihood.jac_negative_log_likelihood()

    def fit(
        self,
        param0: Optional[FloatArray] = None,
        bounds=None,
        method: Optional[str] = None,
        **kwargs,
    ) -> Parameters:

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
            self._func,
            self.param0,
            method=self.method,
            jac=self._jac,
            bounds=self.bounds,
            **kwargs,
        )
        self.likelihood.functions.params.values = opt.x
        self.likelihood.functions.baseline.params.values = opt.x[
            : self.likelihood.functions.baseline.params.size
        ]
        self.likelihood.functions.covar_effect.params.values = opt.x[
            self.likelihood.functions.covar_effect.params.size - 1 :
        ]

        return copy.deepcopy(self.likelihood.functions.params)
