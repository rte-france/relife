"""
This module defines optimizers of likelihoods used in distributions

Copyright (c) 2022, RTE (https://www.rte-france.com)
See AUTHORS.txt
SPDX-License-Identifier: Apache-2.0 (see LICENSE.txt)
"""

import copy
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import Bounds, minimize

from relife2.survival.distributions.types import DistributionOptimizer
from relife2.survival.parameters import Parameters

FloatArray = NDArray[np.float64]


class GenericDistributionOptimizer(DistributionOptimizer):
    """
    BLABLABLABLA
    """

    def init_params(self) -> FloatArray:
        nb_params = self.likelihood.functions.params.size
        param0 = np.ones(nb_params)
        param0[-1] = 1 / np.median(
            np.concatenate(
                [
                    self.likelihood.complete_lifetimes.values,
                    self.likelihood.left_censorships.values,
                    self.likelihood.right_censorships.values,
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
        x,
    ):
        self.likelihood.functions.params.values = x
        return self.likelihood.negative_log_likelihood()

    def _jac(
        self,
        x,
    ):
        self.likelihood.functions.params.values = x
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

        return copy.deepcopy(self.likelihood.functions.params)


class GompertzOptimizer(DistributionOptimizer):
    """
    BLABLABLABLA
    """

    def init_params(self) -> FloatArray:
        nb_params = self.likelihood.functions.params.size
        param0 = np.empty(nb_params, dtype=np.float64)
        rate = np.pi / (
            np.sqrt(6)
            * np.std(
                np.concatenate(
                    (
                        self.likelihood.complete_lifetimes.values,
                        self.likelihood.left_censorships.values,
                        self.likelihood.right_censorships.values,
                    ),
                    axis=0,
                )
            )
        )

        shape = np.exp(
            -rate
            * np.mean(
                np.concatenate(
                    (
                        self.likelihood.complete_lifetimes.values,
                        self.likelihood.left_censorships.values,
                        self.likelihood.right_censorships.values,
                    ),
                    axis=0,
                )
            )
        )

        param0[0] = shape
        param0[1] = rate

        return param0

    def get_params_bounds(self) -> Bounds:
        return Bounds(
            np.full(self.likelihood.functions.params.size, np.finfo(float).resolution),
            np.full(self.likelihood.functions.params.size, np.inf),
        )

    def _func(
        self,
        x,
    ):
        self.likelihood.functions.params.values = x
        return self.likelihood.negative_log_likelihood()

    def _jac(
        self,
        x,
    ):
        self.likelihood.functions.params.values = x
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

        return copy.deepcopy(self.likelihood.functions.params)
