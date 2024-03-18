from typing import Tuple, TypeVar

import numpy as np
from scipy.optimize import Bounds, OptimizeResult, minimize

from ..backbone import ParametricOptimizer
from .functions import DistFunctions
from .likelihood import DistLikelihood

Functions = TypeVar("Functions", bound=DistFunctions)
Likelihood = TypeVar("Likelihood", bound=DistLikelihood)


class DistOptimizer(ParametricOptimizer):
    def __init__(self, functions: Functions, likelihood: Likelihood):
        super().__init__(functions, likelihood)

    def _func(
        self,
        x,
        functions: Functions,
        likelihood: Likelihood,
    ):
        functions.params.values = x
        return likelihood.negative_log_likelihood(functions)

    def _jac(
        self,
        x,
        functions: Functions,
        likelihood: Likelihood,
    ):
        functions.params.values = x
        return likelihood.jac_negative_log_likelihood(functions)

    # relife/parametric.ParametricHazardFunctions
    def fit(
        self,
        functions: Functions,
        likelihood: Likelihood,
        param0: np.ndarray = None,
        bounds=None,
        method: str = None,
        **kwargs,
    ) -> Tuple[Functions, OptimizeResult]:

        if param0 is not None:
            param0 = np.asanyarray(param0, float)
            if param0 != self.param0.size:
                raise ValueError(
                    "Wrong dimension for param0, expected"
                    f" {functions.nb_params} but got {param0.size}"
                )
            self.param0 = param0
        if bounds is not None:
            if not isinstance(bounds, Bounds):
                raise ValueError(
                    "bounds must be scipy.optimize.Bounds instance"
                )
            self.bounds = bounds
        if method is not None:
            if type(method) != str:
                raise ValueError("method must be str")
            self.method = method

        opt = minimize(
            self._func,
            self.param0,
            args=(functions, likelihood),
            method=self.method,
            jac=self._jac,
            bounds=self.bounds,
            **kwargs,
        )
        functions.params.values = opt.x

        return functions, opt


class GompertzOptimizer(DistOptimizer):
    def __init__(self, functions: Functions, likelihood: Likelihood):
        super().__init__(functions, likelihood)

    def _init_param(
        self, likelihood: Likelihood, nb_params: int
    ) -> np.ndarray:
        param0 = np.empty(nb_params)
        rate = np.pi / (
            np.sqrt(6)
            * np.std(
                np.concatenate(
                    [
                        data.values
                        for data in likelihood.data(
                            "complete | right_censored | left_censored"
                        )
                    ]
                )
            )
        )

        c = np.exp(
            -rate
            * np.mean(
                np.concatenate(
                    [
                        data.values
                        for data in likelihood.data(
                            "complete | right_censored | left_censored"
                        )
                    ]
                )
            )
        )

        param0[0] = c
        param0[1] = rate

        return param0
