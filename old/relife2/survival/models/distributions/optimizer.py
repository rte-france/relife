from typing import Tuple

import numpy as np
from scipy.optimize import Bounds, OptimizeResult, minimize

from ..backbone import Optimizer
from .functions import DistFunctions
from .likelihood import DistLikelihood


class DistOptimizer(Optimizer):
    def __init__(self, pf: DistFunctions, likelihood: DistLikelihood):
        super().__init__(pf, likelihood)

    def _func(
        self,
        x,
        pf: DistFunctions,
        likelihood: DistLikelihood,
    ):
        pf.params.values = x
        return likelihood.negative_log_likelihood(pf)

    def _jac(
        self,
        x,
        pf: DistFunctions,
        likelihood: DistLikelihood,
    ):
        pf.params.values = x
        return likelihood.jac_negative_log_likelihood(pf)

    # relife/parametric.ParametricHazardFunctions
    def fit(
        self,
        pf: DistFunctions,
        likelihood: DistLikelihood,
        param0: np.ndarray = None,
        bounds=None,
        method: str = None,
        **kwargs,
    ) -> Tuple[DistFunctions, OptimizeResult]:

        if param0 is not None:
            param0 = np.asanyarray(param0, float)
            if param0 != self.param0.size:
                raise ValueError(
                    "Wrong dimension for param0, expected"
                    f" {pf.nb_params} but got {param0.size}"
                )
            self.param0 = param0
        if bounds is not None:
            if not isinstance(bounds, Bounds):
                raise ValueError("bounds must be scipy.optimize.Bounds instance")
            self.bounds = bounds
        if method is not None:
            if type(method) != str:
                raise ValueError("method must be str")
            self.method = method

        opt = minimize(
            self._func,
            self.param0,
            args=(pf, likelihood),
            method=self.method,
            jac=self._jac,
            bounds=self.bounds,
            **kwargs,
        )
        pf.params.values = opt.x

        return pf, opt


class GompertzOptimizer(DistOptimizer):
    def __init__(self, pf: DistFunctions, likelihood: DistLikelihood):
        super().__init__(pf, likelihood)

    def _init_param(self, likelihood: DistLikelihood, nb_params: int) -> np.ndarray:
        param0 = np.empty(nb_params)
        rate = np.pi / (
            np.sqrt(6)
            * np.std(
                np.concatenate(
                    [
                        likelihood.complete_lifetimes.values,
                        likelihood.left_censorships.values,
                        likelihood.right_censorships.values,
                    ]
                )
            )
        )

        c = np.exp(
            -rate
            * np.mean(
                np.concatenate(
                    [
                        likelihood.complete_lifetimes.values,
                        likelihood.left_censorships.values,
                        likelihood.right_censorships.values,
                    ]
                )
            )
        )

        param0[0] = c
        param0[1] = rate

        return param0
