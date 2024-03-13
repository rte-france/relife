import numpy as np
from scipy.optimize import Bounds, OptimizeResult, minimize

from ..core import ParametricOptimizer
from .function import ParametricFunction
from .likelihood import ParametricLikelihood

MIN_POSITIVE_FLOAT = np.finfo(float).resolution


class DistOptimizer(ParametricOptimizer):
    def __init__(self, likelihood: ParametricLikelihood):
        super().__init__(likelihood)
        # relife/parametric.ParametricHazardFunction
        self._default_method: str = (  #: Default method for minimizing the negative log-likelihood.
            "L-BFGS-B"
        )

    # relife/distribution.ParametricLifetimeDistbution
    def _init_param(self, functions: ParametricFunction) -> np.ndarray:
        param0 = np.ones(functions.params.nb_params)
        param0[-1] = 1 / np.median(
            np.concatenate(
                [
                    data.values
                    for data in self.likelihood.databook(
                        "complete | right_censored | left_censored"
                    )
                ]
            )
        )
        return param0

    def _get_param_bounds(self, functions: ParametricFunction) -> Bounds:
        return Bounds(
            np.full(functions.params.nb_params, MIN_POSITIVE_FLOAT),
            np.full(functions.params.nb_params, np.inf),
        )

    def _func(
        self,
        x,
        functions: ParametricFunction,
    ):
        functions.params.values = x
        return self.likelihood.negative_log_likelihood(functions)

    def _jac(
        self,
        x,
        functions: ParametricFunction,
    ):
        functions.params.values = x
        return self.likelihood.jac_negative_log_likelihood(functions)

    # relife/parametric.ParametricHazardFunctions
    def fit(
        self,
        functions: ParametricFunction,
        param0: np.ndarray = None,
        bounds=None,
        method: str = None,
        **kwargs,
    ) -> OptimizeResult:

        if param0 is not None:
            param0 = np.asanyarray(param0, float)
            if functions.nb_params != param0.size:
                raise ValueError(
                    "Wrong dimension for param0, expected"
                    f" {functions.nb_params} but got {param0.size}"
                )
        else:
            param0 = self._init_param(functions)
        if method is None:
            method = self._default_method
        if bounds is None:
            bounds = self._get_param_bounds(functions)

        opt = minimize(
            self._func,
            param0,
            args=(functions),
            method=method,
            jac=self._jac,
            bounds=bounds,
            **kwargs,
        )
        functions.params.values = opt.x

        return functions, opt


class GompertzOptimizer(DistOptimizer):
    def __init__(self, likelihood: ParametricLikelihood):
        super().__init__(likelihood)

    def _init_param(self, functions: ParametricFunction) -> np.ndarray:
        rate = np.pi / (
            np.sqrt(6)
            * np.std(
                np.concatenate(
                    [
                        data.values
                        for data in self.likelihood.databook(
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
                        for data in self.likelihood.databook(
                            "complete | right_censored | left_censored"
                        )
                    ]
                )
            )
        )
        return np.array([c, rate])
