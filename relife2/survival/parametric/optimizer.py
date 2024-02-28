from abc import ABC, abstractmethod
from typing import Type

import numpy as np
from scipy.optimize import Bounds, OptimizeResult, minimize

from .function import ParametricFunction
from .likelihood import ParametricLikelihood

MIN_POSITIVE_FLOAT = np.finfo(float).resolution


class ParametricOptimizer(ABC):
    def __init__(self, likelihood: Type[ParametricLikelihood]):
        if not isinstance(likelihood, ParametricLikelihood):
            raise TypeError("expected ParametricLikelihood")
        self.likelihood = likelihood

    @abstractmethod
    def fit(self) -> None:
        pass


class DistriOptimizer(ParametricOptimizer):
    def __init__(self, likelihood: Type[ParametricLikelihood]):
        super().__init__(likelihood)
        # relife/parametric.ParametricHazardFunction
        self._default_method: str = (  #: Default method for minimizing the negative log-likelihood.
            "L-BFGS-B"
        )

    # relife/distribution.ParametricLifetimeDistribution
    def _init_param(self, functions: ParametricFunction) -> np.ndarray:
        param0 = np.ones(functions.params.nb_params)
        param0[-1] = 1 / np.median(self.likelihood.databook("complete").values)
        return param0

    def _get_param_bounds(self, functions: ParametricFunction) -> Bounds:
        return Bounds(
            np.full(functions.params.nb_params, MIN_POSITIVE_FLOAT),
            np.full(functions.params.nb_params, np.inf),
        )

    def _func(
        self,
        x,
        functions: Type[ParametricFunction],
    ):
        functions.params.values = x
        return self.likelihood.negative_log_likelihood(functions)

    def _jac(
        self,
        x,
        functions: Type[ParametricFunction],
    ):
        functions.params.values = x
        return self.likelihood.jac_negative_log_likelihood(functions)

    # relife/parametric.ParametricHazardFunctions
    def fit(
        self,
        functions: Type[ParametricFunction],
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

        return opt
