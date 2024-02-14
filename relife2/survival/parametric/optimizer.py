from abc import ABC, abstractmethod

import numpy as np
from scipy.optimize import Bounds, minimize

from .. import SurvivalData
from .function import ParametricFunction
from .likelihood import ParametricLikelihood

MIN_POSITIVE_FLOAT = np.finfo(float).resolution


class ParametricOptimizer(ABC):
    @abstractmethod
    def fit(self) -> None:
        pass


class DistriOptimizer(ParametricOptimizer):
    # relife/parametric.ParametricHazardFunction
    _default_method: str = (
        "L-BFGS-B"  #: Default method for minimizing the negative log-likelihood.
    )
    # relife/distribution.ParametricLifetimeDistribution

    def _init_param(
        self, data: SurvivalData, functions: ParametricFunction
    ) -> np.ndarray:
        param0 = np.ones(functions.nb_param)
        param0[-1] = 1 / np.median(data.lifetimes)
        return param0

    def get_param_bounds(self, functions: ParametricFunction) -> Bounds:
        return Bounds(
            np.full(functions.nb_param, MIN_POSITIVE_FLOAT),
            np.full(functions.nb_param, np.inf),
        )

    # relife/parametric.ParametricHazardFunctions
    def fit(
        self,
        data: SurvivalData,
        functions: ParametricFunction,
        likelihood: ParametricLikelihood,
        param0: np.ndarray = None,
        bounds=None,
        method: str = None,
        eps: float = 1e-6,
        hess_scheme: str = None,
        **kwargs,
    ) -> np.ndarray:

        if param0 is not None:
            param0 = np.asanyarray(param0, float)
            if functions.nb_param != param0.size:
                raise ValueError(
                    f"Wrong dimension for param0, expected {functions.nb_param} but got {param0.size}"
                )
        else:
            param0 = self._init_param(data)
        if method is None:
            method = self._default_method
        if bounds is None:
            bounds = self._get_param_bounds(functions)

        opt = minimize(
            self._negative_log_likelihood,
            param0,
            args=(data, functions),
            method=method,
            jac=likelihood.jac,
            bounds=bounds,
            **kwargs,
        )

        return opt.x

        # var = np.linalg.inv(
        #     self._hess_negative_log_likelihood(opt.x, data, functions, eps, hess_scheme)
        # )
        # self._set_params(opt.x)
