import warnings
from typing import Type

import numpy as np

from .. import DataBook
from .function import ExponentialDistriFunction, ParametricDistriFunction
from .likelihood import ExponentialDistriLikelihood, ParametricDistriLikelihood
from .optimizer import DistriOptimizer, FittingResult


class ParametricDistriModel:
    def __init__(
        self,
        databook: Type[DataBook],
        functions: Type[ParametricDistriFunction],
        likelihood: Type[ParametricDistriLikelihood],
        param_names: list = None,
        # optimizer: DistriOptimizer,
    ):

        if not isinstance(databook, DataBook):
            raise TypeError(
                f"DataBook expected, got '{type(databook).__name__}'"
            )
        if not issubclass(functions, ParametricDistriFunction):
            raise TypeError(
                "ParametricDistriFunction expected, got"
                f" '{type(functions).__name__}'"
            )
        if not issubclass(likelihood, ParametricDistriLikelihood):
            raise TypeError(
                "ParametricDistriLikelihood expected, got"
                f" '{type(likelihood).__name__}'"
            )
        if param_names is not None:
            if {type(name) == str for name in param_names} != {str}:
                raise ValueError("param_names must be string")
            if len(param_names) != functions.nb_params:
                raise ValueError(f"expected {functions.nb_params} params")
            self.param_names = param_names
        else:
            self.param_names = [
                f"param_{i}" for i in range(functions.nb_params)
            ]
        # assert issubclass(optimizer, DistriOptimizer)
        self.databook = databook
        self.functions = functions
        self.likelihood = likelihood
        self.optimizer = DistriOptimizer()
        self.param_names = param_names
        self._fitting_results = None
        self._fitting_params = None
        for i in range(self.param_names):
            setattr(self, self.param_names[i], None)

    def _get_params(self, params: np.ndarray):
        if params is not None:
            if len(params) != self.functions.nb_params:
                raise ValueError(f"expected {self.functions.nb_params}")
            return params
        elif params is None and self.fitting_results is None:
            warnings.warn(
                "No fitted model params. Call fit() or specify params first",
                UserWarning,
            )
            return params
        else:
            return self._fitting_results.opt.x

    def sf(self, elapsed_time: np.ndarray, params: np.ndarray = None):
        params = self._get_params(params)
        if params is None:
            return None
        else:
            return self.functions.sf(params, elapsed_time)

    def cdf(self, elapsed_time: np.ndarray, params: np.ndarray = None):
        params = self._get_params(params)
        if params is None:
            return None
        else:
            return self.functions.cdf(params, elapsed_time)

    def pdf(self, elapsed_time: np.ndarray, params: np.ndarray = None):
        params = self._get_params(params)
        if params is None:
            return None
        else:
            return self.functions.pdf(params, elapsed_time)

    def hf(self, elapsed_time: np.ndarray, params: np.ndarray = None):
        params = self._get_params(params)
        if params is None:
            return None
        else:
            return self.functions.hf(params, elapsed_time)

    def chf(self, elapsed_time: np.ndarray, params: np.ndarray = None):
        params = self._get_params(params)
        if params is None:
            return None
        else:
            return self.functions.chf(params, elapsed_time)

    def mean(self, params: np.ndarray = None):
        params = self._get_params(params)
        if params is None:
            return None
        else:
            return self.functions.mean(params)

    def var(self, params: np.ndarray = None):
        params = self._get_params(params)
        if params is None:
            return None
        else:
            return self.functions.var(params)

    def mrl(self, params: np.ndarray = None):
        params = self._get_params(params)
        if params is None:
            return None
        else:
            return self.functions.mrl(params)

    def fit(
        self,
        **kwargs,
    ):
        opt = self.optimizer.fit(
            self.databook, self.functions, self.likelihood, kwargs
        )
        jac = self.likelihood.jac_negative_log_likelihood(
            opt.x, self.databook, self.functions
        )
        var = np.linalg.inv(
            self.likelihood._hess_negative_log_likelihood(
                self.fitted_param, self.databook, kwargs
            )
        )
        for i in range(self.param_names):
            setattr(self, self.param_names[i], opt.x[i])
        self._results = FittingResult(opt, jac, var, len(self.databook))

    @property
    def fitting_results(self):
        if self._fitting_results is None:
            warnings.warn(
                "Model parameters have not been fitted. Call fit() method"
                " first",
                UserWarning,
            )
        return self._fitting_results

    @property
    def fitting_params(self):
        if self._fitting_results is None:
            warnings.warn(
                "Model parameters have not been fitted. Call fit() method"
                " first",
                UserWarning,
            )
        return self._fitting_results.opt.x


def exponential(databook: Type[DataBook]) -> Type[ParametricDistriModel]:
    return ParametricDistriModel(
        databook,
        ExponentialDistriFunction,
        ExponentialDistriLikelihood,
        param_names=["rate"],
    )


def gompertz(databook: Type[DataBook]) -> Type[ParametricDistriModel]:
    pass
