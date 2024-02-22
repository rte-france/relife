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
            if {type(param) == str for param in param_names} != {str}:
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
        self._results = None
        for i in range(self.param_names):
            setattr(self, self.param_names[i], None)

    def sf(self, param: np.ndarray = None):
        if param is not None:
            if len(param) != self.functions.nb_params:
                raise ValueError(f"expected {self.functions.nb_params}")
            return self.functions.sf()
        elif param is None and self.results is None:
            warnings.warn(
                "No fitted model param. Call fit() or specify param first",
                UserWarning,
            )
        else:
            return self.functions.sf(self.results.opt.x)

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
    def results(self):
        if self._results is None:
            warnings.warn(
                "Model parameters have not been fitted. Call fit() method"
                " first",
                UserWarning,
            )
        return self._results


def exponential(databook: Type[DataBook]) -> Type[ParametricDistriModel]:
    return ParametricDistriModel(
        databook,
        ExponentialDistriFunction,
        ExponentialDistriLikelihood,
        param_names=["rate"],
    )


def gompertz(databook: Type[DataBook]) -> Type[ParametricDistriModel]:
    pass
