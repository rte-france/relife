import copy
import warnings
from typing import Type

import numpy as np

from .. import DataBook
from ..parameter import FittingResult
from .function import ExponentialDistriFunction, ParametricDistriFunction
from .likelihood import ExponentialDistriLikelihood
from .optimizer import DistriOptimizer


class ParametricDistriModel:
    def __init__(
        self,
        functions: Type[ParametricDistriFunction],
        optimizer: Type[DistriOptimizer],
        # optimizer: DistriOptimizer,
    ):

        if not isinstance(functions, ParametricDistriFunction):
            raise TypeError(
                "ParametricDistriFunction expected, got"
                f" '{type(functions).__name__}'"
            )
        if not isinstance(optimizer, DistriOptimizer):
            raise TypeError(
                "ParametricDistriLikelihood expected, got"
                f" '{type(optimizer).__name__}'"
            )

        # assert issubclass(optimizer, DistriOptimizer)
        self.functions = functions
        self.optimizer = optimizer
        self._fitting_results = None
        self._fitting_params = None

    def __getattr__(self, attr):
        """
        called if attr is not found in attributes of the class
        (different from __getattribute__)
        """
        # print("Calling __getattr__: " + attr)
        if hasattr(self.functions, attr):

            def wrapper(*args, **kwargs):
                if "params" in kwargs:
                    if (
                        type(kwargs["params"]) != list
                        and type(kwargs["params"]) != np.ndarray
                    ):
                        input_params = np.array([kwargs["params"]])
                    elif type(kwargs["params"]) == list:
                        input_params = np.array(kwargs["params"])
                    elif type(kwargs["params"]) == np.ndarray:
                        input_params = kwargs["params"]
                    else:
                        raise TypeError("Incorrect params type")

                    if (
                        input_params.shape
                        != self.functions.params.values.shape
                    ):
                        raise ValueError(
                            f"""
                            expected {self.functions.params.values.shape} shape for
                            params but got {input_params.shape}
                            """
                        )
                    else:
                        self.functions.params.values = input_params
                        del kwargs["params"]
                elif "params" not in kwargs and self.fitting_results is None:
                    warnings.warn(
                        "No fitted model params. Call fit() or specify params"
                        " first",
                        UserWarning,
                    )
                    return None
                else:
                    self.functions.params.values = self.fitting_params
                return getattr(self.functions, attr)(*args, **kwargs)

            return wrapper
        else:
            raise AttributeError(f"Model functions has no attr called {attr}")

    def fit(
        self,
        **kwargs,
    ):
        opt = self.optimizer.fit(self.functions, **kwargs)
        self.functions.params.values = opt.x
        jac = self.optimizer.likelihood.jac_negative_log_likelihood(
            self.functions
        )
        var = np.linalg.inv(
            self.optimizer.likelihood.hess_negative_log_likelihood(
                self.functions, **kwargs
            )
        )
        self._fitting_results = FittingResult(
            opt, jac, var, len(self.optimizer.likelihood.databook)
        )
        self._fitting_params = copy.copy(self.functions.params)

    @property
    def params(self):
        return self.functions.params

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
        if self._fitting_params is None:
            warnings.warn(
                "Model parameters have not been fitted. Call fit() method"
                " first",
                UserWarning,
            )
        return self._fitting_params


def exponential(databook: Type[DataBook]) -> Type[ParametricDistriModel]:
    functions = ExponentialDistriFunction(param_names=["rate"])
    likelihood = ExponentialDistriLikelihood(databook)
    optimizer = DistriOptimizer(likelihood)
    return ParametricDistriModel(
        functions,
        optimizer,
    )


def gompertz(databook: Type[DataBook]) -> Type[ParametricDistriModel]:
    pass
