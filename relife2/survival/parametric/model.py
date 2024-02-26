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

        # assert issubclass(optimizer, DistriOptimizer)
        self.databook = databook
        self.functions = functions
        self.likelihood = likelihood
        self.optimizer = DistriOptimizer()
        self._fitting_results = None

    def __getattr__(self, attr):
        """
        called if attr is not found in attributes of the class
        (different from __getattribute__)
        """
        # print("Calling __getattr__: " + attr)
        if hasattr(self.functions, attr):

            def wrapper(*args, **kwargs):
                if "params" in kwargs:
                    input_params = kwargs["params"]
                    if len(input_params) != self.functions.params.nb_params:
                        raise ValueError(
                            f"expected {self.functions.params.nb_params} nb of"
                            f" params but got {input_params}"
                        )
                    else:
                        self.functions.params.params = input_params
                elif "params" not in kwargs and self.fitting_results is None:
                    warnings.warn(
                        "No fitted model params. Call fit() or specify params"
                        " first",
                        UserWarning,
                    )
                    return None
                else:
                    self.functions.params.params = self.fitting_params
                # print("called with %r and %r" % (args, kwargs))
                return getattr(self.functions, attr)(*args, **kwargs)

            return wrapper
        else:
            raise AttributeError(f"Model functions has no attr called {attr}")

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
        self._fitting_results = FittingResult(
            opt, jac, var, len(self.databook)
        )

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
        ExponentialDistriFunction(),
        ExponentialDistriLikelihood(),
    )


def gompertz(databook: Type[DataBook]) -> Type[ParametricDistriModel]:
    pass
