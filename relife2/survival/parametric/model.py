import warnings
from typing import Type

import numpy as np

from .. import databook
from ..parameter import FittingResult, Parameter
from .function import (
    ExponentialDistriFunction,
    GompertzDistriFunction,
    ParametricDistriFunction,
    WeibullDistriFunction,
)
from .likelihood import (
    ExponentialDistriLikelihood,
    GompertzDistriLikelihood,
    ParametricDistriLikelihood,
    WeibullDistriLikelihood,
)
from .optimizer import DistriOptimizer


class ParametricDistriModel:
    def __init__(
        self,
        functions: ParametricDistriFunction,
        Likelihood: Type[ParametricDistriLikelihood],
        Optimizer: Type[DistriOptimizer],
        init_params=False,
        # # optimizer: DistriOptimizer,
    ):

        # assert issubclass(optimizer, DistriOptimizer)
        self.functions = functions
        self.Likelihood = Likelihood
        self.Optimizer = Optimizer
        self._fitting_results = None
        self._fitting_params = Parameter(
            functions.params.nb_params, functions.params.param_names
        )
        self.init_params = init_params
        self.init_params_values = self.functions.params.values

    def __getattr__(self, attr):
        """
        called if attr is not found in attributes of the class
        (different from __getattribute__)
        """
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
                    res = getattr(self.functions, attr)(*args, **kwargs)
                    self.functions.params.values = self.init_params_values

                elif (
                    "params" not in kwargs and self.fitting_params is not None
                ):
                    res = getattr(self.functions, attr)(*args, **kwargs)
                elif "params" not in kwargs and self.init_params:
                    res = getattr(self.functions, attr)(*args, **kwargs)
                else:
                    raise ValueError(
                        "No model params found. Call fit() or specify params"
                        " first"
                    )
                return res

            return wrapper
        else:
            raise AttributeError(f"Model functions has no attr called {attr}")

    def fit(
        self,
        observed_lifetimes: np.ndarray,
        complete_indicators: np.ndarray = None,
        left_censored_indicators: np.ndarray = None,
        right_censored_indicators: np.ndarray = None,
        entry: np.ndarray = None,
        departure: np.ndarray = None,
        **kwargs,
    ):

        db = databook(
            observed_lifetimes,
            complete_indicators,
            left_censored_indicators,
            right_censored_indicators,
            entry,
            departure,
        )

        likelihood = self.Likelihood(db)
        optimizer = self.Optimizer(likelihood)

        self.functions, opt = optimizer.fit(self.functions, **kwargs)
        jac = self.optimizer.likelihood.jac_negative_log_likelihood(
            self.functions
        )
        var = np.linalg.inv(
            self.optimizer.likelihood.hess_negative_log_likelihood(
                self.functions, **kwargs
            )
        )
        self._fitting_results = FittingResult(
            opt,
            jac,
            var,
            len(self.optimizer.likelihood.databook),
        )
        self._fitting_params.values = opt.x

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


def dist(Function, Likelihood, Optimizer):

    if not issubclass(Function, ParametricDistriFunction):
        parent_classes = (Cls.__name__ for Cls in Function.__bases__)
        raise ValueError(
            "ParametricDistriFunction subclass expected, got"
            f" '{parent_classes}'"
        )

    if not issubclass(Likelihood, ParametricDistriLikelihood):
        parent_classes = (Cls.__name__ for Cls in Function.__bases__)
        raise ValueError(
            "ParametricDistriLikelihood subclass expected, got"
            f" '{parent_classes}'"
        )

    if not Optimizer == DistriOptimizer:
        raise ValueError(f"DistriOptimizer expected, got {Optimizer.__name__}")

    def custom_dist(*params, **kparams):
        functions = Function()
        init_params = False
        if len(params) != 0 and len(kparams) != 0:
            raise ValueError("Can't specify key word params and params")
        elif len(params) != 0:
            params = np.array(params)
            params_shape = params.shape
            functions_params_shape = functions.params.values.shape
            if params.shape != functions.params.values.shape:
                raise ValueError(
                    "Incorrect params shape, expected shape"
                    f" {functions_params_shape} but got {params_shape}"
                )
            functions.params.values = params
            init_params = True

        elif len(kparams) != 0:
            missing_param_names = set(functions.params.param_names).difference(
                set(kparams.keys())
            )
            if len(missing_param_names) != 0:
                warnings.warn(
                    f"""{missing_param_names} params are missing and will be initialized randomly"""
                )
            for key, value in kparams.items():
                functions.params[key] = value
            init_params = True

        return ParametricDistriModel(
            functions,
            Likelihood,
            Optimizer,
            init_params=init_params,
        )

    return custom_dist


exponential = dist(
    ExponentialDistriFunction,
    ExponentialDistriLikelihood,
    DistriOptimizer,
)

weibull = dist(
    WeibullDistriFunction,
    WeibullDistriLikelihood,
    DistriOptimizer,
)


gompertz = dist(
    GompertzDistriFunction,
    GompertzDistriLikelihood,
    DistriOptimizer,
)
