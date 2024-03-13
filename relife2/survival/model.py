import warnings
from typing import Type

import numpy as np

from ..data.base import databook
from .distribution import (
    DistOptimizer,
    ParametricDistFunction,
    ParametricDistLikelihood,
)
from .parameter import FittingResult, Parameter


class ParametricDistModel:
    def __init__(
        self,
        functions: ParametricDistFunction,
        Likelihood: Type[ParametricDistLikelihood],
        Optimizer: Type[DistOptimizer],
        init_params=False,
        # # optimizer: DistOptimizer,
    ):

        # assert issubclass(optimizer, DistOptimizer)
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
        jac = optimizer.likelihood.jac_negative_log_likelihood(self.functions)
        var = np.linalg.inv(
            optimizer.likelihood.hess_negative_log_likelihood(
                self.functions, **kwargs
            )
        )
        self._fitting_results = FittingResult(
            opt,
            jac,
            var,
            len(db),
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


def dist(
    Function: Type[ParametricDistFunction],
    Likelihood: Type[ParametricDistLikelihood],
    Optimizer: Type[DistOptimizer],
):
    """Return a distribution object.

    n is the number of input samples

    p is the number of model's parameters

    Args:
        Function (Type[ParametricDistFunction]):
            Function object. Must inherit from **ParametricDistFunction**. Mandatory methods are:

                ``hf(time : np.ndarray) -> np.ndarray``, shape (n,) -> (n,)

                ``chf(time : np.ndarray) -> np.ndarray``, shape (n,) -> (n,)

                ``mean() -> float``

                ``var() -> float``

                ``mrl(time : np.ndarray) -> np.ndarray``, shape (n,) -> (n,)

                ``ichf(cumulative_hazard_rate : np.ndarray) -> np.ndarray``, shape (n,) -> (n,)

        Likelihood (Type[ParametricDistLikelihood]):
            Likelihood object. Must inherit from **ParametricDistLikelihood**.  Mandatory methods are:

                ``jac_hf(time : np.ndarray, functions : ParametricDistFunction) -> np.ndarray``, shape (n,) -> (n,p)

                ``jac_chf(time : np.ndarray, functions : ParametricDistFunction) -> np.ndarray``, shape (n,) -> (n,p)

        Optimizer (Type[DistOptimizer]):
            Optimizer object. Must inherit from **ParametricOptimizer**.  Mandatory methods are:

                ``fit(functions : ParametricDistFunction, *args, **kwargs) -> ParametricDistFunction``

    Examples:
        >>> exponential = dist(
                ExponentialDistFunction,
                ExponentialDistLikelihood,
                DistOptimizer,
            )
        >>> exp_dist = exponential(rate=0.007)

    """

    if not issubclass(Function, ParametricDistFunction):
        parent_classes = (Cls.__name__ for Cls in Function.__bases__)
        raise ValueError(
            f"ParametricDistFunction subclass expected, got '{parent_classes}'"
        )

    if not issubclass(Likelihood, ParametricDistLikelihood):
        parent_classes = (Cls.__name__ for Cls in Function.__bases__)
        raise ValueError(
            "ParametricDistLikelihood subclass expected, got"
            f" '{parent_classes}'"
        )

    if not issubclass(Optimizer, DistOptimizer):
        parent_classes = (Cls.__name__ for Cls in Function.__bases__)
        raise ValueError(
            f"DistOptimizer subclass expected, got {Optimizer.__name__}"
            f" '{parent_classes}'"
        )

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

        return ParametricDistModel(
            functions,
            Likelihood,
            Optimizer,
            init_params=init_params,
        )

    return custom_dist
