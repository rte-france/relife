import warnings
from typing import Type

import numpy as np

from .. import DataBook
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
        optimizer: DistriOptimizer,
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
        self._fitting_params = Parameter(
            functions.params.nb_params, functions.params.param_names
        )

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


def exponential(databook: DataBook) -> ParametricDistriModel:
    """Create exponential distribution.

    Args:
        databook (DataBook): _description_

    Returns:
        ParametricDistriModel: _description_
    """
    functions = ExponentialDistriFunction(param_names=["rate"])
    likelihood = ExponentialDistriLikelihood(databook)
    optimizer = DistriOptimizer(likelihood)
    return ParametricDistriModel(
        functions,
        optimizer,
    )


def weibull(databook: DataBook) -> ParametricDistriModel:
    """Create Weibull distribution.

    Args:
        databook (DataBook): _description_

    Returns:
        ParametricDistriModel: _description_
    """
    functions = WeibullDistriFunction(param_names=["c", "rate"])
    likelihood = WeibullDistriLikelihood(databook)
    optimizer = DistriOptimizer(likelihood)
    return ParametricDistriModel(
        functions,
        optimizer,
    )


def gompertz(databook: DataBook) -> ParametricDistriModel:
    """Create gompertz distribution.

    Args:
        databook (DataBook): _description_

    Returns:
        ParametricDistriModel: _description_
    """
    functions = GompertzDistriFunction(param_names=["c", "rate"])
    likelihood = GompertzDistriLikelihood(databook)
    optimizer = DistriOptimizer(likelihood)
    return ParametricDistriModel(
        functions,
        optimizer,
    )


def custom_distri(
    databook: DataBook,
    functions: Type[ParametricDistriFunction],
    likelihood: Type[ParametricDistriLikelihood],
) -> Type[ParametricDistriModel]:
    """Create custom distribution.

    Args:
        databook (DataBook): _description_
        functions (Type[ParametricDistriFunction]): _description_
        likelihood (Type[ParametricDistriLikelihood]): _description_

    Returns:
        Type[ParametricDistriModel]: _description_
    """
    _functions = functions()
    _likelihood = likelihood(databook)
    _optimizer = DistriOptimizer(_likelihood)
    return ParametricDistriModel(
        _functions,
        _optimizer,
    )
