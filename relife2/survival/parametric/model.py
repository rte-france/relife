import warnings
from typing import Type

import numpy as np

from .. import DataBook, databook
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
        likelihood_cls: Type[ParametricDistriLikelihood],
        optimizer_cls: Type[DistriOptimizer],
        init_params=False,
        # # optimizer: DistriOptimizer,
    ):

        if not isinstance(functions, ParametricDistriFunction):
            raise TypeError(
                "ParametricDistriFunction expected, got"
                f" '{type(functions).__name__}'"
            )

        if not issubclass(likelihood_cls, ParametricDistriLikelihood):
            raise ValueError("ParametricDistriLikelihood subclass expected")

        if not optimizer_cls == DistriOptimizer:
            raise ValueError("DistriOptimizer expected")

        # assert issubclass(optimizer, DistriOptimizer)
        self.functions = functions
        self.likelihood_cls = likelihood_cls
        self.optimizer_cls = optimizer_cls
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

        likelihood = self.likelihood_cls(db)
        optimizer = self.optimizer_cls(likelihood)

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


def exponential(*params) -> ParametricDistriModel:

    functions = ExponentialDistriFunction(param_names=["rate"])
    if len(params) != 0:
        params = np.asarray(params)
        if params.shape != functions.params.values:
            raise ValueError("Incorrect params shape")
        functions.params.values = params

        return ParametricDistriModel(
            functions,
            ExponentialDistriLikelihood,
            DistriOptimizer,
            init_params=True,
        )
    else:
        return ParametricDistriModel(
            functions,
            ExponentialDistriLikelihood,
            DistriOptimizer,
            init_params=False,
        )


def weibull(*params) -> ParametricDistriModel:

    functions = WeibullDistriFunction(param_names=["c", "rate"])
    if len(params) != 0:
        params = np.asarray(params)
        if params.shape != functions.params.values:
            raise ValueError("Incorrect params shape")
        functions.params.values = params

        return ParametricDistriModel(
            functions,
            WeibullDistriLikelihood,
            DistriOptimizer,
            init_params=True,
        )
    else:
        return ParametricDistriModel(
            functions,
            WeibullDistriLikelihood,
            DistriOptimizer,
            init_params=False,
        )


def gompertz(*params) -> ParametricDistriModel:

    functions = GompertzDistriFunction(param_names=["c", "rate"])
    if len(params) != 0:
        params = np.asarray(params)
        if params.shape != functions.params.values:
            raise ValueError("Incorrect params shape")
        functions.params.values = params

        return ParametricDistriModel(
            functions,
            GompertzDistriLikelihood,
            DistriOptimizer,
            init_params=True,
        )
    else:
        return ParametricDistriModel(
            functions,
            GompertzDistriLikelihood,
            DistriOptimizer,
            init_params=False,
        )


# class create_distri:

#     def __call__(self, functions : Type[], likelihood : Type[], optimizer : Type[]):
#         # check functions.__module__ etc.
#         return distri # function like gompertz, etc.
#         # custom_distri = distri(*params)


def custom_distri(
    databook: DataBook,
    functions_cls: Type[ParametricDistriFunction],
    likelihood_cls: Type[ParametricDistriLikelihood],
) -> ParametricDistriModel:
    """Create custom distribution.

    Args:
        databook (DataBook): DataBook instance
        functions_cls (Type[ParametricDistriFunction]): ParametricDistriFunction class definition
        likelihood_cls (Type[ParametricDistriLikelihood]): ParametricDistriLikelihood class definition

    Returns:
        ParametricDistriModel: a parametric distribution


    Note:
        In functions_cls, the following methods are mandatory and must return 1d-array or float:
            | hf(time: 1d-array) -> 1d-array
            | chf(time: 1d-array) -> 1d-array
            | mean() -> float
            | var() -> float
            | mrl(time: 1d-array) -> 1d-array
            | ichf(cumulative_hazard_rate: 1d-array) -> 1d-array
        In likelihood_cls, the following methods are mandatory and must return 2d-array of shape (nb_sample, nb_param)
            | jac_hf(time: 1d-array, functions : ParametricDistriFunction) -> 2d-array
            | jac_chf(time: 1d-array, functions : ParametricDistriFunction) -> 2d-array


    Examples:
        .. code-block:: python

            from relife2.survival.parametric import (
                ParametricDistriFunction,
                ParametricDistriLikelihood,
                custom_distri
            )

            class WeibullDistriFunction(ParametricDistriFunction):
                def __init__(self, param_names=["c", "rate"]):
                    super().__init__(param_names=param_names)

                def hf(self, time: np.ndarray) -> np.ndarray:
                    return (
                        self.params.c
                        * self.params.rate
                        * (self.params.rate * time) ** (self.params.c - 1)
                    )

                def chf(self, time: np.ndarray) -> np.ndarray:
                    return (self.params.rate * time) ** self.params.c

                def mean(self) -> float:
                    return gamma(1 + 1 / self.params.c) / self.params.rate

                def var(self) -> float:
                    return (
                        gamma(1 + 2 / self.params.c) / self.params.rate**2
                        - self.mean() ** 2
                    )

                def mrl(self, time: np.ndarray) -> np.ndarray:
                    return (
                        gamma(1 / self.params.c)
                        / (self.params.rate * self.params.c * self.sf(time))
                        * gammaincc(
                            1 / self.params.c,
                            (self.params.rate * time) ** self.params.c,
                        )
                    )

            class WeibullDistriLikelihood(ParametricDistriLikelihood):
                def __init__(self, databook: DataBook):
                    super().__init__(databook)

                def jac_hf(
                    self,
                    time: np.ndarray,
                    functions: ParametricDistriFunction,
                ) -> np.ndarray:

                    return np.column_stack(
                        (
                            functions.params.rate
                            * (functions.params.rate * time[:, None])
                            ** (functions.params.c - 1)
                            * (
                                1
                                + functions.params.c
                                * np.log(functions.params.rate * time[:, None])
                            ),
                            functions.params.c**2
                            * (functions.params.rate * time[:, None])
                            ** (functions.params.c - 1),
                        )
                    )

                def jac_chf(
                    self,
                    time: np.ndarray,
                    functions: ParametricDistriFunction,
                ) -> np.ndarray:
                    return np.column_stack(
                        (
                            np.log(functions.params.rate * time[:, None])
                            * (functions.params.rate * time[:, None])
                            ** functions.params.c,
                            functions.params.c
                            * time[:, None]
                            * (functions.params.rate * time[:, None])
                            ** (functions.params.c - 1),
                        )
                    )

            my_distri = custom_distri(databook, WeibullDistriFunction, WeibullDistriLikelihood)
    """
    functions = functions_cls()
    likelihood = likelihood_cls(databook)
    optimizer = DistriOptimizer(likelihood)
    return ParametricDistriModel(
        functions,
        optimizer,
    )
