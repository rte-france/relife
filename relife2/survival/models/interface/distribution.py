from abc import abstractmethod

import numpy as np
from scipy.optimize import Bounds, OptimizeResult, approx_fprime, minimize

from ...core.parameter import Parameter
from ...data.base import DataBook
from ...interface.backbone import (
    ParametricFunction,
    ParametricLikelihood,
    ParametricOptimizer,
)


class ParametricDistFunction(ParametricFunction):
    def __init__(self, nb_params: int = None, param_names: list = None):
        params = Parameter(nb_params=nb_params, param_names=param_names)
        super().__init__(params)

    # relife/parametric.ParametricLifetimeModel
    def sf(self, time: np.ndarray) -> np.ndarray:
        """Parametric survival function."""
        return np.exp(-self.chf(time))

    # relife/parametric.ParametricLifetimeModel
    def cdf(self, time: np.ndarray) -> np.ndarray:
        """Parametric cumulative distribution function."""
        return 1 - self.sf(time)

    # relife/parametric.ParametricLifetimeModel
    def pdf(self, time: np.ndarray) -> np.ndarray:
        """Parametric probability density function."""
        return self.hf(time) * self.sf(time)

    @abstractmethod
    def mean(self):
        """only mandatory for ParametricDist as exact expression is known"""
        pass

    @abstractmethod
    def var(self):
        """only mandatory for ParametricDist as exact expression is known"""
        pass

    @abstractmethod
    def mrl(self):
        """only mandatory for ParametricDist as exact expression is known"""
        pass

    @abstractmethod
    def ichf(self, cumulative_hazard_rate: np.ndarray):
        """only mandatory for ParametricDist as exact expression is known"""
        pass

    # relife/model.AbsolutelyContinuousLifetimeModel /!\ dependant of ichf and _ichf
    # /!\ mathematically : -np.log(probability) = cumulative_hazard_rate
    def isf(self, probability: np.ndarray) -> np.ndarray:
        cumulative_hazard_rate = -np.log(probability)
        return self.ichf(cumulative_hazard_rate)


class ParametricDistLikelihood(ParametricLikelihood):
    def __init__(self, databook: DataBook):
        super().__init__(databook)
        # relife/parametric.ParametricHazardFunction
        self._default_hess_scheme: str = (  #: Default method for evaluating the hessian of the negative log-likelihood.
            "cs"
        )

    @abstractmethod
    def jac_hf(self):
        pass

    @abstractmethod
    def jac_chf(self):
        pass

    # relife/parametric.ParametricHazardFunction
    def negative_log_likelihood(
        self,
        functions: ParametricDistFunction,
    ) -> np.ndarray:

        D_contrib = -np.sum(
            np.log(functions.hf(self.databook("complete").values))
        )
        RC_contrib = np.sum(
            functions.chf(
                np.concatenate(
                    (
                        self.databook("complete").values,
                        self.databook("right_censored").values,
                    )
                ),
            )
        )
        LC_contrib = -np.sum(
            np.log(
                -np.expm1(
                    -functions.chf(
                        self.databook("left_censored").values,
                    )
                )
            )
        )
        LT_contrib = -np.sum(
            functions.chf(self.databook("left_truncated").values)
        )
        return D_contrib + RC_contrib + LC_contrib + LT_contrib

    # relife/parametric.ParametricHazardFunction
    def jac_negative_log_likelihood(
        self,
        functions: ParametricDistFunction,
    ) -> np.ndarray:

        jac_D_contrib = -np.sum(
            self.jac_hf(self.databook("complete").values, functions)
            / functions.hf(self.databook("complete").values)[:, None],
            axis=0,
            # keepdims=True,
        )
        # print(jac_D_contrib.shape)
        jac_RC_contrib = np.sum(
            self.jac_chf(
                np.concatenate(
                    (
                        self.databook("complete").values,
                        self.databook("right_censored").values,
                    )
                ),
                functions,
            ),
            axis=0,
            # keepdims=True,
        )
        # print(jac_RC_contrib.shape)
        # print(
        #     np.concatenate(
        #         (
        #             self.databook("complete").values,
        #             self.databook("right_censored").values,
        #         )
        #     ).shape
        # )
        jac_LC_contrib = -np.sum(
            self.jac_chf(self.databook("left_censored").values, functions)
            / np.expm1(
                functions.chf(self.databook("left_censored").values)[:, None]
            ),
            axis=0,
            # keepdims=True,
        )
        # print(jac_LC_contrib.shape)
        # print(
        #     "jac_chf :",
        #     self.jac_chf(self.databook("left_truncated").values).shape,
        # )
        jac_LT_contrib = -np.sum(
            self.jac_chf(self.databook("left_truncated").values, functions),
            axis=0,
            # keepdims=True,
        )
        # print(jac_LT_contrib.shape)

        return jac_D_contrib + jac_RC_contrib + jac_LC_contrib + jac_LT_contrib

    def hess_negative_log_likelihood(
        self,
        functions: ParametricDistFunction,
        eps: float = 1e-6,
        scheme: str = None,
    ) -> np.ndarray:

        size = np.size(functions.params.values)
        # print(size)
        hess = np.empty((size, size))
        params_values = functions.params.values

        if scheme is None:
            scheme = self._default_hess_scheme

        if scheme == "cs":
            u = eps * 1j * np.eye(size)
            for i in range(size):
                for j in range(i, size):
                    # print(type(u[i]))
                    # print(u[i])
                    # print(functions.params.values)
                    # print(functions.params.values + u[i])
                    functions.params.values = functions.params.values + u[i]
                    # print(self.jac_negative_log_likelihood(functions))

                    hess[i, j] = (
                        np.imag(self.jac_negative_log_likelihood(functions)[j])
                        / eps
                    )
                    functions.params.values = params_values
                    if i != j:
                        hess[j, i] = hess[i, j]

        elif scheme == "2-point":

            def f(xk, functions):
                functions.params.values = xk
                return self.jac_negative_log_likelihood(functions)

            xk = functions.params.values

            for i in range(size):
                hess[i] = approx_fprime(
                    xk,
                    lambda x: f(x, functions)[i],
                    eps,
                )
        else:
            raise ValueError("scheme argument must be 'cs' or '2-point'")

        return hess


MIN_POSITIVE_FLOAT = np.finfo(float).resolution


class DistOptimizer(ParametricOptimizer):
    def __init__(self, likelihood: ParametricLikelihood):
        super().__init__(likelihood)
        # relife/parametric.ParametricHazardFunction
        self._default_method: str = (  #: Default method for minimizing the negative log-likelihood.
            "L-BFGS-B"
        )

    # relife/distribution.ParametricLifetimeDistbution
    def _init_param(self, nb_params: int) -> np.ndarray:
        param0 = np.ones(nb_params)
        param0[-1] = 1 / np.median(
            np.concatenate(
                [
                    data.values
                    for data in self.likelihood.databook(
                        "complete | right_censored | left_censored"
                    )
                ]
            )
        )
        return param0

    def _get_param_bounds(self, functions: ParametricFunction) -> Bounds:
        return Bounds(
            np.full(functions.params.nb_params, MIN_POSITIVE_FLOAT),
            np.full(functions.params.nb_params, np.inf),
        )

    def _func(
        self,
        x,
        functions: ParametricFunction,
    ):
        functions.params.values = x
        return self.likelihood.negative_log_likelihood(functions)

    def _jac(
        self,
        x,
        functions: ParametricFunction,
    ):
        functions.params.values = x
        return self.likelihood.jac_negative_log_likelihood(functions)

    # relife/parametric.ParametricHazardFunctions
    def fit(
        self,
        functions: ParametricFunction,
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
            param0 = self._init_param(functions.params.nb_params)
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
        functions.params.values = opt.x

        return functions, opt
