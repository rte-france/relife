from abc import abstractmethod
from typing import TypeVar

import numpy as np
from scipy.optimize import approx_fprime
from scipy.special import digamma

from ..backbone import ParametricLikelihood
from ..integrations import shifted_laguerre
from .functions import DistFunctions

Functions = TypeVar("Functions", bound=DistFunctions)


class DistLikelihood(ParametricLikelihood):
    def __init__(self, *data, **kwdata):
        super().__init__(*data, **kwdata)

    @abstractmethod
    def jac_hf(self):
        pass

    @abstractmethod
    def jac_chf(self):
        pass

    # relife/parametric.ParametricHazardFunction
    def negative_log_likelihood(
        self,
        functions: Functions,
    ) -> float:

        D_contrib = -np.sum(np.log(functions.hf(self.data("complete").values)))
        RC_contrib = np.sum(
            functions.chf(
                np.concatenate(
                    (
                        self.data("complete").values,
                        self.data("right_censored").values,
                    )
                ),
            )
        )
        LC_contrib = -np.sum(
            np.log(
                -np.expm1(
                    -functions.chf(
                        self.data("left_censored").values,
                    )
                )
            )
        )
        LT_contrib = -np.sum(functions.chf(self.data("left_truncated").values))
        return D_contrib + RC_contrib + LC_contrib + LT_contrib

    # relife/parametric.ParametricHazardFunction
    def jac_negative_log_likelihood(
        self,
        functions: Functions,
    ) -> np.ndarray:

        jac_D_contrib = -np.sum(
            self.jac_hf(self.data("complete").values, functions)
            / functions.hf(self.data("complete").values)[:, None],
            axis=0,
            # keepdims=True,
        )
        # print(jac_D_contrib.shape)
        jac_RC_contrib = np.sum(
            self.jac_chf(
                np.concatenate(
                    (
                        self.data("complete").values,
                        self.data("right_censored").values,
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
        #             self.data("complete").values,
        #             self.data("right_censored").values,
        #         )
        #     ).shape
        # )
        jac_LC_contrib = -np.sum(
            self.jac_chf(self.data("left_censored").values, functions)
            / np.expm1(
                functions.chf(self.data("left_censored").values)[:, None]
            ),
            axis=0,
            # keepdims=True,
        )
        # print(jac_LC_contrib.shape)
        # print(
        #     "jac_chf :",
        #     self.jac_chf(self.data("left_truncated").values).shape,
        # )
        jac_LT_contrib = -np.sum(
            self.jac_chf(self.data("left_truncated").values, functions),
            axis=0,
            # keepdims=True,
        )
        # print(jac_LT_contrib.shape)

        return jac_D_contrib + jac_RC_contrib + jac_LC_contrib + jac_LT_contrib

    def hess_negative_log_likelihood(
        self,
        functions: Functions,
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


class ExponentialLikelihood(DistLikelihood):
    def __init__(self, *data, **kwdata):
        super().__init__(*data, **kwdata)

    # relife/distribution.Exponential
    def jac_hf(
        self,
        time: np.ndarray,
        functions: Functions,
    ) -> np.ndarray:
        # shape : (len(sample), nb_param)
        return np.ones((time.size, 1))

    # relife/distribution.Exponential
    def jac_chf(
        self,
        time: np.ndarray,
        functions: Functions,
    ) -> np.ndarray:
        # shape : (len(sample), nb_param)
        return np.ones((time.size, 1)) * time[:, None]


class WeibullLikelihood(DistLikelihood):
    def __init__(self, *data, **kwdata):
        super().__init__(*data, **kwdata)

    def jac_hf(
        self,
        time: np.ndarray,
        functions: Functions,
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
        functions: Functions,
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


class GompertzLikelihood(DistLikelihood):
    def __init__(self, *data, **kwdata):
        super().__init__(*data, **kwdata)

    def jac_hf(
        self,
        time: np.ndarray,
        functions: Functions,
    ) -> np.ndarray:
        return np.column_stack(
            (
                functions.params.rate
                * np.exp(functions.params.rate * time[:, None]),
                functions.params.c
                * np.exp(functions.params.rate * time[:, None])
                * (1 + functions.params.rate * time[:, None]),
            )
        )

    def jac_chf(
        self,
        time: np.ndarray,
        functions: Functions,
    ) -> np.ndarray:
        return np.column_stack(
            (
                np.expm1(functions.params.rate * time[:, None]),
                functions.params.c
                * time[:, None]
                * np.exp(functions.params.rate * time[:, None]),
            )
        )


class GammaLikelihood(DistLikelihood):
    def __init__(self, *data, **kwdata):
        super().__init__(*data, **kwdata)
        self._default_hess_scheme = "2-point"

    @staticmethod
    def _jac_uppergamma_c(functions: Functions, x: np.ndarray) -> np.ndarray:
        return shifted_laguerre(
            lambda s: np.log(s) * s ** (functions.params.c - 1),
            x,
            ndim=np.ndim(x),
        )

    def jac_hf(
        self,
        time: np.ndarray,
        functions: Functions,
    ) -> np.ndarray:

        x = self.params.rate * time[:, None]
        return (
            x ** (self.params.c - 1)
            * np.exp(-x)
            / functions._uppergamma(x) ** 2
            * np.column_stack(
                (
                    functions.params.rate
                    * np.log(x)
                    * functions._uppergamma(x)
                    - functions.params.rate
                    * GammaLikelihood._jac_uppergamma_c(functions, x),
                    (functions.params.c - x) * functions._uppergamma(x)
                    + x**functions.params.c * np.exp(-x),
                )
            )
        )

    def jac_chf(
        self,
        time: np.ndarray,
        functions: Functions,
    ) -> np.ndarray:
        x = functions.params.rate * time[:, None]
        return np.column_stack(
            (
                digamma(functions.params.c)
                - GammaLikelihood._jac_uppergamma_c(functions, x)
                / functions._uppergamma(x),
                x ** (functions.params.c - 1)
                * time[:, None]
                * np.exp(-x)
                / functions._uppergamma(x),
            )
        )
