from abc import ABC, abstractmethod

import numpy as np
from scipy.optimize import approx_fprime

from ...data.base import DataBook
from .function import ParametricDistFunction


class ParametricLikelihood(ABC):
    def __init__(self, databook: DataBook):
        if not isinstance(databook, DataBook):
            raise TypeError("ParametricLikelihood expects databook instance")
        self.databook = databook

    @abstractmethod
    def negative_log_likelihood(self) -> float:
        pass

    @abstractmethod
    def jac_negative_log_likelihood(self) -> np.ndarray:
        pass

    @abstractmethod
    def hess_negative_log_likelihood(self) -> np.ndarray:
        pass


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


class ExponentialDistLikelihood(ParametricDistLikelihood):
    def __init__(self, databook: DataBook):
        super().__init__(databook)

    # relife/distribution.Exponential
    def jac_hf(
        self,
        time: np.ndarray,
        functions: ParametricDistFunction,
    ) -> np.ndarray:
        # shape : (len(sample), nb_param)
        return np.ones((time.size, 1))

    # relife/distribution.Exponential
    def jac_chf(
        self,
        time: np.ndarray,
        functions: ParametricDistFunction,
    ) -> np.ndarray:
        # shape : (len(sample), nb_param)
        return np.ones((time.size, 1)) * time[:, None]


class WeibullDistLikelihood(ParametricDistLikelihood):
    def __init__(self, databook: DataBook):
        super().__init__(databook)

    def jac_hf(
        self,
        time: np.ndarray,
        functions: ParametricDistFunction,
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
        functions: ParametricDistFunction,
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


class GompertzDistLikelihood(ParametricDistLikelihood):
    def __init__(self, databook: DataBook):
        super().__init__(databook)

    def jac_hf(
        self,
        time: np.ndarray,
        functions: ParametricDistFunction,
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
        functions: ParametricDistFunction,
    ) -> np.ndarray:
        return np.column_stack(
            (
                np.expm1(functions.params.rate * time[:, None]),
                functions.params.c
                * time[:, None]
                * np.exp(functions.params.rate * time[:, None]),
            )
        )
