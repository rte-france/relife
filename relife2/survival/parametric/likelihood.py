from abc import ABC, abstractmethod
from typing import Type

import numpy as np
from scipy.optimize import approx_fprime

from .. import DataBook
from .function import ParametricDistriFunction


class ParametricLikelihood(ABC):
    def __init__(self, databook: Type[DataBook]):
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


class ParametricDistriLikelihood(ParametricLikelihood):
    def __init__(self, databook: Type[DataBook]):
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
        functions: Type[ParametricDistriFunction],
    ) -> np.ndarray:
        return (
            -np.sum(np.log(functions.hf(self.databook("complete").values)))
            + np.sum(
                functions.chf(
                    np.concatenate(
                        (
                            self.databook("complete").values,
                            self.databook("right_censored").values,
                        )
                    ),
                )
            )
            - np.sum(functions.chf(self.databook("left_truncated").values))
            - np.sum(
                np.log(
                    -np.expm1(
                        -functions.chf(
                            self.databook("left_censored").values,
                        )
                    )
                )
            )
        )

    # relife/parametric.ParametricHazardFunction
    def jac_negative_log_likelihood(
        self,
        functions: Type[ParametricDistriFunction],
    ) -> np.ndarray:
        return (
            -np.sum(
                self.jac_hf(self.databook("complete").values)
                / functions.hf(self.databook("complete").values),
                axis=0,
            )
            + np.sum(
                self.jac_chf(
                    np.concatenate(
                        (
                            self.databook("complete").values,
                            self.databook("right_censored").values,
                        )
                    ),
                ),
                axis=0,
            )
            - np.sum(
                self.jac_chf(self.databook("left_truncated").values),
                axis=0,
            )
            - np.sum(
                self.jac_chf(self.databook("left_censored").values)
                / np.expm1(
                    functions.chf(self.databook("left_censored").values)
                ),
                axis=0,
            )
        )

    def hess_negative_log_likelihood(
        self,
        functions: Type[ParametricDistriFunction],
        eps: float = 1e-6,
        scheme: str = None,
    ) -> np.ndarray:

        size = np.size(functions.params.values)
        print(size)
        hess = np.empty((size, size))

        if scheme is None:
            scheme = self._default_hess_scheme

        if scheme == "cs":
            u = eps * 1j * np.eye(size)
            for i in range(size):
                for j in range(i, size):
                    print(type(u[i]))
                    print(u[i])
                    print(functions.params.values)
                    functions.params.values += u[i]
                    hess[i, j] = (
                        np.imag(self.jac_negative_log_likelihood(functions)[j])
                        / eps
                    )
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


class ExponentialDistriLikelihood(ParametricDistriLikelihood):
    def __init__(self, databook: Type[DataBook]):
        super().__init__(databook)

    # relife/distribution.Exponential
    def jac_hf(self, time: np.ndarray) -> np.ndarray:
        return np.ones(time.size)

    # relife/distribution.Exponential
    def jac_chf(self, time: np.ndarray) -> np.ndarray:
        return np.ones(time.size) * time
