from abc import ABC, abstractmethod
from typing import Type

import numpy as np
from scipy.optimize.optimize import approx_fprime

from .. import DataBook
from .function import DistributionFunctions


class ParametricLikelihood(ABC):
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

    # relife/parametric.ParametricHazardFunction
    _default_hess_scheme: str = (  #: Default method for evaluating the hessian of the negative log-likelihood.
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
        params: np.ndarray,
        databook: Type[DataBook],
        functions: Type[DistributionFunctions],
    ) -> np.ndarray:
        return (
            -np.sum(np.log(functions.hf(databook("complete").values, params)))
            + np.sum(
                functions.chf(
                    np.contenate(
                        (
                            databook("complete").values,
                            databook("right_censored").values,
                        )
                    ),
                    params,
                )
            )
            - np.sum(functions.chf(databook("left_truncated").values, params))
            - np.sum(
                np.log(
                    -np.expm1(
                        -functions.chf(
                            databook("left_censored").values,
                            params,
                        )
                    )
                )
            )
        )

    # relife/parametric.ParametricHazardFunction
    def jac_negative_log_likelihood(
        self,
        params: np.ndarray,
        databook: Type[DataBook],
        functions: Type[DistributionFunctions],
    ) -> np.ndarray:
        return (
            -np.sum(
                self.jac_hf(databook("complete").values, params)
                / functions.hf(databook("complete").values, params),
                axis=0,
            )
            + np.sum(
                self.jac_chf(
                    np.contenate(
                        (
                            databook("complete").values,
                            databook("right_censored").values,
                        )
                    ),
                    params,
                ),
                axis=0,
            )
            - np.sum(
                self.jac_chf(databook("left_truncated").values),
                params,
                axis=0,
            )
            - np.sum(
                self.jac_chf(databook("left_censored").values, params)
                / np.expm1(
                    self._chf(databook("left_censored").values, params)
                ),
                axis=0,
            )
        )

    def hess_negative_log_likelihood(
        self,
        params: np.ndarray,
        databook: Type[DataBook],
        functions: Type[DistributionFunctions],
        eps: float = 1e-6,
        scheme: str = None,
    ) -> np.ndarray:

        size = np.size(params)
        hess = np.empty((size, size))

        if scheme is None:
            scheme = self._default_hess_scheme

        if scheme == "cs":
            u = eps * 1j * np.eye(size)
            for i in range(size):
                for j in range(i, size):
                    hess[i, j] = (
                        np.imag(
                            self.jac_negative_log_likelihood(
                                params + u[i], databook, functions
                            )[j]
                        )
                        / eps
                    )
                    if i != j:
                        hess[j, i] = hess[i, j]

        elif scheme == "2-point":
            for i in range(size):
                hess[i] = approx_fprime(
                    params,
                    lambda params: self.jac_negative_log_likelihood(
                        params, databook, functions
                    )[i],
                    eps,
                )

        else:
            raise ValueError("scheme argument must be 'cs' or '2-point'")

        return hess


class ExponentialDistriLikelihood(ParametricDistriLikelihood):
    # relife/distribution.Exponential
    def jac_hf(self, time: np.ndarray, params: np.ndarray) -> np.ndarray:
        return np.ones((time.size, 1))

    # relife/distribution.Exponential
    def jac_chf(self, time: np.ndarray, params: np.ndarray) -> np.ndarray:
        return np.ones((time.size, 1)) * time
