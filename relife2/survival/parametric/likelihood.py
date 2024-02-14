from abc import ABC, abstractmethod

import numpy as np
from scipy.optimize.optimize import approx_fprime

from .. import SurvivalData
from .function import DistributionFunction


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
    _default_hess_scheme: str = "cs"  #: Default method for evaluating the hessian of the negative log-likelihood.

    @abstractmethod
    def jac_hf(self):
        pass

    @abstractmethod
    def jac_chf(self):
        pass

    # relife/parametric.ParametricHazardFunction
    def negative_log_likelihood(
        self, param: np.ndarray, data: SurvivalData, functions: DistributionFunction
    ) -> np.ndarray:
        return (
            -np.sum(np.log(functions.hf(param, data.observed(return_values=True))))
            + np.sum(
                functions.chf(
                    param,
                    np.contenate(
                        (
                            data.observed(return_values=True),
                            data.censored(how="right", return_values=True),
                        )
                    ),
                )
            )
            - np.sum(
                functions.chf(param, data.truncated(how="left", return_values=True))
            )
            - np.sum(
                np.log(
                    -np.expm1(
                        -functions.chf(
                            param, data.censored(how="left", return_values=True)
                        )
                    )
                )
            )
        )

    # relife/parametric.ParametricHazardFunction
    def jac_negative_log_likelihood(
        self, param: np.ndarray, data: SurvivalData, functions: DistributionFunction
    ) -> np.ndarray:
        return (
            -np.sum(
                self.jac_hf(param, data.observed(return_values=True))
                / functions.hf(param, data.observed(return_values=True)),
                axis=0,
            )
            + np.sum(
                self.jac_chf(
                    param,
                    np.contenate(
                        (
                            data.observed(return_values=True),
                            data.censored(how="right", return_values=True),
                        )
                    ),
                ),
                axis=0,
            )
            - np.sum(
                self.jac_chf(param, data.truncated(how="left", return_values=True)),
                axis=0,
            )
            - np.sum(
                self.jac_chf(param, data.censored(how="left", return_values=True))
                / np.expm1(
                    self._chf(param, data.censored(how="left", return_values=True))
                ),
                axis=0,
            )
        )

    def hess_negative_log_likelihood(
        self,
        param: np.ndarray,
        data: SurvivalData,
        functions: DistributionFunction,
        eps: float = 1e-6,
        scheme: str = None,
    ) -> np.ndarray:

        size = np.size(param)
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
                                param + u[i], data, functions
                            )[j]
                        )
                        / eps
                    )
                    if i != j:
                        hess[j, i] = hess[i, j]

        elif scheme == "2-point":
            for i in range(size):
                hess[i] = approx_fprime(
                    param,
                    lambda param: self.jac_negative_log_likelihood(
                        param, data, functions
                    )[i],
                    eps,
                )

        else:
            raise ValueError("scheme argument must be 'cs' or '2-point'")

        return hess


class ExponentialDistriLikelihood(ParametricDistriLikelihood):
    # relife/distribution.Exponential
    def dhf(self, elapsed_time: np.ndarray) -> np.ndarray:
        return np.zeros_like(elapsed_time)

    # relife/distribution.Exponential
    def jac_chf(self, param: np.ndarray, elapsed_time: np.ndarray) -> np.ndarray:
        return np.ones((elapsed_time.size, 1)) * elapsed_time

    # relife/distribution.Exponential
    def jac_hf(self, param: np.ndarray, elapsed_time: np.ndarray) -> np.ndarray:
        return np.ones((elapsed_time.size, 1))
