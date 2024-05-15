from abc import abstractmethod

import numpy as np
from scipy.optimize import approx_fprime
from scipy.special import digamma

from relife2.utils.integrations import shifted_laguerre

from ..backbone import Likelihood
from .functions import DistFunctions


class DistLikelihood(Likelihood):
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
        pf: DistFunctions,
    ) -> float:

        D_contrib = -np.sum(np.log(pf.hf(self.complete_lifetimes.values)))
        RC_contrib = np.sum(
            pf.chf(
                np.concatenate(
                    (
                        self.complete_lifetimes.values,
                        self.right_censorships.values,
                    )
                ),
            )
        )
        LC_contrib = -np.sum(
            np.log(
                -np.expm1(
                    -pf.chf(
                        self.left_censorships.values,
                    )
                )
            )
        )
        LT_contrib = -np.sum(pf.chf(self.left_truncations.values))
        return D_contrib + RC_contrib + LC_contrib + LT_contrib

    # relife/parametric.ParametricHazardFunction
    def jac_negative_log_likelihood(
        self,
        pf: DistFunctions,
    ) -> np.ndarray:

        jac_D_contrib = -np.sum(
            self.jac_hf(np.squeeze(self.complete_lifetimes.values), pf)
            / pf.hf(self.complete_lifetimes.values),
            axis=0,
        )

        jac_RC_contrib = np.sum(
            self.jac_chf(
                np.concatenate(
                    (
                        np.squeeze(self.complete_lifetimes.values),
                        np.squeeze(self.right_censorships.values),
                    )
                ),
                pf,
            ),
            axis=0,
        )

        jac_LC_contrib = -np.sum(
            self.jac_chf(np.squeeze(self.left_censorships.values), pf)
            / np.expm1(pf.chf(self.left_censorships.values)),
            axis=0,
        )

        jac_LT_contrib = -np.sum(
            self.jac_chf(np.squeeze(self.left_truncations.values), pf),
            axis=0,
        )

        return jac_D_contrib + jac_RC_contrib + jac_LC_contrib + jac_LT_contrib

    def hess_negative_log_likelihood(
        self,
        pf: DistFunctions,
        eps: float = 1e-6,
        scheme: str = None,
    ) -> np.ndarray:

        size = np.size(pf.params.values)
        # print(size)
        hess = np.empty((size, size))
        params_values = pf.params.values

        if scheme is None:
            scheme = self._default_hess_scheme

        if scheme == "cs":
            u = eps * 1j * np.eye(size)
            for i in range(size):
                for j in range(i, size):
                    # print(type(u[i]))
                    # print(u[i])
                    # print(pf.params.values)
                    # print(pf.params.values + u[i])
                    pf.params.values = pf.params.values + u[i]
                    # print(self.jac_negative_log_likelihood(pf))

                    hess[i, j] = np.imag(self.jac_negative_log_likelihood(pf)[j]) / eps
                    pf.params.values = params_values
                    if i != j:
                        hess[j, i] = hess[i, j]

        elif scheme == "2-point":

            def f(xk, pf):
                pf.params.values = xk
                return self.jac_negative_log_likelihood(pf)

            xk = pf.params.values

            for i in range(size):
                hess[i] = approx_fprime(
                    xk,
                    lambda x: f(x, pf)[i],
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
        pf: DistFunctions,
    ) -> np.ndarray:
        # shape : (len(sample), nb_param)
        return np.ones((time.size, 1))

    # relife/distribution.Exponential
    def jac_chf(
        self,
        time: np.ndarray,
        pf: DistFunctions,
    ) -> np.ndarray:
        # shape : (len(sample), nb_param)
        return np.ones((time.size, 1)) * time[:, None]


class WeibullLikelihood(DistLikelihood):
    def __init__(self, *data, **kwdata):
        super().__init__(*data, **kwdata)

    def jac_hf(
        self,
        time: np.ndarray,
        pf: DistFunctions,
    ) -> np.ndarray:

        return np.column_stack(
            (
                pf.params.rate
                * (pf.params.rate * time[:, None]) ** (pf.params.c - 1)
                * (1 + pf.params.c * np.log(pf.params.rate * time[:, None])),
                pf.params.c**2 * (pf.params.rate * time[:, None]) ** (pf.params.c - 1),
            )
        )

    def jac_chf(
        self,
        time: np.ndarray,
        pf: DistFunctions,
    ) -> np.ndarray:
        return np.column_stack(
            (
                np.log(pf.params.rate * time[:, None])
                * (pf.params.rate * time[:, None]) ** pf.params.c,
                pf.params.c
                * time[:, None]
                * (pf.params.rate * time[:, None]) ** (pf.params.c - 1),
            )
        )


class GompertzLikelihood(DistLikelihood):
    def __init__(self, *data, **kwdata):
        super().__init__(*data, **kwdata)

    def jac_hf(
        self,
        time: np.ndarray,
        pf: DistFunctions,
    ) -> np.ndarray:
        return np.column_stack(
            (
                pf.params.rate * np.exp(pf.params.rate * time[:, None]),
                pf.params.c
                * np.exp(pf.params.rate * time[:, None])
                * (1 + pf.params.rate * time[:, None]),
            )
        )

    def jac_chf(
        self,
        time: np.ndarray,
        pf: DistFunctions,
    ) -> np.ndarray:
        return np.column_stack(
            (
                np.expm1(pf.params.rate * time[:, None]),
                pf.params.c * time[:, None] * np.exp(pf.params.rate * time[:, None]),
            )
        )


class GammaLikelihood(DistLikelihood):
    def __init__(self, *data, **kwdata):
        super().__init__(*data, **kwdata)
        self._default_hess_scheme = "2-point"

    @staticmethod
    def _jac_uppergamma_c(pf: DistFunctions, x: np.ndarray) -> np.ndarray:
        return shifted_laguerre(
            lambda s: np.log(s) * s ** (pf.params.c - 1),
            x,
            ndim=np.ndim(x),
        )

    def jac_hf(
        self,
        time: np.ndarray,
        pf: DistFunctions,
    ) -> np.ndarray:

        x = pf.params.rate * time[:, None]
        return (
            x ** (pf.params.c - 1)
            * np.exp(-x)
            / pf._uppergamma(x) ** 2
            * np.column_stack(
                (
                    pf.params.rate * np.log(x) * pf._uppergamma(x)
                    - pf.params.rate * GammaLikelihood._jac_uppergamma_c(pf, x),
                    (pf.params.c - x) * pf._uppergamma(x) + x**pf.params.c * np.exp(-x),
                )
            )
        )

    def jac_chf(
        self,
        time: np.ndarray,
        pf: DistFunctions,
    ) -> np.ndarray:
        x = pf.params.rate * time[:, None]
        return np.column_stack(
            (
                digamma(pf.params.c)
                - GammaLikelihood._jac_uppergamma_c(pf, x) / pf._uppergamma(x),
                x ** (pf.params.c - 1) * time[:, None] * np.exp(-x) / pf._uppergamma(x),
            )
        )


class LogLogisticLikelihood(DistLikelihood):
    def __init__(self, *data, **kwdata):
        super().__init__(*data, **kwdata)

    def jac_hf(self, time: np.ndarray, pf: DistFunctions) -> np.ndarray:
        x = pf.params.rate * time[:, None]
        return np.column_stack(
            (
                (pf.params.rate * x ** (pf.params.c - 1) / (1 + x**pf.params.c) ** 2)
                * (
                    1
                    + x**pf.params.c
                    + pf.params.c * np.log(pf.params.rate * time[:, None])
                ),
                (pf.params.rate * x ** (pf.params.c - 1) / (1 + x**pf.params.c) ** 2)
                * (pf.params.c**2 / pf.params.rate),
            )
        )

    def jac_chf(self, time: np.ndarray, pf: DistFunctions) -> np.ndarray:
        x = pf.params.rate * time[:, None]
        return np.column_stack(
            (
                (x**pf.params.c / (1 + x**pf.params.c))
                * np.log(pf.params.rate * time[:, None]),
                (x**pf.params.c / (1 + x**pf.params.c))
                * (pf.params.c / pf.params.rate),
            )
        )
