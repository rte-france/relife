from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, TypeVar, Union

import numpy as np
from numpy.typing import NDArray
from scipy import stats
from scipy.optimize import approx_fprime

from relife import ParametricModel


class Likelihood(ABC):
    model: ParametricModel
    data: Any

    @abstractmethod
    def negative_log(self, params: NDArray[np.float64]) -> np.float64:
        """
        Negative log likelihood.

        Parameters
        ----------
        params : ndarray
            Parameters values on which likelihood is evaluated

        Returns
        -------
        float
            Negative log likelihood value
        """

    @abstractmethod
    def maximum_likelihood_estimation(self, **kwargs: Any) -> FittingResults: ...


L = TypeVar("L", bound="LikelihoodFromLifetimes")  # maybe other likelihood in the future


def hessian_cs(
    likelihood: L,
    params: NDArray[np.float64],
    eps: float = 1e-6,
) -> NDArray[np.float64]:

    size = params.size
    hess = np.empty((size, size))
    u = eps * 1j * np.eye(size)
    complex_params = params.astype(np.complex64)  # change params to complex
    for i in range(size):
        for j in range(i, size):
            hess[i, j] = np.imag(likelihood.jac_negative_log(complex_params + u[i])[j]) / eps
            if i != j:
                hess[j, i] = hess[i, j]
    return hess


def hessian_2point(
    likelihood: L,
    params: NDArray[np.float64],
    eps: float = 1e-6,
) -> NDArray[np.float64]:
    size = params.size
    hess = np.empty((size, size))
    for i in range(size):
        hess[i] = approx_fprime(
            params,
            lambda x: likelihood.jac_negative_log(x)[i],
            eps,
        )
    return hess


M = TypeVar("M", bound=Union["LifetimeDistribution", "LifetimeRegression", "MinimumDistribution"])


def approx_hessian(likelihood: L, params: NDArray[np.float64], eps: float = 1e-6) -> NDArray[np.float64]:

    def hessian_scheme(model: M):
        from relife.lifetime_model import Gamma, LifetimeRegression

        if isinstance(model, LifetimeRegression):
            return hessian_scheme(model.baseline)
        if isinstance(model, Gamma):
            return hessian_2point
        return hessian_cs

    return hessian_scheme(likelihood.model)(likelihood, params, eps=eps)


@dataclass
class FittingResults:
    """Fitting results of the parametric_model core."""

    nb_obversations: int  #: Number of observations (samples)
    optimal_params: NDArray[np.float64] = field(repr=False)  #: Optimal parameters values
    neg_log_likelihood: np.float64 = field(repr=False)  #: Negative log likelihood value at optimal parameters values

    covariance_matrix: Optional[NDArray[np.floating[Any]]] = field(
        repr=False, default=None
    )  #: Covariance matrix (computed as the inverse of the Hessian matrix).

    nb_params: int = field(init=False, repr=False)  #: Number of parameters.
    AIC: float = field(init=False)  #: Akaike Information Criterion.
    AICc: float = field(init=False)  #: Akaike Information Criterion with a correction for small sample sizes.
    BIC: float = field(init=False)  #: Bayesian Information Criterion.
    se: Optional[NDArray[np.float64]] = field(
        init=False, repr=False
    )  #: Standard error, square root of the diagonal of the covariance matrix
    IC: Optional[NDArray[np.float64]] = field(init=False, repr=False)  #: 95% IC

    def __post_init__(self):
        nb_params = self.optimal_params.size
        self.AIC = float(2 * nb_params + 2 * self.neg_log_likelihood)
        self.AICc = float(self.AIC + 2 * nb_params * (nb_params + 1) / (self.nb_obversations - nb_params - 1))
        self.BIC = float(np.log(self.nb_obversations) * nb_params + 2 * self.neg_log_likelihood)

        self.se = None
        if self.covariance_matrix is not None:
            self.se = np.sqrt(np.diag(self.covariance_matrix))
            self.IC = self.optimal_params.reshape(-1, 1) + stats.norm.ppf((0.05, 0.95)) * self.se.reshape(
                -1, 1
            ) / np.sqrt(
                self.nb_obversations
            )  # (p, 2)

    def se_estimation_function(self, jac_f: np.ndarray) -> np.float64 | NDArray[np.float64]:
        """Standard error estimation function.

        Parameters
        ----------
        jac_f : 1D array
            The Jacobian of a function f with respect to params.

        Returns
        -------
        1D array
            Standard error for f(params).

        References
        ----------
        .. [1] Meeker, W. Q., Escobar, L. A., & Pascual, F. G. (2022).
            Statistical methods for reliability data. John Wiley & Sons.
        """
        # [1] equation B.10 in Appendix
        # jac_f : (p,), (p, n) or (p, m, n)
        # self.var : (p, p)
        if self.covariance_matrix is not None:
            return np.sqrt(np.einsum("p...,pp,p...", jac_f, self.covariance_matrix, jac_f))  # (), (n,) or (m, n)
        raise ValueError("Can't compute if var is None")

    def __str__(self) -> str:
        """Returns a string representation of FittingResults with fields in a single column."""
        fields = [("fitted params", self.optimal_params), ("AIC", self.AIC), ("AICc", self.AICc), ("BIC", self.BIC)]
        # Find the maximum field name length for alignment
        max_name_length = max(len(name) for name, _ in fields)
        lines = []
        for name, value in fields:
            # Format arrays to be more compact
            if isinstance(value, np.ndarray):
                value_str = f"[{', '.join(f'{x:.6g}' for x in value)}]"
            else:
                value_str = f"{value:.6g}" if isinstance(value, float) else str(value)
            lines.append(f"{name:<{max_name_length}} : {value_str}")
        return "\n".join(lines)
