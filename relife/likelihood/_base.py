from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Unpack

import numpy as np
from numpy.typing import NDArray
from scipy import stats
from scipy.optimize import approx_fprime
from typing_extensions import override

from relife.base import ParametricModel

if TYPE_CHECKING:
    from relife.typing import ScipyMinimizeOptions


class Likelihood(ABC):
    model: ParametricModel

    def __init__(self, model: ParametricModel) -> None:
        # deep copy model to have independent variation of params
        self.model = copy.deepcopy(model)

    @property
    def params(self) -> NDArray[np.float64]:
        return self.model.params

    @params.setter
    def params(self, value: NDArray[np.float64]) -> None:
        self.model.params = value

    @abstractmethod
    def negative_log(self, params: NDArray[np.float64]) -> float:
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
    def maximum_likelihood_estimation(self, **optimizer_options: Unpack[ScipyMinimizeOptions]) -> FittingResults:
        """
        Finds the parameter values that maximize the likelihood.

        Parameters
        ----------
        **optimizer_options
            Extra arguments used by `scipy.minimize`

        Returns
        -------
        FittingResults
            The fitting results.
        """


class DifferentiableLikelihood(Likelihood, ABC):
    @abstractmethod
    def jac_negative_log(self, params: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Jacobian of the negative log likelihood.

        The jacobian (here gradient) is computed with respect to parameters

        Parameters
        ----------
        params : ndarray
            Parameters values on which the jacobian is evaluated

        Returns
        -------
        ndarray
            Jacobian of the negative log likelihood value
        """


def _hessian_scheme(
    likelihood: DifferentiableLikelihood,
    params: NDArray[np.float64],
    method: Literal["2point", "cs"] = "cs",
    eps: float = 1e-6,
) -> NDArray[np.float64]:
    size = params.size
    hess = np.empty((size, size))

    # hessian 2 point
    if method == "2point":
        for i in range(size):
            hess[i] = approx_fprime(
                params,
                lambda x: likelihood.jac_negative_log(x)[i],
                eps,
            )
        return hess
    # hessian cs
    u = eps * 1j * np.eye(size)
    complex_params = params.astype(np.complex64)  # change params to complex
    for i in range(size):
        for j in range(i, size):
            hess[i, j] = np.imag(likelihood.jac_negative_log(complex_params + u[i])[j]) / eps
            if i != j:
                hess[j, i] = hess[i, j]
    return hess


def approx_hessian(
    likelihood: DifferentiableLikelihood,
    params: NDArray[np.float64],
    eps: float = 1e-6,
) -> NDArray[np.float64]:
    from relife.lifetime_model import Gamma
    from relife.lifetime_model._regression import LifetimeRegression

    if isinstance(likelihood.model, LifetimeRegression):
        if isinstance(likelihood.model.baseline, Gamma):
            return _hessian_scheme(likelihood, params, method="2point", eps=eps)
    if isinstance(likelihood.model, Gamma):
        return _hessian_scheme(likelihood, params, method="2point", eps=eps)
    return _hessian_scheme(likelihood, params, eps=eps)


@dataclass
class FittingResults:
    """Fitting results of the parametric_model core."""

    nb_obversations: int  #: Number of observations (samples)
    optimal_params: NDArray[np.float64] = field(repr=False)  #: Optimal parameters values
    neg_log_likelihood: float = field(repr=False)  #: Negative log likelihood value at optimal parameters values

    covariance_matrix: NDArray[np.float64] | None = field(
        repr=False, default=None
    )  #: Covariance matrix (computed as the inverse of the Hessian matrix).

    nb_params: int = field(init=False, repr=False)  #: Number of parameters.
    aic: float = field(init=False)  #: Akaike Information Criterion.
    aicc: float = field(init=False)  #: Akaike Information Criterion with a correction for small sample sizes.
    bic: float = field(init=False)  #: Bayesian Information Criterion.
    se: NDArray[np.float64] | None = field(
        init=False, repr=False
    )  #: Standard error, square root of the diagonal of the covariance matrix
    ic: NDArray[np.float64] | None = field(init=False, repr=False)  #: 95% IC

    def __post_init__(self):
        nb_params = self.optimal_params.size
        self.aic = 2 * nb_params + 2 * self.neg_log_likelihood
        self.aicc = self.aic + 2 * nb_params * (nb_params + 1) / (self.nb_obversations - nb_params - 1)
        self.bic = np.log(self.nb_obversations) * nb_params + 2 * self.neg_log_likelihood

        self.se = None
        if self.covariance_matrix is not None:
            self.se = np.sqrt(np.diag(self.covariance_matrix))
            self.ic = self.optimal_params.reshape(-1, 1) + stats.norm.ppf((0.05, 0.95)) * self.se.reshape(
                -1, 1
            ) / np.sqrt(
                self.nb_obversations
            )  # (p, 2)

    def se_estimation_function(self, jac_f: NDArray[np.float64]) -> np.float64 | NDArray[np.float64]:
        """Standard error estimation function.

        Parameters
        ----------
        jac_f : 1D, 2D or 3D array
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
            if jac_f.ndim == 1:  # jac_f : (p,)
                return np.sqrt(np.einsum("i,ij,j->", jac_f, self.covariance_matrix, jac_f))  # ()
            if jac_f.ndim == 2:  # jac_f : (p, n)
                return np.sqrt(np.einsum("in,ij,jn->n", jac_f, self.covariance_matrix, jac_f))  # (n,)
            if jac_f.ndim == 3:  # jac_f : (p, m, n) if regression with more than one asset
                return np.sqrt(np.einsum("imn,ij,jmn->mn", jac_f, self.covariance_matrix, jac_f))  # (m,n)
            raise ValueError("Invalid jac_f ndim")
        raise ValueError("Can't compute if var is None")

    @override
    def __str__(self) -> str:
        """Returns a string representation of FittingResults with fields in a single column."""
        fields = {
            "fitted params": self.optimal_params,
            "AIC": self.aic,
            "AICc": self.aicc,
            "BIC": self.bic,
        }
        # Find the maximum field name length for alignment
        max_name_length = max(len(name) for name, _ in fields.items())
        lines: list[str] = []
        for name, value in fields.items():
            # Format arrays to be more compact
            if isinstance(value, np.ndarray):
                value_str = f"[{', '.join(f'{x:.6g}' for x in value)}]"
            else:
                value_str = f"{value:.6g}" if isinstance(value, float) else str(value)
            lines.append(f"{name:<{max_name_length}} : {value_str}")
        return "\n".join(lines)
