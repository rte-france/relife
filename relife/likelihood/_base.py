# TODO : déplacer dans relife.base (avoir circular import)
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Generic, TypeVar, Unpack

import numpy as np
from numpy.typing import NDArray
from optype.numpy import Array1D, Array2D, ToFloat
from scipy import stats
from scipy.optimize import Bounds, minimize
from typing_extensions import override

from relife.base import ParametricModel
from relife.typing import MethodMinimize, ScipyMinimizeOptions


# TODO : sortir se_estimation (utile seulement pour les plot, donc à mettre avec)
@dataclass
class FittingResults:
    """Fitting results of the parametric_model core."""

    nb_obversations: int  #: Number of observations (samples)
    optimal_params: NDArray[np.float64] = field(
        repr=False
    )  #: Optimal parameters values
    neg_log_likelihood: float = field(
        repr=False
    )  #: Negative log likelihood value at optimal parameters values

    covariance_matrix: Array2D[np.float64] | None = field(
        repr=False, default=None
    )  #: Covariance matrix (computed as the inverse of the Hessian matrix).

    nb_params: int = field(init=False, repr=False)  #: Number of parameters.
    aic: float = field(init=False)  #: Akaike Information Criterion.
    aicc: float = field(
        init=False
    )  #: Akaike Information Criterion with a correction for small sample sizes.
    bic: float = field(init=False)  #: Bayesian Information Criterion.
    se: NDArray[np.float64] | None = field(
        init=False, repr=False
    )  #: Standard error, square root of the diagonal of the covariance matrix
    ic: NDArray[np.float64] | None = field(init=False, repr=False)  #: 95% IC

    def __post_init__(self):
        self.nb_params = self.optimal_params.size
        self.aic = 2 * self.nb_params + 2 * self.neg_log_likelihood
        self.aicc = self.aic + 2 * self.nb_params * (self.nb_params + 1) / (
            self.nb_obversations - self.nb_params - 1
        )
        self.bic = (
            np.log(self.nb_obversations) * self.nb_params + 2 * self.neg_log_likelihood
        )
        self.se = None
        if self.covariance_matrix is not None:
            self.se = np.sqrt(np.diag(self.covariance_matrix))
            self.ic = self.optimal_params.reshape(-1, 1) + stats.norm.ppf(
                (0.05, 0.95)
            ) * self.se.reshape(-1, 1) / np.sqrt(self.nb_obversations)  # (p, 2)

    def se_estimation_function(
        self, jac_f: NDArray[np.float64]
    ) -> np.float64 | NDArray[np.float64]:
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
                return np.sqrt(
                    np.einsum("i,ij,j->", jac_f, self.covariance_matrix, jac_f)
                )  # ()
            if jac_f.ndim == 2:  # jac_f : (p, n)
                return np.sqrt(
                    np.einsum("in,ij,jn->n", jac_f, self.covariance_matrix, jac_f)
                )  # (n,)
            if (
                jac_f.ndim == 3
            ):  # jac_f : (p, m, n) if regression with more than one asset
                return np.sqrt(
                    np.einsum("imn,ij,jmn->mn", jac_f, self.covariance_matrix, jac_f)
                )  # (m,n)
            raise ValueError("Invalid jac_f ndim")
        raise ValueError("Can't compute if var is None")

    @override
    def __str__(self) -> str:
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


M = TypeVar("M", bound=ParametricModel)
D = TypeVar("D")


# TODO : mettre dans relife.base, à côté de ParametricModel (evite les imports circulaire)
class MaximumLikehoodOptimizer(Generic[M, D], ABC):
    """
    Abstract maximum likelihood optimizer.

    Notes
    -----
    Jacobian and hessian are not required but they can be passed as additional
    arguments to `**optimizer_options` at runtime or in subclass implementions
    by overriding `maximum_likelihood_estimation`.
    """

    model: M
    data: D
    scipy_method: MethodMinimize = "L-BFGS-B"

    @property
    @abstractmethod
    def nb_observations(self) -> int: ...

    @abstractmethod
    def _initialize_model(self) -> M: ...

    @abstractmethod
    def _get_params_bounds(self) -> Bounds: ...

    @abstractmethod
    def negative_log(self, params: Array1D[np.float64]) -> ToFloat:
        """
        Negative log likelihood.

        Parameters
        ----------
        model : parametric model
            A parametrized model, ie. params values must be set.

        Returns
        -------
        out : float
            Negative log likelihood value.
        """

    def maximum_likelihood_estimation(
        self, **optimizer_options: Unpack[ScipyMinimizeOptions]
    ) -> FittingResults:
        """
        Search parameters values that maximize the likelihood given data.

        Parameters
        ----------
        **optimizer_options
            Any additional keyword arguments corresponding to optional
            `scipy.optimize.minimize` arguments.

        Returns
        -------
        out : FittingResults
            An object that encapsulates optimal parameters and fitting
            information (AIC, variance, etc.).
        """
        # set
        model = self._initialize_model()
        x0 = optimizer_options.pop("x0", None)
        if x0 is None:
            x0 = model.params.copy()

        method = optimizer_options.pop("method", self.scipy_method)
        bounds = optimizer_options.pop("bounds", self._get_params_bounds())
        jac = optimizer_options.pop("jac", None)
        hess = optimizer_options.pop("hess", None)

        optimizer = minimize(
            self.negative_log,
            x0=x0,
            jac=jac,
            hess=hess,
            method=method,
            bounds=bounds,
        )
        return FittingResults(
            self.nb_observations,
            np.copy(optimizer.x),
            optimizer.fun,
        )
