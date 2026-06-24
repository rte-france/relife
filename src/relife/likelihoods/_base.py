import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Generic, Literal, TypeVar

import numpy as np
from optype.numpy import Array1D, Array2D, ToFloat, ToFloat1D
from scipy.optimize import approx_fprime, minimize

from relife.base import FittingResults, ParametricModel

__all__ = ["MaximumLikelihoodOptimizer"]


M = TypeVar("M", bound=ParametricModel)
D = TypeVar("D")


@dataclass
class FitConfig:
    x0: ToFloat | ToFloat1D
    scipy_minimize_options: dict[str, Any] = field(default_factory=dict)
    covariance_method: Literal["cs", "2point", "exact", False] = False


class MaximumLikelihoodOptimizer(Generic[M, D], ABC):
    """
    Abstract maximum likelihood optimizer.

    Notes
    -----
    Jacobian and hessian are not required but they can be implemented in
    concrete likelihoods. To use the jacobian or hessian implementations in the
    likelihood, pass them into `config["scipy_minimize_options"]`.

    Attributes
    ----------
    nb_observations : int
        The number of observations.
    """

    model: M
    data: D
    config: FitConfig

    @property
    @abstractmethod
    def nb_observations(self) -> int: ...

    @abstractmethod
    def negative_log(self, params: Array1D[np.float64]) -> float:
        """
        Negative log likelihood.

        Parameters
        ----------
        params : 1d array
            Parameters values.

        Returns
        -------
        out : np.float64
            Negative log likelihood value.
        """

    def optimize(self) -> FittingResults:
        """
        Search parameters values that maximize the likelihood given data.

        Returns
        -------
        out : FittingResults
            An object that encapsulates optimal parameters and fitting
            information (AIC, variance, etc.).
        """

        optimizer = minimize(
            self.negative_log,
            self.config.x0,
            **self.config.scipy_minimize_options,
        )

        fitting_results = FittingResults(
            self.nb_observations,
            np.copy(optimizer.x),
            optimizer.success,
            optimizer.fun,
        )

        if not fitting_results.success:
            warnings.warn(
                "The negative log-likelihood minimization did not exited successfully.",
                stacklevel=2,
            )

        if self.config.covariance_method is False:
            return fitting_results

        jac = self.config.scipy_minimize_options.get("jac", None)
        hess = self.config.scipy_minimize_options.get("hess", None)
        if jac is not None and self.config.covariance_method != "exact":
            fitting_results.covariance_matrix = approx_parameters_covariance(
                fitting_results.optimal_params,
                jac,
                method=self.config.covariance_method,
            )
        if hess is not None and self.config.covariance_method == "exact":
            fitting_results.covariance_matrix = np.linalg.pinv(
                hess(fitting_results.optimal_params)
            )
        return fitting_results


def approx_parameters_covariance(
    params: Array1D[np.float64],
    jac_negative_log: Callable[[Array1D[np.number]], Array1D[np.number]],
    method: Literal["2point", "cs"] = "cs",
) -> Array2D[np.float64] | None:
    """
    Approximate parameters covariance.

    Parameters
    ----------
    params : 1darray of float
        The parameters values.
    jac_negative_log : callable
        A function taking 1d array of numbers and returning 1d array of numbers.
    method : "2point" or "cs", default to "cs"
        The approximation method to use.
    """

    size = params.size
    eps = 1e-6
    hess = np.empty((size, size), dtype=np.float64)

    # hessian 2 point
    if method == "2point":
        for i in range(size):
            hess[i] = approx_fprime(
                params,
                lambda x: jac_negative_log(x)[i],
                eps,
            )
        return hess
    # hessian cs
    u = eps * 1j * np.eye(size)
    complex_params = params.astype(np.complex64)  # change params to complex
    for i in range(size):
        for j in range(i, size):
            hess[i, j] = np.imag(jac_negative_log(complex_params + u[i])[j]) / eps
            if i != j:
                hess[j, i] = hess[i, j]
    covariance_matrix = None
    try:
        covariance_matrix = np.linalg.pinv(hess).astype(np.float64)
    except Exception as err:
        warnings.warn(
            f"""
            Failed to compute parameters covariance due to non-invertible
            hessian matrix. Numpy pseudo-inversion algorithm returned : {err}

            You can skip parameters covariance computation by setting
            covariance_method to False. 
            """,
            stacklevel=2,
        )

    return covariance_matrix
