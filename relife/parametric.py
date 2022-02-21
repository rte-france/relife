"""Generic parametric models."""

# Copyright (c) 2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# SPDX-License-Identifier: Apache-2.0 (see LICENSE.txt)
# This file is part of ReLife, an open source Python library for asset
# management based on reliability theory and lifetime data analysis.

from abc import abstractmethod
from dataclasses import dataclass, field, asdict
import numpy as np
from scipy.optimize import minimize, Bounds, OptimizeResult
from scipy.optimize.optimize import approx_fprime

from .data import LifetimeData
from .model import HazardFunctions, AbsolutelyContinuousLifetimeModel


class ParametricHazardFunctions(HazardFunctions):
    """Generic parametric hazard functions.

    Generic class for the parametric hazard functions with a fitting method based
    on the maximum likelihood estimator.
    """

    _default_method: str = (
        "L-BFGS-B"  #: Default method for minimizing the negative log-likelihood.
    )
    _default_hess_scheme: str = "cs"  #: Default method for evaluating the hessian of the negative log-likelihood.

    @property
    @abstractmethod
    def params(self) -> np.ndarray:
        """Parameters of the hazard functions."""
        pass

    @property
    @abstractmethod
    def n_params(self) -> int:
        """Number of parameters."""
        pass

    @property
    @abstractmethod
    def _param_bounds(self) -> Bounds:
        """Bounds of the parameters for the fit."""
        pass

    @abstractmethod
    def _set_params(self, params: np.ndarray) -> None:
        """Sets the parameters after the fit."""

        pass

    @abstractmethod
    def _init_params(self, data: LifetimeData) -> np.ndarray:
        """Initial guess for the fit."""
        pass

    @abstractmethod
    def _chf(self, params: np.ndarray, t: np.ndarray, *args: np.ndarray) -> np.ndarray:
        """Parametric cumulative hazard function."""
        pass

    @abstractmethod
    def _hf(self, params: np.ndarray, t: np.ndarray, *args: np.ndarray) -> np.ndarray:
        """Parametric hazard function."""
        pass

    @abstractmethod
    def _dhf(self, params: np.ndarray, t: np.ndarray, *args: np.ndarray) -> np.ndarray:
        """Derivative of the parametric hazard function with repspect to t."""
        pass

    @abstractmethod
    def _jac_chf(
        self, params: np.ndarray, t: np.ndarray, *args: np.ndarray
    ) -> np.ndarray:
        """Jacobian of the parametric cumulative hazard function with respect to params."""
        pass

    @abstractmethod
    def _jac_hf(
        self, params: np.ndarray, t: np.ndarray, *args: np.ndarray
    ) -> np.ndarray:
        """Jacobian of the parametric hazard function with respect to params."""
        pass

    @abstractmethod
    def _ichf(self, params: np.ndarray, x: np.ndarray, *args: np.ndarray) -> np.ndarray:
        """Inverse of the parametric cumulative hazard function."""
        pass

    def chf(self, t: np.ndarray, *args: np.ndarray) -> np.ndarray:
        return self._chf(self.params, t, *args)

    def hf(self, t: np.ndarray, *args: np.ndarray) -> np.ndarray:
        return self._hf(self.params, t, *args)

    def ichf(self, v: np.ndarray, *args: np.ndarray) -> np.ndarray:
        return self._ichf(self.params, v, *args)

    def _negative_log_likelihood(
        self, params: np.ndarray, data: LifetimeData
    ) -> np.ndarray:
        """Negative log-likelihood function.

        Parameters
        ----------
        params : 1D array
            Parameters of the hazard functions.
        data : LifetimeData
            Lifetime data with time, event, entry and args attributes.

        Returns
        -------
        float
            The negative log-likelihood for the lifetime `data` evaluated at
            `params`.
        """
        return (
            -np.sum(np.log(self._hf(params, data._time.D, *data._args.D)))
            + np.sum(self._chf(params, data._time.D_RC, *data._args.D_RC))
            - np.sum(self._chf(params, data._time.LT, *data._args.LT))
            - np.sum(
                np.log(-np.expm1(-self._chf(params, data._time.LC, *data._args.LC)))
            )
        )

    def _jac_negative_log_likelihood(
        self, params: np.ndarray, data: LifetimeData
    ) -> np.ndarray:
        """Jacobian of the negative log-likelihood function.

        Parameters
        ----------
        params : 1D array
            Parameters of the hazard functions.
        data : LifetimeData
            Lifetime data with time, event, entry and args attributes.

        Returns
        -------
        1D array
            The Jacobian of the negative log-likelihood evaluated at `params` with
            `data`.
        """
        return (
            -np.sum(
                self._jac_hf(params, data._time.D, *data._args.D)
                / self._hf(params, data._time.D, *data._args.D),
                axis=0,
            )
            + np.sum(self._jac_chf(params, data._time.D_RC, *data._args.D_RC), axis=0)
            - np.sum(self._jac_chf(params, data._time.LT, *data._args.LT), axis=0)
            - np.sum(
                self._jac_chf(params, data._time.LC, *data._args.LC)
                / np.expm1(self._chf(params, data._time.LC, *data._args.LC)),
                axis=0,
            )
        )

    def _hess_negative_log_likelihood(
        self,
        params: np.ndarray,
        data: LifetimeData,
        eps: float = 1e-6,
        scheme: str = None,
    ) -> np.ndarray:
        """Hessian of the negative log-likelihood.

        Parameters
        ----------
        params : 1D array
            Parameters of the hazard functions.
        data : LifetimeData
            Lifetime data with time, event, entry and args attributes.
        eps : float, optional
            Increment to params for computing the numerical approximation,
            by default 1e-6.
        scheme : str, optional
            Approximation method for computing the hessian matrix, 2 options:

                - "cs" (Complex Step derivative approximation)
                - "2-point" (Finite-Difference approximation)

            by default None.

        Returns
        -------
        2D array
            The Hessian of the negative log-likelihood.

        Raises
        ------
        ValueError
            If the scheme argument is not 'cs' or '2-point'.
        """
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
                            self._jac_negative_log_likelihood(params + u[i], data)[j]
                        )
                        / eps
                    )
                    if i != j:
                        hess[j, i] = hess[i, j]

        elif scheme == "2-point":
            for i in range(size):
                hess[i] = approx_fprime(
                    params,
                    lambda params: self._jac_negative_log_likelihood(params, data)[i],
                    eps,
                )

        else:
            raise ValueError("scheme argument must be 'cs' or '2-point'")

        return hess

    def _fit(
        self,
        data: LifetimeData,
        params0: np.ndarray = None,
        method: str = None,
        eps: float = 1e-6,
        hess_scheme: str = None,
        **kwargs,
    ) -> None:
        """Maximum likelihood estimation of the parameters

        Fit a parametric hazard function to lifetime data, by minimizing the
        negative log-likelihood.

        Parameters
        ----------
        data : LifetimeData
            Lifetime data with time, event, entry and args attributes.
        params0 : 1D array, optional
            Initial guess for the parameters, by default None.
        method : str, optional
            Type of solver (see scipy.optimize.minimize documentation), by
            default None.
        eps : float, optional
            Increment to params for computing the numerical approximation,
            by default 1e-6.
        hess_scheme : str, optional
            Approximation method for computing the hessian matrix, 2 options:

                - "cs" (Complex Step derivative approximation)
                - "2-point" (Finite-Difference approximation),

            by default None.

        Raises
        ------
        ValueError
            If `params0` size does not match the expected size of the hazard
            functions parameters.
        """
        if params0 is not None:
            params0 = np.asanyarray(params0, float)
            if self.n_params != params0.size:
                raise ValueError(
                    f"Wrong dimension for params0, expected {self.n_params} got {params0.size}"
                )
        else:
            params0 = self._init_params(data)
        if method is None:
            method = self._default_method
        jac = kwargs.pop("jac", self._jac_negative_log_likelihood)
        bounds = kwargs.pop("bounds", self._param_bounds)
        opt = minimize(
            self._negative_log_likelihood,
            params0,
            args=(data),
            method=method,
            jac=jac,
            bounds=bounds,
            **kwargs,
        )
        jac = self._jac_negative_log_likelihood(opt.x, data)
        var = np.linalg.inv(
            self._hess_negative_log_likelihood(opt.x, data, eps, hess_scheme)
        )
        self._set_params(opt.x)
        self._set_fitting_result(opt, jac, var, data)

    def _set_fitting_result(
        self, opt, jac: np.ndarray, var: np.ndarray, data: LifetimeData
    ) -> None:
        """Set the fitting result.

        Create a `result` attribute which store fitting information in an
        instance of FittingResult.

        Parameters
        ----------
        opt : OptimizeResult
            Represents the optimization result (see
            scipy.optimize.OptimizeResult documentation).
        jac : 1D array
            Jacobian of the negative log-likelihood of the data.
        var : 2D array
            Covariance matrix (computed as the inverse of the Hessian matrix).
        data : LifetimeData
            Lifetime data used for the fit.
        """
        self.result = FittingResult(opt, jac, var, data.size)


@dataclass
class FittingResult:
    """Fitting results of the parametric model."""

    opt: OptimizeResult = field(
        repr=False
    )  #: Optimization result (see scipy.optimize.OptimizeResult doc).
    jac: np.ndarray = field(
        repr=False
    )  #: Jacobian of the negative log-likelihood with the lifetime data.
    var: np.ndarray = field(
        repr=False
    )  #: Covariance matrix (computed as the inverse of the Hessian matrix)
    se: np.ndarray = field(
        init=False, repr=False
    )  #: Standard error, square root of the diagonal of the covariance matrix.
    n_samples: int  #: Number of observations (samples).
    n_params: int = field(init=False)  #: Number of parameters.
    AIC: float = field(init=False)  #: Akaike Information Criterion.
    AICc: float = field(
        init=False
    )  #: Akaike Information Criterion with a correction for small sample sizes.
    BIC: float = field(init=False)  #: Bayesian Information Criterion.

    def __post_init__(self):
        self.se = np.sqrt(np.diag(self.var))
        self.n_params = self.opt.x.size
        self.AIC = 2 * self.n_params + 2 * self.opt.fun
        self.AICc = self.AIC + 2 * self.n_params * (self.n_params + 1) / (
            self.n_samples - self.n_params - 1
        )
        self.BIC = np.log(self.n_samples) * self.n_params + 2 * self.opt.fun

    def standard_error(self, jac_f: np.ndarray) -> np.ndarray:
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
        return np.sqrt(np.einsum("ni,ij,nj->n", jac_f, self.var, jac_f))

    def asdict(self) -> dict:
        """converts FittingResult into a dictionary.

        Returns
        -------
        dict
            Returns the fitting result as a dictionary.
        """
        return asdict(self)


class ParametricLifetimeModel(
    ParametricHazardFunctions, AbsolutelyContinuousLifetimeModel
):
    """Parametric lifetime model."""

    def _sf(self, params: np.ndarray, t: np.ndarray, *args: np.ndarray) -> np.ndarray:
        """Parametric survival function."""
        return np.exp(-self._chf(params, t, *args))

    def _cdf(self, params: np.ndarray, t: np.ndarray, *args: np.ndarray) -> np.ndarray:
        """Parametric cumulative distribution function."""
        return 1 - self._sf(params, t, *args)

    def _pdf(self, params: np.ndarray, t: np.ndarray, *args: np.ndarray) -> np.ndarray:
        """Parametric probability density function."""
        return self._hf(params, t, *args) * self._sf(params, t, *args)

    def _jac_sf(
        self, params: np.ndarray, t: np.ndarray, *args: np.ndarray
    ) -> np.ndarray:
        """Jacobian of the parametric survival function with respect to params."""
        return -self._jac_chf(params, t, *args) * self._sf(params, t, *args)

    def _jac_cdf(
        self, params: np.ndarray, t: np.ndarray, *args: np.ndarray
    ) -> np.ndarray:
        """Jacobian of the parametric cumulative distribution function with
        respect to params."""
        return -self._jac_sf(params, t, *args)

    def _jac_pdf(
        self, params: np.ndarray, t: np.ndarray, *args: np.ndarray
    ) -> np.ndarray:
        """Jacobian of the parametric probability density function with respect to
        params."""
        return self._jac_hf(params, t, *args) * self._sf(
            params, t, *args
        ) + self._jac_sf(params, t, *args) * self._hf(params, t, *args)
