from __future__ import annotations

import copy
from dataclasses import InitVar, asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import Bounds, OptimizeResult, minimize

from .lifetime_likelihood import LikelihoodFromLifetimes

if TYPE_CHECKING:
    from relife import ParametricModel
    from relife.data import FailureData, LifetimeData
    from relife.lifetime_model import ParametricLifetimeModel


@dataclass
class FittingResults:
    """Fitting results of the parametric_model core."""

    nb_samples: InitVar[int]  #: Number of observations (samples).

    opt: InitVar[OptimizeResult] = field(
        repr=False
    )  #: Optimization result (see scipy.optimize.OptimizeResult doc).
    var: Optional[NDArray[np.float64]] = field(
        repr=False, default=None
    )  #: Covariance matrix (computed as the inverse of the Hessian matrix)
    se: NDArray[np.float64] = field(
        init=False, repr=False
    )  #: Standard error, square root of the diagonal of the covariance matrix.

    params: NDArray[np.float64] = field(init=False)  #: Optimal parameters values
    nb_params: int = field(init=False)  #: Number of parameters.
    AIC: float = field(init=False)  #: Akaike Information Criterion.
    AICc: float = field(
        init=False
    )  #: Akaike Information Criterion with a correction for small sample sizes.
    BIC: float = field(init=False)  #: Bayesian Information Criterion.

    def __post_init__(self, nb_samples, opt):
        self.params = opt.x
        self.nb_params = opt.x.size
        self.AIC = 2 * self.nb_params + 2 * opt.fun
        self.AICc = self.AIC + 2 * self.nb_params * (self.nb_params + 1) / (
            nb_samples - self.nb_params - 1
        )
        self.BIC = np.log(nb_samples) * self.nb_params + 2 * opt.fun

        self.se = None
        if self.var is not None:
            self.se = np.sqrt(np.diag(self.var))

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


def params_bounds(model: ParametricModel) -> Bounds:
    from relife.lifetime_model import LifetimeDistribution, LifetimeRegression

    match model:

        case LifetimeDistribution():
            model: LifetimeDistribution
            return Bounds(
                np.full(model.nb_params, np.finfo(float).resolution),
                np.full(model.nb_params, np.inf),
            )

        case LifetimeRegression():
            model: LifetimeRegression
            lb = np.concatenate(
                (
                    np.full(model.covar_effect.nb_params, -np.inf),
                    params_bounds(model.baseline).lb,
                )
            )
            ub = np.concatenate(
                (
                    np.full(model.covar_effect.nb_params, np.inf),
                    params_bounds(model.baseline).ub,
                )
            )
            return Bounds(lb, ub)

        case _:
            raise NotImplemented


def init_params_from_lifetimes(
    model: ParametricLifetimeModel[*tuple[float | NDArray[np.float64], ...]],
    lifetime_data: LifetimeData,
) -> NDArray[np.float64]:
    from relife.lifetime_model import (
        AFT,
        Exponential,
        Gamma,
        Gompertz,
        LifetimeDistribution,
        LifetimeRegression,
        LogLogistic,
        ProportionalHazard,
        Weibull,
    )

    match model:

        case Exponential() | Weibull() | LogLogistic() | Gamma():
            model: LifetimeDistribution
            param0 = np.ones(model.nb_params, dtype=np.float64)
            param0[-1] = 1 / np.median(lifetime_data.complete_or_right_censored.values)
            return param0

        case Gompertz():
            model: LifetimeDistribution
            param0 = np.empty(model.nb_params, dtype=np.float64)
            rate = np.pi / (
                np.sqrt(6) * np.std(lifetime_data.complete_or_right_censored.values)
            )
            shape = np.exp(
                -rate * np.mean(lifetime_data.complete_or_right_censored.values)
            )
            param0[0] = shape
            param0[1] = rate
            return param0

        case ProportionalHazard() | AFT():
            model: LifetimeRegression
            baseline_param0 = init_params_from_lifetimes(model.baseline, lifetime_data)
            param0 = np.zeros_like(model.params, dtype=np.float64)
            param0[-baseline_param0.size :] = baseline_param0
            return param0

        case _:
            raise NotImplemented


def maximum_likelihood_estimation(
    model: ParametricModel, data: FailureData, **kwargs: Any
) -> ParametricModel:
    from relife.lifetime_model import LifetimeDistribution, LifetimeRegression
    from relife.stochastic_process import NonHomogeneousPoissonProcess

    match model:
        case LifetimeDistribution() | LifetimeRegression():
            from relife.data import LifetimeData

            if not isinstance(data, LifetimeData):
                raise ValueError

            # Step 2: Initialize the model and likelihood
            model.params = init_params_from_lifetimes(model, data)
            likelihood = LikelihoodFromLifetimes(model, data)

            try:
                bounds = params_bounds(model)
            except NotImplemented:
                bounds = None

            # Step 3: Configure and run the optimizer
            minimize_kwargs = {
                "method": kwargs.get("method", "L-BFGS-B"),
                "constraints": kwargs.get("constraints", ()),
                "tol": kwargs.get("tol", None),
                "callback": kwargs.get("callback", None),
                "options": kwargs.get("options", None),
                "bounds": kwargs.get("bounds", bounds),
                "x0": kwargs.get("x0", model.params),
            }
            optimizer = minimize(
                likelihood.negative_log,
                minimize_kwargs.pop("x0"),
                jac=None if not likelihood.hasjac else likelihood.jac_negative_log,
                **minimize_kwargs,
            )

            # Step 4: Compute parameters variance (Hessian inverse)
            hessian_inverse = np.linalg.inv(likelihood.hessian())
            model.fitting_results = FittingResults(
                len(data), optimizer, var=hessian_inverse
            )
            model.params = optimizer.x
            return model

        case NonHomogeneousPoissonProcess():
            model: NonHomogeneousPoissonProcess[
                *tuple[float | NDArray[np.float64], ...]
            ]
            from relife.data import NHPPData

            if not isinstance(data, NHPPData):
                raise ValueError

            lifetime_data = data.to_lifetime_data()

            return maximum_likelihood_estimation(
                model.baseline, lifetime_data, **kwargs
            )

        case _:
            raise NotImplemented
