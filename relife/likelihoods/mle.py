import copy
from dataclasses import dataclass, InitVar, field, asdict
from typing import Optional, NewType, TypeVarTuple, Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import OptimizeResult, minimize

from relife.distributions.protocols import ParametricLifetimeDistribution
from .likelihoods import LikelihoodFromLifetimes
from ..data import lifetime_data_factory

Z = TypeVarTuple("Z")
T = NewType("T", NDArray[np.floating] | NDArray[np.integer] | float | int)


@dataclass
class FittingResults:
    """Fitting results of the parametric core."""

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


def maximum_likelihood_estimation(
    distribution: ParametricLifetimeDistribution[*Z],
    time: NDArray[np.float64],
    /,
    *z: *Z,
    event: Optional[NDArray[np.bool_]] = None,
    entry: Optional[NDArray[np.float64]] = None,
    departure: Optional[NDArray[np.float64]] = None,
    **kwargs: Any,
) -> FittingResults:
    # Step 1: Prepare lifetime data
    lifetime_data = lifetime_data_factory(
        time,
        event,
        entry,
        departure,
    )

    # Step 2: Initialize the model and likelihood
    optimized_model = copy.deepcopy(distribution)
    optimized_model.init_params(lifetime_data, *z)
    likelihood = LikelihoodFromLifetimes(optimized_model, lifetime_data, model_args=z)

    # Step 3: Configure and run the optimizer
    minimize_kwargs = {
        "method": kwargs.get("method", "L-BFGS-B"),
        "constraints": kwargs.get("constraints", ()),
        "tol": kwargs.get("tol", None),
        "callback": kwargs.get("callback", None),
        "options": kwargs.get("options", None),
        "bounds": kwargs.get("bounds", optimized_model.params_bounds),
        "x0": kwargs.get("x0", optimized_model.params),
    }
    optimizer = minimize(
        likelihood.negative_log,
        minimize_kwargs.pop("x0"),
        jac=None if not likelihood.hasjac else likelihood.jac_negative_log,
        **minimize_kwargs,
    )

    # Step 4: Compute parameters variance (Hessian inverse)
    hessian_inverse = np.linalg.inv(likelihood.hessian())
    fitting_results = optimized_model.fitting_results = FittingResults(
        len(lifetime_data), optimizer, hessian_inverse
    )
    return fitting_results
