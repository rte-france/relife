from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Type

import numpy as np
from scipy.optimize import Bounds, OptimizeResult, minimize

from .. import DataBook
from .function import ParametricFunction
from .likelihood import ParametricLikelihood

MIN_POSITIVE_FLOAT = np.finfo(float).resolution


class ParametricOptimizer(ABC):
    @abstractmethod
    def fit(self) -> None:
        pass


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
    nb_samples: int  #: Number of observations (samples).
    nb_params: int = field(init=False)  #: Number of parameters.
    AIC: float = field(init=False)  #: Akaike Information Criterion.
    AICc: float = field(
        init=False
    )  #: Akaike Information Criterion with a correction for small sample sizes.
    BIC: float = field(init=False)  #: Bayesian Information Criterion.

    def __post_init__(self):
        self.se = np.sqrt(np.diag(self.var))
        self.nb_params = self.opt.x.size
        self.AIC = 2 * self.nb_params + 2 * self.opt.fun
        self.AICc = self.AIC + 2 * self.nb_params * (self.nb_params + 1) / (
            self.nb_samples - self.nb_params - 1
        )
        self.BIC = np.log(self.nb_samples) * self.nb_params + 2 * self.opt.fun

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


class DistriOptimizer(ParametricOptimizer):
    # relife/parametric.ParametricHazardFunction
    _default_method: str = (  #: Default method for minimizing the negative log-likelihood.
        "L-BFGS-B"
    )
    # relife/distribution.ParametricLifetimeDistribution

    def _init_param(
        self, databook: Type[DataBook], functions: ParametricFunction
    ) -> np.ndarray:
        param0 = np.ones(functions.nb_params)
        param0[-1] = 1 / np.median(databook("complete").values)
        return param0

    def get_param_bounds(self, functions: ParametricFunction) -> Bounds:
        return Bounds(
            np.full(functions.nb_params, MIN_POSITIVE_FLOAT),
            np.full(functions.nb_params, np.inf),
        )

    # relife/parametric.ParametricHazardFunctions
    def fit(
        self,
        databook: Type[DataBook],
        functions: Type[ParametricFunction],
        likelihood: Type[ParametricLikelihood],
        param0: np.ndarray = None,
        bounds=None,
        method: str = None,
        **kwargs,
    ) -> OptimizeResult:

        if param0 is not None:
            param0 = np.asanyarray(param0, float)
            if functions.nb_params != param0.size:
                raise ValueError(
                    "Wrong dimension for param0, expected"
                    f" {functions.nb_params} but got {param0.size}"
                )
        else:
            param0 = self._init_param(databook)
        if method is None:
            method = self._default_method
        if bounds is None:
            bounds = self._get_param_bounds(functions)

        opt = minimize(
            likelihood.negative_log_likelihood,
            param0,
            args=(databook, functions),
            method=method,
            jac=likelihood.jac,
            bounds=bounds,
            **kwargs,
        )

        return opt

        # self._set_params(opt.x)
