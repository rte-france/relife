from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import List, Tuple

import numpy as np
from scipy.optimize import Bounds, OptimizeResult, root_scalar

from ..data import Data


class Parameters:
    def __init__(self, nb_params: int, param_names: List[str] = None):
        if type(nb_params) != int:
            raise TypeError("nb_params must be int")
        self.nb_params = nb_params
        if param_names is not None:
            if {type(name) for name in param_names} != {str}:
                raise ValueError("param_names must be string")
            if len(param_names) != nb_params:
                raise ValueError(
                    "param_names must be have the same number of element than"
                    " parameter numbers"
                )
            self.param_names = param_names
        else:
            self.param_names = [f"param_{i}" for i in range(nb_params)]

        self.values = np.random.rand(self.nb_params)
        self.params_index = {
            name: i for i, name in enumerate(self.param_names)
        }

    def __len__(self):
        return self.nb_params

    def __getitem__(self, i):
        return self.values[i]

    def __setitem__(self, param_name: str, value: float):
        if param_name in self.params_index:
            self.values[self.params_index[param_name]] = value
        else:
            raise AttributeError(
                f"""
                Parameter has no attr called {param_name}
                """
            )

    def __getattr__(self, attr: str):
        """
        called if attr is not found in attributes of the class
        (different from __getattribute__)
        """
        if attr in self.params_index:
            return self.values[self.params_index[attr]]
        else:
            raise AttributeError(
                f"""
                Parameter has no attr called {attr}
                """
            )

    def __str__(self):
        class_name = type(self).__name__
        res = [
            f"{name} = {getattr(self, name)} \n" for name in self.param_names
        ]
        res = "".join(res)
        return f"\n{class_name}\n{res}"


class ProbabilityFunctions(ABC):
    def __init__(self, nb_params: int, param_names: List[str] = None):
        self.params = Parameters(nb_params, param_names=param_names)

    @abstractmethod
    def hf(self, time: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def chf(self, time: np.ndarray) -> np.ndarray:
        pass

    # relife/parametric.ParametricLifetimeModel
    def sf(self, time: np.ndarray) -> np.ndarray:
        """Parametric survival function."""
        return np.exp(-self.chf(time))

    # relife/parametric.ParametricLifetimeModel
    def cdf(self, time: np.ndarray) -> np.ndarray:
        """Parametric cumulative distribution function."""
        return 1 - self.sf(time)

    # relife/parametric.ParametricLifetimeModel
    def pdf(self, time: np.ndarray) -> np.ndarray:
        """Parametric probability density function."""
        return self.hf(time) * self.sf(time)

    def isf(self, probability: np.ndarray) -> np.ndarray:
        """Approx of isf using scipy.optimize in case it is not defined in subclass pf"""
        res = root_scalar(
            lambda x: self.sf(x) - probability,
            method="newton",
            x0=0.0,
        )
        return res.root

    # relife/model.LifetimeModel
    def rvs(self, size: int = 1, random_state: int = None) -> np.ndarray:
        probabilities = np.random.RandomState(seed=random_state).uniform(
            size=size
        )
        return self.isf(probabilities)

    # relife/model.LifetimeModel
    def ls_integrate(self):
        pass

    # relife/model.LifetimeModel
    def moment(self):
        """
        Depends upon ls_integrate
        """
        pass

    # relife/model.LifetimeModel
    def ppf(self, probability: np.ndarray) -> np.ndarray:
        return self.isf(1 - probability)

    # relife/model.LifetimeModel
    def median(self) -> float:
        return self.ppf(0.5)

    def mean(self) -> float:
        """
        Depends upon ls_integrate IF NOT specified in subclass
        """
        pass

    def var(self) -> float:
        """
        Depends upon ls_integrate IF NOT specified in subclass
        """
        pass

    def mrl(self, time: np.ndarray) -> np.ndarray:
        """
        Depends upon ls_integrate IF NOT specified in subclass
        """
        pass


class CovarEffect(ABC):
    def __init__(self, nb_covar: int):
        self.params = Parameters(
            nb_covar, param_names=[f"beta_{i}" for i in range(nb_covar)]
        )

    @abstractmethod
    def g(self, covar: np.ndarray) -> np.ndarray:
        pass


class Likelihood(ABC):
    def __init__(
        self,
        observed_lifetimes: np.ndarray,
        complete_indicators: np.ndarray = None,
        left_censored_indicators: np.ndarray = None,
        right_censored_indicators: np.ndarray = None,
        entry: np.ndarray = None,
        departure: np.ndarray = None,
    ):

        self.data = Data(
            observed_lifetimes,
            complete_indicators,
            left_censored_indicators,
            right_censored_indicators,
            entry,
            departure,
        )

        # relife/parametric.ParametricHazardFunction
        self._default_hess_scheme: str = (  #: Default method for evaluating the hessian of the negative log-likelihood.
            "cs"
        )

    @abstractmethod
    def negative_log_likelihood(self, pf: ProbabilityFunctions) -> float:
        pass

    @abstractmethod
    def jac_negative_log_likelihood(
        self, pf: ProbabilityFunctions
    ) -> np.ndarray:
        pass

    @abstractmethod
    def hess_negative_log_likelihood(
        self, pf: ProbabilityFunctions, **kwargs
    ) -> np.ndarray:
        pass


class Optimizer(ABC):
    def __init__(self, pf: ProbabilityFunctions, likelihood: Likelihood):

        self.method: str = "L-BFGS-B"
        self.param0 = self._init_param(likelihood, pf.params.nb_params)
        self.bounds = self._get_param_bounds(pf)

    # relife/distribution.ParametricLifetimeDistbution
    def _init_param(
        self, likelihood: Likelihood, nb_params: int
    ) -> np.ndarray:
        param0 = np.ones(nb_params)
        param0[-1] = 1 / np.median(
            np.concatenate(
                [
                    data.values
                    for data in likelihood.data(
                        "complete | right_censored | left_censored"
                    )
                ]
            )
        )
        return param0

    def _get_param_bounds(self, pf: ProbabilityFunctions) -> Bounds:
        MIN_POSITIVE_FLOAT = np.finfo(float).resolution
        return Bounds(
            np.full(pf.params.nb_params, MIN_POSITIVE_FLOAT),
            np.full(pf.params.nb_params, np.inf),
        )

    @abstractmethod
    def fit(
        self, pf: ProbabilityFunctions, likelihood: Likelihood, **kwargs
    ) -> Tuple[ProbabilityFunctions, OptimizeResult]:
        pass


@dataclass
class FittingResults:
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

        Parameter
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
