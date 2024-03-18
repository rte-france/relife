import warnings
from abc import ABC, abstractmethod
from dataclasses import InitVar, asdict, dataclass, field
from typing import Type, TypeVar

import numpy as np
from scipy.optimize import Bounds, OptimizeResult, root_scalar

from ..data import Data


@dataclass
class Parameters:
    nb_params: InitVar[int] = None
    param_names: InitVar[list] = None

    def __post_init__(self, nb_params, param_names):
        if nb_params is not None and param_names is not None:
            if {type(name) for name in param_names} != {str}:
                raise ValueError("param_names must be string")
            if len(param_names) != nb_params:
                raise ValueError(
                    "param_names must have same length as nb_params"
                )
            self.nb_params = nb_params
            self.param_names = param_names
        elif nb_params is not None and param_names is None:
            self.nb_params = nb_params
            self.param_names = [f"param_{i}" for i in range(nb_params)]
        elif nb_params is None and param_names is not None:
            if {type(name) for name in param_names} != {str}:
                raise ValueError("param_names must be string")
            self.nb_params = len(param_names)
            self.param_names = param_names
        else:
            raise ValueError(
                """
            Parameter expects at least nb_params or param_names
            """
            )

        self.values = np.random.rand(self.nb_params)
        self.fitting_params = None
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


class ParametricFunctions(ABC):
    def __init__(self, nb_params: int = None, param_names: list = None):
        self.params = Parameters(nb_params=nb_params, param_names=param_names)

    @abstractmethod
    def hf(self):
        pass

    @abstractmethod
    def chf(self):
        pass

    def isf(self, probability: np.ndarray):
        """Approx of isf using scipy.optimize in case it is not defined in subclass functions"""
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
    def ppf(self, probability: np.ndarray):
        return self.isf(1 - probability)

    # relife/model.LifetimeModel
    def median(self):
        return self.ppf(0.5)

    def mean(self):
        """
        Depends upon ls_integrate IF NOT specified in subclass
        """
        pass

    def var(self):
        """
        Depends upon ls_integrate IF NOT specified in subclass
        """
        pass

    def mrl(self):
        """
        Depends upon ls_integrate IF NOT specified in subclass
        """
        pass


class ParametricLikelihood(ABC):
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
    def negative_log_likelihood(self) -> float:
        pass

    @abstractmethod
    def jac_negative_log_likelihood(self) -> np.ndarray:
        pass

    @abstractmethod
    def hess_negative_log_likelihood(self) -> np.ndarray:
        pass


Likelihood = TypeVar("Likelihood", bound=ParametricLikelihood)
Functions = TypeVar("Functions", bound=ParametricFunctions)


class ParametricOptimizer(ABC):
    def __init__(self, functions: Functions, likelihood: Likelihood):

        self.method: str = "L-BFGS-B"
        self.param0 = self._init_param(likelihood, functions.params.nb_params)
        self.bounds = self._get_param_bounds(functions)

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

    def _get_param_bounds(self, functions: Functions) -> Bounds:
        MIN_POSITIVE_FLOAT = np.finfo(float).resolution
        return Bounds(
            np.full(functions.params.nb_params, MIN_POSITIVE_FLOAT),
            np.full(functions.params.nb_params, np.inf),
        )

    @abstractmethod
    def fit(
        self, functions: Functions, likelihood: Likelihood
    ) -> OptimizeResult:
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


class ParametricModel:
    def __init__(
        self,
        Functions: Type[ParametricFunctions],
        Likelihood: Type[ParametricLikelihood] = None,
        Optimizer: Type[ParametricOptimizer] = None,
        *params,
        **kparams,
    ):

        if not issubclass(Functions, ParametricFunctions):
            parent_classes = (Cls.__name__ for Cls in Functions.__bases__)
            raise ValueError(
                f"ParametricFunction subclass expected, got '{parent_classes}'"
            )

        if not issubclass(Likelihood, ParametricLikelihood):
            parent_classes = (Cls.__name__ for Cls in Functions.__bases__)
            raise ValueError(
                "ParametrictLikelihood subclass expected, got"
                f" '{parent_classes}'"
            )

        if not issubclass(Optimizer, ParametricOptimizer):
            parent_classes = (Cls.__name__ for Cls in Functions.__bases__)
            raise ValueError(
                "ParametricOptimizer subclass expected, got"
                f" {Optimizer.__name__} '{parent_classes}'"
            )

        self.functions = Functions()
        self.Likelihood = Likelihood
        self.Optimizer = Optimizer
        self._init_params_values = self.functions.params.values
        self._init_params(*params, **kparams)
        self._fitting_params = Parameters(
            self.functions.params.nb_params, self.functions.params.param_names
        )
        self._fitting_results = None

    def _init_params(self, *params, **kparams):

        if len(params) != 0 and len(kparams) != 0:
            raise ValueError("Can't specify key word params and params")
        elif len(params) != 0:
            params = np.array(params)
            params_shape = params.shape
            functions_params_shape = self.functions.params.values.shape
            if params.shape != self.functions.params.values.shape:
                raise ValueError(
                    "Incorrect params shape, expected shape"
                    f" {functions_params_shape} but got {params_shape}"
                )
            self.functions.params.values = params
            self._init_params_values = self.functions.params.values

        elif len(kparams) != 0:
            missing_param_names = set(
                self.functions.params.param_names
            ).difference(set(kparams.keys()))
            if len(missing_param_names) != 0:
                warnings.warn(
                    f"""{missing_param_names} params are missing and will be initialized randomly"""
                )
            for key, value in kparams.items():
                self.functions.params[key] = value
            self._init_params_values = self.functions.params.values

    def _set_params(self, params):
        if params is None:
            input_params = self._init_params_values
        elif type(params) == list:
            input_params = np.array(params)
        elif type(params) == np.ndarray:
            input_params = params
        elif type(params) == int:
            input_params = np.array(params)
        elif type(params) == float:
            input_params = np.array(params)
        else:
            raise TypeError("Incorrect params type")

        if input_params.shape != self.functions.params.values.shape:
            raise ValueError(
                f"""
                expected {self.functions.params.values.shape} shape for
                params but got {input_params.shape}
                """
            )
        else:
            self.functions.params.values = input_params
        return self.functions

    def _get_func(self, func_name: str, *args, **kwargs):
        if hasattr(self.functions, func_name):
            self.functions = self._set_params(kwargs["params"])
            res = getattr(self.functions, func_name)(*args)
        else:
            raise AttributeError(
                "f{self.functions.__name__} has no attribute {func_name}"
            )
        self.functions.params.values = self._init_params_values
        return res

    def ppf(
        self, probability: np.ndarray, params: np.ndarray = None
    ) -> np.ndarray:
        """_summary_

        Args:
            probability (np.ndarray): _description_
            params (np.ndarray, optional): _description_. Defaults to None.

        Returns:
            np.ndarray: _description_
        """
        return self._get_func("ppf", probability, params=params)

    def median(self, params: np.ndarray = None) -> np.ndarray:
        """_summary_

        Args:
            params (np.ndarray, optional): _description_. Defaults to None.

        Returns:
            np.ndarray: _description_
        """
        return self._get_func("median", params=params)

    def rvs(
        self,
        size: int = 1,
        random_state: int = None,
        params: np.ndarray = None,
    ) -> np.ndarray:
        """_summary_

        Args:
            size (int, optional): _description_. Defaults to 1.
            random_state (int, optional): _description_. Defaults to None.
            params (np.ndarray, optional): _description_. Defaults to None.

        Raises:
            AttributeError: _description_

        Returns:
            np.ndarray: _description_
        """
        return self._get_func("rvs", size, random_state, params=params)

    def sf(self, time: np.ndarray, params: np.ndarray = None) -> np.ndarray:
        """_summary_

        Args:
            time (np.ndarray): _description_
            params (np.ndarray, optional): _description_. Defaults to None.

        Raises:
            AttributeError: _description_

        Returns:
            np.ndarray: _description_
        """
        return self._get_func("sf", time, params=params)

    def cdf(self, time: np.ndarray, params: np.ndarray = None) -> np.ndarray:
        """_summary_

        Args:
            time (np.ndarray): _description_
            params (np.ndarray, optional): _description_. Defaults to None.

        Raises:
            AttributeError: _description_

        Returns:
            np.ndarray: _description_
        """
        return self._get_func("cdf", time, params=params)

    def pdf(self, time: np.ndarray, params: np.ndarray = None) -> np.ndarray:
        """_summary_

        Args:
            time (np.ndarray): _description_
            params (np.ndarray, optional): _description_. Defaults to None.

        Raises:
            AttributeError: _description_

        Returns:
            np.ndarray: _description_
        """
        return self._get_func("pdf", time, params=params)

    def isf(
        self, probability: np.ndarray, params: np.ndarray = None
    ) -> np.ndarray:
        """_summary_

        Args:
            probability (np.ndarray): _description_
            params (np.ndarray, optional): _description_. Defaults to None.

        Raises:
            AttributeError: _description_

        Returns:
            np.ndarray: _description_
        """
        return self._get_func("isf", probability, params=params)

    def hf(self, time: np.ndarray, params: np.ndarray = None):
        """Hazard function

        Args:
            time (np.ndarray): time

        Returns:
            np.ndarray: hazard function values
        """
        return self._get_func("hf", time, params=params)

    def chf(self, time: np.ndarray, params: np.ndarray = None) -> np.ndarray:
        """_summary_

        Args:
            time (np.ndarray): _description_
            params (np.ndarray, optional): _description_. Defaults to None.

        Raises:
            AttributeError: _description_

        Returns:
            np.ndarray: _description_
        """
        return self._get_func("chf", time, params=params)

    def mean(self, params: np.ndarray = None) -> float:
        """_summary_

        Args:
            params (np.ndarray, optional): _description_. Defaults to None.

        Raises:
            AttributeError: _description_

        Returns:
            float: _description_
        """
        return self._get_func("mean", params=params)

    def var(self, params: np.ndarray = None) -> float:
        """_summary_

        Args:
            params (np.ndarray, optional): _description_. Defaults to None.

        Raises:
            AttributeError: _description_

        Returns:
            float: _description_
        """
        return self._get_func("var", params=params)

    def mrl(self, time: np.ndarray, params: np.ndarray = None) -> np.ndarray:
        """_summary_

        Args:
            time (np.ndarray): _description_
            params (np.ndarray, optional): _description_. Defaults to None.

        Raises:
            AttributeError: _description_

        Returns:
            np.ndarray: _description_
        """
        return self._get_func("mrl", time, params=params)

    def ichf(
        self, cumulative_hazard_rate: np.ndarray, params: np.ndarray = None
    ) -> np.ndarray:
        """_summary_

        Args:
            cumulative_hazard_rate (np.ndarray): _description_
            params (np.ndarray, optional): _description_. Defaults to None.

        Raises:
            AttributeError: _description_

        Returns:
            np.ndarray: _description_
        """
        return self._get_func("ichf", cumulative_hazard_rate, params=params)

    def fit(
        self,
        observed_lifetimes: np.ndarray,
        complete_indicators: np.ndarray = None,
        left_censored_indicators: np.ndarray = None,
        right_censored_indicators: np.ndarray = None,
        entry: np.ndarray = None,
        departure: np.ndarray = None,
        **kwargs,
    ):

        if self.Likelihood is None:
            raise ValueError("Model has no Likelihood implemented")

        if self.Optimizer is None:
            raise ValueError("Model has no Optimizer implemented")

        likelihood = self.Likelihood(
            observed_lifetimes,
            complete_indicators,
            left_censored_indicators,
            right_censored_indicators,
            entry,
            departure,
        )
        optimizer = self.Optimizer(self.functions, likelihood)

        self.functions, opt = optimizer.fit(
            self.functions, likelihood, **kwargs
        )
        jac = likelihood.jac_negative_log_likelihood(self.functions)
        var = np.linalg.inv(
            likelihood.hess_negative_log_likelihood(self.functions, **kwargs)
        )
        self._fitting_results = FittingResults(
            opt,
            jac,
            var,
            len(observed_lifetimes),
        )
        self._fitting_params.values = opt.x

    @property
    def params(self):
        return self.functions.params

    @property
    def fitting_results(self):
        if self._fitting_results is None:
            warnings.warn(
                "Model parameters have not been fitted. Call fit() method"
                " first",
                UserWarning,
            )
        return self._fitting_results

    @property
    def fitting_params(self):
        if self._fitting_params is None:
            warnings.warn(
                "Model parameters have not been fitted. Call fit() method"
                " first",
                UserWarning,
            )
        return self._fitting_params
