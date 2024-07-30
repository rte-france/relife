"""
This module defines classes that instanciate facade objects used to create statistical models

Copyright (c) 2022, RTE (https://www.rte-france.com)
See AUTHORS.txt
SPDX-License-Identifier: Apache-2.0 (see LICENSE.txt)
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Union

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import minimize

from relife2.data import (
    Deteriorations,
    ObservedLifetimes,
    Truncations,
    array_factory,
    deteriorations_factory,
    lifetime_factory_template,
)
from relife2.functions import ParametricFunctions
from relife2.likelihoods import LikelihoodFromDeteriorations, LikelihoodFromLifetimes
from relife2.stats.distributions import (
    DistributionFunctions,
    ExponentialFunctions,
    GammaFunctions,
    GompertzFunctions,
    LogLogisticFunctions,
    WeibullFunctions,
)
from relife2.stats.gammaprocess import (
    GPDistributionFunctions,
    GPFunctions,
    PowerShapeFunctions,
)
from relife2.stats.regressions import (
    AFTFunctions,
    CovarEffect,
    ProportionalHazardFunctions,
    RegressionFunctions,
)
from relife2.utils.types import FloatArray

_LIFETIME_FUNCTIONS_NAMES = [
    "sf",
    "isf",
    "hf",
    "chf",
    "cdf",
    "cdf",
    "pdf",
    "ppf",
    "mrl",
    "ichf",
    "rvs",
    "mean",
    "var",
    "median",
]


def are_params_set(functions: ParametricFunctions):
    """
    Args:
        functions ():

    Returns:
    """
    if None in functions.all_params.values():
        params_to_set = " ".join(
            [name for name, value in functions.all_params.items() if value is None]
        )
        raise ValueError(
            f"Params {params_to_set} unset. Please set them first or fit the model."
        )


class ParametricModel:
    """Façade like Model type"""

    def __init__(self, functions: ParametricFunctions):
        self.functions = functions

    @property
    def params(self):
        """
        Returns:
        """
        return self.functions.params

    @params.setter
    def params(self, values: FloatArray):
        """
        Args:
            values ():

        Returns:
        """
        self.functions.params = values

    def __getattr__(self, name: str):
        class_name = type(self).__name__
        if name in self.__dict__:
            value = self.__dict__[name]
        else:
            if not hasattr(self.functions, name):
                raise AttributeError(f"{class_name} has no attribute named {name}")
            value = getattr(self.functions, name)
        return value

    def __setattr__(self, name: str, value: Any):
        if name == "functions":
            super().__setattr__(name, value)
        elif hasattr(self.functions, name):
            setattr(self.functions, name, value)
        else:
            super().__setattr__(name, value)

    # def __repr__(self):
    #     class_name = type(self).__name__
    #     return f"{class_name}(\n" f" params = {self.functions.all_params}\n"

    def __str__(self):
        class_name = type(self).__name__
        return f"{class_name}(\n" f" params = {self.functions.all_params}\n"


class ParametricLifetimeModel(ParametricModel, ABC):
    """
    Façade class for lifetime model (where functions is a LifetimeFunctions)
    """

    @abstractmethod
    def _init_likelihood(
        self,
        observed_lifetimes: ObservedLifetimes,
        truncations: Truncations,
        **kwargs: Any,
    ) -> LikelihoodFromLifetimes:
        """"""

    def __getattribute__(self, item):
        if item in _LIFETIME_FUNCTIONS_NAMES:
            are_params_set(self.functions)
        return super().__getattribute__(item)

    def fit(
        self,
        time: ArrayLike,
        event: Optional[ArrayLike] = None,
        entry: Optional[ArrayLike] = None,
        departure: Optional[ArrayLike] = None,
        inplace: bool = True,
        **kwargs: Any,
    ) -> FloatArray:
        """
        BLABLABLABLA
        Args:
            time (ArrayLike):
            event (Optional[ArrayLike]):
            entry (Optional[ArrayLike]):
            departure (Optional[ArrayLike]):
            inplace (bool): (default is True)

        Returns:
            Parameters: optimum parameters found
        """
        observed_lifetimes, truncations = lifetime_factory_template(
            time,
            event,
            entry,
            departure,
        )

        minimize_kwargs = {
            "method": kwargs.pop("method", "L-BFGS-B"),
            "constraints": kwargs.pop("constraints", ()),
            "tol": kwargs.pop("tol", None),
            "callback": kwargs.pop("callback", None),
            "options": kwargs.pop("options", None),
        }

        likelihood = self._init_likelihood(observed_lifetimes, truncations, **kwargs)
        param0 = kwargs.pop(
            "x0", likelihood.functions.init_params(observed_lifetimes.rlc)
        )
        minimize_kwargs.update(
            {"bounds": kwargs.pop("bounds", likelihood.functions.params_bounds)}
        )

        optimizer = minimize(
            likelihood.negative_log,
            param0,
            jac=None if not likelihood.hasjac else likelihood.jac_negative_log,
            **minimize_kwargs,
        )

        if inplace:
            self.functions = likelihood.functions.copy()
        return optimizer.x


class Distribution(ParametricLifetimeModel):
    """
    Facade implementation for distribution models
    """

    functions: DistributionFunctions

    def sf(self, time: ArrayLike) -> Union[float, FloatArray]:
        """
        Args:
            time ():

        Returns:

        """
        return np.squeeze(self.functions.sf(array_factory(time)))[()]

    def isf(self, probability: ArrayLike) -> Union[float, FloatArray]:
        """

        Args:
            probability ():

        Returns:

        """
        return np.squeeze(self.functions.isf(array_factory(probability)))[()]

    def hf(self, time: ArrayLike) -> Union[float, FloatArray]:
        """

        Args:
            time ():

        Returns:

        """
        return np.squeeze(self.functions.hf(array_factory(time)))[()]

    def chf(self, time: ArrayLike) -> Union[float, FloatArray]:
        """

        Args:
            time ():

        Returns:

        """
        return np.squeeze(self.functions.chf(array_factory(time)))[()]

    def cdf(self, time: ArrayLike) -> Union[float, FloatArray]:
        """

        Args:
            time ():

        Returns:

        """
        return np.squeeze(self.functions.cdf(array_factory(time)))[()]

    def pdf(self, probability: ArrayLike) -> Union[float, FloatArray]:
        """

        Args:
            probability ():

        Returns:

        """
        return np.squeeze(self.functions.pdf(array_factory(probability)))[()]

    def ppf(self, time: ArrayLike) -> Union[float, FloatArray]:
        """

        Args:
            time ():

        Returns:

        """
        return np.squeeze(self.functions.ppf(array_factory(time)))[()]

    def mrl(self, time: ArrayLike) -> Union[float, FloatArray]:
        """

        Args:
            time ():


        Returns:

        """
        return np.squeeze(self.functions.mrl(array_factory(time)))[()]

    def ichf(self, cumulative_hazard_rate: ArrayLike) -> Union[float, FloatArray]:
        """

        Args:
            cumulative_hazard_rate ():

        Returns:

        """
        return np.squeeze(self.functions.ichf(array_factory(cumulative_hazard_rate)))[
            ()
        ]

    def rvs(
        self, size: Optional[int] = 1, seed: Optional[int] = None
    ) -> Union[float, FloatArray]:
        """

        Args:
            size ():
            seed ():

        Returns:

        """
        return np.squeeze(self.functions.rvs(size=size, seed=seed))[()]

    def mean(self) -> Union[float, FloatArray]:
        """

        Returns:

        """
        return self.functions.mean()

    def var(self) -> Union[float, FloatArray]:
        """

        Returns:

        """
        return self.functions.var()

    def median(self) -> Union[float, FloatArray]:
        """

        Returns:

        """
        return self.functions.median()

    def _init_likelihood(
        self,
        observed_lifetimes: ObservedLifetimes,
        truncations: Truncations,
        **kwargs: Any,
    ) -> LikelihoodFromLifetimes:
        return LikelihoodFromLifetimes(
            self.functions.copy(),
            observed_lifetimes,
            truncations,
        )


class Regression(ParametricLifetimeModel):
    """
    Facade implementation for regression models
    """

    functions: RegressionFunctions

    @property
    def coefficients(self) -> FloatArray:
        """
        Returns:
        """
        return self.covar_effect.params

    @coefficients.setter
    def coefficients(self, values: Union[list[float], tuple[float]]):
        """
        Args:
            values ():

        Returns:
        """
        if len(values) != self.functions.nb_covar:
            self.functions = type(self.functions)(
                CovarEffect(**{f"coef_{i}": v for i, v in enumerate(values)}),
                self.functions.baseline.copy(),
            )
        else:
            self.functions.covar_effect.params = values

    def sf(self, time: ArrayLike, covar: ArrayLike) -> Union[float, FloatArray]:
        """
        Args:
            time ():
            covar ():

        Returns:

        """
        self.functions.covar = array_factory(covar)
        # check_params(self.functions)
        return np.squeeze(self.functions.sf(array_factory(time)))[()]

    def isf(self, probability: ArrayLike, covar: ArrayLike) -> Union[float, FloatArray]:
        """

        Args:
            probability ():
            covar ():

        Returns:

        """
        self.functions.covar = array_factory(covar)
        return np.squeeze(self.functions.isf(array_factory(probability)))[()]

    def hf(self, time: ArrayLike, covar: ArrayLike) -> Union[float, FloatArray]:
        """

        Args:
            time ():
            covar ():

        Returns:

        """
        self.functions.covar = array_factory(covar)
        return np.squeeze(self.functions.hf(array_factory(time)))[()]

    def chf(self, time: ArrayLike, covar: ArrayLike) -> Union[float, FloatArray]:
        """

        Args:
            time ():
            covar ():

        Returns:

        """
        self.functions.covar = array_factory(covar)
        return np.squeeze(self.functions.chf(array_factory(time)))[()]

    def cdf(self, time: ArrayLike, covar: ArrayLike) -> Union[float, FloatArray]:
        """

        Args:
            time ():
            covar ():
        Returns:

        """
        self.functions.covar = array_factory(covar)
        return np.squeeze(self.functions.cdf(array_factory(time)))[()]

    def pdf(self, probability: ArrayLike, covar: ArrayLike) -> Union[float, FloatArray]:
        """

        Args:
            probability ():
            covar ():

        Returns:

        """
        self.functions.covar = array_factory(covar)
        return np.squeeze(self.functions.pdf(array_factory(probability)))[()]

    def ppf(self, time: ArrayLike, covar: ArrayLike) -> Union[float, FloatArray]:
        """

        Args:
            time ():
            covar ():

        Returns:

        """
        self.functions.covar = array_factory(covar)
        return np.squeeze(self.functions.ppf(array_factory(time)))[()]

    def mrl(self, time: ArrayLike, covar: ArrayLike) -> Union[float, FloatArray]:
        """

        Args:
            time ():
            covar ():

        Returns:

        """
        self.functions.covar = array_factory(covar)
        return np.squeeze(self.functions.mrl(array_factory(time)))[()]

    def ichf(
        self, cumulative_hazard_rate: ArrayLike, covar: ArrayLike
    ) -> Union[float, FloatArray]:
        """

        Args:
            covar ():
            cumulative_hazard_rate ():

        Returns:

        """
        self.functions.covar = array_factory(covar)
        return np.squeeze(self.functions.ichf(array_factory(cumulative_hazard_rate)))[
            ()
        ]

    def rvs(
        self, covar: ArrayLike, size: Optional[int] = 1, seed: Optional[int] = None
    ) -> Union[float, FloatArray]:
        """

        Args:
            covar ():
            size ():
            seed ():

        Returns:

        """
        self.functions.covar = array_factory(covar)
        return np.squeeze(self.functions.rvs(size=size, seed=seed))[()]

    def mean(self, covar: ArrayLike) -> Union[float, FloatArray]:
        """

        Returns:
            covar ():

        """
        self.functions.covar = array_factory(covar)
        return self.functions.mean()

    def var(self, covar: ArrayLike) -> Union[float, FloatArray]:
        """

        Returns:
            covar ():

        """
        self.functions.covar = array_factory(covar)
        return self.functions.var()

    def median(self, covar: ArrayLike) -> Union[float, FloatArray]:
        """
        Returns:
            covar ():
        """
        self.functions.covar = array_factory(covar)
        return self.functions.median()

    def _init_likelihood(
        self,
        observed_lifetimes: ObservedLifetimes,
        truncations: Truncations,
        **kwargs: Any,
    ) -> LikelihoodFromLifetimes:
        if "covar" not in kwargs:
            raise ValueError(
                """
                Regression likelihood expects covar as data.
                Please add covar values to kwargs.
                """
            )
        covar = kwargs["covar"]
        if covar.shape[-1] != self.functions.covar_effect.nb_params:
            optimized_functions = type(self.functions)(
                CovarEffect(**{f"coef_{i}": None for i in range(covar.shape[-1])}),
                self.functions.baseline.copy(),
            )
        else:
            optimized_functions = self.functions.copy()
        return LikelihoodFromLifetimes(
            optimized_functions,
            observed_lifetimes,
            truncations,
            covar=covar,
        )


class Exponential(Distribution):
    """BLABLABLABLA"""

    def __init__(self, rate: Optional[float] = None):
        super().__init__(ExponentialFunctions(rate=rate))


class Weibull(Distribution):
    """BLABLABLABLA"""

    functions: DistributionFunctions

    def __init__(self, shape: Optional[float] = None, rate: Optional[float] = None):
        super().__init__(WeibullFunctions(shape=shape, rate=rate))


class Gompertz(Distribution):
    """BLABLABLABLA"""

    functions: DistributionFunctions

    def __init__(self, shape: Optional[float] = None, rate: Optional[float] = None):
        super().__init__(GompertzFunctions(shape=shape, rate=rate))


class Gamma(Distribution):
    """BLABLABLABLA"""

    functions: DistributionFunctions

    def __init__(self, shape: Optional[float] = None, rate: Optional[float] = None):
        super().__init__(GammaFunctions(shape=shape, rate=rate))


class LogLogistic(Distribution):
    """BLABLABLABLA"""

    functions: DistributionFunctions

    def __init__(self, shape: Optional[float] = None, rate: Optional[float] = None):
        super().__init__(LogLogisticFunctions(shape=shape, rate=rate))


def control_covar_args(
    coefficients: Optional[
        tuple[float | None] | list[float | None] | dict[str, float | None]
    ] = None,
) -> dict[str, float | None]:
    """

    Args:
        coefficients ():

    Returns:

    """
    if coefficients is None:
        return {"coef_0": None}
    if isinstance(coefficients, (list, tuple)):
        return {f"coef_{i}": v for i, v in enumerate(coefficients)}
    if isinstance(coefficients, dict):
        return coefficients
    raise ValueError("coefficients must be tuple, list or dict")


class ProportionalHazard(Regression):
    """BLABLABLABLA"""

    def __init__(
        self,
        baseline: Distribution,
        coefficients: Optional[
            tuple[float | None] | list[float | None] | dict[str, float | None]
        ] = None,
    ):
        coefficients = control_covar_args(coefficients)
        super().__init__(
            ProportionalHazardFunctions(
                CovarEffect(**coefficients),
                baseline.functions.copy(),
            )
        )


class AFT(Regression):
    """BLABLABLABLA"""

    def __init__(
        self,
        baseline: Distribution,
        coefficients: Optional[
            tuple[float | None] | list[float | None] | dict[str, float | None]
        ] = None,
    ):
        coefficients = control_covar_args(coefficients)
        super().__init__(
            AFTFunctions(
                CovarEffect(**coefficients),
                baseline.functions.copy(),
            )
        )


class GammaProcessDistribution(ParametricLifetimeModel):
    """
    BLABLABLABLA
    """

    shape_names: tuple = ("exponential", "power")

    def __init__(
        self,
        shape: str,
        rate: Optional[float] = None,
        **shape_params: Union[float, None],
    ):

        # if shape == "exponential":
        #     shape_functions = ExponentialShapeFunctions(**shape_params)
        if shape == "power":
            shape_functions = PowerShapeFunctions(**shape_params)
        else:
            raise ValueError(
                f"{shape} is not valid name for shape, only {self.shape_names} are allowed"
            )

        super().__init__(GPDistributionFunctions(shape_functions, rate))

    def sf(
        self, time: ArrayLike, initial_resistance: float, load_threshold: float
    ) -> Union[float, FloatArray]:
        """

        Args:
            time ():
            initial_resistance ():
            load_threshold ():

        Returns:

        """

        self.functions.initial_resistance = float(initial_resistance)
        self.functions.load_threshold = float(load_threshold)
        return np.squeeze(self.functions.sf(array_factory(time)))[()]

    def isf(
        self, probability: ArrayLike, initial_resistance: float, load_threshold: float
    ) -> Union[float, FloatArray]:
        """

        Args:
            probability ():
            initial_resistance ():
            load_threshold ():

        Returns:

        """

        self.functions.initial_resistance = float(initial_resistance)
        self.functions.load_threshold = float(load_threshold)
        return np.squeeze(self.functions.isf(array_factory(probability)))[()]

    def hf(
        self, time: ArrayLike, initial_resistance: float, load_threshold: float
    ) -> Union[float, FloatArray]:
        """

        Args:
            time ():
            initial_resistance ():
            load_threshold ():

        Returns:

        """

        self.functions.initial_resistance = float(initial_resistance)
        self.functions.load_threshold = float(load_threshold)
        return np.squeeze(self.functions.hf(array_factory(time)))[()]

    def chf(
        self, time: ArrayLike, initial_resistance: float, load_threshold: float
    ) -> Union[float, FloatArray]:
        """

        Args:
            time ():
            initial_resistance ():
            load_threshold ():

        Returns:

        """

        self.functions.initial_resistance = float(initial_resistance)
        self.functions.load_threshold = float(load_threshold)
        return np.squeeze(self.functions.chf(array_factory(time)))[()]

    def cdf(
        self, time: ArrayLike, initial_resistance: float, load_threshold: float
    ) -> Union[float, FloatArray]:
        """

        Args:
            time ():
            initial_resistance ():
            load_threshold ():

        Returns:

        """

        self.functions.initial_resistance = float(initial_resistance)
        self.functions.load_threshold = float(load_threshold)
        return np.squeeze(self.functions.cdf(array_factory(time)))[()]

    def pdf(
        self, probability: ArrayLike, initial_resistance: float, load_threshold: float
    ) -> Union[float, FloatArray]:
        """

        Args:
            probability ():
            initial_resistance ():
            load_threshold ():

        Returns:

        """

        self.functions.initial_resistance = float(initial_resistance)
        self.functions.load_threshold = float(load_threshold)
        return np.squeeze(self.functions.pdf(array_factory(probability)))[()]

    def ppf(
        self, time: ArrayLike, initial_resistance: float, load_threshold: float
    ) -> Union[float, FloatArray]:
        """

        Args:
            time ():
            initial_resistance ():
            load_threshold ():

        Returns:

        """

        self.functions.initial_resistance = float(initial_resistance)
        self.functions.load_threshold = float(load_threshold)
        return np.squeeze(self.functions.ppf(array_factory(time)))[()]

    def mrl(
        self, time: ArrayLike, initial_resistance: float, load_threshold: float
    ) -> Union[float, FloatArray]:
        """

        Args:
            time ():
            initial_resistance ():
            load_threshold ():

        Returns:

        """

        self.functions.initial_resistance = float(initial_resistance)
        self.functions.load_threshold = float(load_threshold)
        return np.squeeze(self.functions.mrl(array_factory(time)))[()]

    def ichf(
        self,
        cumulative_hazard_rate: ArrayLike,
        initial_resistance: float,
        load_threshold: float,
    ) -> Union[float, FloatArray]:
        """

        Args:
            cumulative_hazard_rate ():
            initial_resistance ():
            load_threshold ():

        Returns:

        """

        self.functions.initial_resistance = float(initial_resistance)
        self.functions.load_threshold = float(load_threshold)
        return np.squeeze(self.functions.ichf(array_factory(cumulative_hazard_rate)))[
            ()
        ]

    def rvs(
        self,
        initial_resistance: float,
        load_threshold: float,
        size: Optional[int] = 1,
        seed: Optional[int] = None,
    ) -> Union[float, FloatArray]:
        """

        Args:
            initial_resistance ():
            load_threshold ():
            size ():
            seed ():

        Returns:

        """

        self.functions.initial_resistance = float(initial_resistance)
        self.functions.load_threshold = float(load_threshold)
        return np.squeeze(self.functions.rvs(size=size, seed=seed))[()]

    def mean(
        self, initial_resistance: float, load_threshold: float
    ) -> Union[float, FloatArray]:
        """

        Args:
            initial_resistance ():
            load_threshold ():

        Returns:

        """
        self.functions.initial_resistance = float(initial_resistance)
        self.functions.load_threshold = float(load_threshold)
        return self.functions.mean()

    def var(
        self, initial_resistance: float, load_threshold: float
    ) -> Union[float, FloatArray]:
        """

        Args:
            initial_resistance ():
            load_threshold ():

        Returns:

        """

        self.functions.initial_resistance = float(initial_resistance)
        self.functions.load_threshold = float(load_threshold)
        return self.functions.var()

    def _init_likelihood(
        self,
        observed_lifetimes: ObservedLifetimes,
        truncations: Truncations,
        **kwargs: Any,
    ) -> LikelihoodFromLifetimes:
        if "initial_resistance" not in kwargs:
            raise ValueError(
                """
                GammaProcessDistribution likelihood expects initial_resistance as data.
                Please add initial_resistance value to kwargs.
                """
            )
        if "load_threshold" not in kwargs:
            raise ValueError(
                """
                GammaProcessDistribution likelihood expects load_threshold as data.
                Please add load_threshold value to kwargs.
                """
            )

        optimized_functions = self.functions.copy()
        optimized_functions.initial_resistance = kwargs["initial_resistance"]
        optimized_functions.load_threshold = kwargs["load_threshold"]

        return LikelihoodFromLifetimes(
            optimized_functions,
            observed_lifetimes,
            truncations,
        )


# créer un type StochasticProcess (voir reponse Thomas)
class GammaProcess(ParametricModel):
    """
    BLABLABLABLA
    """

    shape_names: tuple = ("exponential", "power")

    def __init__(
        self,
        shape: str,
        rate: Optional[float] = None,
        **shape_params: Union[float, None],
    ):

        # if shape == "exponential":
        #     shape_functions = ExponentialShapeFunctions(**shape_params)
        if shape == "power":
            shape_functions = PowerShapeFunctions(**shape_params)
        else:
            raise ValueError(
                f"{shape} is not valid name for shape, only {self.shape_names} are allowed"
            )

        super().__init__(GPFunctions(shape_functions, rate))

    def sample(
        self,
        time: ArrayLike,
        unit_ids=ArrayLike,
        nb_sample=1,
        seed=None,
        add_death_time=True,
    ):
        """
        Args:
            time ():
            unit_ids ():
            nb_sample ():
            seed ():
            add_death_time ():

        Returns:

        """
        return self.functions.sample(time, unit_ids, nb_sample, seed, add_death_time)

    def _init_likelihood(
        self,
        deterioration_data: Deteriorations,
        first_increment_uncertainty,
        measurement_tol,
        **kwargs: Any,
    ) -> LikelihoodFromDeteriorations:
        if len(kwargs) != 0:
            extra_args_names = tuple(kwargs.keys())
            raise ValueError(
                f"""
                Distribution likelihood does not expect other data than lifetimes
                Remove {extra_args_names} from kwargs.
                """
            )
        return LikelihoodFromDeteriorations(
            self.functions.copy(),
            deterioration_data,
            first_increment_uncertainty=first_increment_uncertainty,
            measurement_tol=measurement_tol,
        )

    def fit(
        self,
        deterioration_measurements: ArrayLike,
        inspection_times: ArrayLike,
        unit_ids: ArrayLike,
        first_increment_uncertainty: Optional[tuple] = None,
        measurement_tol: np.floating[Any] = np.finfo(float).resolution,
        inplace: bool = True,
        **kwargs: Any,
    ) -> FloatArray:
        """
        BLABLABLABLA
        """

        deterioration_data = deteriorations_factory(
            array_factory(deterioration_measurements),
            array_factory(inspection_times),
            array_factory(unit_ids),
            self.functions.process_lifetime_distribution.initial_resistance,
        )

        param0 = kwargs.pop("x0", self.functions.init_params())

        minimize_kwargs = {
            "method": kwargs.pop("method", "Nelder-Mead"),
            "bounds": kwargs.pop("bounds", self.functions.params_bounds),
            "constraints": kwargs.pop("constraints", ()),
            "tol": kwargs.pop("tol", None),
            "callback": kwargs.pop("callback", None),
            "options": kwargs.pop("options", None),
        }

        likelihood = self._init_likelihood(
            deterioration_data, first_increment_uncertainty, measurement_tol, **kwargs
        )

        optimizer = minimize(
            likelihood.negative_log,
            param0,
            jac=None if not likelihood.hasjac else likelihood.jac_negative_log,
            **minimize_kwargs,
        )

        if inplace:
            self.params = optimizer.x
        return optimizer.x
