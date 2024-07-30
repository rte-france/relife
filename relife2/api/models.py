"""
This module defines classes that instanciate facade objects used to create statistical models

Copyright (c) 2022, RTE (https://www.rte-france.com)
See AUTHORS.txt
SPDX-License-Identifier: Apache-2.0 (see LICENSE.txt)
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Union

from numpy.typing import ArrayLike
from scipy.optimize import minimize

from relife2.api.utils import are_params_set, squeeze
from relife2.data import (
    ObservedLifetimes,
    Truncations,
    array_factory,
    lifetime_factory_template,
)
from relife2.stats.distributions import DistributionFunctions
from relife2.stats.functions import ParametricFunctions
from relife2.stats.gammaprocess import GPDistributionFunctions
from relife2.stats.likelihoods import LikelihoodFromLifetimes
from relife2.stats.regressions import CovarEffect, RegressionFunctions
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

    @squeeze
    def sf(self, time: ArrayLike) -> Union[float, FloatArray]:
        """
        Args:
            time ():

        Returns:

        """
        time = array_factory(time)
        return self.functions.sf(time)

    @squeeze
    def isf(self, probability: ArrayLike) -> Union[float, FloatArray]:
        """

        Args:
            probability ():

        Returns:

        """
        probability = array_factory(probability)
        return self.functions.isf(array_factory(probability))

    @squeeze
    def hf(self, time: ArrayLike) -> Union[float, FloatArray]:
        """

        Args:
            time ():

        Returns:

        """
        time = array_factory(time)
        return self.functions.hf(time)

    @squeeze
    def chf(self, time: ArrayLike) -> Union[float, FloatArray]:
        """

        Args:
            time ():

        Returns:

        """
        time = array_factory(time)
        return self.functions.chf(time)

    @squeeze
    def cdf(self, time: ArrayLike) -> Union[float, FloatArray]:
        """

        Args:
            time ():

        Returns:

        """
        time = array_factory(time)
        return self.functions.cdf(time)

    @squeeze
    def pdf(self, probability: ArrayLike) -> Union[float, FloatArray]:
        """

        Args:
            probability ():

        Returns:

        """
        probability = array_factory(probability)
        return self.functions.pdf(probability)

    @squeeze
    def ppf(self, time: ArrayLike) -> Union[float, FloatArray]:
        """

        Args:
            time ():

        Returns:

        """
        time = array_factory(time)
        return self.functions.ppf(time)

    @squeeze
    def mrl(self, time: ArrayLike) -> Union[float, FloatArray]:
        """

        Args:
            time ():


        Returns:

        """
        time = array_factory(time)
        return self.functions.mrl(time)

    @squeeze
    def ichf(self, cumulative_hazard_rate: ArrayLike) -> Union[float, FloatArray]:
        """

        Args:
            cumulative_hazard_rate ():

        Returns:

        """
        cumulative_hazard_rate = array_factory(cumulative_hazard_rate)
        return self.functions.ichf(cumulative_hazard_rate)

    @squeeze
    def rvs(
        self, size: Optional[int] = 1, seed: Optional[int] = None
    ) -> Union[float, FloatArray]:
        """

        Args:
            size ():
            seed ():

        Returns:

        """
        return self.functions.rvs(size=size, seed=seed)

    @squeeze
    def mean(self) -> Union[float, FloatArray]:
        """

        Returns:

        """
        return self.functions.mean()

    @squeeze
    def var(self) -> Union[float, FloatArray]:
        """

        Returns:

        """
        return self.functions.var()

    @squeeze
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

    @squeeze
    def sf(self, time: ArrayLike, covar: ArrayLike) -> Union[float, FloatArray]:
        """
        Args:
            time ():
            covar ():

        Returns:

        """
        time = array_factory(time)
        self.functions.covar = array_factory(covar)
        return self.functions.sf(time)

    @squeeze
    def isf(self, probability: ArrayLike, covar: ArrayLike) -> Union[float, FloatArray]:
        """

        Args:
            probability ():
            covar ():

        Returns:

        """
        probability = array_factory(probability)
        self.functions.covar = array_factory(covar)
        return self.functions.isf(probability)

    @squeeze
    def hf(self, time: ArrayLike, covar: ArrayLike) -> Union[float, FloatArray]:
        """

        Args:
            time ():
            covar ():

        Returns:

        """
        time = array_factory(time)
        self.functions.covar = array_factory(covar)
        return self.functions.hf(time)

    @squeeze
    def chf(self, time: ArrayLike, covar: ArrayLike) -> Union[float, FloatArray]:
        """

        Args:
            time ():
            covar ():

        Returns:

        """
        time = array_factory(time)
        self.functions.covar = array_factory(covar)
        return self.functions.chf(time)

    @squeeze
    def cdf(self, time: ArrayLike, covar: ArrayLike) -> Union[float, FloatArray]:
        """

        Args:
            time ():
            covar ():
        Returns:

        """
        time = array_factory(time)
        self.functions.covar = array_factory(covar)
        return self.functions.cdf(time)

    @squeeze
    def pdf(self, probability: ArrayLike, covar: ArrayLike) -> Union[float, FloatArray]:
        """

        Args:
            probability ():
            covar ():

        Returns:

        """
        probability = array_factory(probability)
        self.functions.covar = array_factory(covar)
        return self.functions.pdf(probability)

    @squeeze
    def ppf(self, time: ArrayLike, covar: ArrayLike) -> Union[float, FloatArray]:
        """

        Args:
            time ():
            covar ():

        Returns:

        """
        time = array_factory(time)
        self.functions.covar = array_factory(covar)
        return self.functions.ppf(time)

    @squeeze
    def mrl(self, time: ArrayLike, covar: ArrayLike) -> Union[float, FloatArray]:
        """

        Args:
            time ():
            covar ():

        Returns:

        """
        time = array_factory(time)
        self.functions.covar = array_factory(covar)
        return self.functions.mrl(time)

    @squeeze
    def ichf(
        self, cumulative_hazard_rate: ArrayLike, covar: ArrayLike
    ) -> Union[float, FloatArray]:
        """

        Args:
            covar ():
            cumulative_hazard_rate ():

        Returns:

        """
        cumulative_hazard_rate = array_factory(cumulative_hazard_rate)
        self.functions.covar = array_factory(covar)
        return self.functions.ichf(cumulative_hazard_rate)

    @squeeze
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
        return self.functions.rvs(size=size, seed=seed)

    @squeeze
    def mean(self, covar: ArrayLike) -> Union[float, FloatArray]:
        """

        Returns:
            covar ():

        """
        self.functions.covar = array_factory(covar)
        return self.functions.mean()

    @squeeze
    def var(self, covar: ArrayLike) -> Union[float, FloatArray]:
        """

        Returns:
            covar ():

        """
        self.functions.covar = array_factory(covar)
        return self.functions.var()

    @squeeze
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


class GammaProcessDistribution(ParametricLifetimeModel):
    """
    BLABLABLABLA
    """

    functions: GPDistributionFunctions

    @squeeze
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
        time = array_factory(time)
        self.functions.initial_resistance = float(initial_resistance)
        self.functions.load_threshold = float(load_threshold)
        return self.functions.sf(time)

    @squeeze
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
        probability = array_factory(probability)
        self.functions.initial_resistance = float(initial_resistance)
        self.functions.load_threshold = float(load_threshold)
        return self.functions.isf(probability)

    @squeeze
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
        time = array_factory(time)
        self.functions.initial_resistance = float(initial_resistance)
        self.functions.load_threshold = float(load_threshold)
        return self.functions.hf(time)

    @squeeze
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
        time = array_factory(time)
        self.functions.initial_resistance = float(initial_resistance)
        self.functions.load_threshold = float(load_threshold)
        return self.functions.chf(time)

    @squeeze
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
        time = array_factory(time)
        self.functions.initial_resistance = float(initial_resistance)
        self.functions.load_threshold = float(load_threshold)
        return self.functions.cdf(time)

    @squeeze
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
        probability = array_factory(probability)
        self.functions.initial_resistance = float(initial_resistance)
        self.functions.load_threshold = float(load_threshold)
        return self.functions.pdf(probability)

    @squeeze
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
        time = array_factory(time)
        self.functions.initial_resistance = float(initial_resistance)
        self.functions.load_threshold = float(load_threshold)
        return self.functions.ppf(time)

    @squeeze
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
        time = array_factory(time)
        self.functions.initial_resistance = float(initial_resistance)
        self.functions.load_threshold = float(load_threshold)
        return self.functions.mrl(time)

    @squeeze
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
        cumulative_hazard_rate = array_factory(cumulative_hazard_rate)
        self.functions.initial_resistance = float(initial_resistance)
        self.functions.load_threshold = float(load_threshold)
        return self.functions.ichf(cumulative_hazard_rate)

    @squeeze
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
        return self.functions.rvs(size=size, seed=seed)

    @squeeze
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

    @squeeze
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
