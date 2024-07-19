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

from relife2 import parametric
from relife2.data import (
    ObservedLifetimes,
    Truncations,
    array_factory,
    lifetime_factory_template,
)
from relife2.distributions import (
    DistributionFunctions,
    ExponentialFunctions,
    GammaFunctions,
    GompertzFunctions,
    LogLogisticFunctions,
    WeibullFunctions,
    GPDistributionFunctions,
)
from relife2.gammaprocess import ExponentialShapeFunctions, PowerShapeFunctions
from relife2.likelihoods import LikelihoodFromLifetimes
from relife2.regressions import (
    AFTEffect,
    AFTFunctions,
    ProportionalHazardEffect,
    ProportionalHazardFunctions,
    RegressionFunctions,
)
from relife2.types import FloatArray


class Model:
    """Façade like Model type"""

    def __init__(self, functions: parametric.Functions):
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


class LifetimeModel(Model, ABC):
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

    def fit(
        self,
        time: ArrayLike,
        entry: Optional[ArrayLike] = None,
        departure: Optional[ArrayLike] = None,
        lc_indicators: Optional[ArrayLike] = None,
        rc_indicators: Optional[ArrayLike] = None,
        inplace: bool = True,
        **kwargs: Any,
    ) -> FloatArray:
        """
        BLABLABLABLA
        Args:
            time (ArrayLike):
            entry (Optional[ArrayLike]):
            departure (Optional[ArrayLike]):
            lc_indicators (Optional[ArrayLike]):
            rc_indicators (Optional[ArrayLike]):
            inplace (bool): (default is True)

        Returns:
            Parameters: optimum parameters found
        """
        observed_lifetimes, truncations = lifetime_factory_template(
            time, entry, departure, lc_indicators, rc_indicators
        )

        param0 = kwargs.pop("x0", self.functions.init_params(observed_lifetimes.rlc))

        minimize_kwargs = {
            "method": kwargs.pop("method", "L-BFGS-B"),
            "bounds": kwargs.pop("bounds", self.functions.params_bounds),
            "constraints": kwargs.pop("constraints", ()),
            "tol": kwargs.pop("tol", None),
            "callback": kwargs.pop("callback", None),
            "options": kwargs.pop("options", None),
        }

        likelihood = self._init_likelihood(observed_lifetimes, truncations, **kwargs)

        optimizer = minimize(
            likelihood.negative_log,
            param0,
            jac=None if not likelihood.hasjac else likelihood.jac_negative_log,
            **minimize_kwargs,
        )

        if inplace:
            self.params = optimizer.x
        return optimizer.x


class Distribution(LifetimeModel):
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
        if len(kwargs) != 0:
            extra_args_names = tuple(kwargs.keys())
            raise ValueError(
                f"""
                Distribution likelihood does not expect other data than lifetimes
                Remove {extra_args_names} from kwargs.
                """
            )
        return LikelihoodFromLifetimes(
            self.functions.copy(),
            observed_lifetimes,
            truncations,
        )


class Regression(LifetimeModel):
    """
    Facade implementation for regression models
    """

    functions: RegressionFunctions

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
        if set(kwargs.keys()) != {"covar"}:
            extra_args_names = tuple(kwargs.keys())
            raise ValueError(
                f"""
                Regression likelihood only expects covar as other data.
                Got {extra_args_names} from kwargs.
                """
            )
        return LikelihoodFromLifetimes(
            self.functions.copy(),
            observed_lifetimes,
            truncations,
            **kwargs,
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
    weights: Optional[
        tuple[float, None] | list[float | None] | dict[str, float | None]
    ] = None,
    nb_covar: Optional[int] = None,
) -> dict[str, float | None]:
    """

    Args:
        nb_covar ():
        weights ():

    Returns:

    """
    if nb_covar is None:
        if weights is None:
            raise ValueError(
                "Regression model expects at least covar weights values or nb_covar"
            )
        if isinstance(weights, (tuple, list)):
            weights = {f"beta_{i}": value for i, value in enumerate(weights)}
    else:
        if weights is not None:
            raise ValueError(
                "When covar weights are specified, nb_covar is useless. Remove nb_covar."
            )
        weights = {f"beta_{i}": None for i in range(nb_covar)}
    return weights


class ProportionalHazard(Regression):
    """BLABLABLABLA"""

    def __init__(
        self,
        baseline: Distribution,
        weights: Optional[
            tuple[float, None] | list[float | None] | dict[str, float | None]
        ] = None,
        nb_covar: Optional[int] = None,
    ):
        weights = control_covar_args(weights, nb_covar)
        super().__init__(
            ProportionalHazardFunctions(
                ProportionalHazardEffect(**weights),
                baseline.functions.copy(),
            )
        )


class AFT(Regression):
    """BLABLABLABLA"""

    def __init__(
        self,
        baseline: Distribution,
        weights: Optional[
            tuple[float, None] | list[float | None] | dict[str, float | None]
        ] = None,
        nb_covar: Optional[int] = None,
    ):
        weights = control_covar_args(weights, nb_covar)
        super().__init__(
            AFTFunctions(
                AFTEffect(**weights),
                baseline.functions.copy(),
            )
        )


class GammaProcessDistribution(Distribution):

    shape_names: tuple = ("exponential", "power")

    def __init__(
        self,
        shape: str,
        rate: Optional[float] = None,
        initial_resistance: Optional[float] = None,
        load_threshold: Optional[float] = None,
        **shape_params: Union[float, None],
    ):

        if shape == "exponential":
            shape_functions = ExponentialShapeFunctions(**shape_params)
        elif shape == "power":
            shape_functions = PowerShapeFunctions(**shape_params)
        else:
            raise ValueError(
                f"{shape} is not valid name for shape, only {self.shape_names} are allowed"
            )

        super().__init__(
            GPDistributionFunctions(
                shape_functions, rate, initial_resistance, load_threshold
            )
        )
