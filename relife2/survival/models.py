"""
This module defines probability functions used in regression

Copyright (c) 2022, RTE (https://www.rte-france.com)
See AUTHORS.txt
SPDX-License-Identifier: Apache-2.0 (see LICENSE.txt)
"""

from typing import Optional, Union, NewType, Any

import numpy as np
from numpy.typing import NDArray, ArrayLike
from scipy.optimize import minimize

from relife2.survival.data import array_factory, lifetime_factory_template
from relife2.survival.distributions import (
    ExponentialFunctions,
    WeibullFunctions,
    GompertzFunctions,
    GammaFunctions,
    LogLogisticFunctions,
)
from relife2.survival.likelihoods import LikelihoodFromLifetimes
from relife2.survival.regressions import (
    ProportionalHazardFunctions,
    ProportionalHazardEffect,
    AFTFunctions,
    AFTEffect,
)
from relife2.survival.types import ParametricHazard

FloatArray = NDArray[np.float64]


class LifetimeModel:
    """
    BLABLABLA
    """

    def __init__(self, functions: ParametricHazard):
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

    def sf(self, time: ArrayLike) -> Union[float, FloatArray]:
        return np.squeeze(self.functions.sf(array_factory(time)))[()]

    def isf(self, probability: ArrayLike) -> Union[float, FloatArray]:
        return np.squeeze(self.functions.isf(array_factory(probability)))[()]

    def hf(self, time: ArrayLike) -> Union[float, FloatArray]:
        return np.squeeze(self.functions.hf(array_factory(time)))[()]

    def chf(self, time: ArrayLike) -> Union[float, FloatArray]:
        return np.squeeze(self.functions.chf(array_factory(time)))[()]

    def cdf(self, time: ArrayLike) -> Union[float, FloatArray]:
        return np.squeeze(self.functions.cdf(array_factory(time)))[()]

    def pdf(self, probability: ArrayLike) -> Union[float, FloatArray]:
        return np.squeeze(self.functions.pdf(array_factory(probability)))[()]

    def ppf(self, time: ArrayLike) -> Union[float, FloatArray]:
        return np.squeeze(self.functions.ppf(array_factory(time)))[()]

    def mrl(self, time: ArrayLike) -> Union[float, FloatArray]:
        return np.squeeze(self.functions.mrl(array_factory(time)))[()]

    def ichf(self, cumulative_hazard_rate: ArrayLike) -> Union[float, FloatArray]:
        return np.squeeze(self.functions.ichf(array_factory(cumulative_hazard_rate)))[
            ()
        ]

    def rvs(
        self, size: Optional[int] = 1, seed: Optional[int] = None
    ) -> Union[float, FloatArray]:
        return np.squeeze(self.functions.rvs(size=size, seed=seed))[()]

    def mean(
        self,
    ) -> float:
        return self.functions.mean()

    def var(
        self,
    ) -> float:
        return self.functions.var()

    def median(
        self,
    ) -> float:
        return self.functions.median()

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

        minimize_kwargs = {
            "method": kwargs.pop("method", "L-BFGS-B"),
            "bounds": kwargs.pop("bounds", None),
            "contraints": kwargs.pop("constraints", ()),
            "tol": kwargs.pop("tol", None),
            "callback": kwargs.pop("callback", None),
            "options": kwargs.pop("options", None),
        }

        likelihood = LikelihoodFromLifetimes(
            self.functions.copy(),
            observed_lifetimes,
            truncations,
            **kwargs,
        )

        optimizer = minimize(
            likelihood.negative_log,
            self.functions.init_params(observed_lifetimes.rlc),
            jac=self.likelihood.jac if self.likelihood.hasjac else None,
            **minimize_kwargs,
        )

        if inplace:
            self.params = optimizer.x
        return optimizer.x

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

    def __repr__(self):
        class_name = type(self).__name__
        return f"{class_name}({self.params.__repr__()})"


class Exponential(LifetimeModel):
    """BLABLABLABLA"""

    def __init__(self, rate: Optional[float] = None):
        super().__init__(ExponentialFunctions(rate=rate))


class Weibull(LifetimeModel):
    """BLABLABLABLA"""

    def __init__(self, shape: Optional[float] = None, rate: Optional[float] = None):
        super().__init__(WeibullFunctions(shape=shape, rate=rate))


class Gompertz(LifetimeModel):
    """BLABLABLABLA"""

    def __init__(self, shape: Optional[float] = None, rate: Optional[float] = None):
        super().__init__(GompertzFunctions(shape=shape, rate=rate))


class Gamma(LifetimeModel):
    """BLABLABLABLA"""

    def __init__(self, shape: Optional[float] = None, rate: Optional[float] = None):
        super().__init__(GammaFunctions(shape=shape, rate=rate))


class LogLogistic(LifetimeModel):
    """BLABLABLABLA"""

    def __init__(self, shape: Optional[float] = None, rate: Optional[float] = None):
        super().__init__(LogLogisticFunctions(shape=shape, rate=rate))


Distribution = NewType(
    "Distribution", Union[Exponential, Gamma, Gompertz, LogLogistic, Weibull]
)


def set_covar_weights(*beta: Union[float, None], **kwargs) -> dict[str, float]:
    nb_covar = kwargs.pop("nb_covar", None)
    if nb_covar is None:
        if len(beta) == 0:
            raise ValueError(
                "Regression model expects at least covar weights values or nb_covar"
            )
        kwbeta = {f"beta_{i}": value for i, value in enumerate(beta)}
    else:
        if len(beta) != 0:
            raise ValueError(
                "When covar weights are specified, nb_covar is useless. Remove nb_covar."
            )
        kwbeta = {f"beta_{i}": np.random.random() for i in range(nb_covar)}
    return kwbeta


class ProportionalHazard(LifetimeModel):
    """BLABLABLABLA"""

    def __init__(
        self,
        baseline: Distribution,
        *beta: Union[float, None],
        nb_covar: Optional[int] = None,
    ):

        super().__init__(
            ProportionalHazardFunctions(
                covar_effect=ProportionalHazardEffect(
                    **set_covar_weights(*beta, nb_covar=nb_covar)
                ),
                baseline=baseline.functions,
            )
        )


class AFT(LifetimeModel):
    """BLABLABLABLA"""

    def __init__(
        self,
        baseline: Distribution,
        *beta: Union[float, None],
        nb_covar: Optional[int] = None,
    ):
        super().__init__(
            AFTFunctions(
                covar_effect=AFTEffect(**set_covar_weights(*beta, nb_covar=nb_covar)),
                baseline=baseline.functions,
            )
        )
