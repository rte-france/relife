"""
This module defines classes that instanciate facade objects used to create statistical models

Copyright (c) 2022, RTE (https://www.rte-france.com)
See AUTHORS.txt
SPDX-License-Identifier: Apache-2.0 (see LICENSE.txt)
"""

from typing import Any, Optional, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize

from relife2.survival.data import array_factory, lifetime_factory_template
from relife2.survival.distributions import (
    DistributionFunctions,
    ExponentialFunctions,
    GammaFunctions,
    GompertzFunctions,
    LogLogisticFunctions,
    WeibullFunctions,
)
from relife2.survival.likelihoods import LikelihoodFromLifetimes
from relife2.survival.regressions import (
    AFTEffect,
    AFTFunctions,
    ProportionalHazardEffect,
    ProportionalHazardFunctions,
)
from relife2.survival.types import FunctionsBridge, ParametricHazard

FloatArray = NDArray[np.float64]


class LifetimeModel(FunctionsBridge):
    """
    Façade class that provides a simplified interface to lifetime model
    """

    functions: ParametricHazard

    def sf(self, time: ArrayLike, **kwargs: Any) -> Union[float, FloatArray]:
        """
        Args:
            time ():
            **kwargs (object):

        Returns:

        """
        self._control_kwargs(**kwargs)
        for name, value in kwargs.items():
            setattr(self.functions, name, value)
        return np.squeeze(self.functions.sf(array_factory(time)))[()]

    def isf(self, probability: ArrayLike, **kwargs: Any) -> Union[float, FloatArray]:
        """

        Args:
            probability ():
            **kwargs (object):

        Returns:

        """
        self._control_kwargs(**kwargs)
        for name, value in kwargs.items():
            setattr(self.functions, name, value)
        return np.squeeze(self.functions.isf(array_factory(probability)))[()]

    def hf(self, time: ArrayLike, **kwargs: Any) -> Union[float, FloatArray]:
        """

        Args:
            time ():
            **kwargs (object):

        Returns:

        """
        self._control_kwargs(**kwargs)
        for name, value in kwargs.items():
            setattr(self.functions, name, value)
        return np.squeeze(self.functions.hf(array_factory(time)))[()]

    def chf(self, time: ArrayLike, **kwargs: Any) -> Union[float, FloatArray]:
        """

        Args:
            time ():
            **kwargs (object):

        Returns:

        """
        self._control_kwargs(**kwargs)
        for name, value in kwargs.items():
            setattr(self.functions, name, value)
        return np.squeeze(self.functions.chf(array_factory(time)))[()]

    def cdf(self, time: ArrayLike, **kwargs: Any) -> Union[float, FloatArray]:
        """

        Args:
            time ():
            **kwargs (object):

        Returns:

        """
        self._control_kwargs(**kwargs)
        for name, value in kwargs.items():
            setattr(self.functions, name, value)
        return np.squeeze(self.functions.cdf(array_factory(time)))[()]

    def pdf(self, probability: ArrayLike, **kwargs: Any) -> Union[float, FloatArray]:
        """

        Args:
            probability ():
            **kwargs (object):

        Returns:

        """
        self._control_kwargs(**kwargs)
        for name, value in kwargs.items():
            setattr(self.functions, name, value)
        return np.squeeze(self.functions.pdf(array_factory(probability)))[()]

    def ppf(self, time: ArrayLike, **kwargs: Any) -> Union[float, FloatArray]:
        """

        Args:
            time ():
            **kwargs (object):

        Returns:

        """
        self._control_kwargs(**kwargs)
        for name, value in kwargs.items():
            setattr(self.functions, name, value)
        return np.squeeze(self.functions.ppf(array_factory(time)))[()]

    def mrl(self, time: ArrayLike, **kwargs: Any) -> Union[float, FloatArray]:
        """

        Args:
            time ():
            **kwargs (object):

        Returns:

        """
        self._control_kwargs(**kwargs)
        for name, value in kwargs.items():
            setattr(self.functions, name, value)
        return np.squeeze(self.functions.mrl(array_factory(time)))[()]

    def ichf(
        self, cumulative_hazard_rate: ArrayLike, **kwargs: Any
    ) -> Union[float, FloatArray]:
        """

        Args:
            cumulative_hazard_rate ():
            **kwargs (object):

        Returns:

        """
        self._control_kwargs(**kwargs)
        for name, value in kwargs.items():
            setattr(self.functions, name, value)
        return np.squeeze(self.functions.ichf(array_factory(cumulative_hazard_rate)))[
            ()
        ]

    def rvs(
        self, size: Optional[int] = 1, seed: Optional[int] = None, **kwargs: Any
    ) -> Union[float, FloatArray]:
        """

        Args:
            size ():
            seed ():

        Returns:

        """
        self._control_kwargs(**kwargs)
        for name, value in kwargs.items():
            setattr(self.functions, name, value)
        return np.squeeze(self.functions.rvs(size=size, seed=seed))[()]

    def mean(self, **kwargs: Any) -> Union[float, FloatArray]:
        """

        Returns:

        """
        self._control_kwargs(**kwargs)
        for name, value in kwargs.items():
            setattr(self.functions, name, value)
        return self.functions.mean()

    def var(self, **kwargs: Any) -> Union[float, FloatArray]:
        """

        Returns:

        """
        self._control_kwargs(**kwargs)
        for name, value in kwargs.items():
            setattr(self.functions, name, value)
        return self.functions.var()

    def median(self, **kwargs: Any) -> Union[float, FloatArray]:
        """

        Returns:

        """
        self._control_kwargs(**kwargs)
        for name, value in kwargs.items():
            setattr(self.functions, name, value)
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

        param0 = kwargs.pop("x0", self.functions.init_params(observed_lifetimes.rlc))
        minimize_kwargs = {
            "method": kwargs.pop("method", "L-BFGS-B"),
            "bounds": kwargs.pop("bounds", self.functions.params_bounds),
            "constraints": kwargs.pop("constraints", ()),
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
            param0,
            jac=None if not likelihood.hasjac else likelihood.jac_negative_log,
            **minimize_kwargs,
        )

        if inplace:
            self.params = optimizer.x
        return optimizer.x


class Exponential(LifetimeModel):
    """BLABLABLABLA"""

    functions: DistributionFunctions

    def __init__(self, rate: Optional[float] = None):
        super().__init__(ExponentialFunctions(rate=rate))


class Weibull(LifetimeModel):
    """BLABLABLABLA"""

    functions: DistributionFunctions

    def __init__(self, shape: Optional[float] = None, rate: Optional[float] = None):
        super().__init__(WeibullFunctions(shape=shape, rate=rate))


class Gompertz(LifetimeModel):
    """BLABLABLABLA"""

    functions: DistributionFunctions

    def __init__(self, shape: Optional[float] = None, rate: Optional[float] = None):
        super().__init__(GompertzFunctions(shape=shape, rate=rate))


class Gamma(LifetimeModel):
    """BLABLABLABLA"""

    functions: DistributionFunctions

    def __init__(self, shape: Optional[float] = None, rate: Optional[float] = None):
        super().__init__(GammaFunctions(shape=shape, rate=rate))


class LogLogistic(LifetimeModel):
    """BLABLABLABLA"""

    functions: DistributionFunctions

    def __init__(self, shape: Optional[float] = None, rate: Optional[float] = None):
        super().__init__(LogLogisticFunctions(shape=shape, rate=rate))


Distribution = Union[Exponential, Gamma, Gompertz, LogLogistic, Weibull]


def control_covar_args(
    weights: Optional[
        tuple[float, None] | list[float | None] | dict[str, float | None]
    ] = None,
    nb_covar: Optional[int] = None,
) -> dict[str, float | None]:
    """

    Args:
        nb_covar ():
        covar ():
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


class ProportionalHazard(LifetimeModel):
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


class AFT(LifetimeModel):
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
