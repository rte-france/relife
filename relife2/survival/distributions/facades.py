"""
This module defines classes seen by clients to use model of distributions.
These classes instanciate facade object design pattern.

Copyright (c) 2022, RTE (https://www.rte-france.com)
See AUTHORS.txt
SPDX-License-Identifier: Apache-2.0 (see LICENSE.txt)
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from relife2.survival.data import (
    LifetimeDataFactory,
    LifetimeDataFactoryFrom1D,
    LifetimeDataFactoryFrom2D,
    array_factory,
)
from relife2.survival.distributions.functions import (
    ExponentialFunctions,
    GammaFunctions,
    GompertzFunctions,
    LogLogisticFunctions,
    WeibullFunctions,
)
from relife2.survival.distributions.likelihoods import GenericDistributionLikelihood
from relife2.survival.distributions.optimizers import (
    GenericDistributionOptimizer,
    GompertzOptimizer,
)
from relife2.survival.distributions.types import Distribution
from relife2.survival.parameters import Parameters

FloatArray = NDArray[np.float64]


class Exponential(Distribution):
    """BLABLABLABLA"""

    def __init__(self, rate: Optional[float] = None):
        self.functions = ExponentialFunctions(rate=rate)

    @property
    def params(self):
        """BLABLABLA"""
        return self.functions.params

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
    ) -> Parameters:

        time = array_factory(time)

        if entry is not None:
            entry = array_factory(entry)

        if departure is not None:
            departure = array_factory(departure)

        if lc_indicators is not None:
            lc_indicators = array_factory(lc_indicators).astype(np.bool_)

        if rc_indicators is not None:
            rc_indicators = array_factory(rc_indicators).astype(np.bool_)

        factory: LifetimeDataFactory
        if time.shape[-1] == 1:
            factory = LifetimeDataFactoryFrom1D(
                time,
                entry,
                departure,
                lc_indicators,
                rc_indicators,
            )
        else:
            factory = LifetimeDataFactoryFrom2D(
                time,
                entry,
                departure,
                lc_indicators,
                rc_indicators,
            )
        observed_lifetimes, truncations = factory()

        likelihood = GenericDistributionLikelihood(
            self.functions, observed_lifetimes, truncations
        )
        optimizer = GenericDistributionOptimizer(likelihood)
        optimum_params = optimizer.fit()
        if inplace:
            self.functions.params.values = optimum_params.values
        return optimum_params


class Weibull(Distribution):
    """BLABLABLABLA"""

    def __init__(self, shape: Optional[float] = None, rate: Optional[float] = None):
        self.functions = WeibullFunctions(shape=shape, rate=rate)

    @property
    def params(self):
        """BLABLABLA"""
        return self.functions.params

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
    ) -> Parameters:

        time = array_factory(time)

        if entry is not None:
            entry = array_factory(entry)

        if departure is not None:
            departure = array_factory(departure)

        if lc_indicators is not None:
            lc_indicators = array_factory(lc_indicators).astype(np.bool_)

        if rc_indicators is not None:
            rc_indicators = array_factory(rc_indicators).astype(np.bool_)

        factory: LifetimeDataFactory
        if time.shape[-1] == 1:
            factory = LifetimeDataFactoryFrom1D(
                time,
                entry,
                departure,
                lc_indicators,
                rc_indicators,
            )
        else:
            factory = LifetimeDataFactoryFrom2D(
                time,
                entry,
                departure,
                lc_indicators,
                rc_indicators,
            )
        observed_lifetimes, truncations = factory()

        likelihood = GenericDistributionLikelihood(
            self.functions, observed_lifetimes, truncations
        )
        optimizer = GenericDistributionOptimizer(likelihood)
        optimum_params = optimizer.fit()
        if inplace:
            self.functions.params.values = optimum_params.values
        return optimum_params


class Gompertz(Distribution):
    """BLABLABLABLA"""

    def __init__(self, shape: Optional[float] = None, rate: Optional[float] = None):
        self.functions = GompertzFunctions(shape=shape, rate=rate)

    @property
    def params(self):
        """BLABLABLA"""
        return self.functions.params

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
    ) -> Parameters:

        time = array_factory(time)

        if entry is not None:
            entry = array_factory(entry)

        if departure is not None:
            departure = array_factory(departure)

        if lc_indicators is not None:
            lc_indicators = array_factory(lc_indicators).astype(np.bool_)

        if rc_indicators is not None:
            rc_indicators = array_factory(rc_indicators).astype(np.bool_)

        factory: LifetimeDataFactory
        if time.shape[-1] == 1:
            factory = LifetimeDataFactoryFrom1D(
                time,
                entry,
                departure,
                lc_indicators,
                rc_indicators,
            )
        else:
            factory = LifetimeDataFactoryFrom2D(
                time,
                entry,
                departure,
                lc_indicators,
                rc_indicators,
            )
        observed_lifetimes, truncations = factory()

        likelihood = GenericDistributionLikelihood(
            self.functions, observed_lifetimes, truncations
        )
        optimizer = GompertzOptimizer(likelihood)
        optimum_params = optimizer.fit()
        if inplace:
            self.functions.params.values = optimum_params.values
        return optimum_params


class Gamma(Distribution):
    """BLABLABLABLA"""

    def __init__(self, shape: Optional[float] = None, rate: Optional[float] = None):
        self.functions = GammaFunctions(shape=shape, rate=rate)

    @property
    def params(self):
        """BLABLABLA"""
        return self.functions.params

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
    ) -> Parameters:

        time = array_factory(time)

        if entry is not None:
            entry = array_factory(entry)

        if departure is not None:
            departure = array_factory(departure)

        if lc_indicators is not None:
            lc_indicators = array_factory(lc_indicators).astype(np.bool_)

        if rc_indicators is not None:
            rc_indicators = array_factory(rc_indicators).astype(np.bool_)

        factory: LifetimeDataFactory
        if time.shape[-1] == 1:
            factory = LifetimeDataFactoryFrom1D(
                time,
                entry,
                departure,
                lc_indicators,
                rc_indicators,
            )
        else:
            factory = LifetimeDataFactoryFrom2D(
                time,
                entry,
                departure,
                lc_indicators,
                rc_indicators,
            )
        observed_lifetimes, truncations = factory()

        likelihood = GenericDistributionLikelihood(
            self.functions, observed_lifetimes, truncations
        )
        optimizer = GenericDistributionOptimizer(likelihood)
        optimum_params = optimizer.fit()
        if inplace:
            self.functions.params.values = optimum_params.values
        return optimum_params


class LogLogistic(Distribution):
    """BLABLABLABLA"""

    def __init__(self, shape: Optional[float] = None, rate: Optional[float] = None):
        self.functions = LogLogisticFunctions(shape=shape, rate=rate)

    @property
    def params(self):
        """BLABLABLA"""
        return self.functions.params

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
    ) -> Parameters:

        time = array_factory(time)

        if entry is not None:
            entry = array_factory(entry)

        if departure is not None:
            departure = array_factory(departure)

        if lc_indicators is not None:
            lc_indicators = array_factory(lc_indicators).astype(np.bool_)

        if rc_indicators is not None:
            rc_indicators = array_factory(rc_indicators).astype(np.bool_)

        factory: LifetimeDataFactory
        if time.shape[-1] == 1:
            factory = LifetimeDataFactoryFrom1D(
                time,
                entry,
                departure,
                lc_indicators,
                rc_indicators,
            )
        else:
            factory = LifetimeDataFactoryFrom2D(
                time,
                entry,
                departure,
                lc_indicators,
                rc_indicators,
            )
        observed_lifetimes, truncations = factory()

        likelihood = GenericDistributionLikelihood(
            self.functions, observed_lifetimes, truncations
        )
        optimizer = GenericDistributionOptimizer(likelihood)
        optimum_params = optimizer.fit()
        if inplace:
            self.functions.params.values = optimum_params.values
        return optimum_params
