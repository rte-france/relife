"""
This module defines classes seen by clients to use model of distributions.
These classes instanciate facade object design pattern.

Copyright (c) 2022, RTE (https://www.rte-france.com)
See AUTHORS.txt
SPDX-License-Identifier: Apache-2.0 (see LICENSE.txt)
"""

from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from relife2.survival.data import array_factory, lifetime_factory_template
from relife2.survival.distributions.functions import (
    ExponentialFunctions,
    GammaFunctions,
    GompertzFunctions,
    LogLogisticFunctions,
    WeibullFunctions,
)
from relife2.survival.distributions.likelihoods import GenericDistributionLikelihood
from relife2.survival.distributions.types import Distribution
from relife2.survival.optimizers import LikelihoodOptimizer
from relife2.survival.parameters import Parameters

FloatArray = NDArray[np.float64]


class Exponential(Distribution):
    """BLABLABLABLA"""

    def __init__(self, rate: Optional[float] = None):
        super().__init__(ExponentialFunctions(rate=rate))

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
    ) -> Parameters:

        observed_lifetimes, truncations = lifetime_factory_template(
            time, entry, departure, lc_indicators, rc_indicators
        )

        likelihood = GenericDistributionLikelihood(
            ExponentialFunctions(rate=self.params.rate), observed_lifetimes, truncations
        )
        optimizer = LikelihoodOptimizer(
            likelihood, param0=self.functions.init_params(observed_lifetimes.rlc)
        )
        optimum_params = optimizer.fit(**kwargs)
        if inplace:
            self.params = optimum_params
        return optimum_params


class Weibull(Distribution):
    """BLABLABLABLA"""

    def __init__(self, shape: Optional[float] = None, rate: Optional[float] = None):
        super().__init__(WeibullFunctions(shape=shape, rate=rate))

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
    ) -> Parameters:

        observed_lifetimes, truncations = lifetime_factory_template(
            time, entry, departure, lc_indicators, rc_indicators
        )

        likelihood = GenericDistributionLikelihood(
            WeibullFunctions(shape=self.params.shape, rate=self.params.rate),
            observed_lifetimes,
            truncations,
        )
        optimizer = LikelihoodOptimizer(
            likelihood, param0=self.functions.init_params(observed_lifetimes.rlc)
        )
        optimum_params = optimizer.fit(**kwargs)
        if inplace:
            self.params = optimum_params
        return optimum_params


class Gompertz(Distribution):
    """BLABLABLABLA"""

    def __init__(self, shape: Optional[float] = None, rate: Optional[float] = None):
        super().__init__(GompertzFunctions(shape=shape, rate=rate))

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
    ) -> Parameters:

        observed_lifetimes, truncations = lifetime_factory_template(
            time, entry, departure, lc_indicators, rc_indicators
        )

        likelihood = GenericDistributionLikelihood(
            GompertzFunctions(shape=self.params.shape, rate=self.params.rate),
            observed_lifetimes,
            truncations,
        )
        optimizer = LikelihoodOptimizer(
            likelihood, param0=self.functions.init_params(observed_lifetimes.rlc)
        )
        optimum_params = optimizer.fit(**kwargs)
        if inplace:
            self.params = optimum_params
        return optimum_params


class Gamma(Distribution):
    """BLABLABLABLA"""

    def __init__(self, shape: Optional[float] = None, rate: Optional[float] = None):
        super().__init__(GammaFunctions(shape=shape, rate=rate))

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
    ) -> Parameters:

        observed_lifetimes, truncations = lifetime_factory_template(
            time, entry, departure, lc_indicators, rc_indicators
        )

        likelihood = GenericDistributionLikelihood(
            GammaFunctions(shape=self.params.shape, rate=self.params.rate),
            observed_lifetimes,
            truncations,
        )
        optimizer = LikelihoodOptimizer(
            likelihood, param0=self.functions.init_params(observed_lifetimes.rlc)
        )
        optimum_params = optimizer.fit(**kwargs)
        if inplace:
            self.params = optimum_params
        return optimum_params


class LogLogistic(Distribution):
    """BLABLABLABLA"""

    def __init__(self, shape: Optional[float] = None, rate: Optional[float] = None):
        super().__init__(LogLogisticFunctions(shape=shape, rate=rate))

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
    ) -> Parameters:

        observed_lifetimes, truncations = lifetime_factory_template(
            time, entry, departure, lc_indicators, rc_indicators
        )

        likelihood = GenericDistributionLikelihood(
            LogLogisticFunctions(shape=self.params.shape, rate=self.params.rate),
            observed_lifetimes,
            truncations,
        )

        optimizer = LikelihoodOptimizer(
            likelihood, param0=self.functions.init_params(observed_lifetimes.rlc)
        )
        optimum_params = optimizer.fit(**kwargs)
        if inplace:
            self.params = optimum_params
        return optimum_params
