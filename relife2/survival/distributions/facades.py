"""
This module defines classes seen by clients to use model of distributions.
These classes instanciate facade object design pattern.

Copyright (c) 2022, RTE (https://www.rte-france.com)
See AUTHORS.txt
SPDX-License-Identifier: Apache-2.0 (see LICENSE.txt)
"""

from typing import Optional, Union

import numpy as np
from numpy.typing import ArrayLike

from relife2.survival.data import array_factory
from relife2.survival.distributions.functions import ExponentialFunctions
from relife2.survival.distributions.likelihoods import GenericDistributionLikelihood
from relife2.survival.distributions.optimizers import GenericDistributionOptimizer
from relife2.survival.distributions.types import Distribution, FloatArray
from relife2.survival.parameters import Parameters


class Exponential(Distribution):
    """BLABLABLABLA"""

    def __init__(self, rate: Optional[float] = None):
        self.functions = ExponentialFunctions(rate=rate)

    @property
    def params(self):
        """BLABLABLA"""
        return self.functions.params

    def sf(self, time: ArrayLike) -> Union[float, FloatArray]:
        return self.functions.sf(array_factory(time))

    def isf(self, probability: ArrayLike) -> Union[float, FloatArray]:
        return self.functions.isf(array_factory(probability))

    def hf(self, time: ArrayLike) -> Union[float, FloatArray]:
        return self.functions.hf(array_factory(time))

    def chf(self, time: ArrayLike) -> Union[float, FloatArray]:
        return self.functions.chf(array_factory(time))

    def cdf(self, time: ArrayLike) -> Union[float, FloatArray]:
        return self.functions.cdf(array_factory(time))

    def pdf(self, probability: ArrayLike) -> Union[float, FloatArray]:
        return self.functions.pdf(array_factory(probability))

    def ppf(self, time: ArrayLike) -> Union[float, FloatArray]:
        return self.functions.ppf(array_factory(time))

    def mrl(self, time: ArrayLike) -> Union[float, FloatArray]:
        return self.functions.mrl(array_factory(time))

    def ichf(self, cumulative_hazard_rate: ArrayLike) -> Union[float, FloatArray]:
        return self.functions.ichf(array_factory(cumulative_hazard_rate))

    def rvs(
        self, size: Optional[int] = 1, seed: Optional[int] = None
    ) -> Union[float, FloatArray]:
        return self.functions.rvs(size=size, seed=seed)

    def mean(
        self,
    ) -> float:
        return self.functions.mean()

    def var(
        self,
    ) -> float:
        return self.functions.var()

    def fit(
        self,
        time: ArrayLike,
        inplace: bool = True,
        entry: Optional[ArrayLike] = None,
        departure: Optional[ArrayLike] = None,
        **indicators: ArrayLike,
    ) -> Parameters:

        time = array_factory(time)
        lc_indicators = array_factory(
            indicators.get("lc_indicators", np.zeros_like(time))
        ).astype(np.bool_)
        rc_indicators = array_factory(
            indicators.get("rc_indicators", np.zeros_like(time))
        ).astype(np.bool_)

        if entry is not None:
            entry = array_factory(entry)
        if departure is not None:
            departure = array_factory(departure)

        likelihood = GenericDistributionLikelihood(
            self.functions,
            array_factory(time),
            entry,
            departure,
            lc_indicators=lc_indicators,
            rc_indicators=rc_indicators,
        )
        optimizer = GenericDistributionOptimizer(likelihood)
        optimum_params = optimizer.fit()
        if inplace:
            self.functions.params.values = optimum_params.values
        return optimum_params