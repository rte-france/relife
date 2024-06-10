"""
This module defines classes seen by clients to use model of regressions.
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
from relife2.survival.distributions.types import Distribution
from relife2.survival.parameters import Parameters
from relife2.survival.regressions.functions import (
    AFTEffect,
    AFTFunctions,
    ProportionalHazardEffect,
    ProportionalHazardFunctions,
)
from relife2.survival.regressions.likelihoods import GenericRegressionLikelihood
from relife2.survival.regressions.optimizers import GenericRegressionOptimizer
from relife2.survival.regressions.types import Regression

FloatArray = NDArray[np.float64]


class ProportionalHazard(Regression):
    """BLABLABLABLA"""

    def __init__(self, baseline: Distribution, **beta: Union[float, None]):
        super().__init__(
            ProportionalHazardFunctions(
                baseline.functions, ProportionalHazardEffect(**beta)
            )
        )

    @property
    def params(self):
        """BLABLABLA"""
        return self.functions.params

    def sf(self, time: ArrayLike, covar: ArrayLike) -> Union[float, FloatArray]:
        return np.squeeze(self.functions.sf(array_factory(time), array_factory(covar)))[
            ()
        ]

    def isf(self, probability: ArrayLike, covar: ArrayLike) -> Union[float, FloatArray]:
        return np.squeeze(
            self.functions.isf(array_factory(probability), array_factory(covar))
        )[()]

    def hf(self, time: ArrayLike, covar: ArrayLike) -> Union[float, FloatArray]:
        return np.squeeze(self.functions.hf(array_factory(time), array_factory(covar)))[
            ()
        ]

    def chf(self, time: ArrayLike, covar: ArrayLike) -> Union[float, FloatArray]:
        return np.squeeze(
            self.functions.chf(array_factory(time), array_factory(covar))
        )[()]

    def cdf(self, time: ArrayLike, covar: ArrayLike) -> Union[float, FloatArray]:
        return np.squeeze(
            self.functions.cdf(array_factory(time), array_factory(covar))
        )[()]

    def pdf(self, probability: ArrayLike, covar: ArrayLike) -> Union[float, FloatArray]:
        return np.squeeze(
            self.functions.pdf(array_factory(probability), array_factory(covar))
        )[()]

    def ppf(self, time: ArrayLike, covar: ArrayLike) -> Union[float, FloatArray]:
        return np.squeeze(
            self.functions.ppf(array_factory(time), array_factory(covar))
        )[()]

    def mrl(self, time: ArrayLike, covar: ArrayLike) -> Union[float, FloatArray]:
        return np.squeeze(
            self.functions.mrl(array_factory(time), array_factory(covar))
        )[()]

    def ichf(
        self, cumulative_hazard_rate: ArrayLike, covar: ArrayLike
    ) -> Union[float, FloatArray]:
        return np.squeeze(
            self.functions.ichf(
                array_factory(cumulative_hazard_rate), array_factory(covar)
            )
        )[()]

    def rvs(
        self, covar: ArrayLike, size: Optional[int] = 1, seed: Optional[int] = None
    ) -> Union[float, FloatArray]:
        return np.squeeze(
            self.functions.rvs(array_factory(covar), size=size, seed=seed)
        )[()]

    def mean(self, covar: ArrayLike) -> float:
        return self.functions.mean(array_factory(covar))

    def var(self, covar: ArrayLike) -> float:
        return self.functions.var(array_factory(covar))

    def median(self, covar: ArrayLike) -> float:
        return self.functions.median(array_factory(covar))

    def fit(
        self,
        time: ArrayLike,
        covar: ArrayLike,
        entry: Optional[ArrayLike] = None,
        departure: Optional[ArrayLike] = None,
        lc_indicators: Optional[ArrayLike] = None,
        rc_indicators: Optional[ArrayLike] = None,
        inplace: bool = True,
    ) -> Parameters:

        time = array_factory(time)
        covar = array_factory(covar)

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

        likelihood = GenericRegressionLikelihood(
            self.functions,
            observed_lifetimes,
            truncations,
            covar,
        )
        optimizer = GenericRegressionOptimizer(likelihood)
        optimum_params = optimizer.fit()
        if inplace:
            self.functions.params.values = optimum_params.values
        return optimum_params


class AFT(Regression):
    """BLABLABLABLA"""

    def __init__(self, baseline: Distribution, **beta: Union[float, None]):
        super().__init__(AFTFunctions(baseline.functions, AFTEffect(**beta)))

    @property
    def params(self):
        """BLABLABLA"""
        return self.functions.params

    def sf(self, time: ArrayLike, covar: ArrayLike) -> Union[float, FloatArray]:
        return np.squeeze(self.functions.sf(array_factory(time), array_factory(covar)))[
            ()
        ]

    def isf(self, probability: ArrayLike, covar: ArrayLike) -> Union[float, FloatArray]:
        return np.squeeze(
            self.functions.isf(array_factory(probability), array_factory(covar))
        )[()]

    def hf(self, time: ArrayLike, covar: ArrayLike) -> Union[float, FloatArray]:
        return np.squeeze(self.functions.hf(array_factory(time), array_factory(covar)))[
            ()
        ]

    def chf(self, time: ArrayLike, covar: ArrayLike) -> Union[float, FloatArray]:
        return np.squeeze(
            self.functions.chf(array_factory(time), array_factory(covar))
        )[()]

    def cdf(self, time: ArrayLike, covar: ArrayLike) -> Union[float, FloatArray]:
        return np.squeeze(
            self.functions.cdf(array_factory(time), array_factory(covar))
        )[()]

    def pdf(self, probability: ArrayLike, covar: ArrayLike) -> Union[float, FloatArray]:
        return np.squeeze(
            self.functions.pdf(array_factory(probability), array_factory(covar))
        )[()]

    def ppf(self, time: ArrayLike, covar: ArrayLike) -> Union[float, FloatArray]:
        return np.squeeze(
            self.functions.ppf(array_factory(time), array_factory(covar))
        )[()]

    def mrl(self, time: ArrayLike, covar: ArrayLike) -> Union[float, FloatArray]:
        return np.squeeze(
            self.functions.mrl(array_factory(time), array_factory(covar))
        )[()]

    def ichf(
        self, cumulative_hazard_rate: ArrayLike, covar: ArrayLike
    ) -> Union[float, FloatArray]:
        return np.squeeze(
            self.functions.ichf(
                array_factory(cumulative_hazard_rate), array_factory(covar)
            )
        )[()]

    def rvs(
        self, covar: ArrayLike, size: Optional[int] = 1, seed: Optional[int] = None
    ) -> Union[float, FloatArray]:
        return np.squeeze(
            self.functions.rvs(array_factory(covar), size=size, seed=seed)
        )[()]

    def mean(self, covar: ArrayLike) -> float:
        return self.functions.mean(array_factory(covar))

    def var(self, covar: ArrayLike) -> float:
        return self.functions.var(array_factory(covar))

    def median(self, covar: ArrayLike) -> float:
        return self.functions.median(array_factory(covar))

    def fit(
        self,
        time: ArrayLike,
        covar: ArrayLike,
        entry: Optional[ArrayLike] = None,
        departure: Optional[ArrayLike] = None,
        lc_indicators: Optional[ArrayLike] = None,
        rc_indicators: Optional[ArrayLike] = None,
        inplace: bool = True,
    ) -> Parameters:

        time = array_factory(time)
        covar = array_factory(covar)

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

        likelihood = GenericRegressionLikelihood(
            self.functions,
            observed_lifetimes,
            truncations,
            covar,
        )
        optimizer = GenericRegressionOptimizer(likelihood)
        optimum_params = optimizer.fit()
        if inplace:
            self.functions.params.values = optimum_params.values
        return optimum_params
