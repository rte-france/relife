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

from relife2.survival.data import array_factory, lifetime_factory_template
from relife2.survival.distributions.types import Distribution
from relife2.survival.optimizers import LikelihoodOptimizer
from relife2.survival.parameters import Parameters
from relife2.survival.regressions.functions import (
    AFTEffect,
    AFTFunctions,
    ProportionalHazardEffect,
    ProportionalHazardFunctions,
)
from relife2.survival.regressions.likelihoods import GenericRegressionLikelihood
from relife2.survival.regressions.types import Regression

FloatArray = NDArray[np.float64]


class ProportionalHazard(Regression):
    """BLABLABLABLA"""

    def __init__(self, baseline: Distribution, *beta: Union[float, None]):
        super().__init__(
            ProportionalHazardFunctions(
                baseline.functions,
                ProportionalHazardEffect(
                    **{f"beta_{i}": value for i, value in enumerate(beta)}
                ),
            )
        )

    def sf(self, time: ArrayLike, covar: ArrayLike) -> Union[float, FloatArray]:
        covar = array_factory(covar)
        self._check_covar_dim(covar)
        return np.squeeze(self.functions.sf(array_factory(time), covar))[()]

    def isf(self, probability: ArrayLike, covar: ArrayLike) -> Union[float, FloatArray]:
        covar = array_factory(covar)
        self._check_covar_dim(covar)
        return np.squeeze(self.functions.isf(array_factory(probability), covar))[()]

    def hf(self, time: ArrayLike, covar: ArrayLike) -> Union[float, FloatArray]:
        covar = array_factory(covar)
        self._check_covar_dim(covar)
        return np.squeeze(self.functions.hf(array_factory(time), covar))[()]

    def chf(self, time: ArrayLike, covar: ArrayLike) -> Union[float, FloatArray]:
        covar = array_factory(covar)
        self._check_covar_dim(covar)
        return np.squeeze(self.functions.chf(array_factory(time), covar))[()]

    def cdf(self, time: ArrayLike, covar: ArrayLike) -> Union[float, FloatArray]:
        covar = array_factory(covar)
        self._check_covar_dim(covar)
        return np.squeeze(self.functions.cdf(array_factory(time), covar))[()]

    def pdf(self, probability: ArrayLike, covar: ArrayLike) -> Union[float, FloatArray]:
        covar = array_factory(covar)
        self._check_covar_dim(covar)
        return np.squeeze(self.functions.pdf(array_factory(probability), covar))[()]

    def ppf(self, time: ArrayLike, covar: ArrayLike) -> Union[float, FloatArray]:
        covar = array_factory(covar)
        self._check_covar_dim(covar)
        return np.squeeze(self.functions.ppf(array_factory(time), covar))[()]

    def mrl(self, time: ArrayLike, covar: ArrayLike) -> Union[float, FloatArray]:
        covar = array_factory(covar)
        self._check_covar_dim(covar)
        return np.squeeze(self.functions.mrl(array_factory(time), covar))[()]

    def ichf(
        self, cumulative_hazard_rate: ArrayLike, covar: ArrayLike
    ) -> Union[float, FloatArray]:
        covar = array_factory(covar)
        self._check_covar_dim(covar)
        return np.squeeze(
            self.functions.ichf(array_factory(cumulative_hazard_rate), covar)
        )[()]

    def rvs(
        self, covar: ArrayLike, size: Optional[int] = 1, seed: Optional[int] = None
    ) -> Union[float, FloatArray]:
        covar = array_factory(covar)
        self._check_covar_dim(covar)
        return np.squeeze(self.functions.rvs(covar, size=size, seed=seed))[()]

    def mean(self, covar: ArrayLike) -> float:
        covar = array_factory(covar)
        self._check_covar_dim(covar)
        return self.functions.mean(covar)

    def var(self, covar: ArrayLike) -> float:
        covar = array_factory(covar)
        self._check_covar_dim(covar)
        return self.functions.var(covar)

    def median(self, covar: ArrayLike) -> float:
        covar = array_factory(covar)
        self._check_covar_dim(covar)
        return self.functions.median(covar)

    def fit(
        self,
        time: ArrayLike,
        entry: Optional[ArrayLike] = None,
        departure: Optional[ArrayLike] = None,
        lc_indicators: Optional[ArrayLike] = None,
        rc_indicators: Optional[ArrayLike] = None,
        inplace: bool = True,
        **kwargs,
    ) -> Parameters:

        if "covar" not in kwargs:
            raise ValueError(
                "fit expects covar values, add covar : fit(..., covar = ...)"
            )
        covar = kwargs.pop("covar")
        covar = array_factory(covar)
        self._check_covar_dim(covar)
        observed_lifetimes, truncations = lifetime_factory_template(
            time, entry, departure, lc_indicators, rc_indicators
        )
        likelihood = GenericRegressionLikelihood(
            ProportionalHazardFunctions(
                self.baseline.__class__(),
                ProportionalHazardEffect(
                    **{f"beta_{i}": None for i in range(self.covar_effect.params.size)}
                ),
            ),
            observed_lifetimes,
            truncations,
            covar,
        )
        optimizer = LikelihoodOptimizer(likelihood)
        optimum_params = optimizer.fit()
        if inplace:
            self.params = optimum_params
        return optimum_params


class AFT(Regression):
    """BLABLABLABLA"""

    def __init__(self, baseline: Distribution, *beta: Union[float, None]):
        super().__init__(
            AFTFunctions(
                baseline.functions,
                AFTEffect(**{f"beta_{i}": value for i, value in enumerate(beta)}),
            )
        )

    def sf(self, time: ArrayLike, covar: ArrayLike) -> Union[float, FloatArray]:
        covar = array_factory(covar)
        self._check_covar_dim(covar)
        return np.squeeze(self.functions.sf(array_factory(time), covar))[()]

    def isf(self, probability: ArrayLike, covar: ArrayLike) -> Union[float, FloatArray]:
        covar = array_factory(covar)
        self._check_covar_dim(covar)
        return np.squeeze(self.functions.isf(array_factory(probability), covar))[()]

    def hf(self, time: ArrayLike, covar: ArrayLike) -> Union[float, FloatArray]:
        covar = array_factory(covar)
        self._check_covar_dim(covar)
        return np.squeeze(self.functions.hf(array_factory(time), covar))[()]

    def chf(self, time: ArrayLike, covar: ArrayLike) -> Union[float, FloatArray]:
        covar = array_factory(covar)
        self._check_covar_dim(covar)
        return np.squeeze(self.functions.chf(array_factory(time), covar))[()]

    def cdf(self, time: ArrayLike, covar: ArrayLike) -> Union[float, FloatArray]:
        covar = array_factory(covar)
        self._check_covar_dim(covar)
        return np.squeeze(self.functions.cdf(array_factory(time), covar))[()]

    def pdf(self, probability: ArrayLike, covar: ArrayLike) -> Union[float, FloatArray]:
        covar = array_factory(covar)
        self._check_covar_dim(covar)
        return np.squeeze(self.functions.pdf(array_factory(probability), covar))[()]

    def ppf(self, time: ArrayLike, covar: ArrayLike) -> Union[float, FloatArray]:
        covar = array_factory(covar)
        self._check_covar_dim(covar)
        return np.squeeze(self.functions.ppf(array_factory(time), covar))[()]

    def mrl(self, time: ArrayLike, covar: ArrayLike) -> Union[float, FloatArray]:
        covar = array_factory(covar)
        self._check_covar_dim(covar)
        return np.squeeze(self.functions.mrl(array_factory(time), covar))[()]

    def ichf(
        self, cumulative_hazard_rate: ArrayLike, covar: ArrayLike
    ) -> Union[float, FloatArray]:
        covar = array_factory(covar)
        self._check_covar_dim(covar)
        return np.squeeze(
            self.functions.ichf(array_factory(cumulative_hazard_rate), covar)
        )[()]

    def rvs(
        self, covar: ArrayLike, size: Optional[int] = 1, seed: Optional[int] = None
    ) -> Union[float, FloatArray]:
        covar = array_factory(covar)
        self._check_covar_dim(covar)
        return np.squeeze(self.functions.rvs(covar, size=size, seed=seed))[()]

    def mean(self, covar: ArrayLike) -> float:
        covar = array_factory(covar)
        self._check_covar_dim(covar)
        return np.squeeze(self.functions.mean(covar))[()]

    def var(self, covar: ArrayLike) -> float:
        covar = array_factory(covar)
        self._check_covar_dim(covar)
        return np.squeeze(self.functions.var(covar))[()]

    def median(self, covar: ArrayLike) -> float:
        covar = array_factory(covar)
        self._check_covar_dim(covar)
        return np.squeeze(self.functions.median(covar))[()]

    def fit(
        self,
        time: ArrayLike,
        entry: Optional[ArrayLike] = None,
        departure: Optional[ArrayLike] = None,
        lc_indicators: Optional[ArrayLike] = None,
        rc_indicators: Optional[ArrayLike] = None,
        inplace: bool = True,
        **kwargs,
    ) -> Parameters:

        if "covar" not in kwargs:
            raise ValueError(
                "fit expects covar values, add covar : fit(..., covar = ...)"
            )
        covar = kwargs.pop("covar")
        covar = array_factory(covar)
        self._check_covar_dim(covar)

        observed_lifetimes, truncations = lifetime_factory_template(
            time, entry, departure, lc_indicators, rc_indicators
        )

        likelihood = GenericRegressionLikelihood(
            AFTFunctions(
                self.baseline.__class__(),
                AFTEffect(
                    **{f"beta_{i}": None for i in range(self.covar_effect.params.size)}
                ),
            ),
            observed_lifetimes,
            truncations,
            covar,
        )
        optimizer = LikelihoodOptimizer(likelihood)
        optimum_params = optimizer.fit(**kwargs)
        if inplace:
            self.params = optimum_params
        return optimum_params
