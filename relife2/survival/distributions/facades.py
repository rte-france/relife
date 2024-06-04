"""
This module defines classes seen by clients to use model of distributions.
These classes instanciate facade object design pattern.

Copyright (c) 2022, RTE (https://www.rte-france.com)
See AUTHORS.txt
SPDX-License-Identifier: Apache-2.0 (see LICENSE.txt)
"""

from __future__ import annotations

from typing import Optional

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


class Exponential(Distribution):
    """BLABLABLABLA"""

    def __init__(self, rate: Optional[float] = None):
        super().__init__(
            ExponentialFunctions(rate=rate),
            GenericDistributionLikelihood,
            GenericDistributionOptimizer,
        )


class Weibull(Distribution):
    """BLABLABLABLA"""

    def __init__(self, shape: Optional[float] = None, rate: Optional[float] = None):

        super().__init__(
            WeibullFunctions(shape=shape, rate=rate),
            GenericDistributionLikelihood,
            GenericDistributionOptimizer,
        )


class Gompertz(Distribution):
    """BLABLABLABLA"""

    def __init__(self, shape: Optional[float] = None, rate: Optional[float] = None):
        super().__init__(
            GompertzFunctions(shape=shape, rate=rate),
            GenericDistributionLikelihood,
            GompertzOptimizer,
        )


class Gamma(Distribution):
    """BLABLABLABLA"""

    def __init__(self, shape: Optional[float] = None, rate: Optional[float] = None):
        super().__init__(
            GammaFunctions(shape=shape, rate=rate),
            GenericDistributionLikelihood,
            GenericDistributionOptimizer,
        )


class LogLogistic(Distribution):
    """BLABLABLABLA"""

    def __init__(self, shape: Optional[float] = None, rate: Optional[float] = None):
        super().__init__(
            LogLogisticFunctions(shape=shape, rate=rate),
            GenericDistributionLikelihood,
            GenericDistributionOptimizer,
        )
