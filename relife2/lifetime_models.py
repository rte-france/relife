"""
This module defines lifetime models objects known by clients
Copyright (c) 2022, RTE (https://www.rte-france.com)
See AUTHORS.txt
SPDX-License-Identifier: Apache-2.0 (see LICENSE.txt)
"""

from typing import Optional

from relife2.api.models import Distribution, GammaProcessDistribution, Regression
from relife2.stats.distributions import (
    DistributionFunctions,
    ExponentialFunctions,
    GammaFunctions,
    GompertzFunctions,
    LogLogisticFunctions,
    WeibullFunctions,
)
from relife2.stats.gammaprocess import GPDistributionFunctions, PowerShapeFunctions
from relife2.stats.regressions import (
    AFTFunctions,
    CovarEffect,
    ProportionalHazardFunctions,
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
    coefficients: Optional[
        tuple[float | None] | list[float | None] | dict[str, float | None]
    ] = None,
) -> dict[str, float | None]:
    """

    Args:
        coefficients ():

    Returns:

    """
    if coefficients is None:
        return {"coef_0": None}
    if isinstance(coefficients, (list, tuple)):
        return {f"coef_{i}": v for i, v in enumerate(coefficients)}
    if isinstance(coefficients, dict):
        return coefficients
    raise ValueError("coefficients must be tuple, list or dict")


class ProportionalHazard(Regression):
    """BLABLABLABLA"""

    def __init__(
        self,
        baseline: Distribution,
        coefficients: Optional[
            tuple[float | None] | list[float | None] | dict[str, float | None]
        ] = None,
    ):
        coefficients = control_covar_args(coefficients)
        super().__init__(
            ProportionalHazardFunctions(
                CovarEffect(**coefficients),
                baseline.functions.copy(),
            )
        )


class AFT(Regression):
    """BLABLABLABLA"""

    def __init__(
        self,
        baseline: Distribution,
        coefficients: Optional[
            tuple[float | None] | list[float | None] | dict[str, float | None]
        ] = None,
    ):
        coefficients = control_covar_args(coefficients)
        super().__init__(
            AFTFunctions(
                CovarEffect(**coefficients),
                baseline.functions.copy(),
            )
        )


class PowerGPDistribution(GammaProcessDistribution):
    """BLABLABLABLA"""

    def __init__(
        self,
        rate: Optional[float] = None,
        shape_rate: Optional[float] = None,
        shape_power: Optional[float] = None,
    ):

        super().__init__(
            GPDistributionFunctions(
                PowerShapeFunctions(shape_rate=shape_rate, shape_power=shape_power),
                rate,
            )
        )
