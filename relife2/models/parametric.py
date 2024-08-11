from typing import Optional

from relife2.functions import (
    AFTFunctions,
    CovarEffect,
    DistributionFunctions,
    ExponentialFunctions,
    GammaFunctions,
    GompertzFunctions,
    GPDistributionFunctions,
    LogLogisticFunctions,
    PowerShapeFunctions,
    ProportionalHazardFunctions,
    WeibullFunctions,
)
from relife2.models.core import ParametricLifetimeModel


class Exponential(ParametricLifetimeModel):
    """BLABLABLABLA"""

    functions: DistributionFunctions

    def __init__(self, rate: Optional[float] = None):
        super().__init__(ExponentialFunctions(rate=rate))


class Weibull(ParametricLifetimeModel):
    """BLABLABLABLA"""

    functions: DistributionFunctions

    def __init__(self, shape: Optional[float] = None, rate: Optional[float] = None):
        super().__init__(WeibullFunctions(shape=shape, rate=rate))


class Gompertz(ParametricLifetimeModel):
    """BLABLABLABLA"""

    functions: DistributionFunctions

    def __init__(self, shape: Optional[float] = None, rate: Optional[float] = None):
        super().__init__(GompertzFunctions(shape=shape, rate=rate))


class Gamma(ParametricLifetimeModel):
    """BLABLABLABLA"""

    functions: DistributionFunctions

    def __init__(self, shape: Optional[float] = None, rate: Optional[float] = None):
        super().__init__(GammaFunctions(shape=shape, rate=rate))


class LogLogistic(ParametricLifetimeModel):
    """BLABLABLABLA"""

    functions: DistributionFunctions

    def __init__(self, shape: Optional[float] = None, rate: Optional[float] = None):
        super().__init__(LogLogisticFunctions(shape=shape, rate=rate))


class PowerGPDistribution(ParametricLifetimeModel):
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
            ),
        )


def check_coefficients_args(
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


class ProportionalHazard(ParametricLifetimeModel):
    """BLABLABLABLA"""

    def __init__(
        self,
        baseline: ParametricLifetimeModel,
        coefficients: Optional[
            tuple[float | None] | list[float | None] | dict[str, float | None]
        ] = None,
    ):
        coefficients = check_coefficients_args(coefficients)
        super().__init__(
            ProportionalHazardFunctions(
                CovarEffect(**coefficients),
                baseline.functions.copy(),
            )
        )


class AFT(ParametricLifetimeModel):
    """BLABLABLABLA"""

    def __init__(
        self,
        baseline: ParametricLifetimeModel,
        coefficients: Optional[
            tuple[float | None] | list[float | None] | dict[str, float | None]
        ] = None,
    ):
        coefficients = check_coefficients_args(coefficients)
        super().__init__(
            AFTFunctions(
                CovarEffect(**coefficients),
                baseline.functions.copy(),
            )
        )
