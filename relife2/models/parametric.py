from typing import Optional, Sequence

from relife2.functions import (
    AFTFunction,
    DistributionFunction,
    ExponentialFunction,
    GammaFunction,
    GompertzFunction,
    GPDistributionFunction,
    LogLogisticFunction,
    PowerShapeFunction,
    ProportionalHazardFunction,
    WeibullFunction,
    LeftTruncatedFunction,
    AgeReplacementFunction,
)
from relife2.models.core import ParametricLifetimeModel


class Exponential(ParametricLifetimeModel):
    """BLABLABLABLA"""

    function: DistributionFunction

    def __init__(self, rate: Optional[float] = None):
        super().__init__(ExponentialFunction(rate=rate))


class Weibull(ParametricLifetimeModel):
    """BLABLABLABLA"""

    function: DistributionFunction

    def __init__(self, shape: Optional[float] = None, rate: Optional[float] = None):
        super().__init__(WeibullFunction(shape=shape, rate=rate))


class Gompertz(ParametricLifetimeModel):
    """BLABLABLABLA"""

    function: DistributionFunction

    def __init__(self, shape: Optional[float] = None, rate: Optional[float] = None):
        super().__init__(GompertzFunction(shape=shape, rate=rate))


class Gamma(ParametricLifetimeModel):
    """BLABLABLABLA"""

    function: DistributionFunction

    def __init__(self, shape: Optional[float] = None, rate: Optional[float] = None):
        super().__init__(GammaFunction(shape=shape, rate=rate))


class LogLogistic(ParametricLifetimeModel):
    """BLABLABLABLA"""

    function: DistributionFunction

    def __init__(self, shape: Optional[float] = None, rate: Optional[float] = None):
        super().__init__(LogLogisticFunction(shape=shape, rate=rate))


class PowerGPDistribution(ParametricLifetimeModel):
    """BLABLABLABLA"""

    def __init__(
        self,
        rate: Optional[float] = None,
        shape_rate: Optional[float] = None,
        shape_power: Optional[float] = None,
    ):

        super().__init__(
            GPDistributionFunction(
                PowerShapeFunction(shape_rate=shape_rate, shape_power=shape_power),
                rate,
            ),
        )


class ProportionalHazard(ParametricLifetimeModel):
    """BLABLABLABLA"""

    def __init__(
        self,
        baseline: ParametricLifetimeModel,
        coef: Optional[Sequence[float | None]] = (),
    ):
        super().__init__(ProportionalHazardFunction(baseline.function.copy(), coef))


class AFT(ParametricLifetimeModel):
    """BLABLABLABLA"""

    def __init__(
        self,
        baseline: ParametricLifetimeModel,
        coef: Optional[Sequence[float | None]] = (),
    ):
        super().__init__(AFTFunction(baseline.function.copy(), coef))


class LeftTruncated(ParametricLifetimeModel):
    """BLABLABLABLA"""

    def __init__(self, baseline: ParametricLifetimeModel):
        super().__init__(LeftTruncatedFunction(baseline.function.copy()))


class AgeReplacementModel(ParametricLifetimeModel):
    """BLABLABLABLA"""

    def __init__(self, baseline: ParametricLifetimeModel):
        super().__init__(AgeReplacementFunction(baseline.function.copy()))
