from .builder import LifetimeModel
from .distributions import (
    DistOptimizer,
    ExponentialFunctions,
    ExponentialLikelihood,
    GammaFunctions,
    GammaLikelihood,
    GompertzFunctions,
    GompertzLikelihood,
    GompertzOptimizer,
    WeibullFunctions,
    WeibullLikelihood,
)


class Exponential(LifetimeModel):
    r"""Exponential parametric lifetime distribution.

    The exponential distribution is a 1-parameter distribution with
    :math:`(\lambda)`. The probability density function is:

    .. math::

        f(t) = \lambda e^{-\lambda t}

    where:

        - :math:`\lambda > 0`, the rate parameter,
        - :math:`t\geq 0`, the operating time, age, cycles, etc.
    """

    def __init__(self, rate: float):
        """Initialize Exponential object

        Args:
            rate (float): _description_
        """
        super().__init__(
            ExponentialFunctions,
            ExponentialLikelihood,
            DistOptimizer,
            rate,
        )


class Weibull(LifetimeModel):
    def __init__(self, c: float, rate: float):
        super().__init__(
            WeibullFunctions,
            WeibullLikelihood,
            DistOptimizer,
            *(c, rate),
        )


class Gompertz(LifetimeModel):
    def __init__(self, c: float, rate: float):
        super().__init__(
            GompertzFunctions,
            GompertzLikelihood,
            GompertzOptimizer,
            *(c, rate),
        )


class Gamma(LifetimeModel):
    def __init__(self, c: float, rate: float):
        super().__init__(
            GammaFunctions,
            GammaLikelihood,
            DistOptimizer,
            *(c, rate),
        )
