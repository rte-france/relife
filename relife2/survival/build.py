from .distribution import (
    DistOptimizer,
    ExponentialDistFunction,
    ExponentialDistLikelihood,
    GompertzDistFunction,
    GompertzDistLikelihood,
    GompertzOptimizer,
    WeibullDistFunction,
    WeibullDistLikelihood,
)
from .model import dist


def exponential(*params, **kparams):
    """Exponential distribution

    Args:
        *params (float): model parameter
        **kwargs (float): model parameter (rate)

    Examples:
        >>> exp_dist = exponential(rate = 0.00795203)
    """
    return dist(
        ExponentialDistFunction,
        ExponentialDistLikelihood,
        DistOptimizer,
    )(*params, **kparams)


def weibull(*params, **kparams):
    """Weilbull distribution

    Args:
        *params (float): model parameters
        **kwargs (float): model parameter (c and rate)

    """
    return dist(
        WeibullDistFunction,
        WeibullDistLikelihood,
        DistOptimizer,
    )(*params, **kparams)


def gompertz(*params, **kparams):
    """Gompertz distribution

    Args:
        *params (float): model parameter
        **kwargs (float): model parameter (c and rate)

    """
    return dist(
        GompertzDistFunction,
        GompertzDistLikelihood,
        GompertzOptimizer,
    )(*params, **kparams)
