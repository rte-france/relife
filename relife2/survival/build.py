from .distribution import (
    DistOptimizer,
    ExponentialDistFunction,
    ExponentialDistLikelihood,
    GompertzDistFunction,
    GompertzDistLikelihood,
    WeibullDistFunction,
    WeibullDistLikelihood,
)
from .model import dist


def exponential(*params, **kparams):
    """Exponential distribution

    Args:
        *params (float): model parameter (rate)
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
        *params (float): model parameter (rate)
        **kwargs (float): model parameter (rate)

    """
    return dist(
        WeibullDistFunction,
        WeibullDistLikelihood,
        DistOptimizer,
    )(*params, **kparams)


def gompertz(*params, **kparams):
    """Gompertz distribution

    Args:
        *params (float): model parameter (rate)
        **kwargs (float): model parameter (rate)

    """
    return dist(
        GompertzDistFunction,
        GompertzDistLikelihood,
        DistOptimizer,
    )(*params, **kparams)
