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
from .model import ParametricModel


class Exponential(ParametricModel):
    def __init__(self, *params, **kparams):
        super().__init__(
            ExponentialDistFunction,
            ExponentialDistLikelihood,
            DistOptimizer,
            *params,
            **kparams
        )


class Weibull(ParametricModel):
    def __init__(self, *params, **kparams):
        super().__init__(
            WeibullDistFunction,
            WeibullDistLikelihood,
            DistOptimizer,
            *params,
            **kparams
        )


class Gompertz(ParametricModel):
    def __init__(self, *params, **kparams):
        super().__init__(
            GompertzDistFunction,
            GompertzDistLikelihood,
            GompertzOptimizer,
            *params,
            **kparams
        )
