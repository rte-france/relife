import numpy as np

from .backbone import ParametricModel
from .distributions import (
    DistOptimizer,
    ExponentialFunctions,
    ExponentialLikelihood,
    GompertzFunctions,
    GompertzLikelihood,
    GompertzOptimizer,
    WeibullFunctions,
    WeibullLikelihood,
)


class Exponential(ParametricModel):
    def __init__(self, *params: np.ndarray, **kparams: float):
        super().__init__(
            ExponentialFunctions,
            ExponentialLikelihood,
            DistOptimizer,
            *params,
            **kparams
        )


class Weibull(ParametricModel):
    def __init__(self, *params: np.ndarray, **kparams: float):
        super().__init__(
            WeibullFunctions,
            WeibullLikelihood,
            DistOptimizer,
            *params,
            **kparams
        )


class Gompertz(ParametricModel):
    def __init__(self, *params: np.ndarray, **kparams: float):
        super().__init__(
            GompertzFunctions,
            GompertzLikelihood,
            GompertzOptimizer,
            *params,
            **kparams
        )
