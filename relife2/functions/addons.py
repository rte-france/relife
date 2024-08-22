from typing import Any

import numpy as np
from scipy.optimize import Bounds

from relife2.functions.core import ParametricLifetimeFunction


class AgeReplacementFunction(ParametricLifetimeFunction):

    def __init__(self, baseline: ParametricLifetimeFunction):
        super().__init__()
        self.add_functions(baseline=baseline)
        self.new_args(ar=None)

    @property
    def support_upper_bound(self):
        return np.minimum(self.ar, self.baseline.support_upper_bound)

    @property
    def support_lower_bound(self):
        return None

    @property
    def params_bounds(self) -> Bounds:
        return None

    def init_params(self, *args: Any) -> np.ndarray:
        pass

    def sf(self, time: np.ndarray) -> np.ndarray:
        return np.where(time < self.ar, self.baseline.sf(time), 0)

    def pdf(self, time: np.ndarray) -> np.ndarray:
        return np.where(time < self.ar, self.baseline.pdf(time), 0)

    def isf(self, probability: np.ndarray) -> np.ndarray:
        return np.minimum(self.baseline.isf(probability), self.ar)

    # def ls_integrate(
    #     self,
    #     func: Callable,
    #     a: np.ndarray,
    #     b: np.ndarray,
    #     ar: np.ndarray,
    #     *args: np.ndarray,
    #     ndim: int = 0,
    #     deg: int = 100
    # ) -> np.ndarray:
    #     ub = self.support_upper_bound(ar, *args)
    #     b = np.minimum(ub, b)
    #     f = lambda x, *args: func(x) * self.baseline.pdf(x, *args)
    #     w = np.where(b == ar, func(ar) * self.baseline.sf(ar, *args), 0)
    #     return gauss_legendre(f, a, b, *args, ndim=ndim, deg=deg) + w


class LeftTruncatedFunction(ParametricLifetimeFunction):
    def __init__(self, baseline: ParametricLifetimeFunction):
        super().__init__()
        self.add_functions(baseline=baseline)
        self.new_args(ar=None)

    @property
    def support_upper_bound(self):
        return None

    @property
    def support_lower_bound(self):
        return None

    def init_params(self, *args: Any) -> np.ndarray:
        pass

    @property
    def params_bounds(self) -> Bounds:
        return None

    def chf(self, time: np.ndarray) -> np.ndarray:
        return self.baseline.chf(self.a0 + time) - self.baseline.chf(self.a0)

    def hf(self, time: np.ndarray) -> np.ndarray:
        return self.baseline.hf(self.a0 + time)

    def ichf(self, cumulative_hazard_rate: np.ndarray) -> np.ndarray:
        return (
            self.baseline.ichf(cumulative_hazard_rate + self.baseline.chf(self.a0))
            - self.a0
        )
