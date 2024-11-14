from typing import Callable, Optional

import numpy as np
from numpy.typing import NDArray

from relife2.fiability.model import LifetimeModel
from relife2.maths.integration import gauss_legendre
from relife2.typing import ModelArgs


class AgeReplacementModel(LifetimeModel[NDArray[np.float64], *ModelArgs]):

    def __init__(self, baseline: LifetimeModel[*ModelArgs]):
        super().__init__()
        self.baseline = baseline

    def sf(
        self,
        time: NDArray[np.float64],
        ar: NDArray[np.float64],
        *args: *ModelArgs,
    ) -> NDArray[np.float64]:
        return np.where(time < ar, self.baseline.sf(time, *args), 0)

    def isf(
        self,
        probability: NDArray[np.float64],
        ar: NDArray[np.float64],
        *args: *ModelArgs,
    ) -> NDArray[np.float64]:
        return np.minimum(self.baseline.isf(probability, *args), ar)

    def pdf(
        self,
        time: NDArray[np.float64],
        ar: NDArray[np.float64],
        *args: *ModelArgs,
    ) -> NDArray[np.float64]:
        return np.where(time < ar, self.baseline.pdf(time, *args), 0)

    def moment(
        self, n: int, ar: NDArray[np.float64], *args: *ModelArgs
    ) -> NDArray[np.float64]:
        return self.ls_integrate(
            lambda x: x**n,
            np.array(0.0),
            ar,
            *args,
        )

    def mrl(
        self, time: NDArray[np.float64], ar: NDArray[np.float64], *args: *ModelArgs
    ) -> NDArray[np.float64]:

        ub = np.array(np.inf)
        mask = time >= ar
        if np.any(mask):
            time, ub = np.broadcast_arrays(time, ub)
            time = np.ma.MaskedArray(time, mask)
            ub = np.ma.MaskedArray(ub, mask)
        mu = self.ls_integrate(lambda x: x - time, time, ub, ar, *args) / self.sf(
            time, ar, *args
        )
        return np.ma.filled(mu, 0)

    def ls_integrate(
        self,
        func: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        a: NDArray[np.float64],
        b: NDArray[np.float64],
        ar: NDArray[np.float64],
        *args: *ModelArgs,
        ndim: int = 0,
        deg: int = 100,
    ) -> NDArray[np.float64]:

        ub = np.minimum(np.inf, ar)
        b = np.minimum(ub, b)
        a, b = np.atleast_2d(*np.broadcast_arrays(a, b))
        args_2d = np.atleast_2d(*args)
        if isinstance(args_2d, np.ndarray):
            args_2d = (args_2d,)

        def integrand(
            x: NDArray[np.float64], *_: * tuple[NDArray[np.float64], ...]
        ) -> NDArray[np.float64]:
            return np.atleast_2d(func(x) * self.baseline.pdf(x, *_))

        w = np.where(b == ar, func(ar) * self.baseline.sf(ar, *args_2d), 0)
        return np.squeeze(
            gauss_legendre(integrand, a, b, *args_2d, ndim=2, deg=deg) + w
        )


class LeftTruncatedModel(LifetimeModel[NDArray[np.float64], *ModelArgs]):
    def __init__(self, baseline: LifetimeModel[*ModelArgs]):
        super().__init__()
        self.baseline = baseline

    def isf(
        self,
        probability: NDArray[np.float64],
        a0: NDArray[np.float64],
        *args: *ModelArgs,
    ) -> NDArray[np.float64]:
        cumulative_hazard_rate = -np.log(probability)
        return self.ichf(cumulative_hazard_rate, a0, *args)

    def chf(
        self,
        time: NDArray[np.float64],
        a0: NDArray[np.float64],
        *args: *ModelArgs,
    ) -> NDArray[np.float64]:
        return self.baseline.chf(a0 + time, *args) - self.baseline.chf(a0, *args)

    def hf(
        self,
        time: NDArray[np.float64],
        a0: NDArray[np.float64],
        *args: *ModelArgs,
    ) -> NDArray[np.float64]:
        return self.baseline.hf(a0 + time, *args)

    def ichf(
        self,
        cumulative_hazard_rate: NDArray[np.float64],
        a0: NDArray[np.float64],
        *args: *ModelArgs,
    ) -> NDArray[np.float64]:
        return (
            self.baseline.ichf(
                cumulative_hazard_rate + self.baseline.chf(a0, *args), *args
            )
            - a0
        )

    def rvs(
        self,
        a0: NDArray[np.float64],
        *args: *ModelArgs,
        size: Optional[int] = 1,
        seed: Optional[int] = None,
    ) -> NDArray[np.float64]:
        return self.baseline.rvs(*(a0, *args), size=size, seed=seed) + a0

    # @property
    # def support_upper_bound(self):
    #     return self.baseline.support_upper_bound
    #
    # @property
    # def support_lower_bound(self):
    #     return self.baseline.support_upper_bound
