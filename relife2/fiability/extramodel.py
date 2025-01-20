from typing import Callable, Optional

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import newton
from typing_extensions import override

from relife2.fiability.model import LifetimeModel
from relife2.utils.integration import gauss_legendre
from relife2.utils.types import ModelArgs


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

    # TODO : correct formula ? if not, does AgeReplacementModel have to be LifetimeModel ?
    def hf(
        self, time: NDArray[np.float64], ar: NDArray[np.float64], *args: *ModelArgs
    ) -> NDArray[np.float64]:
        return self.baseline.hf(time, *args)

    # TODO : correct formula ? if not, does AgeReplacementModel have to be LifetimeModel ?
    def chf(
        self, time: NDArray[np.float64], ar: NDArray[np.float64], *args: *ModelArgs
    ) -> NDArray[np.float64]:
        return self.baseline.chf(time, *args)

    @override
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

    @override
    def moment(
        self, n: int, ar: NDArray[np.float64], *args: *ModelArgs
    ) -> NDArray[np.float64]:
        return self.ls_integrate(
            lambda x: x**n,
            np.array(0.0),
            np.array(np.inf),
            ar,
            *args,
        )

    @override
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

    @override
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
        return gauss_legendre(integrand, a, b, *args_2d, ndim=2, deg=deg) + w


class LeftTruncatedModel(LifetimeModel[NDArray[np.float64], *ModelArgs]):
    def __init__(self, baseline: LifetimeModel[*ModelArgs]):
        super().__init__()
        self.baseline = baseline

    # TODO : correct formula ? if not, does LeftTruncatedModel have to be LifetimeModel ?
    def sf(
        self, time: NDArray[np.float64], a0: NDArray[np.float64], *args: *ModelArgs
    ) -> NDArray[np.float64]:
        return super().sf(time, a0, *args)

    # TODO : correct formula ? if not, does LeftTruncatedModel have to be LifetimeModel ?
    def pdf(
        self, time: NDArray[np.float64], a0: NDArray[np.float64], *args: *ModelArgs
    ) -> NDArray[np.float64]:
        return super().pdf(time, a0, *args)

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


class EquilibriumDistribution(LifetimeModel[*ModelArgs]):
    def __init__(self, baseline: LifetimeModel[*ModelArgs]):
        super().__init__()
        self.baseline = baseline

    def sf(self, time: NDArray[np.float64], *args: *ModelArgs) -> NDArray[np.float64]:
        return 1 - self.cdf(time, *args)

    @override
    def cdf(self, time: NDArray[np.float64], *args: *ModelArgs) -> NDArray[np.float64]:
        args_2d = np.atleast_2d(*args)
        time_2d = np.atleast_2d(time)
        if isinstance(args_2d, np.ndarray):
            args_2d = (args_2d,)
        res = gauss_legendre(
            self.baseline.sf, 0, time_2d, *args_2d, ndim=2
        ) / self.baseline.mean(*args_2d)
        # reshape 2d -> final_dim
        ndim = max(map(np.ndim, (time, *args)), default=0)
        if ndim < 2:
            res = np.squeeze(res)
        return res

    def pdf(self, time: NDArray[np.float64], *args: *ModelArgs) -> NDArray[np.float64]:
        # self.baseline.mean can squeeze -> broadcast error (origin : ls_integrate output shape)
        mean = self.baseline.mean(*args)
        sf = self.baseline.sf(time, *args)
        if mean.ndim < sf.ndim:  # if args is empty, sf can have more dim than mean
            if sf.ndim == 1:
                mean = np.reshape(mean, (-1,))
            if sf.ndim == 2:
                mean = np.broadcast_to(mean, (sf.shape[0], -1))
        return sf / mean

    def hf(self, time: NDArray[np.float64], *args: *ModelArgs) -> NDArray[np.float64]:
        return 1 / self.baseline.mrl(time, *args)

    def chf(self, time: NDArray[np.float64], *args: *ModelArgs) -> NDArray[np.float64]:
        return -np.log(self.sf(time, *args))

    @override
    def isf(
        self, probability: NDArray[np.float64], *args: *ModelArgs
    ) -> NDArray[np.float64]:
        return newton(
            lambda x: self.sf(x, *args) - probability,
            self.baseline.isf(probability, *args),
            args=args,
        )

    @override
    def ichf(
        self, cumulative_hazard_rate: NDArray[np.float64], *args: *ModelArgs
    ) -> NDArray[np.float64]:
        return self.isf(np.exp(-cumulative_hazard_rate), *args)
