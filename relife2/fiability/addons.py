import numpy as np
from numpy.typing import NDArray

from relife2.core import LifetimeModel


class AgeReplacementModel(
    LifetimeModel[*tuple[NDArray[np.float64], tuple[NDArray[np.float64], ...]]]
):

    def __init__(self, baseline: LifetimeModel[tuple[NDArray[np.float64], ...]]):
        super().__init__()
        self.baseline = baseline

    def sf(
        self,
        time: NDArray[np.float64],
        ar: NDArray[np.float64],
        *args: tuple[NDArray[np.float64], ...],
    ) -> NDArray[np.float64]:
        return np.where(time < ar, self.baseline.sf(time, *args), 0)

    def isf(
        self,
        probability: NDArray[np.float64],
        ar: NDArray[np.float64],
        *args: tuple[NDArray[np.float64], ...],
    ) -> NDArray[np.float64]:
        return np.minimum(self.baseline.isf(probability, *args), ar)

    def pdf(
        self,
        time: NDArray[np.float64],
        ar: NDArray[np.float64],
        *args: tuple[NDArray[np.float64], ...],
    ) -> NDArray[np.float64]:
        return np.where(time < ar, self.baseline.pdf(time, *args), 0)

    def support_upper_bound(self, ar: tuple[NDArray[np.float64], ...]):
        return np.minimum(ar, self.baseline.support_upper_bound)

    def support_lower_bound(self):
        return self.baseline.support_upper_bound()

    # def ls_integrate(
    #     self,
    #     func: Callable,
    #     a: NDArray[np.float64],
    #     b: NDArray[np.float64],
    #     ar: NDArray[np.float64],
    #     *args: NDArray[np.float64],
    #     ndim: int = 0,
    #     deg: int = 100
    # ) -> NDArray[np.float64]:
    #     ub = self.support_upper_bound(ar, *args)
    #     b = np.minimum(ub, b)
    #     f = lambda x, *args: func(x) * self.baseline.pdf(x, *args)
    #     w = np.where(b == ar, func(ar) * self.baseline.sf(ar, *args), 0)
    #     return gauss_legendre(f, a, b, *args, ndim=ndim, deg=deg) + w


class LeftTruncated(
    LifetimeModel[*tuple[NDArray[np.float64], tuple[NDArray[np.float64], ...]]]
):
    def __init__(self, baseline: LifetimeModel[tuple[NDArray[np.float64], ...]]):
        super().__init__()
        self.baseline = baseline

    def isf(
        self,
        probability: NDArray[np.float64],
        a0: NDArray[np.float64],
        *args: tuple[NDArray[np.float64], ...],
    ) -> NDArray[np.float64]:
        cumulative_hazard_rate = -np.log(probability)
        return self.ichf(cumulative_hazard_rate, a0, *args)

    def chf(
        self,
        time: NDArray[np.float64],
        a0: NDArray[np.float64],
        *args: tuple[NDArray[np.float64], ...],
    ) -> NDArray[np.float64]:
        return self.baseline.chf(a0 + time, *args) - self.baseline.chf(a0, *args)

    def hf(
        self,
        time: NDArray[np.float64],
        a0: NDArray[np.float64],
        *args: tuple[NDArray[np.float64], ...],
    ) -> NDArray[np.float64]:
        return self.baseline.hf(a0 + time, *args)

    def ichf(
        self,
        cumulative_hazard_rate: NDArray[np.float64],
        a0: NDArray[np.float64],
        *args: tuple[NDArray[np.float64], ...],
    ) -> NDArray[np.float64]:
        return (
            self.baseline.ichf(
                cumulative_hazard_rate + self.baseline.chf(a0, *args), *args
            )
            - a0
        )

    @property
    def support_upper_bound(self):
        return self.baseline.support_upper_bound

    @property
    def support_lower_bound(self):
        return self.baseline.support_upper_bound
