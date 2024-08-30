from typing import Optional

import numpy as np

from .core import (
    LifetimeModel,
)
from .probabilities import default


class AgeReplacementModel:

    def __init__(self, baseline: LifetimeModel):
        super().__init__()
        self.baseline = baseline

    def sf(self, time: np.ndarray, ar: np.ndarray, *args: np.ndarray) -> np.ndarray:
        return np.where(time < ar, self.baseline.sf(time, *args), 0)

    @default
    def isf(
        self, probability: np.ndarray, ar: np.ndarray, *args: np.ndarray
    ) -> np.ndarray:
        """

        Args:
            probability ():
            ar ():
            *args ():

        Returns:

        """
        return np.minimum(self.baseline.isf(probability, *args), ar)

    @default
    def hf(self, time: np.ndarray, ar: np.ndarray, *args: np.ndarray) -> np.ndarray:
        """

        Args:
            time ():
            ar ():
            *args ():

        Returns:

        """

    @default
    def chf(self, time: np.ndarray, ar: np.ndarray, *args: np.ndarray) -> np.ndarray:
        """

        Args:
            time ():
            ar ():
            *args ():

        Returns:

        """

    @default
    def cdf(self, time: np.ndarray, ar: np.ndarray, *args: np.ndarray) -> np.ndarray:
        """

        Args:
            time ():
            ar ():
            *args ():

        Returns:

        """

    @default
    def pdf(self, time: np.ndarray, ar: np.ndarray, *args: np.ndarray) -> np.ndarray:
        """

        Args:
            time ():
            ar ():
            *args ():

        Returns:

        """
        return np.where(time < ar, self.baseline.pdf(time, *args), 0)

    @default
    def ppf(
        self, probability, time: np.ndarray, ar: np.ndarray, *args: np.ndarray
    ) -> np.ndarray:
        """

        Args:
            probability ():
            time ():
            ar ():
            *args ():

        Returns:

        """

    @default
    def mrl(self, time: np.ndarray, ar: np.ndarray, *args: np.ndarray) -> np.ndarray:
        """

        Args:
            time ():
            ar ():
            *args ():

        Returns:

        """

    @default
    def rvs(
        self,
        ar: np.ndarray,
        *args: np.ndarray,
        size: Optional[int] = 1,
        seed: Optional[int] = None,
    ):
        """

        Args:
            ar ():
            *args ():
            size ():
            seed ():

        Returns:

        """

    @default
    def mean(self, ar: np.ndarray, *args: np.ndarray) -> np.ndarray:
        """

        Args:
            ar ():
            *args ():

        Returns:

        """

    @default
    def var(self, ar: np.ndarray, *args: np.ndarray) -> np.ndarray:
        """

        Args:
            ar ():
            *args ():

        Returns:

        """

    @default
    def median(self, ar: np.ndarray, *args: np.ndarray):
        """

        Args:
            ar ():
            *args ():

        Returns:

        """

    def support_upper_bound(self, ar: np.ndarray):
        return np.minimum(ar, self.baseline.support_upper_bound)

    def support_lower_bound(self):
        return self.baseline.support_upper_bound()

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


class LeftTruncated:
    def __init__(self, baseline: LifetimeModel):
        super().__init__()
        self.baseline = baseline

    @default
    def sf(self, time: np.ndarray, a0: np.ndarray, *args: np.ndarray) -> np.ndarray:
        """
        Args:
            time ():
            a0 ():
            *args ():

        Returns:
        """

    def isf(
        self, probability: np.ndarray, a0: np.ndarray, *args: np.ndarray
    ) -> np.ndarray:
        """
        Args:
            probability ():
            a0 ():
            *args ():

        Returns:
        """
        cumulative_hazard_rate = -np.log(probability)
        return self.ichf(cumulative_hazard_rate, a0, *args)

    def chf(self, time: np.ndarray, a0: np.ndarray, *args: np.ndarray) -> np.ndarray:
        """
        Args:
            time ():
            a0 ():
            *args ():

        Returns:

        """
        return self.baseline.chf(a0 + time, *args) - self.baseline.chf(a0, *args)

    def hf(self, time: np.ndarray, a0: np.ndarray, *args: np.ndarray) -> np.ndarray:
        """
        Args:
            time ():
            a0 ():
            *args ():

        Returns:

        """
        return self.baseline.hf(a0 + time, *args)

    def ichf(
        self, cumulative_hazard_rate: np.ndarray, a0: np.ndarray, *args: np.ndarray
    ) -> np.ndarray:
        """

        Args:
            cumulative_hazard_rate ():
            a0 ():
            *args ():

        Returns:

        """
        return (
            self.baseline.ichf(cumulative_hazard_rate + self.baseline.chf(a0, *args))
            - a0
        )

    @default
    def cdf(self, time: np.ndarray, a0: np.ndarray, *args: np.ndarray) -> np.ndarray:
        """

        Args:
            time ():
            a0 ():
            *args ():

        Returns:

        """

    @default
    def pdf(self, time: np.ndarray, a0: np.ndarray, *args: np.ndarray) -> np.ndarray:
        """

        Args:
            time ():
            a0 ():
            *args ():

        Returns:

        """
        return np.where(time < a0, self.baseline.pdf(time, *args), 0)

    @default
    def ppf(
        self, probability, time: np.ndarray, a0: np.ndarray, *args: np.ndarray
    ) -> np.ndarray:
        """

        Args:
            probability ():
            time ():
            a0 ():
            *args ():

        Returns:

        """

    @default
    def mrl(self, time: np.ndarray, a0: np.ndarray, *args: np.ndarray) -> np.ndarray:
        """

        Args:
            time ():
            a0 ():
            *args ():

        Returns:

        """

    @default
    def rvs(
        self,
        a0: np.ndarray,
        *args: np.ndarray,
        size: Optional[int] = 1,
        seed: Optional[int] = None,
    ):
        """

        Args:
            a0 ():
            *args ():
            size ():
            seed ():

        Returns:

        """

    @default
    def mean(self, a0: np.ndarray, *args: np.ndarray) -> np.ndarray:
        """

        Args:
            a0 ():
            *args ():

        Returns:

        """

    @default
    def var(self, a0: np.ndarray, *args: np.ndarray) -> np.ndarray:
        """

        Args:
            a0 ():
            *args ():

        Returns:

        """

    @default
    def median(self, a0: np.ndarray, *args: np.ndarray):
        """
        Args:
            a0 ():
            *args ():

        Returns:

        """

    @property
    def support_upper_bound(self):
        return self.baseline.support_upper_bound

    @property
    def support_lower_bound(self):
        return self.baseline.support_upper_bound
