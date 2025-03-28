from typing import Callable, Optional, NewType, TypeVarTuple

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import newton
from typing_extensions import override

from relife.distributions.abc import (
    SurvivalABC,
)
from relife.distributions.parameters import Parametric
from relife.distributions.protocols import LifetimeDistribution
from relife.distributions.univariates import UnivariateLifetimeDistribution
from relife.quadratures import gauss_legendre

Z = TypeVarTuple("Z")
T = NewType("T", NDArray[np.floating] | NDArray[np.integer] | float | int)
A0 = NewType("A0", NDArray[np.floating] | NDArray[np.integer] | float | int)
Ar = NewType("Ar", NDArray[np.floating] | NDArray[np.integer] | float | int)


class AgeReplacementDistribution(Parametric, SurvivalABC[Ar, *Z]):
    r"""
    Age replacement core.

    Lifetime core where the asset is replaced at age :math:`a_r`.

    Parameters
    ----------
    baseline : LifetimeDistribution
        Underlying lifetime core.

    Notes
    -----
    This is equivalent to the distribution of :math:`\min(X,a_r)` where
    :math:`X` is a baseline lifetime and ar the age of replacement.
    """

    def __init__(self, baseline: LifetimeDistribution[*Z]):
        super().__init__()
        self.compose_with(baseline=baseline)

    def sf(
        self,
        time: T,
        ar: Ar,
        *z: *Z,
    ) -> NDArray[np.float64]:
        return np.where(time < ar, self.baseline.sf(time, *z), 0.0)

    # TODO : check if correct formula
    def hf(
        self,
        time: T,
        ar: Ar,
        *z: *Z,
    ) -> NDArray[np.float64]:

        return np.where(time < ar, self.baseline.hf(time, *z), 0.0)

    # TODO : check if correct formula
    def chf(
        self,
        time: T,
        ar: Ar,
        *z: *Z,
    ) -> NDArray[np.float64]:
        return np.where(time < ar, self.baseline.chf(time, *z), 0.0)

    @override
    def isf(
        self,
        probability: NDArray[np.float64],
        ar: Ar,
        *z: *Z,
    ) -> NDArray[np.float64]:
        return np.minimum(self.baseline.isf(probability, *z), ar)

    @override
    def ichf(
        self,
        probability: NDArray[np.float64],
        ar: Ar,
        *z: *Z,
    ) -> NDArray[np.float64]:
        return np.minimum(self.baseline.ichf(probability, *z), ar)

    def pdf(
        self,
        time: T,
        ar: Ar,
        *z: *Z,
    ) -> NDArray[np.float64]:
        return np.where(time < ar, self.baseline.pdf(time, *z), 0)

    @override
    def moment(self, n: int, ar: Ar, *z: *Z) -> NDArray[np.float64]:
        return self.ls_integrate(
            lambda x: x**n,
            np.array(0.0),
            np.array(np.inf),
            ar,
            *z,
        )

    @override
    def mean(self, ar: Ar, *z: *Z) -> NDArray[np.float64]:
        return self.moment(1, ar, *z)

    @override
    def var(self, ar: Ar, *z: *Z) -> NDArray[np.float64]:
        return self.moment(2, ar, *z) - self.moment(1, ar, *z) ** 2

    def rvs(
        self,
        ar: Ar,
        *z: *Z,
        size: Optional[int] = 1,
        seed: Optional[int] = None,
    ) -> NDArray[np.float64]:
        return np.minimum(self.baseline.rvs(*z, size=size, seed=seed), ar)

    def ppf(
        self,
        probability: float | NDArray[np.float64],
        ar: Ar,
        *z: *Z,
    ) -> NDArray[np.float64]:
        return self.isf(1 - probability, ar, *z)

    def median(self, ar: Ar, *z: *Z) -> NDArray[np.float64]:
        return self.ppf(np.array(0.5), ar, *z)

    def cdf(
        self,
        time: float | NDArray[np.float64],
        ar: Ar,
        *z: *Z,
    ) -> NDArray[np.float64]:
        return np.where(time < ar, self.baseline.cdf(time, *z), 1.0)

    @override
    def mrl(
        self,
        time: T,
        ar: Ar,
        *z: *Z,
    ) -> NDArray[np.float64]:
        ub = np.array(np.inf)
        mask = time >= ar
        if np.any(mask):
            time, ub = np.broadcast_arrays(time, ub)
            time = np.ma.MaskedArray(time, mask)
            ub = np.ma.MaskedArray(ub, mask)
        mu = self.ls_integrate(lambda x: x - time, time, ub, ar, *z) / self.sf(
            time, ar, *z
        )
        return np.ma.filled(mu, 0)

    @override
    def ls_integrate(
        self,
        func: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        a: NDArray[np.float64],
        b: NDArray[np.float64],
        ar: Ar,
        *z: *Z,
        ndim: int = 0,
        deg: int = 100,
    ) -> NDArray[np.float64]:

        ub = np.minimum(np.inf, ar)
        b = np.minimum(ub, b)
        a, b = np.atleast_2d(*np.broadcast_arrays(a, b))
        z_2d = np.atleast_2d(*z)
        if isinstance(z_2d, np.ndarray):
            z_2d = (z_2d,)

        def integrand(
            x: NDArray[np.float64], *_: *tuple[NDArray[np.float64], ...]
        ) -> NDArray[np.float64]:
            return np.atleast_2d(func(x) * self.baseline.pdf(x, *_))

        w = np.where(b == ar, func(ar) * self.baseline.sf(ar, *z_2d), 0)
        return gauss_legendre(integrand, a, b, *z_2d, ndim=2, deg=deg) + w

    @override
    def freeze_zvariables(self, ar: Ar, *z: *Z) -> LifetimeDistribution[()]:
        return UnivariateLifetimeDistribution(self, *(ar, *z))


class LeftTruncatedDistribution(Parametric, SurvivalABC[A0, *Z]):
    r"""Left truncated core.

    Conditional distribution of the lifetime core for an asset having reach age :math:`a_0`.

    Parameters
    ----------
    baseline : LifetimeDistribution
        Underlying lifetime core.
    """

    def __init__(self, baseline: LifetimeDistribution[*Z]):
        super().__init__()
        self.compose_with(baseline=baseline)

    # TODO : correct formula ? if not, does LeftTruncatedModel have to be LifetimeModel ?
    def sf(
        self,
        time: T,
        a0: A0,
        *z: *Z,
    ) -> NDArray[np.float64]:
        return super().sf(time, a0, *z)

    # TODO : correct formula ? if not, does LeftTruncatedModel have to be LifetimeModel ?
    def pdf(
        self,
        time: T,
        a0: A0,
        *z: *Z,
    ) -> NDArray[np.float64]:
        return super().pdf(time, a0, *z)

    def isf(
        self,
        probability: NDArray[np.float64],
        a0: A0,
        *z: *Z,
    ) -> NDArray[np.float64]:
        cumulative_hazard_rate = -np.log(probability)
        return self.ichf(cumulative_hazard_rate, a0, *z)

    def chf(
        self,
        time: T,
        a0: A0,
        *z: *Z,
    ) -> NDArray[np.float64]:
        return self.baseline.chf(a0 + time, *z) - self.baseline.chf(a0, *z)

    def hf(
        self,
        time: T,
        a0: A0,
        *z: *Z,
    ) -> NDArray[np.float64]:
        return self.baseline.hf(a0 + time, *z)

    def ichf(
        self,
        cumulative_hazard_rate: NDArray[np.float64],
        a0: A0,
        *z: *Z,
    ) -> NDArray[np.float64]:
        return (
            self.baseline.ichf(cumulative_hazard_rate + self.baseline.chf(a0, *z), *z)
            - a0
        )

    @override
    def rvs(
        self,
        a0: A0,
        *z: *Z,
        size: Optional[int] = 1,
        seed: Optional[int] = None,
    ) -> NDArray[np.float64]:
        return super().rvs(*(a0, *z), size=size, seed=seed)

    @override
    def freeze_zvariables(self, a0: A0, *z: *Z) -> LifetimeDistribution[()]:
        return UnivariateLifetimeDistribution(self, *(a0, *z))


class EquilibriumDistribution(Parametric, SurvivalABC[*Z]):
    r"""Equilibrium distribution.

    The equilibirum distribution is the distrbution computed from a lifetime
    core that makes the associated delayed renewal processes stationary.

    Parameters
    ----------
    baseline : LifetimeDistribution
        Underlying lifetime core.

    References
    ----------
    .. [1] Ross, S. M. (1996). Stochastic processes. New York: Wiley.
    """

    def __init__(self, baseline: LifetimeDistribution[*Z]):
        super().__init__()
        self.compose_with(baseline=baseline)

    def sf(self, time: T, *z: *Z) -> NDArray[np.float64]:
        return 1 - self.cdf(time, *z)

    @override
    def cdf(self, time: T, *z: *Z) -> NDArray[np.float64]:
        z_2d = np.atleast_2d(*z)
        time_2d = np.atleast_2d(time)
        if isinstance(z_2d, np.ndarray):
            z_2d = (z_2d,)
        res = gauss_legendre(
            self.baseline.sf, 0, time_2d, *z_2d, ndim=2
        ) / self.baseline.mean(*z_2d)
        # reshape 2d -> final_dim
        ndim = max(map(np.ndim, (time, *z)), default=0)
        if ndim < 2:
            res = np.squeeze(res)
        return res

    def pdf(self, time: T, *z: *Z) -> NDArray[np.float64]:
        # self.baseline.mean can squeeze -> broadcast error (origin : ls_integrate output shape)
        mean = self.baseline.mean(*z)
        sf = self.baseline.sf(time, *z)
        if mean.ndim < sf.ndim:  # if args is empty, sf can have more dim than mean
            if sf.ndim == 1:
                mean = np.reshape(mean, (-1,))
            if sf.ndim == 2:
                mean = np.broadcast_to(mean, (sf.shape[0], -1))
        return sf / mean

    def hf(self, time: T, *z: *Z) -> NDArray[np.float64]:
        return 1 / self.baseline.mrl(time, *z)

    def chf(self, time: T, *z: *Z) -> NDArray[np.float64]:
        return -np.log(self.sf(time, *z))

    @override
    def isf(self, probability: NDArray[np.float64], *z: *Z) -> NDArray[np.float64]:
        return newton(
            lambda x: self.sf(x, *z) - probability,
            self.baseline.isf(probability, *z),
            args=z,
        )

    @override
    def ichf(
        self,
        cumulative_hazard_rate: NDArray[np.float64],
        *z: *Z,
    ) -> NDArray[np.float64]:
        return self.isf(np.exp(-cumulative_hazard_rate), *z)
