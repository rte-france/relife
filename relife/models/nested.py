from typing import Callable, Optional, NewType, Union

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import newton
from typing_extensions import override

from relife.core import LifetimeModel
from relife.quadratures import gauss_legendre

Time = NewType(
    "Time",
    Union[NDArray[np.floating], NDArray[np.integer], float, int],
)

Ar = NewType(
    "Ar",
    Union[NDArray[np.floating], NDArray[np.integer], float, int],
)

NumericalArrayLike = NewType(
    "NumericalArrayLike",
    Union[NDArray[np.floating], NDArray[np.integer], float, int],
)


class AgeReplacementModel(LifetimeModel[Ar, *tuple[NumericalArrayLike, ...]]):
    r"""
    Age replacement core.

    Lifetime core where the asset is replaced at age :math:`a_r`.

    Parameters
    ----------
    baseline : LifetimeModel
        Underlying lifetime core.

    Notes
    -----
    This is equivalent to the distribution of :math:`\min(X,a_r)` where
    :math:`X` is a baseline lifetime and ar the age of replacement.
    """

    def __init__(self, baseline: LifetimeModel[*tuple[NumericalArrayLike, ...]]):
        self.baseline = baseline

    def sf(
        self,
        time: Time,
        ar: Ar,
        *args: *tuple[NumericalArrayLike, ...],
    ) -> NDArray[np.float64]:
        return np.where(time < ar, self.baseline.sf(time, *args), 0.0)

    # TODO : check if correct formula
    def hf(
        self,
        time: Time,
        ar: Ar,
        *args: *tuple[NumericalArrayLike, ...],
    ) -> NDArray[np.float64]:

        return np.where(time < ar, self.baseline.hf(time, *args), 0.0)

    # TODO : check if correct formula
    def chf(
        self,
        time: Time,
        ar: Ar,
        *args: *tuple[NumericalArrayLike, ...],
    ) -> NDArray[np.float64]:
        return np.where(time < ar, self.baseline.chf(time, *args), 0.0)

    @override
    def isf(
        self,
        probability: NDArray[np.float64],
        ar: Ar,
        *args: *tuple[NumericalArrayLike, ...],
    ) -> NDArray[np.float64]:
        return np.minimum(self.baseline.isf(probability, *args), ar)

    @override
    def ichf(
        self,
        probability: NDArray[np.float64],
        ar: Ar,
        *args: *tuple[NumericalArrayLike, ...],
    ) -> NDArray[np.float64]:
        return np.minimum(self.baseline.ichf(probability, *args), ar)

    def pdf(
        self,
        time: Time,
        ar: Ar,
        *args: *tuple[NumericalArrayLike, ...],
    ) -> NDArray[np.float64]:
        return np.where(time < ar, self.baseline.pdf(time, *args), 0)

    @override
    def moment(
        self, n: int, ar: Ar, *args: *tuple[NumericalArrayLike, ...]
    ) -> NDArray[np.float64]:
        return self.ls_integrate(
            lambda x: x**n,
            np.array(0.0),
            np.array(np.inf),
            ar,
            *args,
        )

    @override
    def mean(
        self, ar: Ar, *args: *tuple[NumericalArrayLike, ...]
    ) -> NDArray[np.float64]:
        return self.moment(1, ar, *args)

    @override
    def var(
        self, ar: Ar, *args: *tuple[NumericalArrayLike, ...]
    ) -> NDArray[np.float64]:
        return self.moment(2, ar, *args) - self.moment(1, ar, *args) ** 2

    def rvs(
        self,
        ar: Ar,
        *args: *tuple[NumericalArrayLike, ...],
        size: Optional[int] = 1,
        seed: Optional[int] = None,
    ) -> NDArray[np.float64]:
        return np.minimum(self.baseline.rvs(*args, size=size, seed=seed), ar)

    def ppf(
        self,
        probability: float | NDArray[np.float64],
        ar: Ar,
        *args: *tuple[NumericalArrayLike, ...],
    ) -> NDArray[np.float64]:
        return self.isf(1 - probability, ar, *args)

    def median(
        self, ar: Ar, *args: *tuple[NumericalArrayLike, ...]
    ) -> NDArray[np.float64]:
        return self.ppf(np.array(0.5), ar, *args)

    def cdf(
        self,
        time: float | NDArray[np.float64],
        ar: Ar,
        *args: *tuple[NumericalArrayLike, ...],
    ) -> NDArray[np.float64]:
        return np.where(time < ar, self.baseline.cdf(time, *args), 1.0)

    @override
    def mrl(
        self,
        time: Time,
        ar: Ar,
        *args: *tuple[NumericalArrayLike, ...],
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
        ar: Ar,
        *args: *tuple[NumericalArrayLike, ...],
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
            x: NDArray[np.float64], *_: *tuple[NDArray[np.float64], ...]
        ) -> NDArray[np.float64]:
            return np.atleast_2d(func(x) * self.baseline.pdf(x, *_))

        w = np.where(b == ar, func(ar) * self.baseline.sf(ar, *args_2d), 0)
        return gauss_legendre(integrand, a, b, *args_2d, ndim=2, deg=deg) + w


class FixedAgeReplacementModel(LifetimeModel[*tuple[NumericalArrayLike, ...]]):
    r"""
    Age replacement core.

    Lifetime core where the asset is replaced at age :math:`a_r`.

    Parameters
    ----------
    baseline : LifetimeModel
        Underlying lifetime core.

    Notes
    -----
    This is equivalent to the distribution of :math:`\min(X,a_r)` where
    :math:`X` is a baseline lifetime and ar the age of replacement.
    """

    def __init__(
        self, baseline: LifetimeModel[*tuple[NumericalArrayLike, ...]], ar: Ar
    ):
        self.baseline = baseline
        self.ar = ar

    @override
    def sf(
        self,
        time: Time,
        *args: *tuple[NumericalArrayLike, ...],
    ) -> NDArray[np.float64]:
        return np.where(time < self.ar, self.baseline.sf(time, *args), 0.0)

    @override
    def hf(
        self,
        time: Time,
        *args: *tuple[NumericalArrayLike, ...],
    ) -> NDArray[np.float64]:

        return np.where(time < self.ar, self.baseline.hf(time, *args), 0.0)

    @override
    def chf(
        self,
        time: Time,
        *args: *tuple[NumericalArrayLike, ...],
    ) -> NDArray[np.float64]:
        return np.where(time < self.ar, self.baseline.chf(time, *args), 0.0)

    @override
    def isf(
        self,
        probability: NDArray[np.float64],
        *args: *tuple[NumericalArrayLike, ...],
    ) -> NDArray[np.float64]:
        return np.minimum(self.baseline.isf(probability, *args), self.ar)

    @override
    def ichf(
        self,
        probability: NDArray[np.float64],
        *args: *tuple[NumericalArrayLike, ...],
    ) -> NDArray[np.float64]:
        return np.minimum(self.baseline.ichf(probability, *args), self.ar)

    @override
    def pdf(
        self,
        time: Time,
        *args: *tuple[NumericalArrayLike, ...],
    ) -> NDArray[np.float64]:
        return np.where(time < self.ar, self.baseline.pdf(time, *args), 0)

    @override
    def moment(
        self, n: int, *args: *tuple[NumericalArrayLike, ...]
    ) -> NDArray[np.float64]:
        return self.ls_integrate(
            lambda x: x**n,
            np.array(0.0),
            np.array(np.inf),
            self.ar,
            *args,
        )

    @override
    def mean(self, *args: *tuple[NumericalArrayLike, ...]) -> NDArray[np.float64]:
        return self.moment(1, self.ar, *args)

    @override
    def var(self, *args: *tuple[NumericalArrayLike, ...]) -> NDArray[np.float64]:
        return self.moment(2, self.ar, *args) - self.moment(1, self.ar, *args) ** 2

    @override
    def rvs(
        self,
        *args: *tuple[NumericalArrayLike, ...],
        size: Optional[int] = 1,
        seed: Optional[int] = None,
    ) -> NDArray[np.float64]:
        return np.minimum(self.baseline.rvs(*args, size=size, seed=seed), self.ar)

    @override
    def ppf(
        self,
        probability: float | NDArray[np.float64],
        *args: *tuple[NumericalArrayLike, ...],
    ) -> NDArray[np.float64]:
        return self.isf(1 - probability, self.ar, *args)

    @override
    def median(self, *args: *tuple[NumericalArrayLike, ...]) -> NDArray[np.float64]:
        return self.ppf(np.array(0.5), self.ar, *args)

    @override
    def cdf(
        self,
        time: float | NDArray[np.float64],
        *args: *tuple[NumericalArrayLike, ...],
    ) -> NDArray[np.float64]:
        return np.where(time < self.ar, self.baseline.cdf(time, *args), 1.0)

    @override
    def mrl(
        self,
        time: Time,
        *args: *tuple[NumericalArrayLike, ...],
    ) -> NDArray[np.float64]:
        ub = np.array(np.inf)
        mask = time >= self.ar
        if np.any(mask):
            time, ub = np.broadcast_arrays(time, ub)
            time = np.ma.MaskedArray(time, mask)
            ub = np.ma.MaskedArray(ub, mask)
        mu = self.ls_integrate(lambda x: x - time, time, ub, self.ar, *args) / self.sf(
            time, self.ar, *args
        )
        return np.ma.filled(mu, 0)

    @override
    def ls_integrate(
        self,
        func: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        a: NDArray[np.float64],
        b: NDArray[np.float64],
        *args: *tuple[NumericalArrayLike, ...],
        ndim: int = 0,
        deg: int = 100,
    ) -> NDArray[np.float64]:

        ub = np.minimum(np.inf, self.ar)
        b = np.minimum(ub, b)
        a, b = np.atleast_2d(*np.broadcast_arrays(a, b))
        args_2d = np.atleast_2d(*args)
        if isinstance(args_2d, np.ndarray):
            args_2d = (args_2d,)

        def integrand(
            x: NDArray[np.float64], *_: *tuple[NDArray[np.float64], ...]
        ) -> NDArray[np.float64]:
            return np.atleast_2d(func(x) * self.baseline.pdf(x, *_))

        w = np.where(
            b == self.ar, func(self.ar) * self.baseline.sf(self.ar, *args_2d), 0
        )
        return gauss_legendre(integrand, a, b, *args_2d, ndim=2, deg=deg) + w


A0 = NewType(
    "A0",
    Union[NDArray[np.floating], NDArray[np.integer], float, int],
)


class LeftTruncatedModel(LifetimeModel[A0, *tuple[NumericalArrayLike, ...]]):
    r"""Left truncated core.

    Conditional distribution of the lifetime core for an asset having reach age :math:`a_0`.

    Parameters
    ----------
    baseline : LifetimeModel
        Underlying lifetime core.
    """

    def __init__(self, baseline: LifetimeModel[*tuple[NumericalArrayLike, ...]]):
        self.baseline = baseline

    # TODO : correct formula ? if not, does LeftTruncatedModel have to be LifetimeModel ?
    def sf(
        self,
        time: Time,
        a0: A0,
        *args: *tuple[NumericalArrayLike, ...],
    ) -> NDArray[np.float64]:
        return super().sf(time, a0, *args)

    # TODO : correct formula ? if not, does LeftTruncatedModel have to be LifetimeModel ?
    def pdf(
        self,
        time: Time,
        a0: A0,
        *args: *tuple[NumericalArrayLike, ...],
    ) -> NDArray[np.float64]:
        return super().pdf(time, a0, *args)

    def isf(
        self,
        probability: NDArray[np.float64],
        a0: A0,
        *args: *tuple[NumericalArrayLike, ...],
    ) -> NDArray[np.float64]:
        cumulative_hazard_rate = -np.log(probability)
        return self.ichf(cumulative_hazard_rate, a0, *args)

    def chf(
        self,
        time: Time,
        a0: A0,
        *args: *tuple[NumericalArrayLike, ...],
    ) -> NDArray[np.float64]:
        return self.baseline.chf(a0 + time, *args) - self.baseline.chf(a0, *args)

    def hf(
        self,
        time: Time,
        a0: A0,
        *args: *tuple[NumericalArrayLike, ...],
    ) -> NDArray[np.float64]:
        return self.baseline.hf(a0 + time, *args)

    def ichf(
        self,
        cumulative_hazard_rate: NDArray[np.float64],
        a0: A0,
        *args: *tuple[NumericalArrayLike, ...],
    ) -> NDArray[np.float64]:
        return (
            self.baseline.ichf(
                cumulative_hazard_rate + self.baseline.chf(a0, *args), *args
            )
            - a0
        )

    @override
    def rvs(
        self,
        a0: A0,
        *args: *tuple[NumericalArrayLike, ...],
        size: Optional[int] = 1,
        seed: Optional[int] = None,
    ) -> NDArray[np.float64]:
        return super().rvs(*(a0, *args), size=size, seed=seed)


# either LifetimeModel[()] or LifetimeModel[NDArray, ...]
class FixedLeftTruncatedModel(LifetimeModel[*tuple[NumericalArrayLike, ...]]):
    r"""Left truncated core.

    Conditional distribution of the lifetime core for an asset having reach age :math:`a_0`.

    Parameters
    ----------
    baseline : LifetimeModel
        Underlying lifetime core.
    """

    def __init__(
        self, baseline: LifetimeModel[*tuple[NumericalArrayLike, ...]], a0: A0
    ):
        self.baseline = baseline
        self.a0 = a0

    # TODO : correct formula ? if not, does LeftTruncatedModel have to be LifetimeModel ?
    @override
    def sf(
        self,
        time: Time,
        *args: *tuple[NumericalArrayLike, ...],
    ) -> NDArray[np.float64]:
        return super().sf(time, self.a0, *args)

    # TODO : correct formula ? if not, does LeftTruncatedModel have to be LifetimeModel ?
    @override
    def pdf(
        self,
        time: Time,
        *args: *tuple[NumericalArrayLike, ...],
    ) -> NDArray[np.float64]:
        return super().pdf(time, self.a0, *args)

    @override
    def isf(
        self,
        probability: NDArray[np.float64],
        *args: *tuple[NumericalArrayLike, ...],
    ) -> NDArray[np.float64]:
        cumulative_hazard_rate = -np.log(probability)
        return self.ichf(cumulative_hazard_rate, self.a0, *args)

    @override
    def chf(
        self,
        time: Time,
        *args: *tuple[NumericalArrayLike, ...],
    ) -> NDArray[np.float64]:
        return self.baseline.chf(self.a0 + time, *args) - self.baseline.chf(
            self.a0, *args
        )

    @override
    def hf(
        self,
        time: Time,
        *args: *tuple[NumericalArrayLike, ...],
    ) -> NDArray[np.float64]:
        return self.baseline.hf(self.a0 + time, *args)

    @override
    def ichf(
        self,
        cumulative_hazard_rate: NDArray[np.float64],
        *args: *tuple[NumericalArrayLike, ...],
    ) -> NDArray[np.float64]:
        return (
            self.baseline.ichf(
                cumulative_hazard_rate + self.baseline.chf(self.a0, *args), *args
            )
            - self.a0
        )

    @override
    def rvs(
        self,
        a0: A0,
        *args: *tuple[NumericalArrayLike, ...],
        size: Optional[int] = 1,
        seed: Optional[int] = None,
    ) -> NDArray[np.float64]:
        return super().rvs(*(self.a0, *args), size=size, seed=seed)


def left_truncated(
    baseline: LifetimeModel[*tuple[NumericalArrayLike, ...]],
    a0: Optional[A0] = None,
):
    if a0 is None:
        return LeftTruncatedModel(baseline)
    else:
        return FixedLeftTruncatedModel(baseline, a0)


def replace_at_age(
    baseline: LifetimeModel[*tuple[NumericalArrayLike, ...]],
    ar: Optional[Ar] = None,
):
    if ar is None:
        return AgeReplacementModel(baseline)
    else:
        return FixedAgeReplacementModel(baseline, ar)


class EquilibriumDistribution(LifetimeModel[*tuple[NumericalArrayLike, ...]]):
    r"""Equilibrium distribution.

    The equilibirum distribution is the distrbution computed from a lifetime
    core that makes the associated delayed renewal processes stationary.

    Parameters
    ----------
    baseline : LifetimeModel
        Underlying lifetime core.

    References
    ----------
    .. [1] Ross, S. M. (1996). Stochastic processes. New York: Wiley.
    """

    def __init__(self, baseline: LifetimeModel[*tuple[NumericalArrayLike, ...]]):
        super().__init__()
        self.baseline = baseline

    def sf(
        self, time: Time, *args: *tuple[NumericalArrayLike, ...]
    ) -> NDArray[np.float64]:
        return 1 - self.cdf(time, *args)

    @override
    def cdf(
        self, time: Time, *args: *tuple[NumericalArrayLike, ...]
    ) -> NDArray[np.float64]:
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

    def pdf(
        self, time: Time, *args: *tuple[NumericalArrayLike, ...]
    ) -> NDArray[np.float64]:
        # self.baseline.mean can squeeze -> broadcast error (origin : ls_integrate output shape)
        mean = self.baseline.mean(*args)
        sf = self.baseline.sf(time, *args)
        if mean.ndim < sf.ndim:  # if args is empty, sf can have more dim than mean
            if sf.ndim == 1:
                mean = np.reshape(mean, (-1,))
            if sf.ndim == 2:
                mean = np.broadcast_to(mean, (sf.shape[0], -1))
        return sf / mean

    def hf(
        self, time: Time, *args: *tuple[NumericalArrayLike, ...]
    ) -> NDArray[np.float64]:
        return 1 / self.baseline.mrl(time, *args)

    def chf(
        self, time: Time, *args: *tuple[NumericalArrayLike, ...]
    ) -> NDArray[np.float64]:
        return -np.log(self.sf(time, *args))

    @override
    def isf(
        self, probability: NDArray[np.float64], *args: *tuple[NumericalArrayLike, ...]
    ) -> NDArray[np.float64]:
        return newton(
            lambda x: self.sf(x, *args) - probability,
            self.baseline.isf(probability, *args),
            args=args,
        )

    @override
    def ichf(
        self,
        cumulative_hazard_rate: NDArray[np.float64],
        *args: *tuple[NumericalArrayLike, ...],
    ) -> NDArray[np.float64]:
        return self.isf(np.exp(-cumulative_hazard_rate), *args)
