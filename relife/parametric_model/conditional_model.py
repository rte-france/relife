from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional, TypeVarTuple

import numpy as np
from numpy.typing import NDArray
from typing_extensions import override

from relife.model import Parametric, BaseLifetimeModel, FrozenLifetimeModel
from relife.quadratures import gauss_legendre

if TYPE_CHECKING:
    from relife.model import BaseLifetimeModel

Args = TypeVarTuple("Args")


class AgeReplacementModel(
    Parametric, BaseLifetimeModel[float | NDArray[np.float64], *Args]
):
    r"""
    Age replacement core.

    Lifetime core where the asset is replaced at age :math:`a_r`.

    Parameters
    ----------
    baseline : BaseLifetimeModel
        Underlying lifetime core.

    Notes
    -----
    This is equivalent to the distribution of :math:`\min(X,a_r)` where
    :math:`X` is a baseline lifetime and ar the age of replacement.
    """

    def __init__(self, baseline: BaseLifetimeModel[*Args]):
        super().__init__()
        self.compose_with(baseline=baseline)

    def sf(
        self,
        time: float | NDArray[np.float64],
        ar: float | NDArray[np.float64],
        *args: *Args,
    ) -> NDArray[np.float64]:
        return np.where(time < ar, self.baseline.sf(time, *args), 0.0)

    # TODO : check if correct formula
    def hf(
        self,
        time: float | NDArray[np.float64],
        ar: float | NDArray[np.float64],
        *args: *Args,
    ) -> NDArray[np.float64]:

        return np.where(time < ar, self.baseline.hf(time, *args), 0.0)

    # TODO : check if correct formula
    def chf(
        self,
        time: float | NDArray[np.float64],
        ar: float | NDArray[np.float64],
        *args: *Args,
    ) -> NDArray[np.float64]:
        return np.where(time < ar, self.baseline.chf(time, *args), 0.0)

    @override
    def isf(
        self,
        probability: NDArray[np.float64],
        ar: float | NDArray[np.float64],
        *args: *Args,
    ) -> NDArray[np.float64]:
        return np.minimum(self.baseline.isf(probability, *args), ar)

    @override
    def ichf(
        self,
        probability: NDArray[np.float64],
        ar: float | NDArray[np.float64],
        *args: *Args,
    ) -> NDArray[np.float64]:
        return np.minimum(self.baseline.ichf(probability, *args), ar)

    def pdf(
        self,
        time: float | NDArray[np.float64],
        ar: float | NDArray[np.float64],
        *args: *Args,
    ) -> NDArray[np.float64]:
        return np.where(time < ar, self.baseline.pdf(time, *args), 0)

    @override
    def moment(
        self, n: int, ar: float | NDArray[np.float64], *args: *Args
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
        self, ar: float | NDArray[np.float64], *args: *Args
    ) -> NDArray[np.float64]:
        return self.moment(1, ar, *args)

    @override
    def var(self, ar: float | NDArray[np.float64], *args: *Args) -> NDArray[np.float64]:
        return self.moment(2, ar, *args) - self.moment(1, ar, *args) ** 2

    def rvs(
        self,
        ar: float | NDArray[np.float64],
        *args: *Args,
        size: Optional[int] = 1,
        seed: Optional[int] = None,
    ) -> NDArray[np.float64]:
        return np.minimum(self.baseline.rvs(*args, size=size, seed=seed), ar)

    def ppf(
        self,
        probability: float | NDArray[np.float64],
        ar: float | NDArray[np.float64],
        *args: *Args,
    ) -> NDArray[np.float64]:
        return self.isf(1 - probability, ar, *args)

    def median(
        self, ar: float | NDArray[np.float64], *args: *Args
    ) -> NDArray[np.float64]:
        return self.ppf(np.array(0.5), ar, *args)

    def cdf(
        self,
        time: float | NDArray[np.float64],
        ar: float | NDArray[np.float64],
        *args: *Args,
    ) -> NDArray[np.float64]:
        return np.where(time < ar, self.baseline.cdf(time, *args), 1.0)

    @override
    def mrl(
        self,
        time: float | NDArray[np.float64],
        ar: float | NDArray[np.float64],
        *args: *Args,
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
        ar: float | NDArray[np.float64],
        *args: *Args,
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

    @override
    def freeze(
        self, ar: float | NDArray[np.float64], *args: *Args
    ) -> FrozenLifetimeModel[float | NDArray[np.float64], *Args]:
        return FrozenLifetimeModel(self, *(ar, *args))


class LeftTruncatedModel(
    Parametric, BaseLifetimeModel[float | NDArray[np.float64], *Args]
):
    r"""Left truncated core.

    Conditional distribution of the lifetime core for an asset having reach age :math:`a_0`.

    Parameters
    ----------
    baseline : BaseLifetimeModel
        Underlying lifetime core.
    """

    def __init__(self, baseline: BaseLifetimeModel[*Args]):
        super().__init__()
        self.compose_with(baseline=baseline)

    # TODO : correct formula ? if not, does LeftTruncatedModel have to be LifetimeModel ?
    def sf(
        self,
        time: float | NDArray[np.float64],
        a0: float | NDArray[np.float64],
        *args: *Args,
    ) -> NDArray[np.float64]:
        return super().sf(time, a0, *args)

    # TODO : correct formula ? if not, does LeftTruncatedModel have to be LifetimeModel ?
    def pdf(
        self,
        time: float | NDArray[np.float64],
        a0: float | NDArray[np.float64],
        *args: *Args,
    ) -> NDArray[np.float64]:
        return super().pdf(time, a0, *args)

    def isf(
        self,
        probability: NDArray[np.float64],
        a0: float | NDArray[np.float64],
        *args: *Args,
    ) -> NDArray[np.float64]:
        cumulative_hazard_rate = -np.log(probability)
        return self.ichf(cumulative_hazard_rate, a0, *args)

    def chf(
        self,
        time: float | NDArray[np.float64],
        a0: float | NDArray[np.float64],
        *args: *Args,
    ) -> NDArray[np.float64]:
        return self.baseline.chf(a0 + time, *args) - self.baseline.chf(a0, *args)

    def hf(
        self,
        time: float | NDArray[np.float64],
        a0: float | NDArray[np.float64],
        *args: *Args,
    ) -> NDArray[np.float64]:
        return self.baseline.hf(a0 + time, *args)

    def ichf(
        self,
        cumulative_hazard_rate: NDArray[np.float64],
        a0: float | NDArray[np.float64],
        *args: *Args,
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
        a0: float | NDArray[np.float64],
        *args: *Args,
        size: Optional[int] = 1,
        seed: Optional[int] = None,
    ) -> NDArray[np.float64]:
        return super().rvs(*(a0, *args), size=size, seed=seed)

    @override
    def freeze(
        self, a0: float | NDArray[np.float64], *args: *Args
    ) -> FrozenLifetimeModel[*Args]:
        return FrozenLifetimeModel(self, *(a0, *args))
