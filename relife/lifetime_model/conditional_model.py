from typing import Callable, Optional, TypeVarTuple

import numpy as np
from numpy.typing import NDArray
from typing_extensions import override

from relife.quadrature import legendre_quadrature

from ._base import ParametricLifetimeModel
from .frozen_model import FrozenParametricLifetimeModel
from .._args import get_nb_assets

Args = TypeVarTuple("Args")


class AgeReplacementModel(ParametricLifetimeModel[float | NDArray[np.float64], *Args]):
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

    def __init__(self, baseline: ParametricLifetimeModel[*Args]):
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
        cumulative_hazard_rate: NDArray[np.float64],
        ar: float | NDArray[np.float64],
        *args: *Args,
    ) -> NDArray[np.float64]:
        return np.minimum(self.baseline.ichf(cumulative_hazard_rate, *args), ar)

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
        func: Callable[[float | NDArray[np.float64]], NDArray[np.float64]],
        a: float | NDArray[np.float64],
        b: float | NDArray[np.float64],
        ar: float | NDArray[np.float64],
        *args: *Args,
        ndim: int = 0,
        deg: int = 100,
    ) -> NDArray[np.float64]:

        arr_a = np.asarray(a)  #  (m, n) or (n,)
        arr_b = np.asarray(b)  #  (m, n) or (n,)
        arr_a, arr_b = np.broadcast_arrays(arr_a, arr_b)

        frozen_model = self.freeze(ar, *args)
        if get_nb_assets(*frozen_model.args) > 1:
            if arr_a.ndim != 2 and arr_b.ndim != 0:
                raise ValueError

        def integrand(x: float | NDArray[np.float64]) -> NDArray[np.float64]:
            return func(x) * frozen_model.pdf(x)

        arr_ar = frozen_model.args[0] # (m, 1)
        if arr_a.ndim < 2 and arr_b.ndim < 2:
            arr_a = arr_a.reshape(-1, 1)  #  (m, n)
            arr_b = arr_b.reshape(-1, 1) # (m, n)
        if arr_b.shape[0] != arr_ar.shape[0]:
            raise ValueError
        arr_b = np.minimum(arr_ar, arr_b)
        arr_b, arr_ar = np.broadcast_arrays(arr_b, arr_ar)  # same shape (m, n)

        arr_a = arr_a.flatten()  # (m*n,)
        arr_b = arr_b.flatten()  # (m*n,)
        flat_ar = arr_ar.flatten()  # (m*n,)

        assert arr_a.shape == arr_b.shape == arr_ar.shape

        integration = np.empty_like(arr_b)  # (m*n,) or # (n,)

        is_ar = arr_b == flat_ar

        if arr_a[is_ar].size != 0:
            integration[is_ar] = legendre_quadrature(
                integrand, arr_a[is_ar].copy(), arr_b[is_ar].copy(), deg=deg
            ) + func(flat_ar[is_ar]) * frozen_model.sf(flat_ar[is_ar])
        if arr_a[~is_ar].size != 0:
            integration[~is_ar] = legendre_quadrature(
                integrand, arr_a[~is_ar].copy(), arr_b[~is_ar].copy(), deg=deg
            )

        shape = np.asarray(a).shape
        if np.asarray(b).ndim > len(shape):
            shape = np.asarray(b).shape

        return integration.reshape(shape)

    @override
    def freeze(
        self, ar: float | NDArray[np.float64], *args: *Args
    ) -> FrozenParametricLifetimeModel:
        return FrozenParametricLifetimeModel(self, *(ar, *args))


class LeftTruncatedModel(ParametricLifetimeModel[float | NDArray[np.float64], *Args]):
    r"""Left truncated core.

    Conditional distribution of the lifetime core for an asset having reach age :math:`a_0`.

    Parameters
    ----------
    baseline : BaseLifetimeModel
        Underlying lifetime core.
    """

    def __init__(self, baseline: ParametricLifetimeModel[*Args]):
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
    ) -> FrozenParametricLifetimeModel:
        return FrozenParametricLifetimeModel(self, *(a0, *args))
