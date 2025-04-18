import functools
from typing import Callable, Optional, TypeVarTuple, Any, ParamSpec

import numpy as np
from numpy.typing import NDArray
from typing_extensions import override

from relife.quadrature import ls_integrate

from ._base import ParametricLifetimeModel
from .frozen_model import FrozenParametricLifetimeModel

Args = TypeVarTuple("Args")
P = ParamSpec("P")


def _reshape_ar_or_a0(
    method: Callable[P, NDArray[np.float64]],
) -> Callable[[P], NDArray[np.float64]]:
    @functools.wraps(method)
    def wrapper(self, *args: P.args, **kwargs: P.kwargs) -> NDArray[np.float64]:
        match method.__name__:
            case "median"|"var"|"rvs"|"freeze":
                ar_pos = 0
            case "ls_integrate":
                ar_pos = 3
            case _:
                ar_pos = 1
        list_args = list(args)
        ar = np.asarray(list_args[ar_pos], dtype=np.float64)
        ar = np.atleast_2d(ar)
        if ar.ndim > 2 or ar.shape[-1] != 1:
            raise ValueError(f"Incorrect ar shape. Expected shape (), (n,) or (m, 1) only. Got {ar.shape}")
        list_args[ar_pos] = ar
        return method(self, *list_args, **kwargs)
    return wrapper


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

    @_reshape_ar_or_a0
    def sf(
        self,
        time: float | NDArray[np.float64],
        ar: float | NDArray[np.float64],
        *args: *Args,
    ) -> NDArray[np.float64]:
        return np.where(time < ar, self.baseline.sf(time, *args), 0.0)

    @_reshape_ar_or_a0
    def hf(
        self,
        time: float | NDArray[np.float64],
        ar: float | NDArray[np.float64],
        *args: *Args,
    ) -> NDArray[np.float64]:
        ar = np.asarray(ar, dtype=np.float64)
        ar = np.atleast_2d(ar)
        if ar.ndim > 2 or ar.shape[-1] != 1:
            raise ValueError(f"Incorrect ar shape. Expected shape (), (n,) or (m, 1) only. Got {ar.shape}")
        return np.where(time < ar, self.baseline.hf(time, *args), 0.0)

    @_reshape_ar_or_a0
    def chf(
        self,
        time: float | NDArray[np.float64],
        ar: float | NDArray[np.float64],
        *args: *Args,
    ) -> NDArray[np.float64]:
        ar = np.asarray(ar, dtype=np.float64)
        ar = np.atleast_2d(ar)
        if ar.ndim > 2 or ar.shape[-1] != 1:
            raise ValueError(f"Incorrect ar shape. Expected shape (), (n,) or (m, 1) only. Got {ar.shape}")
        return np.where(time < ar, self.baseline.chf(time, *args), 0.0)

    @override
    @_reshape_ar_or_a0
    def isf(
        self,
        probability: NDArray[np.float64],
        ar: float | NDArray[np.float64],
        *args: *Args,
    ) -> NDArray[np.float64]:
        ar = np.asarray(ar, dtype=np.float64)
        ar = np.atleast_2d(ar)
        if ar.ndim > 2 or ar.shape[-1] != 1:
            raise ValueError(f"Incorrect ar shape. Expected shape (), (n,) or (m, 1) only. Got {ar.shape}")
        return np.minimum(self.baseline.isf(probability, *args), ar)

    @override
    @_reshape_ar_or_a0
    def ichf(
        self,
        cumulative_hazard_rate: NDArray[np.float64],
        ar: float | NDArray[np.float64],
        *args: *Args,
    ) -> NDArray[np.float64]:
        ar = np.asarray(ar, dtype=np.float64)
        ar = np.atleast_2d(ar)
        if ar.ndim > 2 or ar.shape[-1] != 1:
            raise ValueError(f"Incorrect ar shape. Expected shape (), (n,) or (m, 1) only. Got {ar.shape}")
        return np.minimum(self.baseline.ichf(cumulative_hazard_rate, *args), ar)

    @_reshape_ar_or_a0
    def pdf(
        self,
        time: float | NDArray[np.float64],
        ar: float | NDArray[np.float64],
        *args: *Args,
    ) -> NDArray[np.float64]:
        return np.where(time < ar, self.baseline.pdf(time, *args), 0)

    @override
    @_reshape_ar_or_a0
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
    @_reshape_ar_or_a0
    def mean(
        self, ar: float | NDArray[np.float64], *args: *Args
    ) -> NDArray[np.float64]:
        return self.moment(1, ar, *args)

    @override
    def var(self, ar: float | NDArray[np.float64], *args: *Args) -> NDArray[np.float64]:
        return self.moment(2, ar, *args) - self.moment(1, ar, *args) ** 2

    @_reshape_ar_or_a0
    def rvs(
        self,
        shape : int|tuple[int, int],
        ar: float | NDArray[np.float64],
        *args: *Args,
        seed: Optional[int] = None,
    ) -> NDArray[np.float64]:
        return np.minimum(self.baseline.rvs(shape, *(ar, *args), seed=seed), ar)

    @_reshape_ar_or_a0
    def ppf(
        self,
        probability: float | NDArray[np.float64],
        ar: float | NDArray[np.float64],
        *args: *Args,
    ) -> NDArray[np.float64]:
        return self.isf(1 - probability, ar, *args)

    @_reshape_ar_or_a0
    def median(
        self, ar: float | NDArray[np.float64], *args: *Args
    ) -> NDArray[np.float64]:
        return self.ppf(np.array(0.5), ar, *args)

    @_reshape_ar_or_a0
    def cdf(
        self,
        time: float | NDArray[np.float64],
        ar: float | NDArray[np.float64],
        *args: *Args,
    ) -> NDArray[np.float64]:
        return np.where(time < ar, self.baseline.cdf(time, *args), 1.0)

    @override
    @_reshape_ar_or_a0
    def mrl(
        self,
        time: float | NDArray[np.float64],
        ar: float | NDArray[np.float64],
        *args: *Args,
    ) -> NDArray[np.float64]:
        ub = np.array(np.inf)
        # ar.shape == (m, 1)
        mask = time >= ar # (m, 1) or (m, n)
        if np.any(mask):
            time, ub = np.broadcast_arrays(time, ub)
            time = np.ma.MaskedArray(time, mask) # (m, 1) or (m, n)
            ub = np.ma.MaskedArray(ub, mask) # (m, 1) or (m, n)
        mu = self.ls_integrate(lambda x: x - time, time, ub, ar, *args, deg=10) / self.sf(
            time, ar, *args
        ) # () or (n,) or (m, n)
        np.ma.filled(mu, 0)
        return np.ma.getdata(mu)

    @override
    @_reshape_ar_or_a0
    def ls_integrate(
        self,
        func: Callable[[float | NDArray[np.float64]], NDArray[np.float64]],
        a: float | NDArray[np.float64],
        b: float | NDArray[np.float64],
        ar: float | NDArray[np.float64],
        *args: *Args,
        deg: int = 10,
    ) -> NDArray[np.float64]:
        return ls_integrate(self.freeze(ar, *args), func, a, b, deg=deg)

    @override
    @_reshape_ar_or_a0
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

    @_reshape_ar_or_a0
    def sf(
        self,
        time: float | NDArray[np.float64],
        a0: float | NDArray[np.float64],
        *args: *Args,
    ) -> NDArray[np.float64]:
        return super().sf(time, a0, *args)

    @_reshape_ar_or_a0
    def pdf(
        self,
        time: float | NDArray[np.float64],
        a0: float | NDArray[np.float64],
        *args: *Args,
    ) -> NDArray[np.float64]:
        return super().pdf(time, a0, *args)

    @_reshape_ar_or_a0
    def isf(
        self,
        probability: NDArray[np.float64],
        a0: float | NDArray[np.float64],
        *args: *Args,
    ) -> NDArray[np.float64]:
        cumulative_hazard_rate = -np.log(probability)
        return self.ichf(cumulative_hazard_rate, a0, *args)

    @_reshape_ar_or_a0
    def chf(
        self,
        time: float | NDArray[np.float64],
        a0: float | NDArray[np.float64],
        *args: *Args,
    ) -> NDArray[np.float64]:
        return self.baseline.chf(a0 + time, *args) - self.baseline.chf(a0, *args)

    @_reshape_ar_or_a0
    def hf(
        self,
        time: float | NDArray[np.float64],
        a0: float | NDArray[np.float64],
        *args: *Args,
    ) -> NDArray[np.float64]:
        return self.baseline.hf(a0 + time, *args)

    @_reshape_ar_or_a0
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
    @_reshape_ar_or_a0
    def rvs(
        self,
        shape : int|tuple[int, int],
        a0: float | NDArray[np.float64],
        *args: *Args,
        seed: Optional[int] = None,
    ) -> NDArray[np.float64]:
        return super().rvs(shape, *(a0, *args), seed=seed)

    @override
    @_reshape_ar_or_a0
    def freeze(
        self, a0: float | NDArray[np.float64], *args: *Args
    ) -> FrozenParametricLifetimeModel:
        return FrozenParametricLifetimeModel(self, *(a0, *args))
