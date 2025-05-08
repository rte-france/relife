from typing import Callable, Optional, ParamSpec, Sequence, TypeVarTuple

import numpy as np
from numpy.typing import NDArray, DTypeLike
from typing_extensions import override

from ._base import ParametricLifetimeModel
from relife.frozen_model import FrozenParametricLifetimeModel

Args = TypeVarTuple("Args")
P = ParamSpec("P")


# necessary to allow user passing 1d ar and a0
def _reshape_ar_or_a0(
    name: str, value: float | Sequence[float] | NDArray[np.float64]
) -> NDArray[np.float64]:  # ndim is 2 (m,n) or (m,1)
    value = np.squeeze(np.asarray(value))
    match value.ndim:
        case 2 if value.shape[1] != 1:
            raise ValueError(
                f"Incorrect {name} shape. If ar has 2 dim, the shape must be (m, 1) only. Got {value.shape}"
            )
        case 1 | 0:
            value = value.reshape(-1, 1)
        case _:
            raise ValueError(
                f"Incorrect {name} shape. Got {value.shape}. Expected (), (m,) or (m, 1)"
            )
    return value


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
        ar: float | Sequence[float] | NDArray[np.float64],
        *args: *Args,
    ) -> NDArray[np.float64]:
        ar = _reshape_ar_or_a0("ar", ar)
        return np.where(time < ar, self.baseline.sf(time, *args), 0.0)

    def hf(
        self,
        time: float | NDArray[np.float64],
        ar: float | Sequence[float] | NDArray[np.float64],
        *args: *Args,
    ) -> NDArray[np.float64]:
        ar = _reshape_ar_or_a0("ar", ar)
        return np.where(time < ar, self.baseline.hf(time, *args), 0.0)

    def chf(
        self,
        time: float | NDArray[np.float64],
        ar: float | Sequence[float] | NDArray[np.float64],
        *args: *Args,
    ) -> NDArray[np.float64]:
        ar = _reshape_ar_or_a0("ar", ar)
        return np.where(time < ar, self.baseline.chf(time, *args), 0.0)

    @override
    def isf(
        self,
        probability: NDArray[np.float64],
        ar: float | Sequence[float] | NDArray[np.float64],
        *args: *Args,
    ) -> NDArray[np.float64]:
        ar = _reshape_ar_or_a0("ar", ar)
        return np.minimum(self.baseline.isf(probability, *args), ar)

    @override
    def ichf(
        self,
        cumulative_hazard_rate: NDArray[np.float64],
        ar: float | Sequence[float] | NDArray[np.float64],
        *args: *Args,
    ) -> NDArray[np.float64]:
        ar = _reshape_ar_or_a0("ar", ar)
        return np.minimum(self.baseline.ichf(cumulative_hazard_rate, *args), ar)

    def pdf(
        self,
        time: float | NDArray[np.float64],
        ar: float | Sequence[float] | NDArray[np.float64],
        *args: *Args,
    ) -> NDArray[np.float64]:
        ar = _reshape_ar_or_a0("ar", ar)
        return np.where(time < ar, self.baseline.pdf(time, *args), 0)

    @override
    def moment(
        self, n: int, ar: float | Sequence[float] | NDArray[np.float64], *args: *Args
    ) -> NDArray[np.float64]:
        ar = _reshape_ar_or_a0("ar", ar)
        return self.ls_integrate(
            lambda x: x**n,
            0,
            np.inf,
            ar,
            *args,
            deg=100,
        )

    @override
    def mean(
        self, ar: float | Sequence[float] | NDArray[np.float64], *args: *Args
    ) -> NDArray[np.float64]:
        ar = _reshape_ar_or_a0("ar", ar)
        return self.moment(1, ar, *args)

    @override
    def var(
        self, ar: float | Sequence[float] | NDArray[np.float64], *args: *Args
    ) -> NDArray[np.float64]:
        ar = _reshape_ar_or_a0("ar", ar)
        return self.moment(2, ar, *args) - self.moment(1, ar, *args) ** 2

    @override
    def rvs(
        self,
        ar: float | Sequence[float] | NDArray[np.float64],
        *args: *Args,
        size: Optional[int | tuple[int] | tuple[int, int]] = None,
        seed: Optional[int] = None,
    ) -> NDArray[DTypeLike]:
        ar = _reshape_ar_or_a0("ar", ar)
        struct_array = super().rvs(*(ar, *args), size=size, seed=seed)
        ar = np.broadcast_to(ar, struct_array["time"].shape).copy()
        struct_array["time"] = np.minimum(self.baseline.rvs(*args, size=size, seed=seed), ar)
        struct_array["event"] = struct_array["event"] != ar
        return struct_array

    def ppf(
        self,
        probability: float | NDArray[np.float64],
        ar: float | Sequence[float] | NDArray[np.float64],
        *args: *Args,
    ) -> NDArray[np.float64]:
        ar = _reshape_ar_or_a0("ar", ar)
        return self.isf(1 - probability, ar, *args)

    def median(
        self, ar: float | Sequence[float] | NDArray[np.float64], *args: *Args
    ) -> NDArray[np.float64]:
        ar = _reshape_ar_or_a0("ar", ar)
        return self.ppf(np.array(0.5), ar, *args)

    def cdf(
        self,
        time: float | NDArray[np.float64],
        ar: float | Sequence[float] | NDArray[np.float64],
        *args: *Args,
    ) -> NDArray[np.float64]:
        ar = _reshape_ar_or_a0("ar", ar)
        return np.where(time < ar, self.baseline.cdf(time, *args), 1.0)

    @override
    def mrl(
        self,
        time: float | NDArray[np.float64],
        ar: float | Sequence[float] | NDArray[np.float64],
        *args: *Args,
    ) -> NDArray[np.float64]:
        ar = _reshape_ar_or_a0("ar", ar)
        ub = np.array(np.inf)
        # ar.shape == (m, 1)
        mask = time >= ar  # (m, 1) or (m, n)
        if np.any(mask):
            time, ub = np.broadcast_arrays(time, ub)
            time = np.ma.MaskedArray(time, mask)  # (m, 1) or (m, n)
            ub = np.ma.MaskedArray(ub, mask)  # (m, 1) or (m, n)
        mu = self.ls_integrate(
            lambda x: x - time, time, ub, ar, *args, deg=10
        ) / self.sf(
            time, ar, *args
        )  # () or (n,) or (m, n)
        np.ma.filled(mu, 0)
        return np.ma.getdata(mu)

    @override
    def ls_integrate(
        self,
        func: Callable[[float | NDArray[np.float64]], NDArray[np.float64]],
        a: float | NDArray[np.float64],
        b: float | NDArray[np.float64],
        ar: float | Sequence[float] | NDArray[np.float64],
        *args: *Args,
        deg: int = 10,
    ) -> NDArray[np.float64]:
        ar = _reshape_ar_or_a0("ar", ar)
        b = np.minimum(ar, b)
        integration = super().ls_integrate(func, a, b, *(ar, *args), deg=deg)
        return integration + np.where(
            b == ar, func(ar) * self.baseline.sf(ar, *args), 0
        )

    @override
    def freeze(
        self, ar: float | NDArray[np.float64], *args: *Args
    ) -> FrozenParametricLifetimeModel:
        ar = _reshape_ar_or_a0("ar", ar)
        return super().freeze(*(ar, *args))


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
        self.baseline = baseline

    def sf(
        self,
        time: float | NDArray[np.float64],
        a0: float | Sequence[float] | NDArray[np.float64],
        *args: *Args,
    ) -> NDArray[np.float64]:
        a0 = _reshape_ar_or_a0("a0", a0)
        return super().sf(time, a0, *args)

    def pdf(
        self,
        time: float | NDArray[np.float64],
        a0: float | Sequence[float] | NDArray[np.float64],
        *args: *Args,
    ) -> NDArray[np.float64]:
        a0 = _reshape_ar_or_a0("a0", a0)
        return super().pdf(time, a0, *args)

    def isf(
        self,
        probability: NDArray[np.float64],
        a0: float | Sequence[float] | NDArray[np.float64],
        *args: *Args,
    ) -> NDArray[np.float64]:
        cumulative_hazard_rate = -np.log(probability + 1e-6) # avoid division by zero
        a0 = _reshape_ar_or_a0("a0", a0)
        return self.ichf(cumulative_hazard_rate, a0, *args)

    def chf(
        self,
        time: float | NDArray[np.float64],
        a0: float | Sequence[float] | NDArray[np.float64],
        *args: *Args,
    ) -> NDArray[np.float64]:
        a0 = _reshape_ar_or_a0("a0", a0)
        return self.baseline.chf(a0 + time, *args) - self.baseline.chf(a0, *args)

    def hf(
        self,
        time: float | NDArray[np.float64],
        a0: float | Sequence[float] | NDArray[np.float64],
        *args: *Args,
    ) -> NDArray[np.float64]:
        a0 = _reshape_ar_or_a0("a0", a0)
        return self.baseline.hf(a0 + time, *args)

    def ichf(
        self,
        cumulative_hazard_rate: NDArray[np.float64],
        a0: float | Sequence[float] | NDArray[np.float64],
        *args: *Args,
    ) -> NDArray[np.float64]:
        a0 = _reshape_ar_or_a0("a0", a0)
        return (
            self.baseline.ichf(
                cumulative_hazard_rate + self.baseline.chf(a0, *args), *args
            )
            - a0
        )

    @override
    def rvs(
        self,
        a0: float | Sequence[float] | NDArray[np.float64],
        *args: *Args,
        size: Optional[int | tuple[int] | tuple[int, int]] = None,
        seed: Optional[int] = None,
    ) -> NDArray[DTypeLike]:
        a0 = _reshape_ar_or_a0("a0", a0) # isf is overriden so rvs is conditional to a0
        struct_array = super().rvs(*(a0, *args), size=size, seed=seed)
        struct_array["entry"] = np.broadcast_to(a0, struct_array["time"].shape).copy()
        return struct_array

    @override
    def freeze(
        self, a0: float | Sequence[float] | NDArray[np.float64], *args: *Args
    ) -> FrozenParametricLifetimeModel:
        a0 = _reshape_ar_or_a0("a0", a0)
        return super().freeze(*(a0, *args))
