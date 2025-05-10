from typing import Callable, Optional, ParamSpec, Sequence, TypeVarTuple

import numpy as np
from numpy.typing import NDArray, DTypeLike
from typing_extensions import override

from ._base import ParametricLifetimeModel

Args = TypeVarTuple("Args")
P = ParamSpec("P")

# necessary to allow user passing 1d ar and a0
def reshape_ar_or_a0(
    name: str, value: float | Sequence[float] | NDArray[np.float64]
) -> NDArray[np.float64]:
    value = np.asarray(value) # in shape : (), (m,) or (m, 1)
    if value.ndim > 2 or (value.ndim == 2 and value.shape[-1] != 1):
        raise ValueError(
            f"Incorrect {name} shape. Got {value.shape}. Expected (), (m,) or (m, 1)"
        )
    if value.ndim == 1:
        value = value.reshape(-1, 1)
    return value  # out shape: () or (m, 1)

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
        ar = reshape_ar_or_a0("ar", ar)
        return np.where(time < ar, self.baseline.sf(time, *args), 0.0)

    def hf(
        self,
        time: float | NDArray[np.float64],
        ar: float | Sequence[float] | NDArray[np.float64],
        *args: *Args,
    ) -> NDArray[np.float64]:
        ar = reshape_ar_or_a0("ar", ar)
        return np.where(time < ar, self.baseline.hf(time, *args), 0.0)

    def chf(
        self,
        time: float | NDArray[np.float64],
        ar: float | Sequence[float] | NDArray[np.float64],
        *args: *Args,
    ) -> NDArray[np.float64]:
        ar = reshape_ar_or_a0("ar", ar)
        return np.where(time < ar, self.baseline.chf(time, *args), 0.0)

    @override
    def isf(
        self,
        probability: NDArray[np.float64],
        ar: float | Sequence[float] | NDArray[np.float64],
        *args: *Args,
    ) -> NDArray[np.float64]:
        ar = reshape_ar_or_a0("ar", ar)
        return np.minimum(self.baseline.isf(probability, *args), ar)

    @override
    def ichf(
        self,
        cumulative_hazard_rate: NDArray[np.float64],
        ar: float | Sequence[float] | NDArray[np.float64],
        *args: *Args,
    ) -> NDArray[np.float64]:
        ar = reshape_ar_or_a0("ar", ar)
        return np.minimum(self.baseline.ichf(cumulative_hazard_rate, *args), ar)

    def pdf(
        self,
        time: float | NDArray[np.float64],
        ar: float | Sequence[float] | NDArray[np.float64],
        *args: *Args,
    ) -> NDArray[np.float64]:
        ar = reshape_ar_or_a0("ar", ar)
        return np.where(time < ar, self.baseline.pdf(time, *args), 0)

    @override
    def moment(
        self, n: int, ar: float | Sequence[float] | NDArray[np.float64], *args: *Args
    ) -> NDArray[np.float64]:
        ar = reshape_ar_or_a0("ar", ar)
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
        ar = reshape_ar_or_a0("ar", ar)
        return self.moment(1, ar, *args)

    @override
    def var(
        self, ar: float | Sequence[float] | NDArray[np.float64], *args: *Args
    ) -> NDArray[np.float64]:
        ar = reshape_ar_or_a0("ar", ar)
        return self.moment(2, ar, *args) - self.moment(1, ar, *args) ** 2

    @override
    def rvs(
        self,
        ar: float | Sequence[float] | NDArray[np.float64],
        *args: *Args,
        size: Optional[int | tuple[int] | tuple[int, int]] = None,
        seed: Optional[int] = None,
    ) -> NDArray[DTypeLike]:
        ar = reshape_ar_or_a0("ar", ar)
        return super().rvs(*(ar, *args), size=size, seed=seed)

    def ppf(
        self,
        probability: float | NDArray[np.float64],
        ar: float | Sequence[float] | NDArray[np.float64],
        *args: *Args,
    ) -> NDArray[np.float64]:
        ar = reshape_ar_or_a0("ar", ar)
        return self.isf(1 - probability, ar, *args)

    def median(
        self, ar: float | Sequence[float] | NDArray[np.float64], *args: *Args
    ) -> NDArray[np.float64]:
        ar = reshape_ar_or_a0("ar", ar)
        return self.ppf(np.array(0.5), ar, *args)

    def cdf(
        self,
        time: float | NDArray[np.float64],
        ar: float | Sequence[float] | NDArray[np.float64],
        *args: *Args,
    ) -> NDArray[np.float64]:
        ar = reshape_ar_or_a0("ar", ar)
        return np.where(time < ar, self.baseline.cdf(time, *args), 1.0)

    @override
    def mrl(
        self,
        time: float | NDArray[np.float64],
        ar: float | Sequence[float] | NDArray[np.float64],
        *args: *Args,
    ) -> NDArray[np.float64]:
        ar = reshape_ar_or_a0("ar", ar)
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
        ar = reshape_ar_or_a0("ar", ar)
        b = np.minimum(ar, b)
        integration = super().ls_integrate(func, a, b, *(ar, *args), deg=deg)
        return integration + np.where(
            b == ar, func(ar) * self.baseline.sf(ar, *args), 0
        )

    def freeze(
        self, ar: float | NDArray[np.float64], *args: *Args
    ) -> ParametricLifetimeModel[()]:
        from .frozen_model import FrozenParametricLifetimeModel
        ar = reshape_ar_or_a0("ar", ar)
        return  FrozenParametricLifetimeModel(self).collect_args(*(ar, *args))


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
        a0 = reshape_ar_or_a0("a0", a0)
        return super().sf(time, a0, *args)

    def pdf(
        self,
        time: float | NDArray[np.float64],
        a0: float | Sequence[float] | NDArray[np.float64],
        *args: *Args,
    ) -> NDArray[np.float64]:
        a0 = reshape_ar_or_a0("a0", a0)
        return super().pdf(time, a0, *args)

    def isf(
        self,
        probability: NDArray[np.float64],
        a0: float | Sequence[float] | NDArray[np.float64],
        *args: *Args,
    ) -> NDArray[np.float64]:
        cumulative_hazard_rate = -np.log(probability + 1e-6) # avoid division by zero
        a0 = reshape_ar_or_a0("a0", a0)
        return self.ichf(cumulative_hazard_rate, a0, *args)

    def chf(
        self,
        time: float | NDArray[np.float64],
        a0: float | Sequence[float] | NDArray[np.float64],
        *args: *Args,
    ) -> NDArray[np.float64]:
        a0 = reshape_ar_or_a0("a0", a0)
        return self.baseline.chf(a0 + time, *args) - self.baseline.chf(a0, *args)

    def hf(
        self,
        time: float | NDArray[np.float64],
        a0: float | Sequence[float] | NDArray[np.float64],
        *args: *Args,
    ) -> NDArray[np.float64]:
        a0 = reshape_ar_or_a0("a0", a0)
        return self.baseline.hf(a0 + time, *args)

    def ichf(
        self,
        cumulative_hazard_rate: NDArray[np.float64],
        a0: float | Sequence[float] | NDArray[np.float64],
        *args: *Args,
    ) -> NDArray[np.float64]:
        a0 = reshape_ar_or_a0("a0", a0)
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
        from relife.sample import sample_failure_data
        a0 = reshape_ar_or_a0("a0", a0)
        nb_assets = None
        if isinstance(size, tuple):
            nb_sample = size[0]
            if len(size) == 2:
                nb_assets = size[-1]
        else:
            nb_sample = size
        time, _, entry, _ = sample_failure_data(self.freeze(a0, *args), nb_sample, (0., np.inf), nb_assets=nb_assets, astuple=True, seed=seed)
        time = time - entry # return residual time
        return time.reshape(size, copy=True)

    def freeze(
        self, a0: float | Sequence[float] | NDArray[np.float64], *args: *Args
    ) -> ParametricLifetimeModel[()]:
        from .frozen_model import FrozenParametricLifetimeModel
        a0 = reshape_ar_or_a0("a0", a0)
        return  FrozenParametricLifetimeModel(self).collect_args(*(a0, *args))
