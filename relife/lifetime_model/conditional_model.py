from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from numpy.typing import DTypeLike, NDArray
from typing_extensions import override

from ._base import FrozenParametricLifetimeModel, ParametricLifetimeModel


def reshape_ar_or_a0(name: str, value: float | NDArray[np.float64]) -> NDArray[np.float64]:
    value = np.asarray(value)  # in shape : (), (m,) or (m, 1)
    if value.ndim > 2 or (value.ndim == 2 and value.shape[-1] != 1):
        raise ValueError(f"Incorrect {name} shape. Got {value.shape}. Expected (), (m,) or (m, 1)")
    if value.ndim == 1:
        value = value.reshape(-1, 1)
    return value  # out shape: () or (m, 1)


# note that AgeReplacementModel does not preserve generic : at the moment, additional args are supposed to be always float | NDArray[np.float64]
class AgeReplacementModel(
    ParametricLifetimeModel[float | NDArray[np.float64], *tuple[float | NDArray[np.float64], ...]]
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

    # can't expect baseline to be FrozenParametricLifetimeModel too because it does not have freeze_args
    def __init__(self, baseline: ParametricLifetimeModel[*tuple[float | NDArray[np.float64], ...]]):
        super().__init__()
        self.baseline = baseline

    def sf(
        self,
        time: float | NDArray[np.float64],
        ar: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
    ) -> NDArray[np.float64]:
        ar = reshape_ar_or_a0("ar", ar)
        return np.where(time < ar, self.baseline.sf(time, *args), 0.0)

    def hf(
        self,
        time: float | NDArray[np.float64],
        ar: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
    ) -> NDArray[np.float64]:
        ar = reshape_ar_or_a0("ar", ar)
        return np.where(time < ar, self.baseline.hf(time, *args), 0.0)

    def chf(
        self,
        time: float | NDArray[np.float64],
        ar: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
    ) -> NDArray[np.float64]:
        ar = reshape_ar_or_a0("ar", ar)
        return np.where(time < ar, self.baseline.chf(time, *args), 0.0)

    @override
    def isf(
        self,
        probability: float | NDArray[np.float64],
        ar: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
    ) -> NDArray[np.float64]:
        ar = reshape_ar_or_a0("ar", ar)
        return np.minimum(self.baseline.isf(probability, *args), ar)

    @override
    def ichf(
        self,
        cumulative_hazard_rate: NDArray[np.float64],
        ar: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
    ) -> NDArray[np.float64]:
        ar = reshape_ar_or_a0("ar", ar)
        return np.minimum(self.baseline.ichf(cumulative_hazard_rate, *args), ar)

    def pdf(
        self,
        time: float | NDArray[np.float64],
        ar: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
    ) -> NDArray[np.float64]:
        ar = reshape_ar_or_a0("ar", ar)
        return np.where(time < ar, self.baseline.pdf(time, *args), 0)

    def ppf(
        self,
        probability: float | NDArray[np.float64],
        ar: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
    ) -> NDArray[np.float64]:
        ar = reshape_ar_or_a0("ar", ar)
        return self.isf(1 - probability, ar, *args)

    def median(self, ar: float | NDArray[np.float64], *args: float | NDArray[np.float64]) -> NDArray[np.float64]:
        ar = reshape_ar_or_a0("ar", ar)
        return self.ppf(np.array(0.5), ar, *args)

    def cdf(
        self,
        time: float | NDArray[np.float64],
        ar: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
    ) -> NDArray[np.float64]:
        ar = reshape_ar_or_a0("ar", ar)
        return np.where(time < ar, self.baseline.cdf(time, *args), 1.0)

    @override
    def mrl(
        self,
        time: float | NDArray[np.float64],
        ar: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
    ) -> NDArray[np.float64]:
        ar = reshape_ar_or_a0("ar", ar)
        ub = np.array(np.inf)
        # ar.shape == (m, 1)
        mask = time >= ar  # (m, 1) or (m, n)
        if np.any(mask):
            time, ub = np.broadcast_arrays(time, ub)
            time = np.ma.MaskedArray(time, mask)  # (m, 1) or (m, n)
            ub = np.ma.MaskedArray(ub, mask)  # (m, 1) or (m, n)
        mu = self.ls_integrate(lambda x: x - time, time, ub, ar, *args, deg=10) / self.sf(
            time, ar, *args
        )  # () or (n,) or (m, n)
        np.ma.filled(mu, 0)
        return np.ma.getdata(mu)

    @override
    def rvs(
        self,
        ar: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
        size: Optional[int | tuple[int] | tuple[int, int]] = None,
        return_event: bool = False,
        return_entry: bool = False,
        seed: Optional[int] = None,
    ) -> NDArray[DTypeLike]:
        ar = reshape_ar_or_a0("ar", ar)
        baseline_rvs = self.baseline.rvs(
            *args, size=size, return_event=return_event, return_entry=return_entry, seed=seed
        )
        time = baseline_rvs[0] if isinstance(baseline_rvs, tuple) else baseline_rvs
        time = np.minimum(time, ar)  # it may change time shape by broadcasting
        if not return_event and not return_entry:
            return time
        elif return_event and not return_entry:
            event = np.broadcast_to(baseline_rvs[1], time.shape).copy()
            event = event != ar
            return time, event
        elif not return_event and return_entry:
            entry = np.broadcast_to(baseline_rvs[1], time.shape).copy()
            return time, entry
        else:
            event, entry = baseline_rvs[1:]
            event = np.broadcast_to(event, time.shape).copy()
            entry = np.broadcast_to(entry, time.shape).copy()
            event = event != ar
            return time, event, entry

    @override
    def ls_integrate(
        self,
        func: Callable[[float | NDArray[np.float64]], NDArray[np.float64]],
        a: float | NDArray[np.float64],
        b: float | NDArray[np.float64],
        ar: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
        deg: int = 10,
    ) -> NDArray[np.float64]:
        ar = reshape_ar_or_a0("ar", ar)
        b = np.minimum(ar, b)
        integration = super().ls_integrate(func, a, b, *(ar, *args), deg=deg)
        return integration + np.where(b == ar, func(ar) * self.baseline.sf(ar, *args), 0)

    @override
    def moment(
        self, n: int, ar: float | NDArray[np.float64], *args: float | NDArray[np.float64]
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
    def mean(self, ar: float | NDArray[np.float64], *args: float | NDArray[np.float64]) -> NDArray[np.float64]:
        ar = reshape_ar_or_a0("ar", ar)
        return self.moment(1, ar, *args)

    @override
    def var(self, ar: float | NDArray[np.float64], *args: float | NDArray[np.float64]) -> NDArray[np.float64]:
        ar = reshape_ar_or_a0("ar", ar)
        return self.moment(2, ar, *args) - self.moment(1, ar, *args) ** 2


class LeftTruncatedModel(
    ParametricLifetimeModel[float | NDArray[np.float64], *tuple[float | NDArray[np.float64], ...]]
):
    r"""Left truncated core.

    Conditional distribution of the lifetime core for an asset having reach age :math:`a_0`.

    Parameters
    ----------
    baseline : BaseLifetimeModel
        Underlying lifetime core.
    """

    # can't expect baseline to be FrozenParametricLifetimeModel too because it does not have freeze_args
    def __init__(self, baseline: ParametricLifetimeModel[*tuple[float | NDArray[np.float64], ...]]):
        super().__init__()
        self.baseline = baseline

    def sf(
        self,
        time: float | NDArray[np.float64],
        a0: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
    ) -> NDArray[np.float64]:
        a0 = reshape_ar_or_a0("a0", a0)
        return super().sf(time, a0, *args)

    def pdf(
        self,
        time: float | NDArray[np.float64],
        a0: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
    ) -> NDArray[np.float64]:
        a0 = reshape_ar_or_a0("a0", a0)
        return super().pdf(time, a0, *args)

    def isf(
        self,
        probability: NDArray[np.float64],
        a0: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
    ) -> NDArray[np.float64]:
        cumulative_hazard_rate = -np.log(probability + 1e-6)  # avoid division by zero
        a0 = reshape_ar_or_a0("a0", a0)
        return self.ichf(cumulative_hazard_rate, a0, *args)

    def chf(
        self,
        time: float | NDArray[np.float64],
        a0: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
    ) -> NDArray[np.float64]:
        a0 = reshape_ar_or_a0("a0", a0)
        return self.baseline.chf(a0 + time, *args) - self.baseline.chf(a0, *args)

    def hf(
        self,
        time: float | NDArray[np.float64],
        a0: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
    ) -> NDArray[np.float64]:
        a0 = reshape_ar_or_a0("a0", a0)
        return self.baseline.hf(a0 + time, *args)

    def ichf(
        self,
        cumulative_hazard_rate: NDArray[np.float64],
        a0: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
    ) -> NDArray[np.float64]:
        a0 = reshape_ar_or_a0("a0", a0)
        return self.baseline.ichf(cumulative_hazard_rate + self.baseline.chf(a0, *args), *args) - a0

    @override
    def rvs(
        self,
        a0: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
        size: Optional[int | tuple[int] | tuple[int, int]] = None,
        return_event: bool = False,
        return_entry: bool = False,
        seed: Optional[int] = None,
    ) -> NDArray[DTypeLike]:
        a0 = reshape_ar_or_a0("a0", a0)
        super_rvs = super().rvs(
            *(a0, *args), size=size, return_event=return_event, return_entry=return_entry, seed=seed
        )
        if not return_event and return_entry:
            time, entry = super_rvs
            entry = np.broadcast_to(a0, entry.shape).copy()
            time = time + a0  #  not residual age
            return time, entry
        elif return_event and return_entry:
            time, event, entry = super_rvs
            entry = np.broadcast_to(a0, entry.shape).copy()
            time = time + a0  #  not residual age
            return time, event, entry
        else:
            return super_rvs



class FrozenAgeReplacementModel(FrozenParametricLifetimeModel[float|NDArray[np.float64], *tuple[float|NDArray[np.float64], ...]]):
    unfrozen_model: AgeReplacementModel
    frozen_args: tuple[float | NDArray[np.float64], *tuple[float | NDArray[np.float64], ...]]

    @override
    def __init__(self, model : AgeReplacementModel, ar : float|NDArray[np.float64], *args : float|NDArray[np.float64]):
        super().__init__(model, *(ar, *args))


    @override
    def unfreeze(self) -> AgeReplacementModel:
        return super().unfreeze()

    @property
    def ar(self):
        return self.frozen_args[0]


class FrozenLeftTruncatedModel(FrozenParametricLifetimeModel[float|NDArray[np.float64], *tuple[float|NDArray[np.float64], ...]]):
    unfrozen_model: LeftTruncatedModel
    frozen_args: tuple[float | NDArray[np.float64], *tuple[float | NDArray[np.float64], ...]]

    @override
    def __init__(self, model : LeftTruncatedModel, a0 : float|NDArray[np.float64], *args : float|NDArray[np.float64]):
        super().__init__(model, *(a0, *args))


    @override
    def unfreeze(self) -> LeftTruncatedModel:
        return super().unfreeze()

    @property
    def a0(self):
        return self.frozen_args[0]
