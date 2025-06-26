from __future__ import annotations

from typing import Callable, Literal, Optional, Union, overload

import numpy as np
from numpy.typing import NDArray
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
    # noinspection PyUnresolvedReferences
    r"""
    Age replacement model.

    Lifetime model where the assets are replaced at age :math:`a_r`. This is equivalent to the model of :math:`\min(X,a_r)` where
    :math:`X` is a baseline lifetime and :math:`a_r` is the age of replacement.

    Parameters
    ----------
    baseline : any parametric lifetime model (frozen lifetime model works)
        The base lifetime model without conditional probabilities

    Attributes
    ----------
    baseline
    nb_params
    params
    params_names
    plot
    """

    def __init__(
        self,
        baseline: (
            ParametricLifetimeModel[*tuple[float | NDArray[np.float64], ...]]
            | FrozenParametricLifetimeModel[*tuple[float | NDArray[np.float64], ...]]
        ),
    ):
        super().__init__()
        self.baseline = baseline

    @property
    def args_names(self) -> tuple[str, *tuple[str, ...]]:
        return ("ar",) + self.baseline.args_names

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

    # def cdf(
    #     self,
    #     time: float | NDArray[np.float64],
    #     ar: float | NDArray[np.float64],
    #     *args: float | NDArray[np.float64],
    # ) -> NDArray[np.float64]:
    #     ar = reshape_ar_or_a0("ar", ar)
    #     return np.where(time < ar, self.baseline.cdf(time, *args), 1.0)

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

    @overload
    def rvs(
        self,
        size: int | tuple[int] | tuple[int, int],
        ar: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
        return_event: Literal[False],
        return_entry: Literal[False],
        seed: Optional[int] = None,
    ) -> np.float64 | NDArray[np.float64]: ...

    @overload
    def rvs(
        self,
        size: int | tuple[int] | tuple[int, int],
        ar: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
        return_event: Literal[True],
        return_entry: Literal[False],
        seed: Optional[int] = None,
    ) -> tuple[np.float64 | NDArray[np.float64], np.bool_ | NDArray[np.bool_]]: ...

    @overload
    def rvs(
        self,
        size: int | tuple[int] | tuple[int, int],
        ar: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
        return_event: Literal[False],
        return_entry: Literal[True],
        seed: Optional[int] = None,
    ) -> tuple[np.float64 | NDArray[np.float64], np.float64 | NDArray[np.float64]]: ...

    @overload
    def rvs(
        self,
        size: int | tuple[int] | tuple[int, int],
        ar: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
        return_event: Literal[True],
        return_entry: Literal[True],
        seed: Optional[int] = None,
    ) -> tuple[np.float64 | NDArray[np.float64], np.bool_ | NDArray[np.bool_], np.float64 | NDArray[np.float64]]: ...

    @overload
    def rvs(
        self,
        size: int | tuple[int] | tuple[int, int],
        ar: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
        return_event: bool = False,
        return_entry: bool = False,
        seed: Optional[int] = None,
    ) -> Union[
        np.float64 | NDArray[np.float64],
        tuple[np.float64 | NDArray[np.float64], np.bool_ | NDArray[np.bool_]],
        tuple[np.float64 | NDArray[np.float64], np.float64 | NDArray[np.float64]],
        tuple[np.float64 | NDArray[np.float64], np.bool_ | NDArray[np.bool_], np.float64 | NDArray[np.float64]],
    ]: ...

    @override
    def rvs(
        self,
        size: int | tuple[int] | tuple[int, int],
        ar: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
        return_event: bool = False,
        return_entry: bool = False,
        seed: Optional[int] = None,
    ) -> Union[
        np.float64 | NDArray[np.float64],
        tuple[np.float64 | NDArray[np.float64], np.bool_ | NDArray[np.bool_]],
        tuple[np.float64 | NDArray[np.float64], np.float64 | NDArray[np.float64]],
        tuple[np.float64 | NDArray[np.float64], np.bool_ | NDArray[np.bool_], np.float64 | NDArray[np.float64]],
    ]:
        ar = reshape_ar_or_a0("ar", ar)
        baseline_rvs = self.baseline.rvs(size, *args, return_event=return_event, return_entry=return_entry, seed=seed)
        time = baseline_rvs[0] if isinstance(baseline_rvs, tuple) else baseline_rvs
        time = np.minimum(time, ar)  # it may change time shape by broadcasting
        if not return_event and not return_entry:
            return time
        elif return_event and not return_entry:
            event = np.broadcast_to(baseline_rvs[1], time.shape).copy()
            event = np.where(time != ar, event, ~event)
            return time, event
        elif not return_event and return_entry:
            entry = np.broadcast_to(baseline_rvs[1], time.shape).copy()
            return time, entry
        else:
            event, entry = baseline_rvs[1:]
            event = np.broadcast_to(event, time.shape).copy()
            entry = np.broadcast_to(entry, time.shape).copy()
            event = np.where(time != ar, event, ~event)
            return time, event, entry

    @override
    def ls_integrate(
        self,
        func: Callable[[float | np.float64 | NDArray[np.float64]], float | np.float64 | NDArray[np.float64]],
        a: float | NDArray[np.float64],
        b: float | NDArray[np.float64],
        ar: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
        deg: int = 10,
    ) -> NDArray[np.float64]:
        ar = reshape_ar_or_a0("ar", ar)
        b = np.minimum(ar, b)
        integration = self.baseline.ls_integrate(func, a, b, *args, deg=deg)
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
    # noinspection PyUnresolvedReferences
    r"""Left truncated model.

    Lifetime model where the assets have already reached the age :math:`a_0`.

    Parameters
    ----------
    baseline : any parametric lifetime model (frozen lifetime model works)
        The base lifetime model without conditional probabilities
    nb_params
    params
    params_names
    plot

    Attributes
    ----------
    baseline
    nb_params
    params
    params_names
    plot
    """

    def __init__(
        self,
        baseline: (
            ParametricLifetimeModel[*tuple[float | NDArray[np.float64], ...]]
            | FrozenParametricLifetimeModel[*tuple[float | NDArray[np.float64], ...]]
        ),
    ):
        super().__init__()
        self.baseline = baseline

    @property
    def args_names(self) -> tuple[str, *tuple[str, ...]]:
        return ("a0",) + self.baseline.args_names

    def sf(
        self,
        time: float | NDArray[np.float64],
        a0: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
    ) -> np.float64 | NDArray[np.float64]:
        a0 = reshape_ar_or_a0("a0", a0)
        return super().sf(time, a0, *args)

    def pdf(
        self,
        time: float | NDArray[np.float64],
        a0: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
    ) -> np.float64 | NDArray[np.float64]:
        a0 = reshape_ar_or_a0("a0", a0)
        return super().pdf(time, a0, *args)

    def isf(
        self,
        probability: NDArray[np.float64],
        a0: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
    ) -> np.float64 | NDArray[np.float64]:
        cumulative_hazard_rate = -np.log(probability + 1e-6)  # avoid division by zero
        a0 = reshape_ar_or_a0("a0", a0)
        return self.ichf(cumulative_hazard_rate, a0, *args)

    def chf(
        self,
        time: float | NDArray[np.float64],
        a0: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
    ) -> np.float64 | NDArray[np.float64]:
        a0 = reshape_ar_or_a0("a0", a0)
        return self.baseline.chf(a0 + time, *args) - self.baseline.chf(a0, *args)

    def cdf(
        self,
        time: float | NDArray[np.float64],
        ar: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
    ) -> np.float64 | NDArray[np.float64]:
        ar = reshape_ar_or_a0("ar", ar)
        return super().cdf(time, *(ar, *args))

    def hf(
        self,
        time: float | NDArray[np.float64],
        a0: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
    ) -> np.float64 | NDArray[np.float64]:
        a0 = reshape_ar_or_a0("a0", a0)
        return self.baseline.hf(a0 + time, *args)

    def ichf(
        self,
        cumulative_hazard_rate: float | NDArray[np.float64],
        a0: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
    ) -> np.float64 | NDArray[np.float64]:
        a0 = reshape_ar_or_a0("a0", a0)
        return self.baseline.ichf(cumulative_hazard_rate + self.baseline.chf(a0, *args), *args) - a0

    @overload
    def rvs(
        self,
        size: int | tuple[int] | tuple[int, int],
        a0: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
        return_event: Literal[False],
        return_entry: Literal[False],
        seed: Optional[int] = None,
    ) -> np.float64 | NDArray[np.float64]: ...

    @overload
    def rvs(
        self,
        size: int | tuple[int] | tuple[int, int],
        a0: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
        return_event: Literal[True],
        return_entry: Literal[False],
        seed: Optional[int] = None,
    ) -> tuple[np.float64 | NDArray[np.float64], np.bool_ | NDArray[np.bool_]]: ...

    @overload
    def rvs(
        self,
        size: int | tuple[int] | tuple[int, int],
        a0: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
        return_event: Literal[False],
        return_entry: Literal[True],
        seed: Optional[int] = None,
    ) -> tuple[np.float64 | NDArray[np.float64], np.float64 | NDArray[np.float64]]: ...

    @overload
    def rvs(
        self,
        size: int | tuple[int] | tuple[int, int],
        a0: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
        return_event: Literal[True],
        return_entry: Literal[True],
        seed: Optional[int] = None,
    ) -> tuple[np.float64 | NDArray[np.float64], np.bool_ | NDArray[np.bool_], np.float64 | NDArray[np.float64]]: ...

    @overload
    def rvs(
        self,
        size: int | tuple[int] | tuple[int, int],
        a0: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
        return_event: bool = False,
        return_entry: bool = False,
        seed: Optional[int] = None,
    ) -> Union[
        np.float64 | NDArray[np.float64],
        tuple[np.float64 | NDArray[np.float64], np.bool_ | NDArray[np.bool_]],
        tuple[np.float64 | NDArray[np.float64], np.float64 | NDArray[np.float64]],
        tuple[np.float64 | NDArray[np.float64], np.bool_ | NDArray[np.bool_], np.float64 | NDArray[np.float64]],
    ]: ...

    @override
    def rvs(
        self,
        size: int | tuple[int] | tuple[int, int],
        a0: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
        return_event: bool = False,
        return_entry: bool = False,
        seed: Optional[int] = None,
    ) -> Union[
        np.float64 | NDArray[np.float64],
        tuple[np.float64 | NDArray[np.float64], np.bool_ | NDArray[np.bool_]],
        tuple[np.float64 | NDArray[np.float64], np.float64 | NDArray[np.float64]],
        tuple[np.float64 | NDArray[np.float64], np.bool_ | NDArray[np.bool_], np.float64 | NDArray[np.float64]],
    ]:
        a0 = reshape_ar_or_a0("a0", a0)
        super_rvs = super().rvs(size, *(a0, *args), return_event=return_event, return_entry=return_entry, seed=seed)
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

    @override
    def ls_integrate(
        self,
        func: Callable[[float | np.float64 | NDArray[np.float64]], float | np.float64 | NDArray[np.float64]],
        a: float | NDArray[np.float64],
        b: float | NDArray[np.float64],
        a0: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
        deg: int = 10,
    ) -> NDArray[np.float64]:
        a0 = reshape_ar_or_a0("a0", a0)
        return super().ls_integrate(func, a, b, *(a0, *args), deg=deg)

    @override
    def mean(
        self, a0: float | NDArray[np.float64], *args: float | NDArray[np.float64]
    ) -> np.float64 | NDArray[np.float64]:
        a0 = reshape_ar_or_a0("a0", a0)
        return super().mean(*(a0, *args))

    @override
    def median(
        self, a0: float | NDArray[np.float64], *args: float | NDArray[np.float64]
    ) -> np.float64 | NDArray[np.float64]:
        a0 = reshape_ar_or_a0("a0", a0)
        return super().median(*(a0, *args))

    @override
    def var(
        self, a0: float | NDArray[np.float64], *args: float | NDArray[np.float64]
    ) -> np.float64 | NDArray[np.float64]:
        a0 = reshape_ar_or_a0("a0", a0)
        return super().var(*(a0, *args))

    @override
    def moment(
        self, n: int, a0: float | NDArray[np.float64], *args: float | NDArray[np.float64]
    ) -> np.float64 | NDArray[np.float64]:
        a0 = reshape_ar_or_a0("a0", a0)
        return super().moment(n, *(a0, *args))

    @override
    def mrl(
        self, time: float | NDArray[np.float64], a0: float | NDArray[np.float64], *args: float | NDArray[np.float64]
    ) -> np.float64 | NDArray[np.float64]:
        a0 = reshape_ar_or_a0("a0", a0)
        return super().mrl(time, *(a0, *args))

    @override
    def ppf(
        self,
        probability: float | NDArray[np.float64],
        a0: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
    ) -> np.float64 | NDArray[np.float64]:
        a0 = reshape_ar_or_a0("a0", a0)
        return super().ppf(probability, *(a0, *args))


class FrozenAgeReplacementModel(
    FrozenParametricLifetimeModel[float | NDArray[np.float64], *tuple[float | NDArray[np.float64], ...]]
):
    r"""
    Frozen age replacement model.

    Parameters
    ----------
    model : AgeReplacementModel
        Any age replacement model.
    args_nb_assets : int
        Number of assets given in frozen arguments. It is automatically computed by ``freeze`` function.
    ar : float or np.ndarray
        Age of replacement values to be frozen.
    *args : float or np.ndarray
        Additional arguments needed by the model to be frozen.

    Attributes
    ----------
    unfrozen_model : AgeReplacementModel
        The unfrozen age replacement model.
    frozen_args : tuple of float or np.ndarray
        All the frozen arguments given and necessary to compute model functions.
    args_nb_assets : int
        Number of assets passed in frozen arguments. The data is mainly used to control numpy broadcasting and may not
        interest an user.

    Warnings
    --------
    The recommanded way to instanciate a frozen model is by using``freeze`` factory function.
    """

    unfrozen_model: AgeReplacementModel
    frozen_args: tuple[float | NDArray[np.float64], *tuple[float | NDArray[np.float64], ...]]

    @override
    def __init__(
        self,
        model: AgeReplacementModel,
        args_nb_assets: int,
        ar: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
    ):
        super().__init__(model, args_nb_assets, *(ar, *args))

    @override
    def unfreeze(self) -> AgeReplacementModel:
        return super().unfreeze()

    @property
    def ar(self) -> float | NDArray[np.float64]:
        return self.frozen_args[0]

    @ar.setter
    def ar(self, value: float | NDArray[np.float64]) -> None:
        self.frozen_args = (value,) + self.frozen_args[1:]


class FrozenLeftTruncatedModel(
    FrozenParametricLifetimeModel[float | NDArray[np.float64], *tuple[float | NDArray[np.float64], ...]]
):
    r"""
    Frozen left truncated model.

    Parameters
    ----------
    model : LeftTruncatedModel
        Any left truncated model.
    args_nb_assets : int
        Number of assets given in frozen arguments. It is automatically computed by ``freeze`` function.
    a0 : float or np.ndarray
        Conditional age values to be frozen.
    *args : float or np.ndarray
        Additional arguments needed by the model to be frozen.


    Attributes
    ----------
    unfrozen_model : LeftTruncatedModel
        The unfrozen left truncated model.
    frozen_args : tuple of float or np.ndarray
        All the frozen arguments given and necessary to compute model functions.
    args_nb_assets : int
        Number of assets passed in frozen arguments. The data is mainly used to control numpy broadcasting and may not
        interest an user.

    Warnings
    --------
    The recommanded way to instanciate a frozen model is by using``freeze`` factory function.
    """

    unfrozen_model: LeftTruncatedModel
    frozen_args: tuple[float | NDArray[np.float64], *tuple[float | NDArray[np.float64], ...]]

    @override
    def __init__(
        self,
        model: LeftTruncatedModel,
        args_nb_assets: int,
        a0: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
    ):
        super().__init__(model, args_nb_assets, *(a0, *args))

    @override
    def unfreeze(self) -> LeftTruncatedModel:
        return super().unfreeze()

    @property
    def a0(self) -> float | NDArray[np.float64]:
        return self.frozen_args[0]

    @a0.setter
    def a0(self, value: float | NDArray[np.float64]) -> None:
        self.frozen_args = (value,) + self.frozen_args[1:]


A0_TIME_BASE_DOCSTRING = """
{name}.

Parameters
----------
time : float or np.ndarray
    Elapsed time value(s) at which to compute the function.
    If ndarray, allowed shapes are ``()``, ``(n_values,)`` or ``(n_assets, n_values)``.
a0 : float or np.ndarray
    Conditional age values. It represents ages reached by assets. If ndarray, shape can only be (m,) 
    as only one age per asset can be given
*args : float or np.ndarray
    Additional arguments needed by the model.

Returns
-------
np.float64 or np.ndarray
    Function values at each given time(s).
"""


A0_MOMENT_BASE_DOCSTRING = """
{name}.

Parameters
----------
a0 : float or np.ndarray
    Conditional age values. It represents ages reached by assets. If ndarray, shape can only be (m,) 
    as only one age per asset can be given
*args : float or np.ndarray
    Additional arguments needed by the model.

Returns
-------
np.float64
    {name} value.
"""

A0_PROBABILITY_BASE_DOCSTRING = """
{name}.

Parameters
----------
probability : float or np.ndarray
    Probability value(s) at which to compute the function.
    If ndarray, allowed shapes are ``()``, ``(n,)`` or ``(m, n)``.
a0 : float or np.ndarray
    Conditional age values. It represents ages reached by assets. If ndarray, shape can only be (m,) 
    as only one age per asset can be given
*args : float or np.ndarray
    Additional arguments needed by the model.

Returns
-------
np.float64 or np.ndarray
    Function values at each given probability value(s).
"""

LeftTruncatedModel.sf.__doc__ = A0_TIME_BASE_DOCSTRING.format(name="The survival function")
LeftTruncatedModel.hf.__doc__ = A0_TIME_BASE_DOCSTRING.format(name="The hazard function")
LeftTruncatedModel.chf.__doc__ = A0_TIME_BASE_DOCSTRING.format(name="The cumulative hazard function")
LeftTruncatedModel.pdf.__doc__ = A0_TIME_BASE_DOCSTRING.format(name="The probability density function")
LeftTruncatedModel.cdf.__doc__ = A0_TIME_BASE_DOCSTRING.format(name="The cumulative distribution function")
LeftTruncatedModel.mrl.__doc__ = A0_TIME_BASE_DOCSTRING.format(name="The mean residual life function")
LeftTruncatedModel.ppf.__doc__ = A0_PROBABILITY_BASE_DOCSTRING.format(name="The percent point function")
LeftTruncatedModel.ppf.__doc__ += f"""
Notes
-----
The ``ppf`` is the inverse of :py:meth:`~LeftTruncatedModel.cdf`.
"""
LeftTruncatedModel.isf.__doc__ = A0_PROBABILITY_BASE_DOCSTRING.format(name="Inverse survival function")

LeftTruncatedModel.rvs.__doc__ = """
Random variable sampling.

Parameters
----------
size : int, (int,) or (int, int)
    Size of the generated sample. If size is ``n`` or ``(n,)``, n samples are generated. If size is ``(m,n)``, a 
    2d array of samples is generated.
a0 : float or np.ndarray
    Conditional age values. It represents ages reached by assets. If ndarray, shape can only be (m,) 
    as only one age per asset can be given
*args : float or np.ndarray
    Additional arguments needed by the model. 
return_event : bool, default is False
    If True, returns event indicators along with the sample time values.
return_entry : bool, default is False
    If True, returns corresponding entry values of the sample time values.
seed : optional int, default is None
    Random seed used to fix random sampling.

Returns
-------
float, ndarray or tuple of float or ndarray
    The sample values. If either ``return_event`` or ``random_entry`` is True, returns a tuple containing
    the time values followed by event values, entry values or both.
"""

LeftTruncatedModel.ichf.__doc__ = """
Inverse cumulative hazard function.

Parameters
----------
cumulative_hazard_rate : float or np.ndarray
    Cumulative hazard rate value(s) at which to compute the function.
    If ndarray, allowed shapes are ``()``, ``(n,)`` or ``(m, n)``.

a0 : float or np.ndarray
    Conditional age values. It represents ages reached by assets. If ndarray, shape can only be (m,) 
    as only one age per asset can be given
*args : float or np.ndarray
    Additional arguments needed by the model.

Returns
-------
np.float64 or np.ndarray
    Function values at each given cumulative hazard rate(s).
"""

LeftTruncatedModel.plot.__doc__ = """
Provides access to plotting functionality for this distribution.
"""

LeftTruncatedModel.ls_integrate.__doc__ = """
Lebesgue-Stieltjes integration.

Parameters
----------
func : callable (in : 1 ndarray , out : 1 ndarray)
    The callable must have only one ndarray object as argument and one ndarray object as output
a : ndarray (maximum number of dimension is 2)
    Lower bound(s) of integration.
b : ndarray (maximum number of dimension is 2)
    Upper bound(s) of integration. If lower bound(s) is infinite, use np.inf as value.)
a0 : float or np.ndarray
    Conditional age values. It represents ages reached by assets. If ndarray, shape can only be (m,) 
    as only one age per asset can be given
*args : float or np.ndarray
    Additional arguments needed by the model.
deg : int, default 10
    Degree of the polynomials interpolation

Returns
-------
np.ndarray
    Lebesgue-Stieltjes integral of func from `a` to `b`.
"""

LeftTruncatedModel.moment.__doc__ = """
n-th order moment

Parameters
----------
n : order of the moment, at least 1.

Returns
-------
np.float64
    n-th order moment.
"""
LeftTruncatedModel.mean.__doc__ = A0_MOMENT_BASE_DOCSTRING.format(name="The mean")
LeftTruncatedModel.var.__doc__ = A0_MOMENT_BASE_DOCSTRING.format(name="The variance")
LeftTruncatedModel.median.__doc__ = A0_MOMENT_BASE_DOCSTRING.format(name="The median")

LeftTruncatedModel.ichf.__doc__ = """
Inverse cumulative hazard function.

Parameters
----------
cumulative_hazard_rate : float or np.ndarray
    Cumulative hazard rate value(s) at which to compute the function.
    If ndarray, allowed shapes are ``()``, ``(n,)`` or ``(m, n)``.

a0 : float or np.ndarray
    Conditional age values. It represents ages reached by assets. If ndarray, shape can only be (m,) 
    as only one age per asset can be given
*args : float or np.ndarray
    Additional arguments needed by the model.

Returns
-------
np.float64 or np.ndarray
    Function values at each given cumulative hazard rate(s).
"""


AR_TIME_BASE_DOCSTRING = """
{name}.

Parameters
----------
time : float or np.ndarray
    Elapsed time value(s) at which to compute the function.
    If ndarray, allowed shapes are ``()``, ``(n_values,)`` or ``(n_assets, n_values)``.
ar : float or np.ndarray
    Age of replacement values. If ndarray, shape can only be (m,) 
    as only one age of replacement per asset can be given
*args : float or np.ndarray
    Additional arguments needed by the model.

Returns
-------
np.float64 or np.ndarray
    Function values at each given time(s).
"""


AR_MOMENT_BASE_DOCSTRING = """
{name}.

Parameters
----------
ar : float or np.ndarray
    Age of replacement values. If ndarray, shape can only be (m,) 
    as only one age of replacement per asset can be given
*args : float or np.ndarray
    Additional arguments needed by the model.

Returns
-------
np.float64
    {name} value.
"""

AR_PROBABILITY_BASE_DOCSTRING = """
{name}.

Parameters
----------
probability : float or np.ndarray
    Probability value(s) at which to compute the function.
    If ndarray, allowed shapes are ``()``, ``(n,)`` or ``(m, n)``.
ar : float or np.ndarray
    Age of replacement values. If ndarray, shape can only be (m,) 
    as only one age of replacement per asset can be given
*args : float or np.ndarray
    Additional arguments needed by the model.

Returns
-------
np.float64 or np.ndarray
    Function values at each given probability value(s).
"""

AgeReplacementModel.sf.__doc__ = AR_TIME_BASE_DOCSTRING.format(name="The survival function")
AgeReplacementModel.hf.__doc__ = AR_TIME_BASE_DOCSTRING.format(name="The hazard function")
AgeReplacementModel.chf.__doc__ = AR_TIME_BASE_DOCSTRING.format(name="The cumulative hazard function")
AgeReplacementModel.pdf.__doc__ = AR_TIME_BASE_DOCSTRING.format(name="The probability density function")
AgeReplacementModel.cdf.__doc__ = AR_TIME_BASE_DOCSTRING.format(name="The cumulative distribution function")
AgeReplacementModel.mrl.__doc__ = AR_TIME_BASE_DOCSTRING.format(name="The mean residual life function")
AgeReplacementModel.ppf.__doc__ = AR_PROBABILITY_BASE_DOCSTRING.format(name="The percent point function")
AgeReplacementModel.isf.__doc__ = AR_PROBABILITY_BASE_DOCSTRING.format(name="Inverse survival function")

AgeReplacementModel.rvs.__doc__ = """
Random variable sampling.

Parameters
----------
size : int, (int,) or (int, int)
    Size of the generated sample. If size is ``n`` or ``(n,)``, n samples are generated. If size is ``(m,n)``, a 
    2d array of samples is generated.
ar : float or np.ndarray
    Age of replacement values. If ndarray, shape can only be (m,) 
    as only one age of replacement per asset can be given
*args : float or np.ndarray
    Additional arguments needed by the model. 
return_event : bool, default is False
    If True, returns event indicators along with the sample time values.
return_entry : bool, default is False
    If True, returns corresponding entry values of the sample time values.
seed : optional int, default is None
    Random seed used to fix random sampling.

Returns
-------
float, ndarray or tuple of float or ndarray
    The sample values. If either ``return_event`` or ``random_entry`` is True, returns a tuple containing
    the time values followed by event values, entry values or both.

Notes
-----
If ``return_entry`` is true, returned time values are not residual time. Otherwise, the times are residuals
"""

AgeReplacementModel.ichf.__doc__ = """
Inverse cumulative hazard function.

Parameters
----------
cumulative_hazard_rate : float or np.ndarray
    Cumulative hazard rate value(s) at which to compute the function.
    If ndarray, allowed shapes are ``()``, ``(n,)`` or ``(m, n)``.

ar : float or np.ndarray
    Age of replacement values. If ndarray, shape can only be (m,) 
    as only one age of replacement per asset can be given
*args : float or np.ndarray
    Additional arguments needed by the model.

Returns
-------
np.float64 or np.ndarray
    Function values at each given cumulative hazard rate(s).
"""


AgeReplacementModel.plot.__doc__ = """
Provides access to plotting functionality for this distribution.
"""

AgeReplacementModel.ls_integrate.__doc__ = """
Lebesgue-Stieltjes integration.

Parameters
----------
func : callable (in : 1 ndarray , out : 1 ndarray)
    The callable must have only one ndarray object as argument and one ndarray object as output
a : ndarray (maximum number of dimension is 2)
    Lower bound(s) of integration.
b : ndarray (maximum number of dimension is 2)
    Upper bound(s) of integration. If lower bound(s) is infinite, use np.inf as value.)
ar : float or np.ndarray
    Age of replacement values. If ndarray, shape can only be (m,) 
    as only one age of replacement per asset can be given
*args : float or np.ndarray
    Additional arguments needed by the model.
deg : int, default 10
    Degree of the polynomials interpolation

Returns
-------
np.ndarray
    Lebesgue-Stieltjes integral of func from `a` to `b`.
"""

AgeReplacementModel.moment.__doc__ = """
n-th order moment

Parameters
----------
n : order of the moment, at least 1.
ar : float or np.ndarray
    Age of replacement values. If ndarray, shape can only be (m,) 
    as only one age of replacement per asset can be given
*args : float or np.ndarray
    Additional arguments needed by the model.

Returns
-------
np.float64 or np.ndarray
    n-th order moment.
"""
AgeReplacementModel.mean.__doc__ = A0_MOMENT_BASE_DOCSTRING.format(name="The mean")
AgeReplacementModel.var.__doc__ = A0_MOMENT_BASE_DOCSTRING.format(name="The variance")
AgeReplacementModel.median.__doc__ = A0_MOMENT_BASE_DOCSTRING.format(name="The median")

AgeReplacementModel.ichf.__doc__ = """
Inverse cumulative hazard function.

Parameters
----------
cumulative_hazard_rate : float or np.ndarray
    Cumulative hazard rate value(s) at which to compute the function.
    If ndarray, allowed shapes are ``()``, ``(n,)`` or ``(m, n)``.
ar : float or np.ndarray
    Age of replacement values. If ndarray, shape can only be (m,) 
    as only one age of replacement per asset can be given
*args : float or np.ndarray
    Additional arguments needed by the model.

Returns
-------
np.float64 or np.ndarray
    Function values at each given cumulative hazard rate(s).
"""
