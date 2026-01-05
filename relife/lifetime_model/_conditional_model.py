from __future__ import annotations

from typing import Callable, Literal, TypeVarTuple, overload

import numpy as np
from numpy.typing import NDArray
from typing_extensions import override

from relife.typing import (
    AnyFloat,
    AnyParametricLifetimeModel,
    NumpyBool,
    NumpyFloat,
    Seed,
)
from relife.utils import get_args_nb_assets, is_frozen, reshape_1d_arg

from ._base import ParametricLifetimeModel
from ._frozen import FrozenParametricLifetimeModel

__all__: list[str] = ["AgeReplacementModel", "LeftTruncatedModel"]

Ts = TypeVarTuple("Ts")


class AgeReplacementModel(ParametricLifetimeModel[*tuple[AnyFloat, *Ts]]):
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

    baseline: AnyParametricLifetimeModel[*Ts]

    def __init__(self, baseline: AnyParametricLifetimeModel[*Ts]):
        super().__init__()
        self.baseline = baseline

    @override
    def sf(self, time: AnyFloat, ar: AnyFloat, *args: *Ts) -> NumpyFloat:
        """
        The survival function.

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
        ar = reshape_1d_arg(ar)
        return np.where(time < ar, self.baseline.sf(time, *args), 0.0)

    @override
    def hf(self, time: AnyFloat, ar: AnyFloat, *args: *Ts) -> NumpyFloat:
        """
        The hazard function.

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
        ar = reshape_1d_arg(ar)
        return np.where(time < ar, self.baseline.hf(time, *args), 0.0)

    @override
    def cdf(self, time: AnyFloat, ar: AnyFloat, *args: *Ts) -> NumpyFloat:
        """
        The cumulative density function.

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
        ar = reshape_1d_arg(ar)
        return super().cdf(time, *(ar, *args))

    @override
    def chf(self, time: AnyFloat, ar: AnyFloat, *args: *Ts) -> NumpyFloat:
        """
        The cumulative hazard function.

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
        ar = reshape_1d_arg(ar)
        return np.where(time < ar, self.baseline.chf(time, *args), 0.0)

    @override
    def isf(self, probability: AnyFloat, ar: AnyFloat, *args: *Ts) -> NumpyFloat:
        """
        The inverse of the survival function.

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
        ar = reshape_1d_arg(ar)
        return np.minimum(self.baseline.isf(probability, *args), ar)

    @override
    def ichf(self, cumulative_hazard_rate: AnyFloat, ar: AnyFloat, *args: *Ts) -> NumpyFloat:
        """
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
        ar = reshape_1d_arg(ar)
        return np.minimum(self.baseline.ichf(cumulative_hazard_rate, *args), ar)

    @override
    def pdf(self, time: AnyFloat, ar: AnyFloat, *args: *Ts) -> NumpyFloat:
        """
        The probability density function.

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
        ar = reshape_1d_arg(ar)
        return np.where(time < ar, self.baseline.pdf(time, *args), 0)

    @override
    def mrl(self, time: AnyFloat, ar: AnyFloat, *args: *Ts) -> NumpyFloat:
        """
        The mean residual life function.

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
        ar = reshape_1d_arg(ar)
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
    def ppf(self, probability: AnyFloat, ar: AnyFloat, *args: *Ts) -> NumpyFloat:
        """
        The percent point function.

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
        ar = reshape_1d_arg(ar)
        return self.isf(1 - probability, ar, *args)

    # def cdf(self, time, ar, *args):
    #     ar = reshape_ar_or_a0("ar", ar)
    #     return np.where(time < ar, self.baseline.cdf(time, *args), 1.0)

    @override
    def median(self, ar: AnyFloat, *args: *Ts) -> NumpyFloat:
        """
        The median.

        Parameters
        ----------
        ar : float or np.ndarray
            Age of replacement values. If ndarray, shape can only be (m,)
            as only one age of replacement per asset can be given
        *args : float or np.ndarray
            Additional arguments needed by the model.

        Returns
        -------
        np.float64 or np.ndarray
        """
        ar = reshape_1d_arg(ar)
        return self.ppf(np.array(0.5), ar, *args)

    @overload
    def rvs(
        self,
        size: int,
        ar: AnyFloat,
        *args: *Ts,
        nb_assets: int | None = None,
        return_event: Literal[False],
        return_entry: Literal[False],
        seed: Seed | None = None,
    ) -> NumpyFloat: ...
    @overload
    def rvs(
        self,
        size: int,
        ar: AnyFloat,
        *args: *Ts,
        nb_assets: int | None = None,
        return_event: Literal[True],
        return_entry: Literal[False],
        seed: Seed | None = None,
    ) -> tuple[NumpyFloat, NumpyBool]: ...
    @overload
    def rvs(
        self,
        size: int,
        ar: AnyFloat,
        *args: *Ts,
        nb_assets: int | None = None,
        return_event: Literal[False],
        return_entry: Literal[True],
        seed: Seed | None = None,
    ) -> tuple[NumpyFloat, NumpyFloat]: ...
    @overload
    def rvs(
        self,
        size: int,
        ar: AnyFloat,
        *args: *Ts,
        nb_assets: int | None = None,
        return_event: Literal[True],
        return_entry: Literal[True],
        seed: Seed | None = None,
    ) -> tuple[NumpyFloat, NumpyBool, NumpyFloat]: ...
    @overload
    def rvs(
        self,
        size: int,
        ar: AnyFloat,
        *args: *Ts,
        nb_assets: int | None = None,
        return_event: bool = False,
        return_entry: bool = False,
        seed: Seed | None = None,
    ) -> (
        NumpyFloat
        | tuple[NumpyFloat, NumpyBool]
        | tuple[NumpyFloat, NumpyFloat]
        | tuple[NumpyFloat, NumpyBool, NumpyFloat]
    ): ...
    @override
    def rvs(
        self,
        size: int,
        ar: AnyFloat,
        *args: *Ts,
        nb_assets: int | None = None,
        return_event: bool = False,
        return_entry: bool = False,
        seed: Seed | None = None,
    ) -> (
        NumpyFloat
        | tuple[NumpyFloat, NumpyBool]
        | tuple[NumpyFloat, NumpyFloat]
        | tuple[NumpyFloat, NumpyBool, NumpyFloat]
    ):
        """
        Random variable sampling.

        Parameters
        ----------
        size : int
            Size of the generated sample.
        ar : float or np.ndarray
            Age of replacement values. If ndarray, shape can only be (m,)
            as only one age of replacement per asset can be given
        *args : float or np.ndarray
            Additional arguments needed by the model.
        nb_assets : int, optional
            If nb_assets is not None, 2d arrays of samples are generated.
        return_event : bool, default is False
            If True, returns event indicators along with the sample time values.
        return_entry : bool, default is False
            If True, returns corresponding entry values of the sample time values.
        seed : optional int, np.random.BitGenerator, np.random.Generator, np.random.RandomState, default is None
            If int or BitGenerator, seed for random number generator. If np.random.RandomState or np.random.Generator, use as given.

        Returns
        -------
        float, ndarray or tuple of float or ndarray
            The sample values. If either ``return_event`` or ``return_entry`` is True, returns a tuple containing
            the time values followed by event values, entry values or both.

        Notes
        -----
        If ``return_entry`` is true, returned time values are not residual time. Otherwise, the times are residuals
        """

        ar = reshape_1d_arg(ar)
        if nb_assets is None:
            nb_assets = get_args_nb_assets(ar, *args)
            if nb_assets == 1:
                nb_assets = None
        baseline_rvs = self.baseline.rvs(
            size,
            *args,
            nb_assets=nb_assets,
            return_event=return_event,
            return_entry=return_entry,
            seed=seed,
        )
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
        func: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        a: AnyFloat,
        b: AnyFloat,
        ar: AnyFloat,
        *args: *Ts,
        deg: int = 10,
    ) -> NumpyFloat:
        """
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
        ar = reshape_1d_arg(ar)
        b = np.minimum(ar, b)
        integration = self.baseline.ls_integrate(func, a, b, *args, deg=deg)
        if func(ar).ndim == 2 and integration.ndim == 1:
            integration = integration.reshape(-1, 1)
        return integration + np.where(b == ar, func(ar) * self.baseline.sf(ar, *args), 0)

    @override
    def moment(self, n: int, ar: AnyFloat, *args: *Ts) -> NumpyFloat:
        """
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
        """
        ar = reshape_1d_arg(ar)
        return self.ls_integrate(
            lambda x: x**n,
            np.float64(0),
            np.inf,
            ar,
            *args,
            deg=100,
        )

    @override
    def mean(self, ar: AnyFloat, *args: *Ts) -> NumpyFloat:
        """
        The mean.

        Parameters
        ----------
        ar : float or np.ndarray
            Age of replacement values. If ndarray, shape can only be (m,)
            as only one age of replacement per asset can be given
        *args : float or np.ndarray
            Additional arguments needed by the model.

        Returns
        -------
        np.float64 or np.ndarray
        """
        ar = reshape_1d_arg(ar)
        return self.moment(1, ar, *args)

    @override
    def var(self, ar: AnyFloat, *args: *Ts) -> NumpyFloat:
        """
        The variance.

        Parameters
        ----------
        ar : float or np.ndarray
            Age of replacement values. If ndarray, shape can only be (m,)
            as only one age of replacement per asset can be given
        *args : float or np.ndarray
            Additional arguments needed by the model.

        Returns
        -------
        np.float64 or np.ndarray
        """
        ar = reshape_1d_arg(ar)
        return self.moment(2, ar, *args) - self.moment(1, ar, *args) ** 2

    def freeze(self, ar: AnyFloat, *args: *Ts) -> FrozenParametricLifetimeModel[*tuple[AnyFloat, *Ts]]:
        """
        Freeze age replacement values and other arguments into the object data.

        Parameters
        ----------
        ar : float or np.ndarray
            Age of replacement values. If ndarray, shape can only be (m,)
            as only one age of replacement per asset can be given
        *args : float or np.ndarray
            Additional arguments needed by the model.

        Returns
        -------
        FrozenParametricModel
        """
        return FrozenParametricLifetimeModel(self, ar, *args)


class LeftTruncatedModel(ParametricLifetimeModel[*tuple[AnyFloat, *Ts]]):
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

    baseline: AnyParametricLifetimeModel[*Ts]

    def __init__(self, baseline: AnyParametricLifetimeModel[*Ts]):
        super().__init__()
        self.baseline = baseline

    @override
    def sf(self, time: AnyFloat, a0: AnyFloat, *args: *Ts) -> NumpyFloat:
        """
        The survival function.

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
        a0 = reshape_1d_arg(a0)
        return super().sf(time, a0, *args)

    @override
    def pdf(self, time: AnyFloat, a0: AnyFloat, *args: *Ts) -> NumpyFloat:
        """
        The probability density function.

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
        a0 = reshape_1d_arg(a0)
        return super().pdf(time, a0, *args)

    @override
    def isf(self, probability: AnyFloat, a0: AnyFloat, *args: *Ts) -> NumpyFloat:
        """
        The inverse of the survival function.

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
        cumulative_hazard_rate = -np.log(probability + 1e-6)  # avoid division by zero
        a0 = reshape_1d_arg(a0)
        return self.ichf(cumulative_hazard_rate, a0, *args)

    @override
    def chf(self, time: AnyFloat, a0: AnyFloat, *args: *Ts) -> NumpyFloat:
        """
        The cumulative hazard function.

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
        a0 = reshape_1d_arg(a0)
        return self.baseline.chf(a0 + time, *args) - self.baseline.chf(a0, *args)

    @override
    def cdf(self, time: AnyFloat, a0: AnyFloat, *args: *Ts) -> NumpyFloat:
        """
        The cumulative density function.

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
        a0 = reshape_1d_arg(a0)
        return super().cdf(time, *(a0, *args))

    @override
    def hf(self, time: AnyFloat, a0: AnyFloat, *args: *Ts) -> NumpyFloat:
        """
        The hazard function.

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
        a0 = reshape_1d_arg(a0)
        return self.baseline.hf(a0 + time, *args)

    @override
    def ichf(self, cumulative_hazard_rate: AnyFloat, a0: AnyFloat, *args: *Ts) -> NumpyFloat:
        """
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
        a0 = reshape_1d_arg(a0)
        return self.baseline.ichf(cumulative_hazard_rate + self.baseline.chf(a0, *args), *args) - a0

    @overload
    def rvs(
        self,
        size: int,
        a0: AnyFloat,
        *args: *Ts,
        nb_assets: int | None = None,
        return_event: Literal[False],
        return_entry: Literal[False],
        seed: Seed | None = None,
    ) -> NumpyFloat: ...
    @overload
    def rvs(
        self,
        size: int,
        a0: AnyFloat,
        *args: *Ts,
        nb_assets: int | None = None,
        return_event: Literal[True],
        return_entry: Literal[False],
        seed: Seed | None = None,
    ) -> tuple[NumpyFloat, NumpyBool]: ...
    @overload
    def rvs(
        self,
        size: int,
        a0: AnyFloat,
        *args: *Ts,
        nb_assets: int | None = None,
        return_event: Literal[False],
        return_entry: Literal[True],
        seed: Seed | None = None,
    ) -> tuple[NumpyFloat, NumpyFloat]: ...
    @overload
    def rvs(
        self,
        size: int,
        a0: AnyFloat,
        *args: *Ts,
        nb_assets: int | None = None,
        return_event: Literal[True],
        return_entry: Literal[True],
        seed: Seed | None = None,
    ) -> tuple[NumpyFloat, NumpyBool, NumpyFloat]: ...
    @overload
    def rvs(
        self,
        size: int,
        a0: AnyFloat,
        *args: *Ts,
        nb_assets: int | None = None,
        return_event: bool = False,
        return_entry: bool = False,
        seed: Seed | None = None,
    ) -> (
        NumpyFloat
        | tuple[NumpyFloat, NumpyBool]
        | tuple[NumpyFloat, NumpyFloat]
        | tuple[NumpyFloat, NumpyBool, NumpyFloat]
    ): ...
    @override
    def rvs(
        self,
        size: int,
        a0: AnyFloat,
        *args: *Ts,
        nb_assets: int | None = None,
        return_event: bool = False,
        return_entry: bool = False,
        seed: Seed | None = None,
    ) -> (
        NumpyFloat
        | tuple[NumpyFloat, NumpyBool]
        | tuple[NumpyFloat, NumpyFloat]
        | tuple[NumpyFloat, NumpyBool, NumpyFloat]
    ):
        """
        Random variable sampling.

        Parameters
        ----------
        size : int
            Size of the generated sample.
        a0 : float or np.ndarray
            Conditional age values. It represents ages reached by assets. If ndarray, shape can only be (m,)
            as only one age per asset can be given
        *args : float or np.ndarray
            Additional arguments needed by the model.
        nb_assets : int, optional
            If nb_assets is not None, 2d arrays of samples are generated.
        return_event : bool, default is False
            If True, returns event indicators along with the sample time values.
        return_entry : bool, default is False
            If True, returns corresponding entry values of the sample time values.
        seed : optional int, np.random.BitGenerator, np.random.Generator, np.random.RandomState, default is None
            If int or BitGenerator, seed for random number generator. If np.random.RandomState or np.random.Generator, use as given.

        Returns
        -------
        float, ndarray or tuple of float or ndarray
            The sample values. If either ``return_event`` or ``return_entry`` is True, returns a tuple containing
            the time values followed by event values, entry values or both.
        """
        a0 = reshape_1d_arg(a0)
        if nb_assets is None:
            nb_assets = get_args_nb_assets(a0, *args)
            if nb_assets == 1:
                nb_assets = None
        super_rvs = super().rvs(
            size,
            *(a0, *args),
            nb_assets=nb_assets,
            return_event=return_event,
            return_entry=return_entry,
            seed=seed,
        )
        time = super_rvs[0] if isinstance(super_rvs, tuple) else super_rvs
        complete_ages = time + a0
        output = [
            time,
        ]  # at least time in output
        if return_event:
            event = super_rvs[1]  # event always at index 1
            # reconstruct event for AgeReplacementModel c omposition as super skips this info
            if isinstance(self.baseline, AgeReplacementModel):
                ar = reshape_1d_arg(args[0])
                event = np.where(complete_ages < ar, event, ~event)
            if is_frozen(self.baseline):
                ar = reshape_1d_arg(self.baseline.args[0])
                event = np.where(complete_ages < ar, event, ~event)
            output.append(event)
        if return_entry:
            output[0] = complete_ages  # don't return residual ages
            entry = super_rvs[-1]  # entry always at last index
            entry = np.broadcast_to(a0, entry.shape).copy()
            output.append(entry)
        if len(output) > 1:
            return tuple(output)  # return tuple, not list
        return output[0]

    @override
    def ls_integrate(
        self,
        func: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        a: AnyFloat,
        b: AnyFloat,
        a0: AnyFloat,
        *args: *Ts,
        deg: int = 10,
    ) -> NumpyFloat:
        """
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
        a0 = reshape_1d_arg(a0)
        return super().ls_integrate(func, a, b, *(a0, *args), deg=deg)

    @override
    def mean(self, a0: AnyFloat, *args: *Ts) -> NumpyFloat:
        """
        The mean.

        Parameters
        ----------
        a0 : float or np.ndarray
            Conditional age values. It represents ages reached by assets. If ndarray, shape can only be (m,)
            as only one age per asset can be given
        *args : float or np.ndarray
            Additional arguments needed by the model.

        Returns
        -------
        np.float64 or np.ndarray
        """
        a0 = reshape_1d_arg(a0)
        return super().mean(*(a0, *args))

    @override
    def median(self, a0: AnyFloat, *args: *Ts) -> NumpyFloat:
        """
        The median.

        Parameters
        ----------
        a0 : float or np.ndarray
            Conditional age values. It represents ages reached by assets. If ndarray, shape can only be (m,)
            as only one age per asset can be given
        *args : float or np.ndarray
            Additional arguments needed by the model.

        Returns
        -------
        np.float64 or np.ndarray
        """
        a0 = reshape_1d_arg(a0)
        return super().median(*(a0, *args))

    @override
    def var(self, a0: AnyFloat, *args: *Ts) -> NumpyFloat:
        """
        The variance.

        Parameters
        ----------
        a0 : float or np.ndarray
            Conditional age values. It represents ages reached by assets. If ndarray, shape can only be (m,)
            as only one age per asset can be given
        *args : float or np.ndarray
            Additional arguments needed by the model.

        Returns
        -------
        np.float64 or np.ndarray
        """
        a0 = reshape_1d_arg(a0)
        return super().var(*(a0, *args))

    @override
    def moment(self, n: int, a0: AnyFloat, *args: *Ts) -> NumpyFloat:
        """
        n-th order moment

        Parameters
        ----------
        n : int
            Order of the moment, at least 1
        a0 : float or np.ndarray
            Conditional age values. It represents ages reached by assets. If ndarray, shape can only be (m,)
            as only one age per asset can be given
        *args : float or np.ndarray
            Additional arguments needed by the model.

        Returns
        -------
        np.float64 or np.ndarray
        """
        a0 = reshape_1d_arg(a0)
        return super().moment(n, *(a0, *args))

    @override
    def mrl(self, time: AnyFloat, a0: AnyFloat, *args: *Ts) -> NumpyFloat:
        """
        The mean residual life function.

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
        a0 = reshape_1d_arg(a0)
        return super().mrl(time, *(a0, *args))

    @override
    def ppf(self, probability: AnyFloat, a0: AnyFloat, *args: *Ts) -> NumpyFloat:
        """
        The percent point function.

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

        Notes
        -----
        The ``ppf`` is the inverse of :py:meth:`~LeftTruncatedModel.cdf`.

        """
        a0 = reshape_1d_arg(a0)
        return super().ppf(probability, *(a0, *args))

    def freeze(self, a0: AnyFloat, *args: *Ts) -> FrozenParametricLifetimeModel[*tuple[AnyFloat, *Ts]]:
        """
        Freeze conditional age values and other arguments into the object data.

        Parameters
        ----------
        a0 : float or np.ndarray
            Conditional age values. It represents ages reached by assets. If ndarray, shape can only be (m,)
            as only one age per asset can be given
        *args : float or np.ndarray
            Additional arguments needed by the model.

        Returns
        -------
        FrozenLeftTruncatedModel
        """
        return FrozenParametricLifetimeModel(self, a0, *args)
