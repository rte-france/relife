"""Base classes for all parametric lifetime models."""

from __future__ import annotations

import copy
import functools
from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    Literal,
    ParamSpec,
    Self,
    TypedDict,
    TypeVar,
    TypeVarTuple,
    Unpack,
    overload,
    final,
)

import numpy as np
import numpydoc.docscrape as docscrape  # pyright: ignore[reportMissingTypeStubs]
from numpy.typing import NDArray
from optype.numpy import Array1D, Array2D, ToFloat
from scipy.optimize import approx_fprime, newton, Bounds
from typing_extensions import override

from relife.base import (
    FittingResults,
    FrozenParametricModel,
    MaximumLikehoodOptimizer,
    ParametricModel,
)
from relife.typing import (
    AnyFloat,
    NumpyBool,
    NumpyFloat,
    ScipyMinimizeOptions,
    Seed,
)
from relife.utils import reshape_1d_arg
from relife.utils.quadrature import legendre_quadrature, unweighted_laguerre_quadrature

if TYPE_CHECKING:
    pass

Ts = TypeVarTuple("Ts")


class ParametricLifetimeModel(ParametricModel, ABC, Generic[*Ts]):
    r"""Base class for parametric lifetime models in ReLife.

    This class is a blueprint for implementing parametric lifetime models. The
    interface is generic and can define a variadic set of arguments. It expects
    implementation of the hazard function (`hf`), the cumulative hazard
    function (`chf`), the probability density function (`pdf`) and the survival
    function (`sf`). Other functions are implemented by default but can be
    overridden by the derived classes.

    Note:
        The abstract methods also provides a default implementation. One may
        not have to implement `hf`, `chf`, `pdf` and `sf` and just call
        `super()` to access the base implementation.

    Methods:
        hf: Abstract method to compute the hazard function.
        chf: Abstract method to compute the cumulative hazard function.
        sf: Abstract method to compute the survival function.
        pdf: Abstract method to compute the probability density function.
    """

    @abstractmethod
    def sf(self, time: AnyFloat, *args: *Ts) -> NumpyFloat:
        """
        The survival function.

        Parameters
        ----------
        time : float or np.ndarray
            Elapsed time value(s) at which to compute the function.
            If ndarray, allowed shapes are `()`, `(n,)` or `(m, n)`.
        *args
            Any additonal args.

        Returns
        -------
        out : np.float64 or np.ndarray
            sf values at each given time(s).
        """
        if hasattr(self, "chf"):
            return np.exp(
                -self.chf(
                    time,
                    *args,
                )
            )
        elif hasattr(self, "pdf") and hasattr(self, "hf"):
            return self.pdf(time, *args) / self.hf(time, *args)
        else:
            class_name = type(self).__name__
            raise NotImplementedError(
                f"""
                {class_name} must implement concrete sf function
                """
            )

    @abstractmethod
    def hf(self, time: AnyFloat, *args: *Ts) -> NumpyFloat:
        """
        The hazard function.

        Parameters
        ----------
        time : float or np.ndarray
            Elapsed time value(s) at which to compute the function.
            If ndarray, allowed shapes are `()`, `(n,)` or `(m, n)`.
        *args
            Any additonal args.

        Returns
        -------
        out : np.float64 or np.ndarray
            hf values at each given time(s).
        """
        if hasattr(self, "pdf") and hasattr(self, "sf"):
            return self.pdf(time, *args) / self.sf(time, *args)
        else:
            class_name = type(self).__name__
            raise NotImplementedError(
                f"""
                    {class_name} must implement concrete hf function
                    """
            )

    @abstractmethod
    def chf(self, time: AnyFloat, *args: *Ts) -> NumpyFloat:
        """
        The cumulative hazard function.

        Parameters
        ----------
        time : float or np.ndarray
            Elapsed time value(s) at which to compute the function.
            If ndarray, allowed shapes are `()`, `(n,)` or `(m, n)`.
        *args
            Any additonal args.

        Returns
        -------
        out : np.float64 or np.ndarray
            chf values at each given time(s).
        """
        if hasattr(self, "sf"):
            return -np.log(self.sf(time, *args))
        elif hasattr(self, "pdf") and hasattr(self, "hf"):
            return -np.log(self.pdf(time, *args) / self.hf(time, *args))
        else:
            class_name = type(self).__name__
            raise NotImplementedError(
                f"""
                {class_name} must implement concrete chf or at least concrete hf function
                """
            )

    @abstractmethod
    def pdf(self, time: AnyFloat, *args: *Ts) -> NumpyFloat:
        """
        The probability density function.

        Parameters
        ----------
        time : float or np.ndarray
            Elapsed time value(s) at which to compute the function.
            If ndarray, allowed shapes are `()`, `(n,)` or `(m, n)`.
        *args
            Any additonal args.

        Returns
        -------
        out : np.float64 or np.ndarray
            pdf values at each given time(s).
        """
        try:
            return self.sf(time, *args) * self.hf(time, *args)
        except NotImplementedError as err:
            class_name = type(self).__name__
            raise NotImplementedError(
                f"""
            {class_name} must implement pdf or the above functions
            """
            ) from err

    def cdf(self, time: AnyFloat, *args: *Ts) -> NumpyFloat:
        """
        The cumulative density function.

        Parameters
        ----------
        time : float or np.ndarray
            Elapsed time value(s) at which to compute the function.
            If ndarray, allowed shapes are `()`, `(n,)` or `(m, n)`.
        *args
            Any additonal args.

        Returns
        -------
        out : np.float64 or np.ndarray
            cdf values at each given time(s).
        """
        return 1 - self.sf(time, *args)

    def ppf(self, probability: AnyFloat, *args: *Ts) -> NumpyFloat:
        """
        The percent point function.

        Parameters
        ----------
        probability : float or np.ndarray
            Probability value(s) at which to compute the function.
            If ndarray, allowed shapes are `()`, `(n,)` or `(m, n)`.
        *args
            Any additonal args.

        Returns
        -------
        out : np.float64 or np.ndarray
            ppf values at each given probability value(s).
        """
        probability = np.asarray(probability)
        return self.isf(1 - probability, *args)

    def median(self, *args: *Ts) -> NumpyFloat:
        """
        The median.

        Parameters
        ----------
        *args
            Any additonal args.

        Returns
        -------
        out : np.float64 or np.ndarray
        """
        return self.ppf(0.5, *args)

    def isf(self, probability: AnyFloat, *args: *Ts) -> NumpyFloat:
        """
        The inverse survival function.

        Parameters
        ----------
        probability : float or np.ndarray
            Probability value(s) at which to compute the function.
            If ndarray, allowed shapes are `()`, `(n,)` or `(m, n)`.
        *args
            Any additonal args.

        Returns
        -------
        out : np.float64 or np.ndarray
            isf values at each given probability value(s).
        """

        def func(x: NDArray[np.float64]) -> NumpyFloat:
            return self.sf(x, *args) - probability

        # no idea on how to type func
        return newton(  # pyright: ignore[reportCallIssue, reportUnknownVariableType]
            func,  # pyright: ignore[reportArgumentType]
            x0=np.zeros_like(probability),
            args=args,
        )

    def ichf(self, cumulative_hazard_rate: AnyFloat, *args: *Ts) -> NumpyFloat:
        """
        Inverse cumulative hazard function.

        Parameters
        ----------
        cumulative_hazard_rate : float or np.ndarray
            Cumulative hazard rate value(s) at which to compute the function.
            If ndarray, allowed shapes are `()`, `(n,)` or `(m, n)`.
        *args
            Any additonal args.

        Returns
        -------
        out : np.float64 or np.ndarray
            ichf values at each given cumulative hazard rate(s).
        """

        def func(x: NDArray[np.float64]) -> NumpyFloat:
            return self.chf(x, *args) - cumulative_hazard_rate

        # no idea on how to type func
        return newton(  # pyright: ignore[reportCallIssue, reportUnknownVariableType]
            func,  # pyright: ignore[reportArgumentType]
            x0=np.zeros_like(cumulative_hazard_rate),
        )

    @overload
    def rvs(
        self,
        size: int | tuple[int, int],
        *args: *Ts,
        return_event: Literal[False],
        return_entry: Literal[False],
        seed: Seed | None = None,
    ) -> NumpyFloat: ...
    @overload
    def rvs(
        self,
        size: int | tuple[int, int],
        *args: *Ts,
        return_event: Literal[True],
        return_entry: Literal[False],
        seed: Seed | None = None,
    ) -> tuple[NumpyFloat, NumpyBool]: ...
    @overload
    def rvs(
        self,
        size: int | tuple[int, int],
        *args: *Ts,
        return_event: Literal[False],
        return_entry: Literal[True],
        seed: Seed | None = None,
    ) -> tuple[NumpyFloat, NumpyFloat]: ...
    @overload
    def rvs(
        self,
        size: int | tuple[int, int],
        *args: *Ts,
        return_event: Literal[True],
        return_entry: Literal[True],
        seed: Seed | None = None,
    ) -> tuple[NumpyFloat, NumpyBool, NumpyFloat]: ...
    @overload
    def rvs(
        self,
        size: int | tuple[int, int],
        *args: *Ts,
        return_event: bool = False,
        return_entry: bool = False,
        seed: Seed | None = None,
    ) -> (
        NumpyFloat
        | tuple[NumpyFloat, NumpyBool]
        | tuple[NumpyFloat, NumpyFloat]
        | tuple[NumpyFloat, NumpyBool, NumpyFloat]
    ): ...
    def rvs(
        self,
        size: int | tuple[int, int],
        *args: *Ts,
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
        size : int or tuple (m, n) of int
            Size of the generated sample.
        *args
            Any additonal args.
        return_event : bool, default is False
            If True, returns event indicators along with the sample time
            values.
        return_entry : bool, default is False
            If True, returns corresponding entry values of the sample time
            values.
        seed : optional int, np.random.BitGenerator, np.random.Generator, np.random.RandomState, default is None
            If int or BitGenerator, seed for random number generator. If
            np.random.RandomState or np.random.Generator, use as given.

        Returns
        -------
        out : float, ndarray or tuple of float or ndarray
            The sample values. If either `return_event` or `return_entry` is
            True, returns a tuple containing the time values followed by event
            values, entry values or both.
        """
        rng = np.random.default_rng(seed)
        probability = rng.uniform(size=size)
        if size == 1:
            probability = np.squeeze(probability)
        time = self.isf(probability, *args)
        event = (
            np.ones_like(time, dtype=np.bool_)
            if isinstance(time, np.ndarray)
            else np.bool_(True)
        )
        entry = (
            np.zeros_like(time, dtype=np.float64)
            if isinstance(time, np.ndarray)
            else np.float64(0)
        )
        if not return_event and not return_entry:
            return time
        elif return_event and not return_entry:
            return time, event
        elif not return_event and return_entry:
            return time, entry
        else:
            return time, event, entry

    def ls_integrate(
        self,
        func: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        a: AnyFloat,
        b: AnyFloat,
        *args: *Ts,
        deg: int = 10,
    ) -> NumpyFloat:
        """
        Lebesgue-Stieltjes integration.

        Parameters
        ----------
        func : callable (in : 1 ndarray , out : 1 ndarray)
            The callable must have only one ndarray object as argument and one
            ndarray object as output.
        a : ndarray (maximum number of dimension is 2)
            Lower bound(s) of integration.
        b : ndarray (maximum number of dimension is 2)
            Upper bound(s) of integration. If lower bound(s) is infinite, use
            np.inf as value.
        *args
            Any additonal args.
        deg : int, default 10
            Degree of the polynomials interpolation.

        Returns
        -------
        out : np.ndarray
            Lebesgue-Stieltjes integral of func from `a` to `b`.
        """

        def integrand(x: NDArray[np.float64]) -> NDArray[np.float64]:
            #  x.shape == (deg,), (deg, n) or (deg, m, n), ie points of quadratures
            # fx : (d_1, ..., d_i, deg), (d_1, ..., d_i, deg, n) or (d_1, ..., d_i, deg, m, n)
            x = np.asarray(x)
            fx = func(x)

            try:
                _ = np.broadcast_shapes(fx.shape[-len(x.shape) :], x.shape)
            except ValueError:
                raise ValueError(
                    """
                    func can't squeeze input dimensions. If x has shape (d_1, ..., d_i), func(x) must have shape (..., d_1, ..., d_i).
                    Ex : if x.shape == (m, n), func(x).shape == (..., m, n).
                    """
                )
            if (
                x.ndim == 3
            ):  # reshape because model.pdf is tested only for input ndim <= 2
                x_shape: tuple[int, int, int] = x.shape
                xdeg, m, n = x_shape
                x = np.rollaxis(x, 1).reshape(
                    m, -1
                )  # (m, deg*n), roll on m because axis 0 must align with m of args
                pdf = self.pdf(x, *args)  # (m, deg*n)
                pdf = np.rollaxis(pdf.reshape(m, xdeg, n), 1, 0)  #  (deg, m, n)
            else:  # ndim == 1 | 2
                # reshape to (1, deg*n) or (1, deg), ie place 1 on axis 0 to allow broadcasting with m of args
                pdf = self.pdf(x.reshape(1, -1), *args)  # (1, deg*n) or (1, deg)
                pdf = pdf.reshape(x.shape)  # (deg, n) or (deg,)

            # (d_1, ..., d_i, deg) or (d_1, ..., d_i, deg, n) or (d_1, ..., d_i, deg, m, n)
            return fx * pdf

        arr_a, arr_b = np.broadcast_arrays(a, b)  # (), (n,) or (m, n)
        if np.any(arr_a > arr_b):
            raise ValueError("Bound values a must be lower than values of b")

        bound_b = self.isf(
            1e-4, *args
        )  #  () or (m, 1), if (m, 1) then arr_b.shape == (m, 1) or (m, n)
        broadcasted_arrs = np.broadcast_arrays(arr_a, arr_b, bound_b)
        arr_a = broadcasted_arrs[
            0
        ].copy()  # arr_a.shape == arr_b.shape == bound_b.shape
        arr_b = broadcasted_arrs[
            1
        ].copy()  # arr_a.shape == arr_b.shape == bound_b.shape
        bound_b = broadcasted_arrs[
            2
        ].copy()  # arr_a.shape == arr_b.shape == bound_b.shape
        is_inf = np.isinf(arr_b)  # () or (n,) or (m, n)
        arr_b = np.where(is_inf, bound_b, arr_b)
        integration = legendre_quadrature(
            integrand, arr_a, arr_b, deg=deg
        )  #  (d_1, ..., d_i), (d_1, ..., d_i, n) or (d_1, ..., d_i, m, n)
        is_inf, _ = np.broadcast_arrays(is_inf, integration)
        return np.where(
            is_inf,
            integration + unweighted_laguerre_quadrature(integrand, arr_b, deg=deg),
            integration,
        )

    def moment(self, n: int, *args: *Ts) -> NumpyFloat:
        """
        n-th order moment.

        Parameters
        ----------
        n : int
            order of the moment, at least 1.
        *args
            Any additonal args.

        Returns
        -------
        out : np.float64
        """
        if n < 1:
            raise ValueError("order of the moment must be at least 1")

        def func(x: NDArray[np.float64]) -> NDArray[np.float64]:
            return np.power(x, n)

        return self.ls_integrate(
            func,
            0.0,
            np.inf,
            *args,
            deg=100,
        )  #  high degree of polynome to ensure high precision

    def mean(self, *args: *Ts) -> NumpyFloat:
        """
        The mean of the distribution.

        Parameters
        ----------
        *args
            Any additonal args.

        Returns
        -------
        out : np.float64 or np.ndarray
        """
        return self.moment(1, *args)

    def var(self, *args: *Ts) -> NumpyFloat:
        """
        The variance of the distribution.

        Parameters
        ----------
        *args
            Any additonal args.

        Returns
        -------
        out : np.float64 or np.ndarray
        """
        return self.moment(2, *args) - self.moment(1, *args) ** 2

    def mrl(self, time: AnyFloat, *args: *Ts) -> NumpyFloat:
        """
        The mean residual life function.

        Parameters
        ----------
        time : float or np.ndarray
            Elapsed time value(s) at which to compute the function.
            If ndarray, allowed shapes are `()`, `(n,)` or `(m, n)`.
        *args
            Any additonal args.

        Returns
        -------
        out : np.float64 or np.ndarray
            Function values at each given time(s).
        """

        sf = self.sf(time, *args)

        def func(x: NDArray[np.float64]) -> NDArray[np.float64]:
            return np.asarray(x) - time

        ls = self.ls_integrate(func, time, np.inf, *args)
        if sf.ndim < 2:  # 2d to 1d or 0d
            ls = np.squeeze(ls)
        return ls / sf


class FrozenParametricLifetimeModel(
    FrozenParametricModel[ParametricLifetimeModel[*Ts], *Ts]
):
    _args: tuple[*Ts]
    _unfrozen_model: ParametricLifetimeModel[*Ts]

    def __init__(self, model: ParametricLifetimeModel[*Ts], *args: *Ts) -> None:
        super().__init__(model, *args)

    def sf(self, time: AnyFloat) -> NumpyFloat:
        return self._unfrozen_model.sf(time, *self._args)

    def hf(self, time: AnyFloat) -> NumpyFloat:
        return self._unfrozen_model.hf(time, *self._args)

    def chf(self, time: AnyFloat) -> NumpyFloat:
        return self._unfrozen_model.chf(time, *self._args)

    def pdf(self, time: AnyFloat) -> NumpyFloat:
        return self._unfrozen_model.pdf(time, *self._args)

    def cdf(self, time: AnyFloat) -> NumpyFloat:
        return self._unfrozen_model.cdf(time, *self._args)

    def ppf(self, probability: AnyFloat) -> NumpyFloat:
        return self._unfrozen_model.ppf(probability, *self._args)

    def median(self) -> NumpyFloat:
        return self._unfrozen_model.median(*self._args)

    def isf(self, probability: AnyFloat) -> NumpyFloat:
        return self._unfrozen_model.isf(probability, *self._args)

    def ichf(self, cumulative_hazard_rate: AnyFloat) -> NumpyFloat:
        return self._unfrozen_model.ichf(cumulative_hazard_rate, *self._args)

    @overload
    def rvs(
        self,
        size: int | tuple[int, int],
        return_event: Literal[False],
        return_entry: Literal[False],
        seed: Seed | None = None,
    ) -> NumpyFloat: ...
    @overload
    def rvs(
        self,
        size: int | tuple[int, int],
        return_event: Literal[True],
        return_entry: Literal[False],
        seed: Seed | None = None,
    ) -> tuple[NumpyFloat, NumpyBool]: ...
    @overload
    def rvs(
        self,
        size: int | tuple[int, int],
        return_event: Literal[False],
        return_entry: Literal[True],
        seed: Seed | None = None,
    ) -> tuple[NumpyFloat, NumpyFloat]: ...
    @overload
    def rvs(
        self,
        size: int | tuple[int, int],
        return_event: Literal[True],
        return_entry: Literal[True],
        seed: Seed | None = None,
    ) -> tuple[NumpyFloat, NumpyBool, NumpyFloat]: ...
    @overload
    def rvs(
        self,
        size: int | tuple[int, int],
        return_event: bool = False,
        return_entry: bool = False,
        seed: Seed | None = None,
    ) -> (
        NumpyFloat
        | tuple[NumpyFloat, NumpyBool]
        | tuple[NumpyFloat, NumpyFloat]
        | tuple[NumpyFloat, NumpyBool, NumpyFloat]
    ): ...
    def rvs(
        self,
        size: int | tuple[int, int],
        return_event: bool = False,
        return_entry: bool = False,
        seed: Seed | None = None,
    ) -> (
        NumpyFloat
        | tuple[NumpyFloat, NumpyBool]
        | tuple[NumpyFloat, NumpyFloat]
        | tuple[NumpyFloat, NumpyBool, NumpyFloat]
    ):
        return self._unfrozen_model.rvs(
            size,
            *self._args,
            return_event=return_event,
            return_entry=return_entry,
            seed=seed,
        )

    def ls_integrate(
        self,
        func: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        a: NumpyFloat,
        b: NumpyFloat,
        *,
        deg: int = 10,
    ) -> NumpyFloat:
        return self._unfrozen_model.ls_integrate(func, a, b, *self._args, deg=deg)

    def moment(self, n: int) -> NumpyFloat:
        return self._unfrozen_model.moment(n, *self._args)

    def mean(self) -> NumpyFloat:
        return self._unfrozen_model.mean(*self._args)

    def var(self) -> NumpyFloat:
        return self._unfrozen_model.var(*self._args)

    def mrl(self, time: AnyFloat) -> NumpyFloat:
        return self._unfrozen_model.mrl(time, *self._args)


@overload
def is_lifetime_model(model: FrozenParametricLifetimeModel[*Ts]) -> Literal[True]: ...
@overload
def is_lifetime_model(
    model: ParametricLifetimeModel[*Ts]
    | FrozenParametricLifetimeModel[ParametricModel, *Ts],
) -> Literal[True]: ...
@overload
def is_lifetime_model(
    model: Any
    | ParametricLifetimeModel[*Ts]
    | FrozenParametricLifetimeModel[ParametricModel, *Ts],
) -> bool: ...
def is_lifetime_model(
    model: Any
    | ParametricLifetimeModel[*Ts]
    | FrozenParametricLifetimeModel[ParametricModel, *Ts],
) -> bool:
    """
    Checks if model is a lifetime model.
    """
    return isinstance(model, (ParametricLifetimeModel, FrozenParametricLifetimeModel))


P = ParamSpec("P")
T = TypeVar("T")


def document_args(
    *,
    base_cls: type,
    args_docstring: list[docscrape.Parameter],
    returns: list[docscrape.Parameter] | None = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    def decorator_extend_docstring(
        method: Callable[P, T],
    ) -> Callable[P, T]:
        base_doc = getattr(base_cls, method.__name__).__doc__
        numpy_docstring = docscrape.NumpyDocString(base_doc)
        new_parameters_docstring: list[docscrape.Parameter] = []
        for param in numpy_docstring["Parameters"]:  # pyright: ignore[reportUnknownVariableType]
            if param.name != "*args":  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
                new_parameters_docstring.append(param)  # pyright: ignore[reportArgumentType]
            else:
                new_parameters_docstring += args_docstring
        numpy_docstring["Parameters"] = new_parameters_docstring
        if returns is not None:
            numpy_docstring["Returns"] = returns
        method.__doc__ = str(numpy_docstring)

        @functools.wraps(method)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            return method(*args, **kwargs)

        return wrapper

    return decorator_extend_docstring


class FittableParametricLifetimeModel(ParametricLifetimeModel[*Ts], ABC):
    fitting_results: FittingResults | None
    approx_hessian_method: Literal["2point", "cs"] = "cs"

    def __init__(self, **kwparams: float | None):
        super().__init__(**kwparams)
        self.fitting_results = None

    @abstractmethod
    def jac_hf(
        self,
        time: AnyFloat,
        *args: *Ts,
    ) -> NumpyFloat:
        """
        The jacobian of the hazard function.

        Parameters
        ----------
        time : float or np.ndarray
            Elapsed time value(s) at which to compute the function.
            If ndarray, allowed shapes are `()`, `(n,)` or `(m, n)`.
        *args
            Any additonal args.

        Returns
        -------
        out : np.float64 or np.ndarray
            The derivatives with respect to each parameter. If the result is
            an `np.ndarray`, the first dimension holds the number of parameters.
        """

    @abstractmethod
    def jac_chf(self, time: AnyFloat, *args: *Ts) -> NumpyFloat:
        """
        The jacobian of the cumulative hazard function.

        Parameters
        ----------
        time : float or np.ndarray
            Elapsed time value(s) at which to compute the function.
            If ndarray, allowed shapes are `()`, `(n,)` or `(m, n)`.
        *args
            Any additonal args.

        Returns
        -------
        out : np.float64 or np.ndarray
            The derivatives with respect to each parameter. If the result is
            an `np.ndarray`, the first dimension holds the number of parameters.
        """

    @abstractmethod
    def jac_sf(self, time: AnyFloat, *args: *Ts) -> NumpyFloat:
        """
        The jacobian of the survival function.

        Parameters
        ----------
        time : float or np.ndarray
            Elapsed time value(s) at which to compute the function.
            If ndarray, allowed shapes are `()`, `(n,)` or `(m, n)`.
        *args
            Any additonal args.

        Returns
        -------
        out : np.float64 or np.ndarray
            The derivatives with respect to each parameter. If the result is
            an `np.ndarray`, the first dimension holds the number of parameters.
        """

    def jac_cdf(self, time: AnyFloat, *args: *Ts) -> NumpyFloat:
        """
        The jacobian of the cumulative density function.

        Parameters
        ----------
        time : float or np.ndarray
            Elapsed time value(s) at which to compute the function.
            If ndarray, allowed shapes are `()`, `(n,)` or `(m, n)`.
        *args
            Any additonal args.

        Returns
        -------
        out : np.float64 or np.ndarray
            The derivatives with respect to each parameter. If the result is
            an `np.ndarray`, the first dimension holds the number of parameters.
        """
        return -self.jac_sf(time, *args)

    @abstractmethod
    def jac_pdf(self, time: AnyFloat, *args: *Ts) -> NumpyFloat:
        """
        The jacobian of the probability density function.

        Parameters
        ----------
        time : float or np.ndarray
            Elapsed time value(s) at which to compute the function.
            If ndarray, allowed shapes are `()`, `(n,)` or `(m, n)`.
        *args
            Any additonal args.

        Returns
        -------
        out : np.float64 or np.ndarray
            The derivatives with respect to each parameter. If the result is
            an `np.ndarray`, the first dimension holds the number of parameters.

        """

    @abstractmethod
    def dhf(self, time: AnyFloat, *args: *Ts) -> NumpyFloat:
        """
        The derivate of the hazard function.

        Parameters
        ----------
        time : float or np.ndarray
            Elapsed time value(s) at which to compute the function.
            If ndarray, allowed shapes are `()`, `(n,)` or `(m, n)`.
        *args
            Any additonal args.

        Returns
        -------
        out : np.float64 or np.ndarray
            Function values at each given time(s).
        """

    @abstractmethod
    def fit(
        self,
        time: NDArray[np.float64],
        model_args: NDArray[Any] | tuple[NDArray[Any], ...] | None = None,
        event: NDArray[np.bool_] | None = None,
        entry: NDArray[np.float64] | None = None,
        **optimizer_options: Unpack[ScipyMinimizeOptions],
    ) -> Self:
        """
        Estimation of the distribution parameters from lifetime data.

        Parameters
        ----------
        time : 1d array
            Observed lifetime values.
        model_args : any ndarray or tuple of ndarray, default is None
            Any additional arguments required by the model.
        event : 1d array of bool, default is None
            Boolean indicators tagging lifetime values as right censored or complete.
        entry : 1d array, default is None
            Left truncations applied to lifetime values.
        optimizer_options : dict, default is None
            Extra arguments used by `scipy.minimize`. Default values are:
                - `method` : `"L-BFGS-B"`
                - `contraints` : `()`
                - `tol` : `None`
                - `callback` : `None`
                - `options` : `None`
                - `bounds` : `self.params_bounds`
                - `x0` : `self.init_params`

        Returns
        -------
        out : the object instance
            The estimated parameters are setted inplace.
        """


class LifetimeData(TypedDict):
    complete_time: NDArray[np.float64]
    censored_time: NDArray[np.float64]  # 1d array or 2d
    left_truncations: NDArray[np.float64]
    complete_time_args: tuple[NDArray[Any], ...]
    censored_time_args: tuple[NDArray[Any], ...]
    left_truncations_args: tuple[NDArray[Any], ...]
    nb_observations: int


@final
class LifetimeLikelihood(MaximumLikehoodOptimizer[FittableParametricLifetimeModel, LifetimeData]):
    """
    Likelihood from lifetime data.

    Parameters
    ----------
    model : generic FittableParametricLifetimeModel
        All model parameters must exist first. Its values are initialized by
        model.fit with respect to data.
    time: numpy array of lifetime durations
    model_args: numpy array or tuple thereof with additional model arguments (e.g. covar)
    event: numpy array of boolean indicating event occurrence or not
    entry: numpy array with assets lifetime duration at the beginning of observation

    Attributes
    ----------
    model: a copy of the original model object
    data: a LifetimeData object with processed data information for model fitting purposes
    nb_observations: number of samples
    """

    model: FittableParametricLifetimeModel
    data: LifetimeData

    def __init__(
        self,
        model: FittableParametricLifetimeModel,
        time: NDArray[np.float64],
        model_args: NDArray[Any] | tuple[NDArray[Any], ...] | None = None,
        event: NDArray[np.bool_] | None = None,
        entry: NDArray[np.float64] | None = None,
    ):
        self.model = copy.deepcopy(model)
        self.data = _init_lifetime_data(
            time, model_args=model_args, event=event, entry=entry
        )

    @property
    @override
    def nb_observations(self) -> int:
        return self.data["nb_observations"]

    @override
    def negative_log(self, params: Array1D[np.float64]) -> ToFloat:
        self.model.params = params
        return (
            _complete_time_contrib(self.model, self.data)
            + _censored_time_contrib(self.model, self.data)
            + _left_truncations_contrib(self.model, self.data)
        )

    def jac_negative_log(self, params: Array1D[np.float64]) -> Array1D[np.float64]:
        """
        Jacobian of the negative log likelihood.

        The jacobian is computed with respect to parameters.

        Parameters
        ----------
        model : parametric model
            A parametrized model with appropriate parameters values.

        Returns
        -------
        out : ndarray
        """
        self.model.params = params
        return (
            _jac_complete_time_contrib(self.model, self.data)
            + _jac_censored_time_contrib(self.model, self.data)
            + _jac_left_truncations_contrib(self.model, self.data)
        )

    @override
    def maximum_likelihood_estimation(
        self, **optimizer_options: Unpack[ScipyMinimizeOptions]
    ) -> FittingResults:
        if "jac" not in optimizer_options:
            optimizer_options["jac"] = self.jac_negative_log
        return super().maximum_likelihood_estimation(**optimizer_options)

        # hessian = approx_hessian(self, fitting_results.optimal_params)
        # fitting_results.covariance_matrix = np.linalg.pinv(hessian)
        # return fitting_results


def approx_parameters_covariance(
    likelihood: LifetimeLikelihood,
    optimal_params: NDArray[np.float64],
    method: Literal["2point", "cs"] = "cs",
    eps: float = 1e-6,
) -> Array2D[np.float64]:
    size = optimal_params.size
    hess = np.empty((size, size))

    # hessian 2 point
    if method == "2point":
        for i in range(size):
            hess[i] = approx_fprime(
                optimal_params,
                lambda x: likelihood.jac_negative_log(x)[i],
                eps,
            )
        return hess
    # hessian cs
    u = eps * 1j * np.eye(size)
    complex_params = optimal_params.astype(np.complex64)  # change params to complex
    for i in range(size):
        for j in range(i, size):
            hess[i, j] = (
                np.imag(likelihood.jac_negative_log(complex_params + u[i])[j]) / eps
            )
            if i != j:
                hess[j, i] = hess[i, j]
    covariance_matrix = np.linalg.pinv(hess).astype(np.float64)
    return covariance_matrix


def _init_lifetime_data(
    time: NDArray[np.float64],  # 1d array or 2d
    model_args: NDArray[Any] | tuple[NDArray[Any], ...] | None = None,
    event: NDArray[np.bool_] | None = None,
    entry: NDArray[np.float64] | None = None,
) -> LifetimeData:
    time = reshape_1d_arg(time)
    if time.shape[-1] == 2 and event is not None:
        raise ValueError("If time is given as intervals, event must be None")
    if time.shape[-1] == 1:
        event = (
            reshape_1d_arg(event)
            if event is not None
            else np.ones_like(time, dtype=np.bool_)
        )
    entry = (
        reshape_1d_arg(entry)
        if entry is not None
        else np.zeros(len(time), dtype=np.float64)
    )
    if np.any(time <= entry):
        raise ValueError("All time values must be greater than entry values")
    if isinstance(model_args, tuple):
        args = tuple((reshape_1d_arg(arg) for arg in model_args))
    elif isinstance(model_args, np.ndarray):
        args = (reshape_1d_arg(model_args),)
    elif model_args is None:
        args = ()
    sizes = [len(x) for x in (time, event, entry, *args) if x is not None]
    if len(set(sizes)) != 1:
        raise ValueError(
            f"""
            All lifetime data must have the same number of values. Fields
            length are different. Got {tuple(sizes)}
            """
        )
    non_zero_entry = np.flatnonzero(entry)
    if event is not None:
        non_zero_event = np.flatnonzero(event)
        zero_event = np.flatnonzero(event == 0)
        data = LifetimeData(
            complete_time=time[non_zero_event],
            censored_time=time[zero_event],
            left_truncations=entry[non_zero_entry],
            complete_time_args=tuple(arg[non_zero_event] for arg in args),
            censored_time_args=tuple(arg[zero_event] for arg in args),
            left_truncations_args=tuple(arg[non_zero_entry] for arg in args),
            nb_observations=time.size,
        )
        return data

    complete_time_index = np.flatnonzero(time[:, 0] == time[:, 1])
    non_complete_time_index = np.flatnonzero(time[:, 0] != time[:, 1])
    data = LifetimeData(
        complete_time=time[:, 1][complete_time_index],
        censored_time=time[non_complete_time_index],
        left_truncations=entry[non_zero_entry],
        complete_time_args=tuple(arg[complete_time_index] for arg in args),
        censored_time_args=tuple(arg[non_complete_time_index] for arg in args),
        left_truncations_args=tuple(arg[non_zero_entry] for arg in args),
        nb_observations=time.size,
    )
    return data


def _complete_time_contrib(
    model: FittableParametricLifetimeModel[*tuple[Any, ...]],
    data: LifetimeData,
) -> float:
    if data["complete_time"].size == 0.0:
        return 0.0
    return -np.sum(
        np.log(model.pdf(data["complete_time"], *data["complete_time_args"]))
    )


def _jac_complete_time_contrib(
    model: FittableParametricLifetimeModel[*tuple[Any, ...]],
    data: LifetimeData,
) -> NDArray[np.float64]:
    if data["complete_time"].size == 0:
        return np.zeros_like(model.params)
    jac = -model.jac_pdf(
        data["complete_time"], *data["complete_time_args"]
    ) / model.pdf(data["complete_time"], *data["complete_time_args"])

    return np.sum(jac, axis=(1, 2))


def _censored_time_contrib(
    model: FittableParametricLifetimeModel[*tuple[Any, ...]],
    data: LifetimeData,
) -> float:
    if data["censored_time"].size == 0:
        return 0.0
    if data["censored_time"].shape[-1] > 1:
        # interval censored time
        return np.sum(
            -np.log(
                10**-10
                + model.cdf(data["censored_time"][:, 1], *data["censored_time_args"])
                - model.cdf(data["censored_time"][:, 0], *data["censored_time_args"])
            ),
        )
    else:
        # right censored time
        return np.sum(model.chf(data["censored_time"], *data["censored_time_args"]))


def _jac_censored_time_contrib(
    model: FittableParametricLifetimeModel[*tuple[Any, ...]],
    data: LifetimeData,
) -> NDArray[np.float64]:
    if data["censored_time"].size == 0:
        return np.zeros_like(model.params)
    if data["censored_time"].shape[-1] > 1:
        # interval censored time
        jac_interval_censored = (
            model.jac_sf(data["censored_time"][:, 1], *data["censored_time_args"])
            - model.jac_sf(data["censored_time"][:, 0], *data["censored_time_args"])
        ) / (
            10**-10
            + model.cdf(data["censored_time"][:, 1], *data["censored_time_args"])
            - model.cdf(data["censored_time"][:, 0], *data["censored_time_args"])
        )

        return np.sum(jac_interval_censored, axis=(1, 2))
    else:
        # right censored time
        return np.sum(
            model.jac_chf(data["censored_time"], *data["censored_time_args"]), axis=(1, 2)
        )


def _left_truncations_contrib(
    model: FittableParametricLifetimeModel[*tuple[Any, ...]],
    data: LifetimeData,
) -> float:
    if data["left_truncations"].size == 0.0:
        return 0.0
    return -np.sum(model.chf(data["left_truncations"], *data["left_truncations_args"]))


def _jac_left_truncations_contrib(
    model: FittableParametricLifetimeModel[*tuple[Any, ...]],
    data: LifetimeData,
) -> NDArray[np.float64]:
    if data["left_truncations"].size == 0.0:
        return np.zeros_like(model.params)
    jac = -model.jac_chf(data["left_truncations"], *data["left_truncations_args"])
    return np.sum(jac, axis=(1, 2))
