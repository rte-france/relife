"""Base classes for all parametric lifetime models."""

from __future__ import annotations

import copy
import functools
import inspect
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import field
from typing import (
    Any,
    Generic,
    Literal,
    ParamSpec,
    Self,
    TypeAlias,
    TypeVar,
    TypeVarTuple,
    final,
    overload,
)

import matplotlib.pyplot as plt
import numpy as np
import numpydoc.docscrape as docscrape  # pyright: ignore[reportMissingTypeStubs]
from matplotlib.axes import Axes
from numpy.typing import NDArray
from optype.numpy import (
    Array,
    Array1D,
    Array2D,
    ArrayND,
    is_array_1d,
)
from scipy import stats
from scipy.optimize import newton
from typing_extensions import TypeIs, override

from relife.base import (
    FittingResults,
    MaximumLikelihoodOptimizer,
    OptimizerConfig,
    ParametricModel,
)
from relife.quadratures import legendre_quadrature, unweighted_laguerre_quadrature
from relife.utils import to_column_2d_if_1d

__all__ = [
    "ParametricLifetimeModel",
    "FittableParametricLifetimeModel",
    "LifetimeData",
    "LifetimeLikelihood",
    "is_frozen_parametric_lifetime_model",
]

Ts = TypeVarTuple("Ts")
ST: TypeAlias = int | float
NumpyST: TypeAlias = np.floating | np.uint


@overload
def plot_probability_function(
    x: Array1D[np.float64],
    y: Array1D[np.float64],
    se: Literal[None],
    ci_bounds: Literal[None],
    ax: Axes | None = None,
    **kwargs: Any,
) -> Axes: ...
@overload
def plot_probability_function(
    x: Array1D[np.float64],
    y: Array1D[np.float64],
    se: Array1D[np.float64],
    ci_bounds: tuple[float, float],
    ax: Axes | None = None,
    **kwargs: Any,
) -> Axes: ...
def plot_probability_function(
    x: Array1D[np.float64],
    y: Array1D[np.float64],
    se: Array1D[np.float64] | None = None,
    ci_bounds: tuple[float, float] | None = None,
    ax: Axes | None = None,
    **kwargs: Any,
) -> Axes:
    if ax is None:
        ax = plt.gca()
    ax.plot(x, y, **kwargs)
    if se is not None and ci_bounds is not None:
        alpha_ci = kwargs.get("alpha_ci", 0.95)
        assert isinstance(alpha_ci, float)
        z = stats.norm.ppf((1 + alpha_ci) / 2)
        yl = np.clip(y - z * se, ci_bounds[0], ci_bounds[1])
        yu = np.clip(y + z * se, ci_bounds[0], ci_bounds[1])
        drawstyle = kwargs.get("drawstyle", "default")
        step = drawstyle.split("-")[1] if "steps-" in drawstyle else None
        _ = ax.fill_between(
            x,
            yl,
            yu,
            facecolors=[ax.lines[-1].get_color()],
            step=step,
            alpha=0.25,
            label=f"IC-{alpha_ci}",
        )
        ax.legend()
    if kwargs.get("label") is not None:
        ax.legend()
    ax.set_ylim(bottom=0.0)
    ax.set_xlim(left=0.0, right=np.max(x))
    return ax


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
    def sf(
        self, time: ST | NumpyST | ArrayND[NumpyST], *args: *Ts
    ) -> np.float64 | ArrayND[np.float64]:
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
            return np.divide(self.pdf(time, *args), self.hf(time, *args))
        else:
            class_name = type(self).__name__
            raise NotImplementedError(
                f"""
                {class_name} must implement concrete sf function
                """
            )

    @abstractmethod
    def hf(
        self, time: ST | NumpyST | ArrayND[NumpyST], *args: *Ts
    ) -> np.float64 | ArrayND[np.float64]:
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
            return np.divide(self.pdf(time, *args), self.sf(time, *args))
        else:
            class_name = type(self).__name__
            raise NotImplementedError(
                f"""
                {class_name} must implement concrete hf function.
                """
            )

    @abstractmethod
    def chf(
        self, time: ST | NumpyST | ArrayND[NumpyST], *args: *Ts
    ) -> np.float64 | ArrayND[np.float64]:
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
                {class_name} must implement concrete chf or at least concrete
                hf function.
                """
            )

    @abstractmethod
    def pdf(
        self, time: ST | NumpyST | ArrayND[NumpyST], *args: *Ts
    ) -> np.float64 | ArrayND[np.float64]:
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

    def cdf(
        self, time: ST | NumpyST | ArrayND[NumpyST], *args: *Ts
    ) -> np.float64 | ArrayND[np.float64]:
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

    def ppf(
        self, probability: ST | NumpyST | ArrayND[NumpyST], *args: *Ts
    ) -> np.float64 | ArrayND[np.float64]:
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

    def median(self, *args: *Ts) -> np.float64 | ArrayND[np.float64]:
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

    def isf(
        self, probability: ST | NumpyST | ArrayND[NumpyST], *args: *Ts
    ) -> np.float64 | ArrayND[np.float64]:
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

        def func(x: ArrayND[np.float64]) -> ArrayND[np.float64]:
            return np.asarray(self.sf(x, *args) - probability)

        return newton(  # pyright: ignore[reportCallIssue, reportUnknownVariableType]
            func,  # pyright: ignore[reportArgumentType]
            x0=np.zeros_like(probability),
            args=args,
        )

    def ichf(
        self,
        cumulative_hazard_rate: ST | NumpyST | ArrayND[NumpyST],
        *args: *Ts,
    ) -> np.float64 | ArrayND[np.float64]:
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

        def func(x: NDArray[np.float64]) -> np.float64 | ArrayND[np.float64]:
            return self.chf(x, *args) - cumulative_hazard_rate

        # no idea on how to type func
        return newton(  # pyright: ignore[reportCallIssue, reportUnknownVariableType]
            func,  # pyright: ignore[reportArgumentType]
            x0=np.zeros_like(cumulative_hazard_rate),
        )

    def rvs(
        self,
        size: int | tuple[int, int],
        *args: *Ts,
        seed: int
        | np.random.Generator
        | np.random.BitGenerator
        | np.random.RandomState
        | None = None,
    ) -> np.float64 | ArrayND[np.float64]:
        """
        Random variable sampling.

        Parameters
        ----------
        size : int or tuple (m, n) of int
            Size of the generated sample.
        *args
            Any additonal args.
        seed : optional int, np.random.BitGenerator, np.random.Generator, np.random.RandomState, default is None
            If int or BitGenerator, seed for random number generator. If
            np.random.RandomState or np.random.Generator, use as given.

        Returns
        -------
        out : float or ndarray
            The sample values.
        """  # noqa: E501
        rng = np.random.default_rng(seed)
        probability = rng.uniform(size=size)
        if size == 1:
            probability = np.squeeze(probability)
        return self.isf(probability, *args)

    def ls_integrate(
        self,
        func: Callable[
            [ST | NumpyST | ArrayND[NumpyST]],
            np.float64 | ArrayND[np.float64],
        ],
        a: ST | NumpyST | ArrayND[NumpyST],
        b: ST | NumpyST | ArrayND[NumpyST],
        *args: *Ts,
        deg: int = 10,
    ) -> np.float64 | ArrayND[np.float64]:
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

        def integrand(
            x: ST | NumpyST | ArrayND[NumpyST],
        ) -> np.float64 | ArrayND[np.float64]:
            #  x.shape == (deg,), (deg, n) or (deg, m, n), ie points of quadratures
            # fx : (d_1, ..., d_i, deg), (d_1, ..., d_i, deg, n) or (d_1, ..., d_i, deg, m, n)  # noqa: E501
            x = np.asarray(x)
            fx = func(x)

            try:
                _ = np.broadcast_shapes(fx.shape[-len(x.shape) :], x.shape)
            except ValueError as err:
                raise ValueError(
                    """
                    func can't squeeze input dimensions. If x has shape (d_1, ..., d_i), func(x) must have shape (..., d_1, ..., d_i).
                    Ex : if x.shape == (m, n), func(x).shape == (..., m, n).
                    """  # noqa: E501
                ) from err
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
                # reshape to (1, deg*n) or (1, deg), ie place 1 on axis 0 to allow broadcasting with m of args  # noqa: E501
                pdf = self.pdf(x.reshape(1, -1), *args)  # (1, deg*n) or (1, deg)
                pdf = pdf.reshape(x.shape)  # (deg, n) or (deg,)

            # (d_1, ..., d_i, deg) or (d_1, ..., d_i, deg, n) or (d_1, ..., d_i, deg, m, n)  # noqa: E501
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

    def moment(self, n: int, *args: *Ts) -> np.float64 | ArrayND[np.float64]:
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

        def func(
            x: ST | NumpyST | ArrayND[NumpyST],
        ) -> np.float64 | ArrayND[np.float64]:
            return np.power(x, n, dtype=np.float64)

        return self.ls_integrate(
            func,
            0.0,
            np.inf,
            *args,
            deg=100,
        )  #  high degree of polynome to ensure high precision

    def mean(self, *args: *Ts) -> np.float64 | ArrayND[np.float64]:
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

    def var(self, *args: *Ts) -> np.float64 | ArrayND[np.float64]:
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

    def mrl(
        self, time: ST | NumpyST | ArrayND[NumpyST], *args: *Ts
    ) -> np.float64 | ArrayND[np.float64]:
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

        def func(
            x: ST | NumpyST | ArrayND[NumpyST],
        ) -> np.float64 | ArrayND[np.float64]:
            return np.asarray(x, dtype=np.float64) - time

        ls = self.ls_integrate(func, time, np.inf, *args)
        if sf.ndim < 2:  # 2d to 1d or 0d
            ls = np.squeeze(ls)
        return ls / sf

    def plot(
        self,
        fname: Literal["sf", "cdf", "chf", "hf", "pdf"],
        time: Array1D[np.float64],
        *args: *Ts,
        ax: Axes | None = None,
        **kwargs: Any,
    ) -> Axes:
        if kwargs.get("ci", False) is True:
            raise ValueError("ci is available for fitted models only.")
        y = getattr(self, fname)(time, *args)
        assert is_array_1d(y)  # typeguards
        return plot_probability_function(time, y, ax=ax, **kwargs)


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


class FrozenParametricLifetimeModel(ParametricLifetimeModel[()], Generic[*Ts]):
    args: tuple[*Ts]
    unfrozen: ParametricLifetimeModel[*Ts]

    def __init__(self, model: ParametricLifetimeModel[*Ts], *args: *Ts) -> None:
        super().__init__()
        self.unfrozen = model
        self.args = args

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=[])
    def sf(
        self, time: ST | NumpyST | ArrayND[NumpyST]
    ) -> np.float64 | ArrayND[np.float64]:
        return self.unfrozen.sf(time, *self.args)

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=[])
    def hf(
        self, time: ST | NumpyST | ArrayND[NumpyST]
    ) -> np.float64 | ArrayND[np.float64]:
        return self.unfrozen.hf(time, *self.args)

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=[])
    def chf(
        self, time: ST | NumpyST | ArrayND[NumpyST]
    ) -> np.float64 | ArrayND[np.float64]:
        return self.unfrozen.chf(time, *self.args)

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=[])
    def pdf(
        self, time: ST | NumpyST | ArrayND[NumpyST]
    ) -> np.float64 | ArrayND[np.float64]:
        return self.unfrozen.pdf(time, *self.args)

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=[])
    def cdf(
        self, time: ST | NumpyST | ArrayND[NumpyST]
    ) -> np.float64 | ArrayND[np.float64]:
        return self.unfrozen.cdf(time, *self.args)

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=[])
    def ppf(
        self, probability: ST | NumpyST | ArrayND[NumpyST]
    ) -> np.float64 | ArrayND[np.float64]:
        return self.unfrozen.ppf(probability, *self.args)

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=[])
    def median(self) -> np.float64 | ArrayND[np.float64]:
        return self.unfrozen.median(*self.args)

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=[])
    def isf(
        self, probability: ST | NumpyST | ArrayND[NumpyST]
    ) -> np.float64 | ArrayND[np.float64]:
        return self.unfrozen.isf(probability, *self.args)

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=[])
    def ichf(
        self, cumulative_hazard_rate: ST | NumpyST | ArrayND[NumpyST]
    ) -> np.float64 | ArrayND[np.float64]:
        return self.unfrozen.ichf(cumulative_hazard_rate, *self.args)

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=[])
    def rvs(
        self,
        size: int | tuple[int, int],
        *,
        seed: int
        | np.random.Generator
        | np.random.BitGenerator
        | np.random.RandomState
        | None = None,
    ) -> np.float64 | ArrayND[np.float64]:
        return self.unfrozen.rvs(
            size,
            *self.args,
            seed=seed,
        )

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=[])
    def ls_integrate(
        self,
        func: Callable[
            [ST | NumpyST | ArrayND[NumpyST]],
            np.float64 | ArrayND[np.float64],
        ],
        a: ST | NumpyST | ArrayND[NumpyST],
        b: ST | NumpyST | ArrayND[NumpyST],
        *,
        deg: int = 10,
    ) -> np.float64 | ArrayND[np.float64]:
        return self.unfrozen.ls_integrate(func, a, b, *self.args, deg=deg)

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=[])
    def moment(self, n: int) -> np.float64 | ArrayND[np.float64]:
        return self.unfrozen.moment(n, *self.args)

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=[])
    def mean(self) -> np.float64 | ArrayND[np.float64]:
        return self.unfrozen.mean(*self.args)

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=[])
    def var(self) -> np.float64 | ArrayND[np.float64]:
        return self.unfrozen.var(*self.args)

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=[])
    def mrl(
        self, time: ST | NumpyST | ArrayND[NumpyST]
    ) -> np.float64 | ArrayND[np.float64]:
        return self.unfrozen.mrl(time, *self.args)

    @override
    def __getattr__(self, key: str) -> Any:
        # __getattr__ needed to catch jac_<func> if it exists
        frozen_type = self.unfrozen.__class__.__name__
        try:
            attr = getattr(self.unfrozen, key)
        except AttributeError as err:
            raise AttributeError(
                f"Frozen {frozen_type} has no attribute {key}"
            ) from err

        def wrapper(*args: Any, **kwargs: Any):
            return attr(*(*args, *self.args), **kwargs)

        if inspect.ismethod(attr):
            return wrapper
        return attr


# typeguard, narrowing type
def is_frozen_parametric_lifetime_model(
    model: ParametricLifetimeModel[*tuple[Any, ...]],
) -> TypeIs[FrozenParametricLifetimeModel[*tuple[Any, ...]]]:
    return isinstance(model, FrozenParametricLifetimeModel)


M = TypeVar(
    "M",
    bound="FittableParametricLifetimeModel[*tuple[ST | NumpyST | ArrayND[NumpyST], ...]]",  # noqa: E501
)


class FittableParametricLifetimeModel(ParametricLifetimeModel[*Ts], ABC):
    fitting_results: FittingResults | None

    def __init__(self, **kwparams: ST | None):
        super().__init__(**kwparams)
        self.fitting_results = None

    @abstractmethod
    def jac_hf(
        self,
        time: ST | NumpyST | ArrayND[NumpyST],
        *args: *Ts,
    ) -> ArrayND[np.float64]:
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
    def jac_chf(
        self, time: ST | NumpyST | ArrayND[NumpyST], *args: *Ts
    ) -> ArrayND[np.float64]:
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
    def jac_sf(
        self, time: ST | NumpyST | ArrayND[NumpyST], *args: *Ts
    ) -> ArrayND[np.float64]:
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

    def jac_cdf(
        self, time: ST | NumpyST | ArrayND[NumpyST], *args: *Ts
    ) -> ArrayND[np.float64]:
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
    def jac_pdf(
        self, time: ST | NumpyST | ArrayND[NumpyST], *args: *Ts
    ) -> ArrayND[np.float64]:
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
    def dhf(
        self, time: ST | NumpyST | ArrayND[NumpyST], *args: *Ts
    ) -> ArrayND[np.float64]:
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
    def init_likelihood(
        self,
        time: Array1D[np.float64],
        args: Array1D[Any]
        | Array2D[Any]
        | tuple[Array1D[Any] | Array2D[Any], ...]
        | None = None,
        event: Array1D[np.bool_] | None = None,
        entry: Array1D[np.float64] | None = None,
        **kwargs: Any,
    ) -> LifetimeLikelihood[M]:
        r"""
        Initialize the lifetime likelihood used to fit the parameters.

        `fit` method is the preferred way to fit model parameters. However,
        users can also interact with the likelihood returned by
        `init_likelihood` to study the optimization process.

        This method implementation is usally composed of 3 steps:
            1. Initialize an object to preprocess and encapsulate observation values.
            2. Create a `OptimizerConfig` config instance depending on the model needs.
            3. Instanciate and return a LifetimeLikelihood.

        `init_likelihood` is separated from `fit` in order to reuse existing
        likelihood parametrization in case of model composition. Any parameters
        initialization needed by the likelihood optimizer (e.g. `x0` or
        `bounds` as required in step 2.) are left to specific functions
        alongside concrete model implementations. These functions are invoked
        within `init_likelihood`.

        Parameters
        ----------
        time : 1d array
            Observed lifetime values.
        args : any ndarray or tuple of ndarray, default is None
            Additional arguments required by the model (e.g. covar).
        event : 1d array of bool, default is None
            Boolean indicators tagging lifetime values as right censored or complete.
        entry : 1d array, default is None
            Left truncations applied to lifetime values.
        **kwargs
            Extra arguments to control the parameters optimization. It can be:

                - those used by `scipy.optimize.minimize
                  <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`_
                  to search for the paremeters that minimize the negative
                  log-likelihood.
                - `covariance_method` to control the method used to estimate
                  parameters covariance. Values can be `"cs"`, `"2point"`,
                  `"exact"` or `False`. To skip parameters covariance
                  estimation, set `covariance_method` to `False`, otherwise the
                  default method associated to the model will be used. If
                  `covariance_method` is `"exact"` the `hess` must be passed
                  too.

        Returns
        -------
        out : LifetimeLikelihood instance
        """

    def fit(
        self,
        time: Array1D[np.float64],
        args: Array1D[Any]
        | Array2D[Any]
        | tuple[Array1D[Any] | Array2D[Any], ...]
        | None = None,
        event: Array1D[np.bool_] | None = None,
        entry: Array1D[np.float64] | None = None,
        **kwargs: Any,
    ) -> Self:
        """
        Estimation of parameters from lifetime data.

        Parameters
        ----------
        time : 1d array
            Observed lifetime values.
        args : any ndarray or tuple of ndarray, default is None
            Additional arguments required by the model (e.g. covar).
        event : 1d array of bool, default is None
            Boolean indicators tagging lifetime values as right censored or complete.
        entry : 1d array, default is None
            Left truncations applied to lifetime values.
        **kwargs
            Extra arguments used by `scipy.optimize.minimize
            <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`_
            to search for the paremeters that minimize the negative
            log-likelihood. `covariance_method` can also be passed to control
            the method used to estimate parameters covariance. Values can be
            `"cs"`, `"2point"`, `"exact"` or `False`. To skip parameters
            covariance estimation, set `covariance_method` to `False`,
            otherwise the default method associated to the model will be used.
            If `covariance_method` is `"exact"` the `hess` must be passed too.

        Returns
        -------
        out : the object instance
            The estimated parameters are setted inplace.
        """
        optimizer: LifetimeLikelihood[Self] = self.init_likelihood(
            time, args, event, entry, **kwargs
        )
        assert id(optimizer.model) != id(self)
        self.fitting_results = optimizer.optimize()
        self.set_params(self.fitting_results.optimal_params)

        return self

    @override
    def plot(
        self,
        fname: Literal["sf", "cdf", "chf", "hf", "pdf"],
        time: Array1D[np.float64],
        *args: *Ts,
        ax: Axes | None = None,
        **kwargs: Any,
    ) -> Axes:
        """
        Plot function.

        Parameters
        ----------
        fname : str
            The function name to plot. Allowed names are sf, cdf, chf, hf, pdf.
        time : 1d array
            The timeline used for x-axis.
        *args
            Any additional args required to compute the function.
        ax : plt.Axes, optional
            An optional existing matplotlib.axes.
        **kwargs
            Extra arguments to configure the plot:
                - ci : bool, default is True, if False the CI is not plotted
                - alpha_ci :
                - any arguments allowed by matplotlib.plot
        """
        y = getattr(self, fname)(time, *args)
        assert is_array_1d(y)  # typeguards
        ci = kwargs.pop("ci", True)
        if ci:
            time = np.asarray(time, dtype=float)
            se = estimate_se(self, fname, time, *args)
            if se is not None:
                ci_bounds = (0.0, np.inf)
                if fname in ("sf", "chf"):
                    ci_bounds = (0.0, 1.0)
                return plot_probability_function(
                    time, y, se=se, ci_bounds=ci_bounds, ax=ax, **kwargs
                )
        return plot_probability_function(time, y, ax=ax, **kwargs)


def estimate_se(
    model: FittableParametricLifetimeModel[*Ts],
    fname: str,
    time: Array1D[np.float64],
    *args: *Ts,
) -> Array1D[np.float64] | None:
    """

    References
    ----------
    .. [1] Meeker, W. Q., Escobar, L. A., & Pascual, F. G. (2022).
        Statistical methods for reliability data. John Wiley & Sons.
    """
    # [1] equation B.10 in Appendix
    if (
        model.fitting_results is not None
        and model.fitting_results.covariance_matrix is not None
    ):
        se = np.zeros_like(time)
        jac_f = getattr(model, f"jac_{fname}")(time[1:], *args)
        se[1:] = np.sqrt(
            np.einsum(
                "i...,ij,j...->...",
                jac_f,
                model.fitting_results.covariance_matrix,
                jac_f,
            )
        )
        return se
    return None


class LifetimeData:
    nb_observations: int = field(init=False)
    complete_time: Array[tuple[int, Literal[1]], np.float64] = field(
        init=False, repr=False
    )
    censored_time: (
        Array[tuple[int, Literal[1]], np.float64]
        | Array[tuple[int, Literal[2]], np.float64]
    ) = field(init=False, repr=False)
    left_truncations: Array[tuple[int, Literal[1]], np.float64] = field(
        init=False, repr=False
    )
    complete_time_args: tuple[Array2D[Any], ...] = field(init=False, repr=False)
    censored_time_args: tuple[Array2D[Any], ...] = field(init=False, repr=False)
    left_truncations_args: tuple[Array2D[Any], ...] = field(init=False, repr=False)

    def __init__(
        self,
        time: Array1D[np.float64] | Array[tuple[int, Literal[2]], np.float64],
        args: (
            Array1D[Any] | Array2D[Any] | tuple[Array1D[Any] | Array2D[Any], ...] | None
        ) = None,
        event: Array1D[np.bool_] | None = None,
        entry: Array1D[np.float64] | None = None,
    ) -> None:
        column_time = to_column_2d_if_1d(time)
        if column_time.shape[-1] == 2 and event is not None:
            raise ValueError("If time is given as intervals, event must be None")
        column_event = None
        if column_time.shape[-1] == 1:
            column_event = (
                to_column_2d_if_1d(event)
                if event is not None
                else np.ones_like(time, dtype=np.bool_)
            )
        column_entry = (
            to_column_2d_if_1d(entry)
            if entry is not None
            else np.zeros(len(time), dtype=np.float64)
        )
        if np.any(column_time <= column_entry):
            raise ValueError("All time values must be greater than entry values")
        if isinstance(args, tuple):
            column_args = tuple(to_column_2d_if_1d(arg) for arg in args)
        elif isinstance(args, np.ndarray):
            column_args = (to_column_2d_if_1d(args),)
        else:
            column_args = ()
        sizes = [
            len(x)
            for x in (column_time, column_event, column_entry, *column_args)
            if x is not None
        ]
        if len(set(sizes)) != 1:
            raise ValueError(
                f"""
                All lifetime data must have the same number of values. Fields
                length are different. Got {tuple(sizes)}
                """
            )
        non_zero_entry = np.flatnonzero(column_entry)
        if column_event is not None:
            non_zero_event = np.flatnonzero(column_event)
            zero_event = np.flatnonzero(column_event == 0)
            self.nb_observations = len(time)
            self.complete_time = column_time[non_zero_event]
            self.censored_time = column_time[zero_event]
            self.left_truncations = column_entry[non_zero_entry]
            self.complete_time_args = tuple(arg[non_zero_event] for arg in column_args)
            self.censored_time_args = tuple(arg[zero_event] for arg in column_args)
            self.left_truncations_args = tuple(
                arg[non_zero_entry] for arg in column_args
            )
        else:
            complete_time_index = np.flatnonzero(column_time[:, 0] == column_time[:, 1])
            non_complete_time_index = np.flatnonzero(
                column_time[:, 0] != column_time[:, 1]
            )
            self.nb_observations = len(time)
            self.complete_time = column_time[:, 1][complete_time_index]
            self.censored_time = column_time[non_complete_time_index]
            self.left_truncations = column_entry[non_zero_entry]
            self.complete_time_args = tuple(
                arg[complete_time_index] for arg in column_args
            )
            self.censored_time_args = tuple(
                arg[non_complete_time_index] for arg in column_args
            )
            self.left_truncations_args = tuple(
                arg[non_zero_entry] for arg in column_args
            )


@final
class LifetimeLikelihood(MaximumLikelihoodOptimizer[M, LifetimeData]):
    """
    Maximum likelihood estimator from lifetime data.

    Parameters
    ----------
    model : generic FittableParametricLifetimeModel
        Every model parameters must be initialized before passing it to the
        likelihood.
    data : LifetimeData
        An object that encapsulate and preprocess lifetime observations and
        truncations.
    config : OptimizerConfig
        An object that groups configurations used by the optimizer.

    Attributes
    ----------
    model: FittableParametricLifetimeModel
        A copy of the original model.
    data : LifetimeData
        An object that encapsulate and preprocess lifetime observations and
        truncations.
    config : OptimizerConfig
        An object that groups configurations used by the optimizer.
    """

    model: M
    data: LifetimeData

    def __init__(
        self,
        model: M,
        data: LifetimeData,
        config: OptimizerConfig,
    ):
        self.model = copy.deepcopy(model)
        self.data = data
        self.config = config
        if "jac" not in self.config.scipy_minimize_options:
            self.config.scipy_minimize_options["jac"] = self.jac_negative_log

    @property
    @override
    def nb_observations(self) -> int:
        return self.data.nb_observations

    @override
    def negative_log(self, params: Array1D[np.float64]) -> float:
        self.model.set_params(params)
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
        self.model.set_params(params)
        return (
            _jac_complete_time_contrib(self.model, self.data)
            + _jac_censored_time_contrib(self.model, self.data)
            + _jac_left_truncations_contrib(self.model, self.data)
        )


def _complete_time_contrib(
    model: FittableParametricLifetimeModel[
        *tuple[ST | NumpyST | ArrayND[NumpyST], ...]
    ],
    data: LifetimeData,
) -> float:
    if data.complete_time.size == 0.0:
        return 0.0
    return -np.sum(np.log(model.pdf(data.complete_time, *data.complete_time_args)))


def _jac_complete_time_contrib(
    model: FittableParametricLifetimeModel[
        *tuple[ST | NumpyST | ArrayND[NumpyST], ...]
    ],
    data: LifetimeData,
) -> NDArray[np.float64]:
    if data.complete_time.size == 0:
        return np.zeros_like(model.get_params())
    jac = -model.jac_pdf(data.complete_time, *data.complete_time_args) / model.pdf(
        data.complete_time, *data.complete_time_args
    )

    return np.sum(jac, axis=(1, 2))


def _censored_time_contrib(
    model: FittableParametricLifetimeModel[
        *tuple[ST | NumpyST | ArrayND[NumpyST], ...]
    ],
    data: LifetimeData,
) -> float:
    if data.censored_time.size == 0:
        return 0.0
    if data.censored_time.shape[-1] > 1:
        # interval censored time
        return np.sum(
            -np.log(
                10**-10
                + model.cdf(data.censored_time[:, 1], *data.censored_time_args)
                - model.cdf(data.censored_time[:, 0], *data.censored_time_args)
            ),
        )
    else:
        # right censored time
        return np.sum(model.chf(data.censored_time, *data.censored_time_args))


def _jac_censored_time_contrib(
    model: FittableParametricLifetimeModel[
        *tuple[ST | NumpyST | ArrayND[NumpyST], ...]
    ],
    data: LifetimeData,
) -> NDArray[np.float64]:
    if data.censored_time.size == 0:
        return np.zeros_like(model.get_params())
    if data.censored_time.shape[-1] > 1:
        # interval censored time
        jac_interval_censored = (
            model.jac_sf(data.censored_time[:, 1], *data.censored_time_args)
            - model.jac_sf(data.censored_time[:, 0], *data.censored_time_args)
        ) / (
            10**-10
            + model.cdf(data.censored_time[:, 1], *data.censored_time_args)
            - model.cdf(data.censored_time[:, 0], *data.censored_time_args)
        )

        return np.sum(jac_interval_censored, axis=(1, 2))
    else:
        # right censored time
        return np.sum(
            model.jac_chf(data.censored_time, *data.censored_time_args),
            axis=(1, 2),
        )


def _left_truncations_contrib(
    model: FittableParametricLifetimeModel[
        *tuple[ST | NumpyST | ArrayND[NumpyST], ...]
    ],
    data: LifetimeData,
) -> float:
    if data.left_truncations.size == 0.0:
        return 0.0
    return -np.sum(model.chf(data.left_truncations, *data.left_truncations_args))


def _jac_left_truncations_contrib(
    model: FittableParametricLifetimeModel[
        *tuple[ST | NumpyST | ArrayND[NumpyST], ...]
    ],
    data: LifetimeData,
) -> NDArray[np.float64]:
    if data.left_truncations.size == 0.0:
        return np.zeros_like(model.get_params())
    jac = -model.jac_chf(data.left_truncations, *data.left_truncations_args)
    return np.sum(jac, axis=(1, 2))
