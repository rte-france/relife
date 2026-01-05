"""Base classes for all parametric lifetime models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    Literal,
    Self,
    TypeVarTuple,
    overload,
)

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import Bounds, newton

from relife.base import ParametricModel
from relife.typing import (
    AnyFloat,
    NumpyBool,
    NumpyFloat,
    ScipyMinimizeOptions,
    Seed,
)
from relife.utils.quadrature import legendre_quadrature, unweighted_laguerre_quadrature

from ._plot import PlotParametricLifetimeModel

if TYPE_CHECKING:
    from relife.likelihood._base import FittingResults

Ts = TypeVarTuple("Ts")


class ParametricLifetimeModel(ParametricModel, ABC, Generic[*Ts]):
    r"""Base class for parametric lifetime models in ReLife.

    This class is a blueprint for implementing parametric lifetime models.
    The interface is generic and can define a variadic set of arguments.
    It expects implementation of the hazard function (``hf``), the cumulative hazard function (``chf``),
    the probability density function (``pdf``) and the survival function (``sf``).
    Other functions are implemented by default but can be overridden by the derived classes.

    Note:
        The abstract methods also provides a default implementation. One may not have to implement
        ``hf``, ``chf``, ``pdf`` and ``sf`` and just call ``super()`` to access the base implementation.

    Methods:
        hf: Abstract method to compute the hazard function.
        chf: Abstract method to compute the cumulative hazard function.
        sf: Abstract method to compute the survival function.
        pdf: Abstract method to compute the probability density function.
    """

    @abstractmethod
    def sf(self, time: AnyFloat, *args: *Ts) -> NumpyFloat:
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
        return 1 - self.sf(time, *args)

    def ppf(self, probability: AnyFloat, *args: *Ts) -> NumpyFloat:
        probability = np.asarray(probability)
        return self.isf(1 - probability, *args)

    def median(self, *args: *Ts) -> NumpyFloat:
        return self.ppf(0.5, *args)

    def isf(self, probability: AnyFloat, *args: *Ts) -> NumpyFloat:

        def func(x: NDArray[np.float64]) -> NumpyFloat:
            return self.sf(x, *args) - probability

        # no idea on how to type func
        return newton(  # pyright: ignore[reportCallIssue, reportUnknownVariableType]
            func,  # pyright: ignore[reportArgumentType]
            x0=np.zeros_like(probability),
            args=args,
        )

    def ichf(self, cumulative_hazard_rate: AnyFloat, *args: *Ts) -> NumpyFloat:

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
        size: int,
        *args: *Ts,
        nb_assets: int | None = None,
        return_event: Literal[False],
        return_entry: Literal[False],
        seed: int | None = None,
    ) -> NumpyFloat: ...
    @overload
    def rvs(
        self,
        size: int,
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
    def rvs(
        self,
        size: int,
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
        rng = np.random.default_rng(seed)
        if nb_assets is not None:
            np_size = (nb_assets, size)
        else:
            np_size = size
        probability = rng.uniform(size=np_size)
        if np_size == 1:
            probability = np.squeeze(probability)
        time = self.isf(probability, *args)
        event = np.ones_like(time, dtype=np.bool_) if isinstance(time, np.ndarray) else np.bool_(True)
        entry = np.zeros_like(time, dtype=np.float64) if isinstance(time, np.ndarray) else np.float64(0)
        if not return_event and not return_entry:
            return time
        elif return_event and not return_entry:
            return time, event
        elif not return_event and return_entry:
            return time, entry
        else:
            return time, event, entry

    @property
    def plot(  # pyright: ignore[reportUnknownParameterType]
        self,
    ) -> PlotParametricLifetimeModel:  # pyright: ignore[reportMissingTypeArgument]
        """Provides access to plotting functionnalities"""
        return PlotParametricLifetimeModel(self)

    def ls_integrate(
        self,
        func: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        a: AnyFloat,
        b: AnyFloat,
        *args: *Ts,
        deg: int = 10,
    ) -> NumpyFloat:
        def integrand(x: NDArray[np.float64]) -> NDArray[np.float64]:
            #  x.shape == (deg,), (deg, n) or (deg, m, n), ie points of quadratures
            # fx : (d_1, ..., d_i, deg), (d_1, ..., d_i, deg, n) or (d_1, ..., d_i, deg, m, n)
            x = np.asarray(x)
            fx = func(x)

            try:
                _ = np.broadcast_shapes(fx.shape[-len(x.shape) :], x.shape)
            except ValueError:
                raise ValueError(
                    f"""
                    func can't squeeze input dimensions. If x has shape (d_1, ..., d_i), func(x) must have shape (..., d_1, ..., d_i).
                    Ex : if x.shape == (m, n), func(x).shape == (..., m, n).
                    """
                )
            if x.ndim == 3:  # reshape because model.pdf is tested only for input ndim <= 2
                x_shape: tuple[int, int, int] = x.shape
                xdeg, m, n = x_shape
                x = np.rollaxis(x, 1).reshape(m, -1)  # (m, deg*n), roll on m because axis 0 must align with m of args
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

        bound_b = self.isf(1e-4, *args)  #  () or (m, 1), if (m, 1) then arr_b.shape == (m, 1) or (m, n)
        broadcasted_arrs = np.broadcast_arrays(arr_a, arr_b, bound_b)
        arr_a = broadcasted_arrs[0].copy()  # arr_a.shape == arr_b.shape == bound_b.shape
        arr_b = broadcasted_arrs[1].copy()  # arr_a.shape == arr_b.shape == bound_b.shape
        bound_b = broadcasted_arrs[2].copy()  # arr_a.shape == arr_b.shape == bound_b.shape
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
        return self.moment(1, *args)

    def var(self, *args: *Ts) -> NumpyFloat:
        return self.moment(2, *args) - self.moment(1, *args) ** 2

    def mrl(self, time: AnyFloat, *args: *Ts) -> NumpyFloat:
        sf = self.sf(time, *args)

        def func(x: NDArray[np.float64]) -> NDArray[np.float64]:
            return np.asarray(x) - time

        ls = self.ls_integrate(func, time, np.inf, *args)
        if sf.ndim < 2:  # 2d to 1d or 0d
            ls = np.squeeze(ls)
        return ls / sf


class FittableParametricLifetimeModel(ParametricLifetimeModel[*Ts], ABC):
    fitting_results: FittingResults | None

    def __init__(self, **kwparams: float | None):
        super().__init__(**kwparams)
        self.fitting_results = None

    @overload
    def jac_hf(
        self,
        time: AnyFloat,
        *args: *Ts,
        asarray: Literal[False],
    ) -> tuple[NumpyFloat, ...]: ...
    @overload
    def jac_hf(
        self,
        time: AnyFloat,
        *args: *Ts,
        asarray: Literal[True],
    ) -> NumpyFloat: ...
    @overload
    def jac_hf(
        self,
        time: AnyFloat,
        *args: *Ts,
        asarray: bool,
    ) -> tuple[NumpyFloat, ...] | NumpyFloat: ...
    @abstractmethod
    def jac_hf(
        self,
        time: AnyFloat,
        *args: *Ts,
        asarray: bool = True,
    ) -> tuple[NumpyFloat, ...] | NumpyFloat: ...
    @overload
    def jac_chf(
        self,
        time: AnyFloat,
        *args: *Ts,
        asarray: Literal[False],
    ) -> tuple[NumpyFloat, ...]: ...
    @overload
    def jac_chf(
        self,
        time: AnyFloat,
        *args: *Ts,
        asarray: Literal[True],
    ) -> NumpyFloat: ...
    @overload
    def jac_chf(
        self,
        time: AnyFloat,
        *args: *Ts,
        asarray: bool,
    ) -> tuple[NumpyFloat, ...] | NumpyFloat: ...
    @abstractmethod
    def jac_chf(
        self,
        time: AnyFloat,
        *args: *Ts,
        asarray: bool = True,
    ) -> tuple[NumpyFloat, ...] | NumpyFloat: ...
    @overload
    def jac_sf(
        self,
        time: AnyFloat,
        *args: *Ts,
        asarray: Literal[False],
    ) -> tuple[NumpyFloat, ...]: ...
    @overload
    def jac_sf(
        self,
        time: AnyFloat,
        *args: *Ts,
        asarray: Literal[True],
    ) -> NumpyFloat: ...
    @overload
    def jac_sf(
        self,
        time: AnyFloat,
        *args: *Ts,
        asarray: bool,
    ) -> tuple[NumpyFloat, ...] | NumpyFloat: ...
    @abstractmethod
    def jac_sf(
        self,
        time: AnyFloat,
        *args: *Ts,
        asarray: bool = True,
    ) -> tuple[NumpyFloat, ...] | NumpyFloat: ...
    @overload
    def jac_pdf(
        self,
        time: AnyFloat,
        *args: *Ts,
        asarray: Literal[False],
    ) -> tuple[NumpyFloat, ...]: ...
    @overload
    def jac_pdf(
        self,
        time: AnyFloat,
        *args: *Ts,
        asarray: Literal[True],
    ) -> NumpyFloat: ...
    @overload
    def jac_pdf(
        self,
        time: AnyFloat,
        *args: *Ts,
        asarray: bool,
    ) -> tuple[NumpyFloat, ...] | NumpyFloat: ...
    @abstractmethod
    def jac_pdf(
        self,
        time: AnyFloat,
        *args: *Ts,
        asarray: bool = True,
    ) -> tuple[NumpyFloat, ...] | NumpyFloat: ...

    @abstractmethod
    def dhf(self, time: AnyFloat, *args: *Ts) -> NumpyFloat: ...

    @abstractmethod
    def get_initial_params(
        self,
        time: NDArray[np.float64],
        model_args: NDArray[Any] | tuple[NDArray[Any], ...] | None = None,
    ) -> NDArray[np.float64]:
        """Get the inital params values used in 'fit'.
        To change this value, pass 'x0' in 'optimizer_options' of 'fit'.
        """

    @property
    @abstractmethod
    def params_bounds(self) -> Bounds:
        """Parameters bounds"""

    @abstractmethod
    def fit(
        self,
        time: NDArray[np.float64],
        model_args: NDArray[Any] | tuple[NDArray[Any], ...] | None = None,
        event: NDArray[np.bool_] | None = None,
        entry: NDArray[np.float64] | None = None,
        optimizer_options: ScipyMinimizeOptions | None = None,
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
        Self
            The current object with the estimated parameters setted inplace.
        """
        from relife.likelihood import DefaultLifetimeLikelihood

        self.params = self.get_initial_params(  # pyright: ignore[reportUnannotatedClassAttribute]
            time, model_args=model_args
        )
        if optimizer_options is None:
            optimizer_options = {}
        if "bounds" not in optimizer_options:
            optimizer_options["bounds"] = self.params_bounds
        likelihood = DefaultLifetimeLikelihood(self, time, model_args=model_args, event=event, entry=entry)
        fitting_results = likelihood.maximum_likelihood_estimation(**optimizer_options)
        self.params = fitting_results.optimal_params
        self.fitting_results = fitting_results
        return self

    @abstractmethod
    def fit_from_interval_censored_lifetimes(
        self,
        time_inf: NDArray[np.float64],
        time_sup: NDArray[np.float64],
        model_args: NDArray[Any] | tuple[NDArray[Any], ...] | None = None,
        entry: NDArray[np.float64] | None = None,
        optimizer_options: ScipyMinimizeOptions | None = None,
    ) -> Self:
        """
        Estimation of the distribution parameters from interval censored lifetime data.

        Parameters
        ----------
        time_inf : 1d array
            Observed lifetime lower bounds.
        time_sup : 1d array
            Observed lifetime upper bounds.
        model_args : any ndarray or tuple of ndarray, default is None
            Any additional arguments required by the model.
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

        Notes
        -----
        Where `time_inf == time_sup`, lifetimes are complete.

        Returns
        -------
        Self
            The current object with the estimated parameters setted inplace.
        """
        from relife.likelihood import IntervalLifetimeLikelihood

        self.params = self.get_initial_params(time_sup, model_args=model_args)
        if optimizer_options is None:
            optimizer_options = {}
        if "bounds" not in optimizer_options:
            optimizer_options["bounds"] = self.params_bounds
        likelihood = IntervalLifetimeLikelihood(self, time_inf, time_sup, entry=entry)
        fitting_results = likelihood.maximum_likelihood_estimation(**optimizer_options)
        self.params = fitting_results.optimal_params
        self.fitting_results = fitting_results
        return self
