from __future__ import annotations

from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    NewType,
    Optional,
    Self,
    TypeVarTuple,
)

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import newton
from typing_extensions import override

from relife import FrozenMixin, ParametricModel
from relife._plots import PlotSurvivalFunc
from relife.data import lifetime_data_factory
from relife.likelihood import maximum_likelihood_estimation
from relife.likelihood.maximum_likelihood_estimation import FittingResults

if TYPE_CHECKING:
    from relife.lifetime_model import (
        FrozenLifetimeDistribution,
        FrozenLifetimeRegression,
        FrozenParametricLifetimeModel,
    )

    from ._structural_type import FittableParametricLifetimeModel

Args = TypeVarTuple("Args")


class ParametricLifetimeModel(ParametricModel, Generic[*Args], ABC):
    r"""Base class for lifetime model.

    This class defines the structure for creating lifetime model. It is s a blueprint
    for implementing lifetime model parametrized by a variadic set of arguments.
    It provides the framework for implementing hazard functions (``hf``), cumulative hazard functions (``chf``),
    probability density function (``pdf``) and survival function (``sf``).
    Other functions are implemented by default but can be overridden by derived classes.

    Methods:
        hf: Abstract method to compute the hazard function.
        chf: Abstract method to compute the cumulative hazard function.
        sf: Abstract method to compute the survival function.
        pdf: Abstract method to compute the probability density function.

    Raises:
        NotImplementedError: Raised when an abstract method or feature in this
        class has not been implemented in a derived class.
    """

    @abstractmethod
    def hf(
        self, time: float | NDArray[np.float64], *args: *Args
    ) -> np.float64 | NDArray[np.float64]:
        match self:
            case ParametricLifetimeModel(pdf=_, sf=_):
                return self.pdf(time, *args) / self.sf(time, *args)
            case ParametricLifetimeModel(sf=_):
                raise NotImplementedError(
                    """
                    ReLife does not implement hf as the derivate of chf yet. Consider adding it in future versions
                    see: https://docs.scipy.org/doc/scipy-1.11.4/reference/generated/scipy.misc.derivative.html
                    or : https://github.com/maroba/findiff
                    """
                )
            case _:
                class_name = type(self).__name__
                raise NotImplementedError(
                    f"""
                        {class_name} must implement concrete hf function
                        """
                )

    @abstractmethod
    def sf(
        self, time: float | NDArray[np.float64], *args: *Args
    ) -> np.float64 | NDArray[np.float64]:
        match self:
            case ParametricLifetimeModel(chf=_):
                return np.exp(
                    -self.chf(
                        time,
                        *args,
                    )
                )
            case ParametricLifetimeModel(pdf=_, hf=_):
                return self.pdf(time, *args) / self.hf(time, *args)
            case _:
                class_name = type(self).__name__
                raise NotImplementedError(
                    f"""
                    {class_name} must implement concrete sf function
                    """
                )

    @abstractmethod
    def chf(
        self, time: float | NDArray[np.float64], *args: *Args
    ) -> np.float64 | NDArray[np.float64]:
        match self:
            case ParametricLifetimeModel(sf=_):
                return -np.log(self.sf(time, *args))
            case ParametricLifetimeModel(pdf=_, hf=_):
                return -np.log(self.pdf(time, *args) / self.hf(time, *args))
            case ParametricLifetimeModel(hf=_):
                raise NotImplementedError(
                    """
                    ReLife does not implement chf as the integration of hf yet. Consider adding it in future versions
                    """
                )
            case _:
                class_name = type(self).__name__
                raise NotImplementedError(
                    f"""
                    {class_name} must implement concrete chf or at least concrete hf function
                    """
                )

    @abstractmethod
    def pdf(
        self, time: float | NDArray[np.float64], *args: *Args
    ) -> np.float64 | NDArray[np.float64]:
        try:
            return self.sf(time, *args) * self.hf(time, *args)
        except NotImplementedError as err:
            class_name = type(self).__name__
            raise NotImplementedError(
                f"""
            {class_name} must implement pdf or the above functions
            """
            ) from err

    def mrl(
        self, time: float | NDArray[np.float64], *args: *Args
    ) -> np.float64 | NDArray[np.float64]:
        sf = self.sf(time, *args)
        ls = self.ls_integrate(lambda x: x - time, time, np.array(np.inf), *args)
        if sf.ndim < 2:  # 2d to 1d or 0d
            ls = np.squeeze(ls)
        return ls / sf

    def isf(
        self,
        probability: float | NDArray[np.float64],
        *args: *Args,
    ) -> np.float64 | NDArray[np.float64]:
        """Inverse survival function.

        Parameters
        ----------
        probability : float or ndarray, shape (n, ) or (m, n)
        *args : variadic arguments required by the function

        Returns
        -------
        ndarray of shape (), (n, ) or (m, n)
            Complement quantile corresponding to probability.
        """
        return newton(
            lambda x: self.sf(x, *args) - probability,
            x0=np.zeros_like(probability),
            args=args,
        )

    def ichf(
        self,
        cumulative_hazard_rate: float | NDArray[np.float64],
        *args: *Args,
    ) -> np.float64 | NDArray[np.float64]:
        return newton(
            lambda x: self.chf(x, *args) - cumulative_hazard_rate,
            x0=np.zeros_like(cumulative_hazard_rate),
        )

    def cdf(
        self, time: float | NDArray[np.float64], *args: *Args
    ) -> np.float64 | NDArray[np.float64]:
        return 1 - self.sf(time, *args)

    def rvs(
        self,
        *args: *Args,
        size: int | tuple[int, int] = 1,
        seed: Optional[int] = None,
    ) -> np.float64 | NDArray[np.float64]:
        """Random variable sample.

        Parameters
        ----------
        *args : variadic arguments required by the function
        size : int or (int, int), default 1
            Output shape of the generated sample.
        seed : int, default None
            Random seed.

        Returns
        -------
        ndarray of shape (size, )
            Sample of random lifetimes.
        """
        frozen_model = self.freeze(*args)
        return frozen_model.rvs(size=size, seed=seed)

    def ppf(
        self,
        probability: float | NDArray[np.float64],
        *args: *Args,
    ) -> np.float64 | NDArray[np.float64]:
        return self.isf(1 - probability, *args)

    def ls_integrate(
        self,
        func: Callable[[float | NDArray[np.float64]], NDArray[np.float64]],
        a: float | NDArray[np.float64] = 0.0,
        b: float | NDArray[np.float64] = np.inf,
        *args: *Args,
        deg: int = 10,
    ) -> np.float64 | NDArray[np.float64]:
        r"""
        Lebesgue-Stieltjes integration.

        Parameters
        ----------
        func : callable (in : 1 ndarray , out : 1 ndarray)
            The callable must have only one ndarray object as argument and returns one ndarray object
        a : ndarray (max dim of 2)
            Lower bound(s) of integration.
        b : ndarray (max dim of 2)
            Upper bound(s) of integration. If lower bound(s) is infinite, use np.inf as value.)
        deg : int, default 10
            Degree of the polynomials interpolation

        Returns
        -------
        np.ndarray
            Lebesgue-Stieltjes integral of func from `a` to `b`.
        """
        from relife._quadrature import (
            check_and_broadcast_bounds,
            legendre_quadrature,
            unweighted_laguerre_quadrature,
        )

        frozen_model = self.freeze(*args)

        def integrand(x: NDArray[np.float64]) -> NDArray[np.float64]:
            #  x.shape == (deg,), (deg, n) or (deg, m, n)
            # fx : (d_1, ..., d_i, deg), (d_1, ..., d_i, deg, n) or (d_1, ..., d_i, deg, m, n)
            fx = func(x)
            if fx.shape[-len(x.shape):] != x.shape:
                raise ValueError(
                    f"""
                    func can't squeeze input dimensions. If x has shape (d_1, ..., d_i), func(x) must have shape (..., d_1, ..., d_i).
                    Ex : if x.shape == (m, n), func(x).shape == (..., m, n).
                    """
                )
            if x.ndim == 3: # reshape because model.pdf is tested only for input ndim <= 2
                deg, m, n = x.shape
                x = np.rollaxis(x, 1).reshape(m, -1) # (m, deg*n), roll on m because axis 0 must align with m of args
                pdf = frozen_model.pdf(x)  # (m, deg*n)
                pdf = np.rollaxis(pdf.reshape(m, deg, n), 1, 0)  #  (deg, m, n)
            else: # ndim == 1 | 2
                # reshape to (1, deg*n) or (1, deg), ie place 1 on axis 0 to allow broadcasting with m of args
                pdf = frozen_model.pdf(x.reshape(1, -1)) # (1, deg*n) or (1, deg)
                pdf = pdf.reshape(x.shape) # (deg, n) or (deg,)

            # (d_1, ..., d_i, deg) or (d_1, ..., d_i, deg, n) or (d_1, ..., d_i, deg, m, n)
            return fx * pdf

        # if isinstance(model, AgeReplacementModel):
        #     ar, args = frozen_model.args[0], frozen_model.args[1:]
        #     b = np.minimum(ar, b)
        #     w = np.where(b == ar, func(ar) * model.baseline.sf(ar, *args), 0.)

        arr_a, arr_b = check_and_broadcast_bounds(a, b)  # (), (n,) or (m, n)
        if np.any(arr_a >= arr_b):
            raise ValueError("Bound values a must be strictly lower than values of b")
        if arr_a.ndim == 2:
            if arr_a.shape[0] not in (
                1,
                frozen_model.args_nb_assets,
            ) and frozen_model.args_nb_assets not in (1, arr_a.shape[0]):
                raise ValueError(
                    f"Incompatible bounds with model. Model has {frozen_model.nb_assets} nb_assets but a and b have shape {a.shape}, {b.shape}"
                )

        bound_b = frozen_model.isf(
            1e-4
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

    def moment(self, n: int, *args: *Args) -> np.float64 | NDArray[np.float64]:
        """n-th order moment

        Parameters
        ----------
        n : order of the moment, at least 1.
        *args : variadic arguments required by the function

        Returns
        -------
        ndarray of shape (0, )
            n-th order moment.
        """
        if n < 1:
            raise ValueError("order of the moment must be at least 1")
        return self.ls_integrate(
            lambda x: x**n,
            0.0,
            np.inf,
            *args,
            deg=100,
        )  #  high degree of polynome to ensure high precision

    def mean(self, *args: *Args) -> np.float64 | NDArray[np.float64]:
        return self.moment(1, *args)

    def var(self, *args: *Args) -> np.float64 | NDArray[np.float64]:
        return self.moment(2, *args) - self.moment(1, *args) ** 2

    def median(self, *args: *Args) -> np.float64 | NDArray[np.float64]:
        return self.ppf(np.array(0.5), *args)

    def freeze(
        self,
        *args: *Args,
    ) -> ParametricLifetimeModel[()]:
        from .frozen_model import FrozenParametricLifetimeModel

        args_names = self.args_names
        if len(args) != len(args_names):
            raise ValueError(
                f"Expected {args_names} positional arguments but got only {len(args)} arguments"
            )
        frozen_model = FrozenParametricLifetimeModel(self)
        frozen_model.freeze_args(**{k: v for (k, v) in zip(args_names, args)})
        return frozen_model

    @property
    def plot(self) -> PlotSurvivalFunc:
        """Plot"""
        return PlotSurvivalFunc(self)

    @property
    def args_names(self) -> tuple[str, ...]:
        from relife.lifetime_model import (
            AcceleratedFailureTime,
            AgeReplacementModel,
            LeftTruncatedModel,
            ProportionalHazard,
        )

        try:
            next(self.nested_models())
            _, nested_models = zip(*self.nested_models())
        except StopIteration:
            return ()
        args_names = ()
        #  iterate on self instance and every components
        for model in (self, *nested_models):
            match model:
                case ProportionalHazard() | AcceleratedFailureTime():
                    args_names += ("covar",)
                case AgeReplacementModel():
                    args_names += ("ar",)
                case LeftTruncatedModel():
                    args_names += ("a0",)
                #  break because other args are frozen in frozen instance
                case FrozenMixin():
                    break
                case _:
                    continue
        return args_names

    def __getattribute__(self, item):
        match item:
            case (
                "sf"
                | "hf"
                | "chf"
                | "pdf"
                | "mrl"
                | "isf"
                | "ichf"
                | "cdf"
                | "rvs"
                | "ppf"
                | "ls_integrate"
                | "moment"
                | "mean"
                | "var"
                | "median"
                | "freeze"
                | "plot"
            ):
                if np.any(np.isnan(self.params)):
                    raise ValueError(
                        f"Can't call {item} if params are not set. Got {self.params} params"
                    )
                return super().__getattribute__(item)
            case _:
                return super().__getattribute__(item)


class LifetimeDistribution(ParametricLifetimeModel[()], ABC):
    """
    Base class for distribution model.
    """

    @property
    def fitting_results(self) -> Optional[FittingResults]:
        return self._fitting_results

    @fitting_results.setter
    def fitting_results(self, value: FittingResults):
        self._fitting_results = value

    @override
    def sf(self, time: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]:
        return super().sf(time)

    @override
    def isf(
        self, probability: float | NDArray[np.float64]
    ) -> np.float64 | NDArray[np.float64]:
        """Inverse survival function.

        Parameters
        ----------
        probability : float or np.ndarray
            Probability value(s) at which to compute the function.
            If ndarray, allowed shapes are ``()``, ``(n_values,)`` or ``(n_assets, n_values)``.

        Returns
        -------
        np.float64 or np.ndarray
            Function values at each given time(s).
        """
        cumulative_hazard_rate = -np.log(probability + 1e-6) # avoid division by zero
        return self.ichf(cumulative_hazard_rate)

    @override
    def cdf(self, time: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]:
        return super().cdf(time)

    def pdf(self, time: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]:
        return super().pdf(time)

    @override
    def ppf(
        self, probability: float | NDArray[np.float64]
    ) -> np.float64 | NDArray[np.float64]:
        """Percent point function.

        The percent point corresponds to the inverse of the cumulative distribution function.

        Parameters
        ----------
        probability : float or np.ndarray
            Probability value(s) at which to compute the function.
            If ndarray, allowed shapes are ``()``, ``(n_values,)`` or ``(n_assets, n_values)``.

        Returns
        -------
        np.float64 or np.ndarray
            Function values at each given time(s).
        """
        return super().ppf(probability)

    @override
    def rvs(
        self, size: int | tuple[int, int] = 1, seed: Optional[int] = None
    ) -> np.float64 | NDArray[np.float64]:
        """Random variable sampling.

        Parameters
        ----------
        size : int or (int, int), default 1
            Shape of the sample.
        seed : int, default None
            Random seed.

        Returns
        -------
        np.ndarray
            Sample of random lifetimes.
        """

        return super().rvs(size=size, seed=seed)

    @override
    def moment(self, n: int) -> np.float64:
        """
        n-th order moment of the distribution.

        Parameters
        ----------
        n : int
            Order of the moment, at least 1.

        Returns
        -------
        np.float64
            n-th order moment of the distribution.
        """

        return super().moment(n)

    @override
    def median(self) -> np.float64:
        return super().median()

    @abstractmethod
    def dhf(
        self,
        time: float | NDArray[np.float64],
    ) -> np.float64 | NDArray[np.float64]: ...

    @abstractmethod
    def jac_hf(
        self,
        time: float | NDArray[np.float64],
        *,
        asarray : bool = False,
    ) -> (
        np.float64
        | NDArray[np.float64]
        | tuple[np.float64, ...]
        | tuple[NDArray[np.float64], ...]
    ): ...

    @abstractmethod
    def jac_chf(
        self,
        time: float | NDArray[np.float64],
        *,
        asarray: bool = False,
    ) -> (
        np.float64
        | NDArray[np.float64]
        | tuple[np.float64, ...]
        | tuple[NDArray[np.float64], ...]
    ): ...

    def jac_sf(
        self, time: float | NDArray[np.float64],
        *,
        asarray: bool = False,
    ) -> (
        np.float64
        | NDArray[np.float64]
        | tuple[np.float64, ...]
        | tuple[NDArray[np.float64], ...]
    ):
        jac_chf, sf = self.jac_chf(time, asarray=True), self.sf(time)
        jac = -jac_chf * sf
        if not asarray:
            return np.unstack(jac)
        return jac

    def jac_cdf(
        self, time: float | NDArray[np.float64],
        *,
        asarray: bool = False,
    ) -> (
        np.float64
        | NDArray[np.float64]
        | tuple[np.float64, ...]
        | tuple[NDArray[np.float64], ...]
    ):
        jac = -self.jac_sf(time, asarray=True)
        if not asarray:
            return np.unstack(jac)
        return jac

    def jac_pdf(
        self, time: float | NDArray[np.float64],
        *,
        asarray: bool = False,
    ) -> (
        np.float64
        | NDArray[np.float64]
        | tuple[np.float64, ...]
        | tuple[NDArray[np.float64], ...]
    ):
        jac_hf, hf = self.jac_hf(time, asarray=True), self.hf(time)
        jac_sf, sf = self.jac_sf(time, asarray=True), self.sf(time)
        jac = jac_hf * sf + jac_sf * hf
        if not asarray:
            return np.unstack(jac)
        return jac

    @override
    def ls_integrate(
        self,
        func: Callable[[float | NDArray[np.float64]], NDArray[np.float64]],
        a: float | NDArray[np.float64],
        b: float | NDArray[np.float64],
        deg: int = 10,
    ) -> NDArray[np.float64]:

        return super().ls_integrate(func, a, b, deg=deg)

    @override
    def freeze(self) -> FrozenLifetimeDistribution:
        from relife.lifetime_model import FrozenLifetimeDistribution

        return FrozenLifetimeDistribution(self)

    def fit(
        self,
        time: NDArray[np.float64],
        /,
        event: Optional[NDArray[np.bool_]] = None,
        entry: Optional[NDArray[np.float64]] = None,
        departure: Optional[NDArray[np.float64]] = None,
        **kwargs: Any,
    ) -> Self:
        lifetime_data = lifetime_data_factory(
            time,
            event=event,
            entry=entry,
            departure=departure,
        )
        maximum_likelihood_estimation(
            self,
            lifetime_data,
            **kwargs,
        )
        return self

    def __getattribute__(self, item):
        match item:
            case "dhf" | "jac_hf" | "jac_chf" | "jac_pdf" | "jac_sf" | "jac_cdf":
                if np.any(np.isnan(self.params)):
                    raise ValueError(
                        f"Can't call {item} if params are not set. Got {self.params} params"
                    )
                return super().__getattribute__(item)
            case _:
                return super().__getattribute__(item)


class CovarEffect(ParametricModel):
    """
    Covariates effect.

    Parameters
    ----------
    *coefficients : float
        Coefficients of the covariates effect.
    """

    def __init__(self, coefficients: tuple[Optional[float], ...] = (None,)):
        super().__init__(**{f"coef_{i + 1}": v for i, v in enumerate(coefficients)})

    @property
    def nb_coef(self) -> int:
        return self.nb_params

    def g(self, covar: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]:
        """
        Compute the covariates effect.
        If covar.shape : () or (nb_coef,) => out.shape : (), float
        If covar.shape : (m, nb_coef) => out.shape : (m, 1)
        """
        covar: NDArray[np.float64] = np.asarray(covar) # (nb_coef,) or (m, nb_coef)
        if covar.ndim > 2:
            raise ValueError(
                f"Invalid covar shape. Expected (nb_coef,) or (m, nb_coef) but got {covar.shape}"
            )
        covar_nb_coef = covar.size if covar.ndim <= 1 else covar.shape[-1]
        if covar_nb_coef != self.nb_coef:
            raise ValueError(
                f"Invalid covar. Number of covar does not match number of coefficients. Got {self.nb_coef} nb_coef but covar shape is {covar.shape}"
            )
        g = np.exp(np.sum(self.params * covar, axis=-1, keepdims=True))  # (m, 1)
        if covar.ndim <= 1:
            return np.float64(g.item())
        return g

    def jac_g(
        self, covar: float | NDArray[np.float64], *, asarray : bool = False
    ) -> (
        np.float64
        | NDArray[np.float64]
        | tuple[np.float64, ...]
        | tuple[NDArray[np.float64], ...]
    ):
        """
        Compute the Jacobian of the covariates effect.
        If covar.shape : () or (nb_coef,) => out.shape : (nb_coef,)
        If covar.shape : (m, nb_coef) => out.shape : (nb_coef, m, 1)
        """
        covar: NDArray[np.float64] = np.asarray(covar) # (), (nb_coef,) or (m, nb_coef)
        g = self.g(covar) # () or (m, 1)
        jac = covar.T.reshape(self.nb_coef, -1, 1) * g # (nb_coef, m, 1)
        if covar.ndim <= 1:
            jac = jac.reshape(self.nb_coef) # (nb_coef,) or (nb_coef, m, 1)
        if not asarray:
            return np.unstack(jac, axis=0) # tuple
        return jac # (nb_coef, m, 1)


class LifetimeRegression(
    ParametricLifetimeModel[float | NDArray[np.float64], *Args], ABC
):
    """
    Base class for regression model.
    """

    def __init__(
        self,
        baseline: FittableParametricLifetimeModel[*Args],
        coefficients: tuple[Optional[float], ...] = (None,),
    ):
        super().__init__()
        self.covar_effect = CovarEffect(coefficients)
        self.baseline = baseline

    @property
    def fitting_results(self) -> Optional[FittingResults]:
        return self._fitting_results

    @fitting_results.setter
    def fitting_results(self, value: FittingResults):
        self._fitting_results = value

    @property
    def nb_coef(self) -> int:
        return self.covar_effect.nb_coef

    @override
    def sf(
        self,
        time: float | NDArray[np.float64],
        covar: float | NDArray[np.float64],
        *args: *Args,
    ) -> np.float64 | NDArray[np.float64]:
        return super().sf(time, covar, *args)

    @override
    def isf(
        self,
        probability: float | NDArray[np.float64],
        covar: float | NDArray[np.float64],
        *args: *Args,
    ) -> np.float64 | NDArray[np.float64]:
        """Inverse survival function.

        Parameters
        ----------
        probability : float or np.ndarray
            Probability value(s) at which to compute the function.
            If ndarray, allowed shapes are ``()``, ``(n_values,)`` or ``(n_assets, n_values)``.
        covar : np.ndarray
            Covariate values. The ndarray must be broadcastable with ``time``.
        *args : variable number of np.ndarray
            Any variables needed to compute the function. Those variables must be
            broadcastable with ``time``. They may exist and result from method chaining due to nested class instantiation.

        Returns
        -------
        ndarray of shape (), (n, ) or (m, n)
            Time values corresponding to the given survival probabilities.
        """
        cumulative_hazard_rate = -np.log(probability + 1e-6) # avoid division by zero
        return self.ichf(cumulative_hazard_rate, covar, *args)

    @override
    def cdf(
        self,
        time: float | NDArray[np.float64],
        covar: float | NDArray[np.float64],
        *args: *Args,
    ) -> np.float64 | NDArray[np.float64]:
        return super().cdf(time, *(covar, *args))

    def pdf(
        self,
        time: float | NDArray[np.float64],
        covar: float | NDArray[np.float64],
        *args: *Args,
    ) -> np.float64 | NDArray[np.float64]:
        return super().pdf(time, *(covar, *args))

    @override
    def ppf(
        self,
        probability: float | NDArray[np.float64],
        covar: float | NDArray[np.float64],
        *args: *Args,
    ) -> np.float64 | NDArray[np.float64]:
        return super().ppf(probability, *(covar, *args))

    @override
    def mrl(
        self,
        time: float | NDArray[np.float64],
        covar: float | NDArray[np.float64],
        *args: *Args,
    ) -> np.float64 | NDArray[np.float64]:
        return super().mrl(time, *(covar, *args))

    @override
    def rvs(
        self,
        covar: float | NDArray[np.float64],
        *args: *Args,
        size: int | tuple[int, int] = 1,
        seed: Optional[int] = None,
    ) -> np.float64 | NDArray[np.float64]:
        """
        Random variable sampling.

        Parameters
        ----------
        size : int or (int, int)
            Shape of the sample.
        covar : np.ndarray
            Covariate values. Shapes can be ``(n_values,)`` or ``(n_assets, n_values)``.
        *args : variable number of np.ndarray
            Any variables needed to compute the function. Those variables must be
            broadcastable with ``covar``. They may exist and result from method chaining due to nested class instantiation.

        seed : int, default None
            Random seed.

        Returns
        -------
        np.ndarray
            Sample of random lifetimes.
        """
        return super().rvs(*(covar, *args), size=size, seed=seed)

    @override
    def ls_integrate(
        self,
        func: Callable[[float | NDArray[np.float64]], NDArray[np.float64]],
        a: float | NDArray[np.float64],
        b: float | NDArray[np.float64],
        covar: float | NDArray[np.float64],
        *args: *Args,
        deg: int = 10,
    ) -> np.float64 | NDArray[np.float64]:
        return super().ls_integrate(func, a, b, *(covar, *args), deg=deg)

    @override
    def mean(
        self, covar: float | NDArray[np.float64], *args: *Args
    ) -> np.float64 | NDArray[np.float64]:
        return super().mean(*(covar, *args))

    @override
    def var(
        self, covar: float | NDArray[np.float64], *args: *Args
    ) -> np.float64 | NDArray[np.float64]:
        return super().var(*(covar, *args))

    @override
    def median(
        self, covar: float | NDArray[np.float64], *args: *Args
    ) -> np.float64 | NDArray[np.float64]:
        return super().median(*(covar, *args))

    @abstractmethod
    def dhf(
        self,
        time: float | NDArray[np.float64],
        covar: float | NDArray[np.float64],
        *args: *Args,
    ) -> np.float64 | NDArray[np.float64]: ...


    @abstractmethod
    def jac_hf(
        self,
        time: float | NDArray[np.float64],
        covar: float | NDArray[np.float64],
        *args: *Args,
        asarray : bool = False,
    ) -> np.float64 | NDArray[np.float64] | tuple[np.float64, ...] | tuple[NDArray[np.float64], ...]: ...

    @abstractmethod
    def jac_chf(
        self,
        time: float | NDArray[np.float64],
        covar: float | NDArray[np.float64],
        *args: *Args,
        asarray: bool = False,
    ) -> np.float64 | NDArray[np.float64] | tuple[np.float64, ...] | tuple[NDArray[np.float64], ...]: ...

    def jac_sf(
        self,
        time: float | NDArray[np.float64],
        covar: float | NDArray[np.float64],
        *args: *Args,
        asarray: bool = False,
    ) -> np.float64 | NDArray[np.float64] | tuple[np.float64, ...] | tuple[NDArray[np.float64], ...]:
        jac = -self.jac_chf(time, covar, *args, asarray=True) * self.sf(time, covar, *args)
        if not asarray:
            return np.unstack(jac)
        return jac

    def jac_cdf(
        self,
        time: float | NDArray[np.float64],
        covar: float | NDArray[np.float64],
        *args: *Args,
        asarray: bool = False,
    ) -> np.float64 | NDArray[np.float64] | tuple[np.float64, ...] | tuple[NDArray[np.float64], ...]:
        jac = -self.jac_sf(time, covar, *args, asarray=True)
        if not asarray:
            return np.unstack(jac)
        return jac

    def jac_pdf(
        self,
        time: float | NDArray[np.float64],
        covar: float | NDArray[np.float64],
        *args: *Args,
        asarray: bool = False,
    ) -> np.float64 | NDArray[np.float64] | tuple[np.float64, ...] | tuple[NDArray[np.float64], ...]:
        jac = self.jac_hf(time, covar, *args, asarray=True) * self.sf(
            time, covar, *args
        ) + self.jac_sf(time, covar, *args, asarray=True) * self.hf(time, covar, *args)
        if not asarray:
            return np.unstack(jac)
        return jac

    @override
    def freeze(
        self, covar: float | NDArray[np.float64], *args: *Args
    ) -> FrozenLifetimeRegression:
        from relife.lifetime_model import FrozenLifetimeRegression

        args_names = self.args_names
        if len((covar, *args)) != len(args_names):
            raise ValueError(
                f"Expected {args_names} positional arguments but got only {len((covar, *args))} arguments"
            )
        frozen_model = FrozenLifetimeRegression(self)
        frozen_model.freeze_args(**{k: v for (k, v) in zip(args_names, (covar, *args))})
        return frozen_model

    def fit(
        self,
        time: NDArray[np.float64],
        covar: float | NDArray[np.float64],
        /,
        *args: *Args,
        event: Optional[NDArray[np.bool_]] = None,
        entry: Optional[NDArray[np.float64]] = None,
        departure: Optional[NDArray[np.float64]] = None,
        **kwargs: Any,
    ) -> Self:
        lifetime_data = lifetime_data_factory(
            time,
            covar,
            *args,
            event=event,
            entry=entry,
            departure=departure,
        )
        covar = np.atleast_2d(np.asarray(covar, dtype=np.float64))
        self.covar_effect._parameters.nodedata = {f"coef_{i}": 0.0 for i in range(covar.shape[-1])}
        maximum_likelihood_estimation(
            self,
            lifetime_data,
            **kwargs,
        )
        return self

    def __getattribute__(self, item):
        match item:
            case "dhf" | "jac_hf" | "jac_chf" | "jac_pdf" | "jac_sf" | "jac_cdf":
                if np.any(np.isnan(self.params)):
                    raise ValueError(
                        f"Can't call {item} if params are not set. Got {self.params} params"
                    )
                return super().__getattribute__(item)
            case _:
                return super().__getattribute__(item)


NonParametricEstimation = NewType(
    "NonParametricEstimation",
    dict[
        str,
        tuple[NDArray[np.float64], NDArray[np.float64], Optional[NDArray[np.float64]]],
    ],
)


class NonParametricLifetimeModel(ABC):
    estimations: Optional[NonParametricEstimation]

    def __init__(self):
        self.estimations = None

    @abstractmethod
    def fit(
        self,
        time: float | NDArray[np.float64],
        /,
        event: Optional[NDArray[np.bool_]] = None,
        entry: Optional[NDArray[np.float64]] = None,
        departure: Optional[NDArray[np.float64]] = None,
    ) -> Self: ...

    @property
    def plot(self) -> PlotSurvivalFunc:
        if self.estimations is None:
            raise ValueError
        return PlotSurvivalFunc(self)
