from __future__ import annotations

from abc import ABC, abstractmethod
from typing import (
    Any,
    Callable,
    Generic,
    Literal,
    Optional,
    Self,
    TypeVarTuple,
    overload,
)

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import Bounds, minimize, newton
from typing_extensions import override

from relife import FittingResults, ParametricModel
from relife.data import LifetimeData
from relife.likelihood import LikelihoodFromLifetimes
from relife.quadrature import (
    check_and_broadcast_bounds,
    legendre_quadrature,
    unweighted_laguerre_quadrature,
)

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
    def sf(self, time: float | NDArray[np.float64], *args: *Args) -> np.float64 | NDArray[np.float64]:
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
    def hf(self, time: float | NDArray[np.float64], *args: *Args) -> np.float64 | NDArray[np.float64]:
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
    def chf(self, time: float | NDArray[np.float64], *args: *Args) -> np.float64 | NDArray[np.float64]:
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
    def pdf(self, time: float | NDArray[np.float64], *args: *Args) -> np.float64 | NDArray[np.float64]:
        try:
            return self.sf(time, *args) * self.hf(time, *args)
        except NotImplementedError as err:
            class_name = type(self).__name__
            raise NotImplementedError(
                f"""
            {class_name} must implement pdf or the above functions
            """
            ) from err

    def cdf(self, time: float | NDArray[np.float64], *args: *Args) -> np.float64 | NDArray[np.float64]:
        return 1 - self.sf(time, *args)

    def ppf(
        self,
        probability: float | NDArray[np.float64],
        *args: *Args,
    ) -> np.float64 | NDArray[np.float64]:
        return self.isf(1 - probability, *args)

    def median(self, *args: *Args) -> np.float64 | NDArray[np.float64]:
        return self.ppf(np.array(0.5), *args)

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

    @overload
    def rvs(
        self,
        *args: *Args,
        size: int | tuple[int] | tuple[int, int] = 1,
        return_event: Literal[False] = False,
        return_entry: Literal[False] = False,
        seed: Optional[int] = None,
    ) -> np.float64 | NDArray[np.float64]: ...

    @overload
    def rvs(
        self,
        *args: *Args,
        size: int | tuple[int] | tuple[int, int] = 1,
        return_event: Literal[True] = True,
        return_entry: Literal[False] = False,
        seed: Optional[int] = None,
    ) -> tuple[np.float64 | NDArray[np.float64], np.float64 | NDArray[np.float64]]: ...

    @overload
    def rvs(
        self,
        *args: *Args,
        size: int | tuple[int] | tuple[int, int] = 1,
        return_event: Literal[False] = False,
        return_entry: Literal[True] = True,
        seed: Optional[int] = None,
    ) -> tuple[np.float64 | NDArray[np.float64], np.float64 | NDArray[np.float64]]: ...

    @overload
    def rvs(
        self,
        *args: *Args,
        size: int | tuple[int] | tuple[int, int] = 1,
        return_event: Literal[True] = True,
        return_entry: Literal[True] = True,
        seed: Optional[int] = None,
    ) -> tuple[
        np.float64 | NDArray[np.float64], np.float64 | NDArray[np.float64], np.float64 | NDArray[np.float64]
    ]: ...

    def rvs(
        self,
        *args: *Args,
        size: int | tuple[int] | tuple[int, int] = 1,
        return_event: bool = False,
        return_entry: bool = False,
        seed: Optional[int] = None,
    ) -> (
        np.float64
        | NDArray[np.float64]
        | tuple[np.float64 | NDArray[np.float64], np.float64 | NDArray[np.float64]]
        | tuple[np.float64 | NDArray[np.float64], np.float64 | NDArray[np.float64], np.float64 | NDArray[np.float64]]
    ):
        rs = np.random.RandomState(seed=seed)
        probability = rs.uniform(size=size)
        time = self.isf(probability, *args)
        event = np.ones_like(time, dtype=np.bool_) if isinstance(time, np.ndarray) else True
        entry = np.zeros_like(time, dtype=np.float64) if isinstance(time, np.ndarray) else np.float64(0)
        if not return_event and not return_entry:
            return time
        elif return_event and not return_entry:
            return time, event
        elif not return_event and return_entry:
            return time, entry
        else:
            return time, event, entry

    def sample_lifetime_data(
        self,
        *args: *Args,
        size: int | tuple[int] | tuple[int, int] = 1,
        window: tuple[float, float] = (0.0, np.inf),
        seed: Optional[int] = None,
    ) -> LifetimeData:
        time, event, entry = self.rvs(*args, size=size, return_event=True, return_entry=True, seed=seed)
        t0, tf = window
        entry = np.where(time > t0, np.full_like(time, t0), entry)
        time = np.where(time > tf, np.full_like(time, tf), time)
        event[time > tf] = False
        selection = t0 <= time <= tf
        asset_id, sample_id = np.where(selection)
        args = tuple((np.take(arg, asset_id) for arg in args))
        return LifetimeData(
            time[selection].copy(), event=event[selection].copy(), entry=entry[selection].copy(), args=args
        )

    # @property
    # def plot(self) -> PlotSurvivalFunc:
    #     """Plot"""
    #     return PlotSurvivalFunc(self)

    @property
    @abstractmethod
    def args_names(self) -> tuple[str, ...]:
        ...

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
        from relife import freeze
        from relife.lifetime_model import (
            FrozenParametricLifetimeModel,
            LifetimeDistribution,
        )

        model = self
        if bool(args):
            model = freeze(self, *args)
        model: LifetimeDistribution | FrozenParametricLifetimeModel

        def integrand(x: NDArray[np.float64]) -> NDArray[np.float64]:
            #  x.shape == (deg,), (deg, n) or (deg, m, n)
            # fx : (d_1, ..., d_i, deg), (d_1, ..., d_i, deg, n) or (d_1, ..., d_i, deg, m, n)
            fx = func(x)
            if fx.shape[-len(x.shape) :] != x.shape:
                raise ValueError(
                    f"""
                    func can't squeeze input dimensions. If x has shape (d_1, ..., d_i), func(x) must have shape (..., d_1, ..., d_i).
                    Ex : if x.shape == (m, n), func(x).shape == (..., m, n).
                    """
                )
            if x.ndim == 3:  # reshape because model.pdf is tested only for input ndim <= 2
                deg, m, n = x.shape
                x = np.rollaxis(x, 1).reshape(m, -1)  # (m, deg*n), roll on m because axis 0 must align with m of args
                pdf = model.pdf(x)  # (m, deg*n)
                pdf = np.rollaxis(pdf.reshape(m, deg, n), 1, 0)  #  (deg, m, n)
            else:  # ndim == 1 | 2
                # reshape to (1, deg*n) or (1, deg), ie place 1 on axis 0 to allow broadcasting with m of args
                pdf = model.pdf(x.reshape(1, -1))  # (1, deg*n) or (1, deg)
                pdf = pdf.reshape(x.shape)  # (deg, n) or (deg,)

            # (d_1, ..., d_i, deg) or (d_1, ..., d_i, deg, n) or (d_1, ..., d_i, deg, m, n)
            return fx * pdf

        # if isinstance(model, AgeReplacementModel):
        #     ar, args = frozen_model.args[0], frozen_model.args[1:]
        #     b = np.minimum(ar, b)
        #     w = np.where(b == ar, func(ar) * model.baseline.sf(ar, *args), 0.)

        arr_a, arr_b = check_and_broadcast_bounds(a, b)  # (), (n,) or (m, n)
        if np.any(arr_a > arr_b):
            raise ValueError("Bound values a must be lower than values of b")

        # model_args_nb_assets = getattr(frozen_model, "args_nb_assets", 1)
        # if arr_a.ndim == 2:
        #     if arr_a.shape[0] not in (
        #         1,
        #         model_args_nb_assets,
        #     ) and model_args_nb_assets not in (1, arr_a.shape[0]):
        #         raise ValueError(
        #             f"Incompatible bounds with model. Model has {model_args_nb_assets} nb_assets but a and b have shape {a.shape}, {b.shape}"
        #         )

        bound_b = model.isf(1e-4)  #  () or (m, 1), if (m, 1) then arr_b.shape == (m, 1) or (m, n)
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

    def mrl(self, time: float | NDArray[np.float64], *args: *Args) -> np.float64 | NDArray[np.float64]:
        sf = self.sf(time, *args)
        ls = self.ls_integrate(lambda x: x - time, time, np.array(np.inf), *args)
        if sf.ndim < 2:  # 2d to 1d or 0d
            ls = np.squeeze(ls)
        return ls / sf


class FittableParametricLifetimeModel(ParametricLifetimeModel[*Args], ABC):

    @abstractmethod
    def init_params_structure(self, *args: *Args) -> None:
        """
        Initialize the number of parameters with respect to addtional args
        Eg: For LifetimeDistribution, it changes nothing. For LifetimeRegression, covar changes the number of parameters
        in CovarEffect.
        """

    @abstractmethod
    def init_params_values(self, lifetime_data: LifetimeData) -> None: ...

    @property
    def params_bounds(self) -> Optional[Bounds]:
        return None

    def fit(
        self,
        time: NDArray[np.float64],
        *args: *Args,
        event: Optional[NDArray[np.bool_]] = None,
        entry: Optional[NDArray[np.float64]] = None,
        departure: Optional[NDArray[np.float64]] = None,
        **kwargs: Any,
    ) -> Self:
        # initialize params structure (number of parameters in params tree)
        self.init_params_structure(*args)
        lifetime_data = LifetimeData(time, event=event, entry=entry, departure=departure, args=args)
        return self.fit_from_lifetime_data(lifetime_data, **kwargs)

    def fit_from_lifetime_data(self, lifetime_data: LifetimeData, **kwargs) -> Self:

        # initialize params values
        self.init_params_values(lifetime_data)
        likelihood = LikelihoodFromLifetimes(self, lifetime_data)

        # configure and run the optimizer
        minimize_kwargs = {
            "method": kwargs.get("method", "L-BFGS-B"),
            "constraints": kwargs.get("constraints", ()),
            "bounds": kwargs.get("bounds", self.params_bounds),
            "x0": kwargs.get("x0", self.params),
        }
        optimizer = minimize(
            likelihood.negative_log,
            minimize_kwargs.pop("x0"),
            jac=None if not likelihood.hasjac else likelihood.jac_negative_log,
            callback=lambda x: print(x),
            **minimize_kwargs,
        )
        # set fitted params
        self.params = optimizer.x

        # compute parameters variance (Hessian inverse)
        hessian_inverse = np.linalg.inv(likelihood.hessian())

        # set fitting_results
        self.fitting_results = FittingResults(len(lifetime_data), optimizer, var=hessian_inverse)

        return self


class LifetimeDistribution(FittableParametricLifetimeModel[()], ABC):
    """
    Base class for distribution model.
    """
    @property
    def args_names(self) -> tuple[()]:
        return ()

    def init_params_structure(self) -> None:
        pass

    def init_params_values(self, lifetime_data: LifetimeData) -> None:
        param0 = np.ones(self.nb_params, dtype=np.float64)
        param0[-1] = 1 / np.median(lifetime_data.complete_or_right_censored.lifetime_values)
        self.params = param0

    @property
    def params_bounds(self) -> Bounds:
        return Bounds(
            np.full(self.nb_params, np.finfo(float).resolution),
            np.full(self.nb_params, np.inf),
        )

    @override
    def sf(self, time: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]:
        return super().sf(time)

    @override
    def isf(self, probability: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]:
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
        cumulative_hazard_rate = -np.log(probability + 1e-6)  # avoid division by zero
        return self.ichf(cumulative_hazard_rate)

    @override
    def cdf(self, time: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]:
        return super().cdf(time)

    def pdf(self, time: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]:
        return super().pdf(time)

    @override
    def ppf(self, probability: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]:
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
    def median(self) -> np.float64:
        return super().median()

    @abstractmethod
    def dhf(
        self,
        time: float | NDArray[np.float64],
    ) -> np.float64 | NDArray[np.float64]: ...

    @overload
    def jac_hf(
        self,
        time: float | NDArray[np.float64],
        *,
        asarray: Literal[False],
    ) -> tuple[np.float64 | NDArray[np.float64], ...]: ...

    @overload
    def jac_hf(
        self,
        time: float | NDArray[np.float64],
        *,
        asarray: Literal[True],
    ) -> np.float64 | NDArray[np.float64]: ...

    @abstractmethod
    def jac_hf(
        self,
        time: float | NDArray[np.float64],
        *,
        asarray: bool = False,
    ) -> np.float64 | NDArray[np.float64] | tuple[np.float64 | NDArray[np.float64], ...]: ...

    @overload
    def jac_chf(
        self,
        time: float | NDArray[np.float64],
        *,
        asarray: Literal[False],
    ) -> tuple[np.float64 | NDArray[np.float64], ...]: ...

    @overload
    def jac_chf(
        self,
        time: float | NDArray[np.float64],
        *,
        asarray: Literal[True],
    ) -> np.float64 | NDArray[np.float64]: ...

    @abstractmethod
    def jac_chf(
        self,
        time: float | NDArray[np.float64],
        *,
        asarray: bool = False,
    ) -> np.float64 | NDArray[np.float64] | tuple[np.float64 | NDArray[np.float64], ...]: ...

    @overload
    def jac_sf(
        self,
        time: float | NDArray[np.float64],
        *,
        asarray: Literal[False],
    ) -> tuple[np.float64 | NDArray[np.float64], ...]: ...

    @overload
    def jac_sf(
        self,
        time: float | NDArray[np.float64],
        *,
        asarray: Literal[True],
    ) -> np.float64 | NDArray[np.float64]: ...

    def jac_sf(
        self,
        time: float | NDArray[np.float64],
        *,
        asarray: bool = False,
    ) -> np.float64 | NDArray[np.float64] | tuple[np.float64 | NDArray[np.float64], ...]:
        jac_chf, sf = self.jac_chf(time, asarray=True), self.sf(time)
        jac = -jac_chf * sf
        if not asarray:
            return np.unstack(jac)
        return jac

    @overload
    def jac_cdf(
        self,
        time: float | NDArray[np.float64],
        *,
        asarray: Literal[False] = False,
    ) -> tuple[np.float64 | NDArray[np.float64], ...]: ...

    @overload
    def jac_cdf(
        self,
        time: float | NDArray[np.float64],
        *,
        asarray: Literal[True] = True,
    ) -> np.float64 | NDArray[np.float64]: ...

    def jac_cdf(
        self,
        time: float | NDArray[np.float64],
        *,
        asarray: bool = False,
    ) -> np.float64 | NDArray[np.float64] | tuple[np.float64 | NDArray[np.float64], ...]:
        jac = -self.jac_sf(time, asarray=True)
        if not asarray:
            return np.unstack(jac)
        return jac

    @overload
    def jac_pdf(
        self,
        time: float | NDArray[np.float64],
        *,
        asarray: Literal[False] = False,
    ) -> tuple[np.float64 | NDArray[np.float64], ...]: ...

    @overload
    def jac_pdf(
        self,
        time: float | NDArray[np.float64],
        *,
        asarray: Literal[True] = True,
    ) -> np.float64 | NDArray[np.float64]: ...

    def jac_pdf(
        self,
        time: float | NDArray[np.float64],
        *,
        asarray: bool = False,
    ) -> np.float64 | NDArray[np.float64] | tuple[np.float64 | NDArray[np.float64], ...]:
        jac_hf, hf = self.jac_hf(time, asarray=True), self.hf(time)
        jac_sf, sf = self.jac_sf(time, asarray=True), self.sf(time)
        jac = jac_hf * sf + jac_sf * hf
        if not asarray:
            return np.unstack(jac)
        return jac

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
    def rvs(
        self,
        size: int | tuple[int] | tuple[int, int] = 1,
        return_event: bool = False,
        return_entry: bool = False,
        seed: Optional[int] = None,
    ) -> (
        np.float64
        | NDArray[np.float64]
        | tuple[np.float64 | NDArray[np.float64], np.float64 | NDArray[np.float64]]
        | tuple[np.float64 | NDArray[np.float64], np.float64 | NDArray[np.float64], np.float64 | NDArray[np.float64]]
    ):
        """Random variable sampling.

        Parameters
        ----------
        return_entry
        return_event
        size : int or (int, int), default 1
            Shape of the sample.
        seed : int, default None
            Random seed.

        Returns
        -------
        np.ndarray
            Sample of random lifetimes.
        """

        return super().rvs(size=size, return_event=return_event, return_entry=return_entry, seed=seed)

    @override
    def sample_lifetime_data(
        self,
        size: int | tuple[int] | tuple[int, int] = 1,
        window: tuple[float, float] = (0.0, np.inf),
        seed: Optional[int] = None,
    ) -> LifetimeData:
        return super().sample_lifetime_data(size=size, window=window, seed=seed)

    @override
    def ls_integrate(
        self,
        func: Callable[[float | NDArray[np.float64]], NDArray[np.float64]],
        a: float | NDArray[np.float64],
        b: float | NDArray[np.float64],
        deg: int = 10,
    ) -> NDArray[np.float64]:

        return super().ls_integrate(func, a, b, deg=deg)


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
        covar: NDArray[np.float64] = np.asarray(covar)  # (nb_coef,) or (m, nb_coef)
        if covar.ndim > 2:
            raise ValueError(f"Invalid covar shape. Expected (nb_coef,) or (m, nb_coef) but got {covar.shape}")
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
        self, covar: float | NDArray[np.float64], *, asarray: bool = False
    ) -> np.float64 | NDArray[np.float64] | tuple[np.float64, ...] | tuple[NDArray[np.float64], ...]:
        """
        Compute the Jacobian of the covariates effect.
        If covar.shape : () or (nb_coef,) => out.shape : (nb_coef,)
        If covar.shape : (m, nb_coef) => out.shape : (nb_coef, m, 1)
        """
        covar: NDArray[np.float64] = np.asarray(covar)  # (), (nb_coef,) or (m, nb_coef)
        g = self.g(covar)  # () or (m, 1)
        jac = covar.T.reshape(self.nb_coef, -1, 1) * g  # (nb_coef, m, 1)
        if covar.ndim <= 1:
            jac = jac.reshape(self.nb_coef)  # (nb_coef,) or (nb_coef, m, 1)
        if not asarray:
            return np.unstack(jac, axis=0)  # tuple
        return jac  # (nb_coef, m, 1)


# note that LifetimeRegression does not preserve generic : at the moment, additional args are supposed to be always float | NDArray[np.float64]
class LifetimeRegression(
    FittableParametricLifetimeModel[float | NDArray[np.float64], *tuple[float | NDArray[np.float64], ...]], ABC
):
    """
    Base class for regression model.

    At least one positional covar arg and 0 or more additional args (variable number) : https://peps.python.org/pep-0646/#unpacking-unbounded-tuple-types
    """

    def __init__(
        self,
        baseline: FittableParametricLifetimeModel[*tuple[float | NDArray[np.float64], ...]],
        coefficients: tuple[Optional[float], ...] = (None,),
    ):
        super().__init__()
        self.covar_effect = CovarEffect(coefficients)
        self.baseline = baseline

    @property
    def args_names(self) -> tuple[str, *tuple[str, ...]]:
        return ("covar",) + self.baseline.args_names


    def init_params_structure(self, covar: float | NDArray[np.float64], *args: float | NDArray[np.float64]) -> None:
        covar = np.atleast_2d(np.asarray(covar, dtype=np.float64))
        self.covar_effect._parameters.nodedata = {f"coef_{i}": 0.0 for i in range(covar.shape[-1])}
        self.baseline.init_params_structure(*args)

    def init_params_values(self, lifetime_data: LifetimeData) -> None:
        self.baseline.init_params_values(lifetime_data)
        param0 = np.zeros_like(self.params, dtype=np.float64)
        param0[-self.baseline.params.size :] = self.baseline.params
        self.params = param0

    @property
    def params_bounds(self) -> Bounds:
        lb = np.concatenate(
            (
                np.full(self.covar_effect.nb_params, -np.inf),
                self.baseline.params_bounds.lb,
            )
        )
        ub = np.concatenate(
            (
                np.full(self.covar_effect.nb_params, np.inf),
                self.baseline.params_bounds.ub,
            )
        )
        return Bounds(lb, ub)

    @property
    def nb_coef(self) -> int:
        return self.covar_effect.nb_coef

    @override
    def sf(
        self,
        time: float | NDArray[np.float64],
        covar: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
    ) -> np.float64 | NDArray[np.float64]:
        return super().sf(time, covar, *args)

    @override
    def isf(
        self,
        probability: float | NDArray[np.float64],
        covar: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
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
        cumulative_hazard_rate = -np.log(probability + 1e-6)  # avoid division by zero
        return self.ichf(cumulative_hazard_rate, covar, *args)

    @override
    def cdf(
        self,
        time: float | NDArray[np.float64],
        covar: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
    ) -> np.float64 | NDArray[np.float64]:
        return super().cdf(time, *(covar, *args))

    def pdf(
        self,
        time: float | NDArray[np.float64],
        covar: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
    ) -> np.float64 | NDArray[np.float64]:
        return super().pdf(time, *(covar, *args))

    @override
    def ppf(
        self,
        probability: float | NDArray[np.float64],
        covar: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
    ) -> np.float64 | NDArray[np.float64]:
        return super().ppf(probability, *(covar, *args))

    @override
    def mrl(
        self,
        time: float | NDArray[np.float64],
        covar: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
    ) -> np.float64 | NDArray[np.float64]:
        return super().mrl(time, *(covar, *args))

    @override
    def ls_integrate(
        self,
        func: Callable[[float | NDArray[np.float64]], NDArray[np.float64]],
        a: float | NDArray[np.float64],
        b: float | NDArray[np.float64],
        covar: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
        deg: int = 10,
    ) -> np.float64 | NDArray[np.float64]:
        return super().ls_integrate(func, a, b, *(covar, *args), deg=deg)

    @override
    def mean(
        self, covar: float | NDArray[np.float64], *args: float | NDArray[np.float64]
    ) -> np.float64 | NDArray[np.float64]:
        return super().mean(*(covar, *args))

    @override
    def var(
        self, covar: float | NDArray[np.float64], *args: float | NDArray[np.float64]
    ) -> np.float64 | NDArray[np.float64]:
        return super().var(*(covar, *args))

    @override
    def median(
        self, covar: float | NDArray[np.float64], *args: float | NDArray[np.float64]
    ) -> np.float64 | NDArray[np.float64]:
        return super().median(*(covar, *args))

    @abstractmethod
    def dhf(
        self,
        time: float | NDArray[np.float64],
        covar: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
    ) -> np.float64 | NDArray[np.float64]: ...

    @overload
    def jac_hf(
        self,
        time: float | NDArray[np.float64],
        covar: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
        asarray: Literal[False] = False,
    ) -> tuple[np.float64 | NDArray[np.float64], ...]: ...

    @overload
    def jac_hf(
        self,
        time: float | NDArray[np.float64],
        covar: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
        asarray: Literal[True] = True,
    ) -> np.float64 | NDArray[np.float64]: ...

    @abstractmethod
    def jac_hf(
        self,
        time: float | NDArray[np.float64],
        covar: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
        asarray: bool = False,
    ) -> np.float64 | NDArray[np.float64] | tuple[np.float64 | NDArray[np.float64], ...]: ...

    @overload
    def jac_chf(
        self,
        time: float | NDArray[np.float64],
        covar: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
        asarray: Literal[False] = False,
    ) -> tuple[np.float64 | NDArray[np.float64], ...]: ...

    @overload
    def jac_chf(
        self,
        time: float | NDArray[np.float64],
        covar: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
        asarray: Literal[True] = True,
    ) -> np.float64 | NDArray[np.float64]: ...

    @abstractmethod
    def jac_chf(
        self,
        time: float | NDArray[np.float64],
        covar: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
        asarray: bool = False,
    ) -> np.float64 | NDArray[np.float64] | tuple[np.float64 | NDArray[np.float64], ...]: ...

    @overload
    def jac_sf(
        self,
        time: float | NDArray[np.float64],
        covar: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
        asarray: Literal[False] = False,
    ) -> tuple[np.float64 | NDArray[np.float64], ...]: ...

    @overload
    def jac_sf(
        self,
        time: float | NDArray[np.float64],
        covar: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
        asarray: Literal[True] = True,
    ) -> np.float64 | NDArray[np.float64]: ...

    def jac_sf(
        self,
        time: float | NDArray[np.float64],
        covar: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
        asarray: bool = False,
    ) -> np.float64 | NDArray[np.float64] | tuple[np.float64 | NDArray[np.float64], ...]:
        jac = -self.jac_chf(time, covar, *args, asarray=True) * self.sf(time, covar, *args)
        if not asarray:
            return np.unstack(jac)
        return jac

    @overload
    def jac_cdf(
        self,
        time: float | NDArray[np.float64],
        covar: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
        asarray: Literal[False] = False,
    ) -> tuple[np.float64 | NDArray[np.float64], ...]: ...

    @overload
    def jac_cdf(
        self,
        time: float | NDArray[np.float64],
        covar: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
        asarray: Literal[True] = True,
    ) -> np.float64 | NDArray[np.float64]: ...

    def jac_cdf(
        self,
        time: float | NDArray[np.float64],
        covar: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
        asarray: bool = False,
    ) -> np.float64 | NDArray[np.float64] | tuple[np.float64 | NDArray[np.float64], ...]:
        jac = -self.jac_sf(time, covar, *args, asarray=True)
        if not asarray:
            return np.unstack(jac)
        return jac

    @overload
    def jac_pdf(
        self,
        time: float | NDArray[np.float64],
        covar: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
        asarray: Literal[False] = False,
    ) -> tuple[np.float64 | NDArray[np.float64], ...]: ...

    @overload
    def jac_pdf(
        self,
        time: float | NDArray[np.float64],
        covar: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
        asarray: Literal[True] = True,
    ) -> np.float64 | NDArray[np.float64]: ...

    def jac_pdf(
        self,
        time: float | NDArray[np.float64],
        covar: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
        asarray: bool = False,
    ) -> np.float64 | NDArray[np.float64] | tuple[np.float64 | NDArray[np.float64], ...]:
        jac = self.jac_hf(time, covar, *args, asarray=True) * self.sf(time, covar, *args) + self.jac_sf(
            time, covar, *args, asarray=True
        ) * self.hf(time, covar, *args)
        if not asarray:
            return np.unstack(jac)
        return jac

    @override
    def rvs(
        self,
        covar: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
        return_event: bool = False,
        return_entry: bool = False,
        size: int | tuple[int, int] = 1,
        seed: Optional[int] = None,
    ) -> (
        np.float64
        | NDArray[np.float64]
        | tuple[np.float64 | NDArray[np.float64], np.float64 | NDArray[np.float64]]
        | tuple[np.float64 | NDArray[np.float64], np.float64 | NDArray[np.float64], np.float64 | NDArray[np.float64]]
    ):
        """
        Random variable sampling.

        Parameters
        ----------
        return_entry
        return_event
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
        return super().rvs(*(covar, *args), size=size, return_event=return_event, return_entry=return_entry, seed=seed)

    @override
    def sample_lifetime_data(
        self,
        covar: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
        size: int | tuple[int] | tuple[int, int] = 1,
        window: tuple[float, float] = (0.0, np.inf),
        seed: Optional[int] = None,
    ) -> LifetimeData:
        return super().sample_lifetime_data(*(covar, *args), size=size, window=window, seed=seed)


class NonParametricLifetimeModel(ABC):

    @abstractmethod
    def fit(
        self,
        time: float | NDArray[np.float64],
        /,
        event: Optional[NDArray[np.bool_]] = None,
        entry: Optional[NDArray[np.float64]] = None,
        departure: Optional[NDArray[np.float64]] = None,
    ) -> Self: ...

    # @property
    # def plot(self) -> PlotSurvivalFunc:
    #     if self.estimations is None:
    #         raise ValueError
    #     return PlotSurvivalFunc(self)


class FrozenParametricLifetimeModel(ParametricModel, Generic[*Args]):

    frozen_args: tuple[*Args]

    def __init__(self, model: ParametricLifetimeModel[*Args], *args: *Args):
        super().__init__()
        if np.any(np.isnan(model.params)):
            raise ValueError("You try to freeze a model with unsetted parameters. Set params first")
        self.unfrozen_model = model
        self.frozen_args = args

    def unfreeze(self) -> ParametricLifetimeModel[*Args]:
        return self.unfrozen_model

    def hf(self, time: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]:
        return self.unfrozen_model.hf(time, *self.frozen_args)

    def chf(self, time: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]:
        return self.unfrozen_model.chf(time, *self.frozen_args)

    def sf(self, time: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]:
        return self.unfrozen_model.sf(time, *self.frozen_args)

    def pdf(self, time: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]:
        return self.unfrozen_model.pdf(time, *self.frozen_args)

    def mrl(self, time: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]:
        return self.unfrozen_model.mrl(time, *self.frozen_args)

    def moment(self, n: int) -> np.float64 | NDArray[np.float64]:
        return self.unfrozen_model.moment(n, *self.frozen_args)

    def mean(self) -> np.float64 | NDArray[np.float64]:
        return self.unfrozen_model.moment(1, *self.frozen_args)

    def var(self) -> np.float64 | NDArray[np.float64]:
        return self.unfrozen_model.moment(2, *self.frozen_args) - self.unfrozen_model.moment(1, *self.frozen_args) ** 2

    def isf(self, probability: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]:
        return self.unfrozen_model.isf(probability, *self.frozen_args)

    def ichf(self, cumulative_hazard_rate: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]:
        return self.unfrozen_model.ichf(cumulative_hazard_rate, *self.frozen_args)

    def cdf(self, time: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]:
        return self.unfrozen_model.cdf(time, *self.frozen_args)

    @overload
    def rvs(
        self,
        size: int | tuple[int] | tuple[int, int] = 1,
        return_event: Literal[False] = False,
        return_entry: Literal[False] = False,
        seed: Optional[int] = None,
    ) -> np.float64 | NDArray[np.float64]: ...

    @overload
    def rvs(
        self,
        size: int | tuple[int] | tuple[int, int] = 1,
        return_event: Literal[True] = True,
        return_entry: Literal[False] = True,
        seed: Optional[int] = None,
    ) -> tuple[np.float64 | NDArray[np.float64], np.float64 | NDArray[np.float64]]: ...

    @overload
    def rvs(
        self,
        size: int | tuple[int] | tuple[int, int] = 1,
        return_event: Literal[False] = False,
        return_entry: Literal[True] = True,
        seed: Optional[int] = None,
    ) -> tuple[np.float64 | NDArray[np.float64], np.float64 | NDArray[np.float64]]: ...

    @overload
    def rvs(
        self,
        size: int | tuple[int] | tuple[int, int] = 1,
        return_event: Literal[True] = True,
        return_entry: Literal[True] = True,
        seed: Optional[int] = None,
    ) -> tuple[
        np.float64 | NDArray[np.float64], np.float64 | NDArray[np.float64], np.float64 | NDArray[np.float64]
    ]: ...

    def rvs(
        self,
        size: int | tuple[int] | tuple[int, int] = 1,
        return_event: bool = False,
        return_entry: bool = False,
        seed: Optional[int] = None,
    ) -> (
        np.float64
        | NDArray[np.float64]
        | tuple[np.float64 | NDArray[np.float64], np.float64 | NDArray[np.float64]]
        | tuple[np.float64 | NDArray[np.float64], np.float64 | NDArray[np.float64], np.float64 | NDArray[np.float64]]
    ):
        return self.unfrozen_model.rvs(
            *self.frozen_args, size=size, return_event=return_event, return_entry=return_entry, seed=seed
        )

    def ppf(self, probability: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]:
        return self.unfrozen_model.ppf(probability, *self.frozen_args)

    def median(self) -> np.float64 | NDArray[np.float64]:
        return self.unfrozen_model.median(*self.frozen_args)

    def ls_integrate(
        self,
        func: Callable[[float | NDArray[np.float64]], NDArray[np.float64]],
        a: float | NDArray[np.float64],
        b: float | NDArray[np.float64],
        deg: int = 10,
    ) -> np.float64 | NDArray[np.float64]:
        return self.unfrozen_model.ls_integrate(func, a, b, *self.frozen_args, deg=deg)


#
# class FrozenLifetimeDistribution(FrozenParametricLifetimeModel[()]):
#
#     unfrozen_model: LifetimeDistribution
#     frozen_args: tuple[()]
#
#     @override
#     def __init__(self, model : LifetimeDistribution):
#         super().__init__(model)
#
#     @override
#     def unfreeze(self) -> LifetimeDistribution:
#         return super().unfreeze()
#
#     def dhf(
#         self,
#         time: float | NDArray[np.float64],
#     ) -> np.float64 | NDArray[np.float64]:
#         return self.unfrozen_model.dhf(time)
#
#     @overload
#     def jac_hf(
#         self,
#         time: float | NDArray[np.float64],
#         asarray: Literal[False] = False,
#     ) -> tuple[np.float64 | NDArray[np.float64], ...]: ...
#
#     @overload
#     def jac_hf(
#         self,
#         time: float | NDArray[np.float64],
#         asarray: Literal[True] = True,
#     ) -> np.float64 | NDArray[np.float64]: ...
#
#     def jac_hf(
#         self,
#         time: float | NDArray[np.float64],
#         asarray: bool = False,
#     ) -> np.float64 | NDArray[np.float64] | tuple[np.float64 | NDArray[np.float64], ...]:
#         return self.unfrozen_model.jac_hf(time, asarray=asarray)
#
#     @overload
#     def jac_chf(
#         self,
#         time: float | NDArray[np.float64],
#         asarray: Literal[False] = False,
#     ) -> tuple[np.float64 | NDArray[np.float64], ...]: ...
#
#     @overload
#     def jac_chf(
#         self,
#         time: float | NDArray[np.float64],
#         asarray: Literal[True] = True,
#     ) -> np.float64 | NDArray[np.float64]: ...
#
#     def jac_chf(
#         self,
#         time: float | NDArray[np.float64],
#         asarray: bool = False,
#     ) -> np.float64 | NDArray[np.float64] | tuple[np.float64 | NDArray[np.float64], ...]:
#         return self.unfrozen_model.jac_chf(time, asarray=asarray)
#
#     @overload
#     def jac_sf(
#         self,
#         time: float | NDArray[np.float64],
#         asarray: Literal[False] = False,
#     ) -> tuple[np.float64 | NDArray[np.float64], ...]: ...
#
#     @overload
#     def jac_sf(
#         self,
#         time: float | NDArray[np.float64],
#         asarray: Literal[True] = True,
#     ) -> np.float64 | NDArray[np.float64]: ...
#
#     def jac_sf(
#         self,
#         time: float | NDArray[np.float64],
#         asarray: bool = False,
#     ) -> np.float64 | NDArray[np.float64] | tuple[np.float64 | NDArray[np.float64], ...]:
#         return self.unfrozen_model.jac_sf(time, asarray=asarray)
#
#     @overload
#     def jac_cdf(
#         self,
#         time: float | NDArray[np.float64],
#         asarray: Literal[False] = False,
#     ) -> tuple[np.float64 | NDArray[np.float64], ...]: ...
#
#     @overload
#     def jac_cdf(
#         self,
#         time: float | NDArray[np.float64],
#         asarray: Literal[True] = True,
#     ) -> np.float64 | NDArray[np.float64]: ...
#
#     def jac_cdf(
#         self,
#         time: float | NDArray[np.float64],
#         asarray: bool = False,
#     ) -> np.float64 | NDArray[np.float64] | tuple[np.float64 | NDArray[np.float64], ...]:
#         return self.unfrozen_model.jac_cdf(time, asarray=asarray)
#
#     @overload
#     def jac_pdf(
#         self,
#         time: float | NDArray[np.float64],
#         asarray: Literal[False] = False,
#     ) -> tuple[np.float64 | NDArray[np.float64], ...]: ...
#
#     @overload
#     def jac_pdf(
#         self,
#         time: float | NDArray[np.float64],
#         asarray: Literal[True] = True,
#     ) -> np.float64 | NDArray[np.float64]: ...
#
#     def jac_pdf(
#         self,
#         time: float | NDArray[np.float64],
#         asarray: bool = False,
#     ) -> np.float64 | NDArray[np.float64] | tuple[np.float64 | NDArray[np.float64], ...]:
#         return self.unfrozen_model.jac_pdf(time, asarray=asarray)


class FrozenLifetimeRegression(
    FrozenParametricLifetimeModel[float | NDArray[np.float64], *tuple[float | NDArray[np.float64], ...]]
):

    unfrozen_model: LifetimeRegression
    frozen_args: tuple[float | NDArray[np.float64], *tuple[float | NDArray[np.float64], ...]]

    @override
    def __init__(
        self, model: LifetimeRegression, covar: float | NDArray[np.float64], *args: float | NDArray[np.float64]
    ):
        super().__init__(model, *(covar, *args))

    @override
    def unfreeze(self) -> LifetimeRegression:
        return super().unfreeze()

    @property
    def nb_coef(self) -> int:
        return self.unfrozen_model.nb_coef

    @property
    def covar(self) -> float | NDArray[np.float64]:
        return self.frozen_args[0]

    @covar.setter
    def covar(self, value: float | NDArray[np.float64]) -> None:
        self.frozen_args = (value,) + self.frozen_args[1:]

    def dhf(
        self,
        time: float | NDArray[np.float64],
    ) -> np.float64 | NDArray[np.float64]:
        return self.unfrozen_model.dhf(time, self.frozen_args[0], *self.frozen_args[1:])

    @overload
    def jac_hf(
        self,
        time: float | NDArray[np.float64],
        asarray: Literal[False] = False,
    ) -> tuple[np.float64 | NDArray[np.float64], ...]: ...

    @overload
    def jac_hf(
        self,
        time: float | NDArray[np.float64],
        asarray: Literal[True] = True,
    ) -> np.float64 | NDArray[np.float64]: ...

    def jac_hf(
        self,
        time: float | NDArray[np.float64],
        asarray: bool = False,
    ) -> np.float64 | NDArray[np.float64] | tuple[np.float64, ...] | tuple[NDArray[np.float64], ...]:
        return self.unfrozen_model.jac_hf(time, self.frozen_args[0], *self.frozen_args[1:], asarray=asarray)

    @overload
    def jac_chf(
        self,
        time: float | NDArray[np.float64],
        asarray: Literal[False] = False,
    ) -> tuple[np.float64 | NDArray[np.float64], ...]: ...

    @overload
    def jac_chf(
        self,
        time: float | NDArray[np.float64],
        asarray: Literal[True] = True,
    ) -> np.float64 | NDArray[np.float64]: ...

    def jac_chf(
        self,
        time: float | NDArray[np.float64],
        asarray: bool = False,
    ) -> np.float64 | NDArray[np.float64] | tuple[np.float64, ...] | tuple[NDArray[np.float64], ...]:
        return self.unfrozen_model.jac_chf(time, self.frozen_args[0], *self.frozen_args[1:], asarray=asarray)

    @overload
    def jac_sf(
        self,
        time: float | NDArray[np.float64],
        asarray: Literal[False] = False,
    ) -> tuple[np.float64 | NDArray[np.float64], ...]: ...

    @overload
    def jac_sf(
        self,
        time: float | NDArray[np.float64],
        asarray: Literal[True] = True,
    ) -> np.float64 | NDArray[np.float64]: ...

    def jac_sf(
        self,
        time: float | NDArray[np.float64],
        asarray: bool = False,
    ) -> np.float64 | NDArray[np.float64] | tuple[np.float64, ...] | tuple[NDArray[np.float64], ...]:
        return self.unfrozen_model.jac_sf(time, self.frozen_args[0], *self.frozen_args[1:], asarray=asarray)

    @overload
    def jac_cdf(
        self,
        time: float | NDArray[np.float64],
        asarray: Literal[False] = False,
    ) -> tuple[np.float64 | NDArray[np.float64], ...]: ...

    @overload
    def jac_cdf(
        self,
        time: float | NDArray[np.float64],
        asarray: Literal[True] = True,
    ) -> np.float64 | NDArray[np.float64]: ...

    def jac_cdf(
        self,
        time: float | NDArray[np.float64],
        asarray: bool = False,
    ) -> np.float64 | NDArray[np.float64] | tuple[np.float64, ...] | tuple[NDArray[np.float64], ...]:
        return self.unfrozen_model.jac_cdf(time, self.frozen_args[0], *self.frozen_args[1:], asarray=asarray)

    @overload
    def jac_pdf(
        self,
        time: float | NDArray[np.float64],
        asarray: Literal[False] = False,
    ) -> tuple[np.float64 | NDArray[np.float64], ...]: ...

    @overload
    def jac_pdf(
        self,
        time: float | NDArray[np.float64],
        asarray: Literal[True] = True,
    ) -> np.float64 | NDArray[np.float64]: ...

    def jac_pdf(
        self,
        time: float | NDArray[np.float64],
        asarray: bool = False,
    ) -> np.float64 | NDArray[np.float64] | tuple[np.float64, ...] | tuple[NDArray[np.float64], ...]:
        return self.unfrozen_model.jac_pdf(time, self.frozen_args[0], *self.frozen_args[1:], asarray=asarray)
