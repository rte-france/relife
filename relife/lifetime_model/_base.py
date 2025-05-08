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
from numpy._typing import DTypeLike
from numpy.typing import NDArray
from scipy.optimize import newton
from typing_extensions import override

from relife import ParametricModel
from relife._plots import PlotSurvivalFunc
from relife.data import lifetime_data_factory
from relife.likelihood import maximum_likelihood_estimation
from relife.likelihood.maximum_likelihood_estimation import FittingResults
from relife.sample import RandomTimeSamplerMixin
from relife.quadrature import LebesgueStieltjesMixin

if TYPE_CHECKING:
    from relife.lifetime_model import (
        FrozenLifetimeDistribution,
        FrozenLifetimeRegression,
        FrozenParametricLifetimeModel,
    )

    from ._structural_type import FittableParametricLifetimeModel

Args = TypeVarTuple("Args")


class ParametricLifetimeModel(
    ParametricModel,
    RandomTimeSamplerMixin[*Args],
    LebesgueStieltjesMixin[*Args],
    Generic[*Args],
    ABC
):
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

    def ppf(
        self,
        probability: float | NDArray[np.float64],
        *args: *Args,
    ) -> np.float64 | NDArray[np.float64]:
        return self.isf(1 - probability, *args)

    def median(self, *args: *Args) -> np.float64 | NDArray[np.float64]:
        return self.ppf(np.array(0.5), *args)

    def rvs(self, *args: *Args, size: int | tuple[int, int] = 1, seed: Optional[int] = None) -> NDArray[DTypeLike]:
        return self.freeze(*args).rvs(siz=size, seed=seed)

    @property
    def plot(self) -> PlotSurvivalFunc:
        """Plot"""
        return PlotSurvivalFunc(self)

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

    # @override
    # def freeze(self) -> FrozenLifetimeDistribution:
    #     from relife.lifetime_model import FrozenLifetimeDistribution
    #
    #     return FrozenLifetimeDistribution(self)

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

    def freeze(
        self, covar: float | NDArray[np.float64], *args: *Args
    ) -> FrozenLifetimeRegression:
        from .frozen import FrozenLifetimeRegression
        return  FrozenLifetimeRegression(self).collect_args(covar, *args)

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
