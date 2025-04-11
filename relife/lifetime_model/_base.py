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

from relife import ParametricModel
from relife._plots import PlotSurvivalFunc
from relife.data import lifetime_data_factory
from relife.likelihood import maximum_likelihood_estimation
from relife.likelihood.maximum_likelihood_estimation import FittingResults
from relife.quadrature import (
    legendre_quadrature,
    ls_integrate,
    squeeze_like,
    unweighted_laguerre_quadrature,
)

from .frozen_model import FrozenParametricLifetimeModel
from .._args import get_nb_assets

if TYPE_CHECKING:
    from ._fittable_type import FittableParametricLifetimeModel

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

    frozen: bool = False

    def __init__(self):
        super().__init__()

    @abstractmethod
    def hf(
        self, time: float | NDArray[np.float64], *args: *Args
    ) -> NDArray[np.float64]:
        if hasattr(self, "pdf") and hasattr(self, "sf"):
            return self.pdf(time, *args) / self.sf(time, *args)
        if hasattr(self, "sf"):
            raise NotImplementedError(
                """
                ReLife does not implement hf as the derivate of chf yet. Consider adding it in future versions
                see: https://docs.scipy.org/doc/scipy-1.11.4/reference/generated/scipy.misc.derivative.html
                or : https://github.com/maroba/findiff
                """
            )
        class_name = type(self).__name__
        raise NotImplementedError(
            f"""
            {class_name} must implement concrete hf function
            """
        )

    @abstractmethod
    def chf(
        self, time: float | NDArray[np.float64], *args: *Args
    ) -> NDArray[np.float64]:
        if hasattr(self, "sf"):
            return -np.log(self.sf(time, *args))
        if hasattr(self, "pdf") and hasattr(self, "hf"):
            return -np.log(self.pdf(time, *args) / self.hf(time, *args))
        if hasattr(self, "hf"):
            raise NotImplementedError(
                """
                ReLife does not implement chf as the integration of hf yet. Consider adding it in future versions
                """
            )
        class_name = type(self).__name__
        raise NotImplementedError(
            f"""
        {class_name} must implement concrete chf or at least concrete hf function
        """
        )

    @abstractmethod
    def sf(
        self, time: float | NDArray[np.float64], *args: *Args
    ) -> NDArray[np.float64]:
        if hasattr(self, "chf"):
            return np.exp(
                -self.chf(
                    time,
                    *args,
                )
            )
        if hasattr(self, "pdf") and hasattr(self, "hf"):
            return self.pdf(time, *args) / self.hf(time, *args)

        class_name = type(self).__name__
        raise NotImplementedError(
            f"""
        {class_name} must implement concrete sf function
        """
        )

    @abstractmethod
    def pdf(
        self, time: float | NDArray[np.float64], *args: *Args
    ) -> NDArray[np.float64]:
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
    ) -> NDArray[np.float64]:
        sf = self.sf(time, *args)
        ls = self.ls_integrate(lambda x: x - time, time, np.array(np.inf), *args)
        if sf.ndim < 2:  # 2d to 1d or 0d
            ls = np.squeeze(ls)
        return ls / sf

    def isf(
        self,
        probability: float | NDArray[np.float64],
        *args: *Args,
    ) -> NDArray[np.float64]:
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
    ) -> NDArray[np.float64]:
        return newton(
            lambda x: self.chf(x, *args) - cumulative_hazard_rate,
            x0=np.zeros_like(cumulative_hazard_rate),
        )

    def cdf(
        self, time: float | NDArray[np.float64], *args: *Args
    ) -> NDArray[np.float64]:
        return 1 - self.sf(time, *args)

    def rvs(
        self,
        *args: *Args,
        size: int = 1,
        seed: Optional[int] = None,
    ) -> NDArray[np.float64]:
        """Random variable sample.

        Parameters
        ----------
        *args : variadic arguments required by the function
        size : int, default 1
            Sized of the generated sample.
        seed : int, default None
            Random seed.

        Returns
        -------
        ndarray of shape (size, )
            Sample of random lifetimes.
        """
        generator = np.random.RandomState(seed=seed)
        probability = generator.uniform(size=size)
        return self.isf(probability, *args)

    def ppf(
        self,
        probability: float | NDArray[np.float64],
        *args: *Args,
    ) -> NDArray[np.float64]:
        return self.isf(1 - probability, *args)

    def ls_integrate(
        self,
        func: Callable[[float | NDArray[np.float64]], NDArray[np.float64]],
        a: float | NDArray[np.float64],
        b: float | NDArray[np.float64],
        *args: *Args,
        deg: int = 100,
    ) -> NDArray[np.float64]:
        r"""
        Lebesgue-Stieltjes integration.

        The Lebesgue-Stieljes intregration of a function with respect to the lifetime core
        taking into account the probability density function and jumps

        The Lebesgue-Stieltjes integral is:

        .. math::

            \int_a^b g(x) \mathrm{d}F(x) = \int_a^b g(x) f(x)\mathrm{d}x +
            \sum_i g(a_i) w_i

        where:

        - :math:`F` is the cumulative distribution function,
        - :math:`f` the probability density function of the lifetime core,
        - :math:`a_i` and :math:`w_i` are the points and weights of the jumps.

        Parameters
        ----------
        func : callable (in : 1 ndarray, out : 1 ndarray)
            The callable must have only one ndarray object as argument and returns one ndarray object
        a : ndarray (max dim of 2)
            Lower bound(s) of integration.
        b : ndarray (max dim of 2)
            Upper bound(s) of integration. If lower bound(s) is infinite, use np.inf as value.)
        deg : int, default 100
            Degree of the polynomials interpolation

        Returns
        -------
        2d ndarray
            Lebesgue-Stieltjes integral of func with respect to `cdf` from `a`
            to `b`.

        Notes
        -----
        `ls_integrate` operations rely on arguments number of dimensions passed in `a`, `b`, `*args` or
        any other variable referenced in `func`. Because `func` callable is not easy to inspect, either one must specify
        the maximum number of dimensions used (0, 1 or 2), or `ls_integrate` converts all these objects to 2d-array.
        Currently, the second option is prefered. That's why, returns are always 2d-array.
        """

        arr_a = np.asarray(a)  #  (m, n) or (n,)
        arr_b = np.asarray(b)  #  (m, n) or (n,)
        arr_a, arr_b = np.broadcast_arrays(arr_a, arr_b)

        frozen_model = self.freeze(self, *args)
        if get_nb_assets(*frozen_model.args) > 1:
            if arr_a.ndim != 2 and arr_b.ndim != 0:
                raise ValueError

        def integrand(x: float | NDArray[np.float64]) -> NDArray[np.float64]:
            return func(x) * frozen_model.pdf(x)

        arr_a = arr_a.flatten()  # (m*n,) or # (n,)
        arr_b = arr_b.flatten()  # (m*n,) or # (n,)

        assert arr_a.shape == arr_b.shape

        integration = np.empty_like(arr_b)  # (m*n,) or # (n,)

        is_inf = np.isinf(arr_b)
        arr_b[is_inf] = frozen_model.isf(1e-4)

        if arr_a[is_inf].size != 0:
            integration[is_inf] = legendre_quadrature(
                integrand, arr_a[is_inf].copy(), arr_b[is_inf].copy(), deg=deg
            ) + unweighted_laguerre_quadrature(integrand, b[is_inf].copy(), deg=deg)
        if arr_a[~is_inf].size != 0:
            integration[~is_inf] = legendre_quadrature(
                integrand, arr_a[~is_inf].copy(), arr_b[~is_inf].copy(), deg=deg
            )

        shape = np.asarray(a).shape
        if np.asarray(b).ndim > len(shape):
            shape = np.asarray(b).shape

        return integration.reshape(shape)

    def moment(self, n: int, *args: *Args) -> NDArray[np.float64]:
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
        ls = self.ls_integrate(
            lambda x: x**n,
            np.array(0.0),
            np.array(np.inf),
            *args,
        )
        ndim = max(map(np.ndim, *args), default=0)
        if ndim < 2:  # 2d to 1d or 0d
            ls = np.squeeze(ls)
        return ls

    def mean(self, *args: *Args) -> NDArray[np.float64]:
        return self.moment(1, *args)

    def var(self, *args: *Args) -> NDArray[np.float64]:
        return self.moment(2, *args) - self.moment(1, *args) ** 2

    def median(self, *args: *Args) -> NDArray[np.float64]:
        return self.ppf(np.array(0.5), *args)

    def freeze(
        self,
        *args: *Args,
    ) -> FrozenParametricLifetimeModel:
        return FrozenParametricLifetimeModel(self, *args)

    @property
    def plot(self) -> PlotSurvivalFunc:
        """Plot"""
        return PlotSurvivalFunc(self)


class LifetimeDistribution(ParametricLifetimeModel[()], ABC):
    """
    Base class for distribution model.
    """

    frozen: bool = True

    @property
    def fitting_results(self) -> Optional[FittingResults]:
        return self._fitting_results

    @fitting_results.setter
    def fitting_results(self, value: FittingResults):
        self._fitting_results = value

    @override
    def sf(self, time: float | NDArray[np.float64]) -> NDArray[np.float64]:
        return super().sf(time)

    @override
    def isf(self, probability: float | NDArray[np.float64]) -> NDArray[np.float64]:
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
        cumulative_hazard_rate = -np.log(probability)
        return self.ichf(cumulative_hazard_rate)

    @override
    def cdf(self, time: float | NDArray[np.float64]) -> NDArray[np.float64]:
        return super().cdf(time)

    def pdf(self, time: float | NDArray[np.float64]) -> NDArray[np.float64]:
        return super().pdf(time)

    @override
    def ppf(self, probability: float | NDArray[np.float64]) -> NDArray[np.float64]:
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
    def rvs(self, *, size: int = 1, seed: Optional[int] = None):
        """Random variable sampling.

        Parameters
        ----------
        size : int, default 1
            Size of the sample.
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
    def jac_hf(
        self,
        time: float | NDArray[np.float64],
    ) -> NDArray[np.float64]: ...

    @abstractmethod
    def jac_chf(
        self,
        time: float | NDArray[np.float64],
    ) -> NDArray[np.float64]: ...

    @abstractmethod
    def dhf(
        self,
        time: float | NDArray[np.float64],
    ) -> NDArray[np.float64]: ...

    def jac_sf(self, time: float | NDArray[np.float64]) -> NDArray[np.float64]:
        return -self.jac_chf(time) * self.sf(time)

    def jac_cdf(self, time: float | NDArray[np.float64]) -> NDArray[np.float64]:
        return -self.jac_sf(time)

    def jac_pdf(self, time: float | NDArray[np.float64]) -> NDArray[np.float64]:
        return self.jac_hf(time) * self.sf(time) + self.jac_sf(time) * self.hf(time)

    @override
    def freeze(self) -> FrozenParametricLifetimeModel:
        return FrozenParametricLifetimeModel(self)

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


class CovarEffect(ParametricModel):
    """
    Covariates effect.

    Parameters
    ----------
    coef : tuple of float or tuple of None, optional
        Coefficients of the covariates effect. Values can be None.
    """

    def __init__(self, coef: tuple[float, ...] | tuple[None] = (None,)):
        super().__init__()
        self.set_params(**{f"coef_{i}": v for i, v in enumerate(coef)})

    def g(self, covar: float | NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute the covariates effect.

        Parameters
        ----------
        covar : np.ndarray of shape (k, ) or (m, k)
            Covariate values. Should have shape (k, ) or (m, k) where m is
            the number of assets and k is the number of covariates.

        Returns
        -------
        np.ndarray
            The covariate effect values, with shape (1,) or (m, 1).

        Raises
        ------
        ValueError
            If the number of covariates does not match the number of parameters.
        """
        covar: NDArray[np.float64] = np.asarray(covar)
        covar = np.atleast_2d(covar)
        if covar.ndim > 2:
            raise ValueError
        if covar.shape[-1] != self.nb_params:
            raise ValueError(
                f"Invalid number of covar : expected {self.nb_params}, got {covar.shape[-1]}"
            )
        return np.exp(np.sum(self.params * covar, axis=1, keepdims=True))

    def jac_g(self, covar: float | NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute the Jacobian of the covariates effect.

        Parameters
        ----------
        covar : np.ndarray of shape (k, ) or (m, k)
            Covariate values. Should have shape (k, ) or (m, k) where m is
            the number of assets and k is the number of covariates.

        Returns
        -------
        np.ndarray of shape (nb_params, ) or (m, nb_params)
            The values of the Jacobian (eventually for m assets).
        """
        covar: NDArray[np.float64] = np.asarray(covar)
        if covar.ndim > 2:
            raise ValueError
        return covar * self.g(covar)


class LifetimeRegression(
    ParametricLifetimeModel[float | NDArray[np.float64], *Args], ABC
):
    """
    Base class for regression model.
    """

    baseline: FittableParametricLifetimeModel[*Args]
    covar_effect: CovarEffect

    def __init__(
        self,
        baseline: FittableParametricLifetimeModel[*Args],
        coef: tuple[float, ...] | tuple[None] = (None,),
    ):
        super().__init__()
        self.compose_with(
            covar_effect=CovarEffect(coef),
            baseline=baseline,
        )

    @property
    def fitting_results(self) -> Optional[FittingResults]:
        return self._fitting_results

    @fitting_results.setter
    def fitting_results(self, value: FittingResults):
        self._fitting_results = value

    @override
    def sf(
        self,
        time: float | NDArray[np.float64],
        covar: float | NDArray[np.float64],
        *args: *Args,
    ) -> NDArray[np.float64]:
        return np.squeeze(super().sf(time, covar, *args))

    @override
    def isf(
        self,
        probability: float | NDArray[np.float64],
        covar: float | NDArray[np.float64],
        *args: *Args,
    ) -> NDArray[np.float64]:
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
        cumulative_hazard_rate = -np.log(probability)
        return self.ichf(cumulative_hazard_rate, covar, *args)

    @override
    def cdf(
        self,
        time: float | NDArray[np.float64],
        covar: float | NDArray[np.float64],
        *args: *Args,
    ) -> NDArray[np.float64]:
        return super().cdf(time, covar, *args)

    def pdf(
        self,
        time: float | NDArray[np.float64],
        covar: float | NDArray[np.float64],
        *args: *Args,
    ) -> NDArray[np.float64]:
        return super().pdf(time, covar, *args)

    @override
    def ppf(
        self,
        probability: float | NDArray[np.float64],
        covar: float | NDArray[np.float64],
        *args: *Args,
    ) -> NDArray[np.float64]:
        return super().ppf(probability, covar, *args)

    @override
    def mrl(
        self,
        time: float | NDArray[np.float64],
        covar: float | NDArray[np.float64],
        *args: *Args,
    ) -> NDArray[np.float64]:
        return super().mrl(time, covar, *args)

    @override
    def rvs(
        self,
        covar: float | NDArray[np.float64],
        *args: *Args,
        size: Optional[int] = 1,
        seed: Optional[int] = None,
    ):
        """
        Random variable sampling.

        Parameters
        ----------
        covar : np.ndarray
            Covariate values. Shapes can be ``(n_values,)`` or ``(n_assets, n_values)``.
        *args : variable number of np.ndarray
            Any variables needed to compute the function. Those variables must be
            broadcastable with ``covar``. They may exist and result from method chaining due to nested class instantiation.
        size : int, default 1
            Size of the sample.
        seed : int, default None
            Random seed.

        Returns
        -------
        np.ndarray
            Sample of random lifetimes.
        """
        return super().rvs(covar, *args, size=size, seed=seed)

    @override
    def mean(
        self, covar: float | NDArray[np.float64], *args: *Args
    ) -> NDArray[np.float64]:
        return super().mean(covar, *args)

    @override
    def var(
        self, covar: float | NDArray[np.float64], *args: *Args
    ) -> NDArray[np.float64]:
        return super().var(covar, *args)

    @override
    def median(
        self, covar: float | NDArray[np.float64], *args: *Args
    ) -> NDArray[np.float64]:
        return super().median(covar, *args)

    @abstractmethod
    def jac_hf(
        self,
        time: float | NDArray[np.float64],
        covar: float | NDArray[np.float64],
        *args: *Args,
    ) -> NDArray[np.float64]: ...

    @abstractmethod
    def jac_chf(
        self,
        time: float | NDArray[np.float64],
        covar: float | NDArray[np.float64],
        *args: *Args,
    ) -> NDArray[np.float64]: ...

    @abstractmethod
    def dhf(
        self,
        time: float | NDArray[np.float64],
        covar: float | NDArray[np.float64],
        *args: *Args,
    ) -> NDArray[np.float64]: ...

    def jac_sf(
        self,
        time: float | NDArray[np.float64],
        covar: float | NDArray[np.float64],
        *args: *Args,
    ) -> NDArray[np.float64]:
        return -self.jac_chf(time, covar, *args) * self.sf(time, covar, *args)

    def jac_cdf(
        self,
        time: float | NDArray[np.float64],
        covar: float | NDArray[np.float64],
        *args: *Args,
    ) -> NDArray[np.float64]:
        return -self.jac_sf(time, covar, *args)

    def jac_pdf(
        self,
        time: float | NDArray[np.float64],
        covar: float | NDArray[np.float64],
        *args: *Args,
    ) -> NDArray[np.float64]:
        return self.jac_hf(time, covar, *args) * self.sf(
            time, covar, *args
        ) + self.jac_sf(time, covar, *args) * self.hf(time, covar, *args)

    @override
    def freeze(
        self, covar: float | NDArray[np.float64], *args: *Args
    ) -> FrozenParametricLifetimeModel:
        return FrozenParametricLifetimeModel(self, *(covar, *args))

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
        self.covar_effect.set_params(
            **{f"coef_{i}": 0.0 for i in range(covar.shape[-1])}
        )
        maximum_likelihood_estimation(
            self,
            lifetime_data,
            **kwargs,
        )
        return self


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
