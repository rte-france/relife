from abc import ABC, abstractmethod
from itertools import chain
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    NewType,
    Optional,
    Self,
    TypeVarTuple, Iterator,
)

import numpy as np
from numpy._typing import NDArray
from numpy.typing import NDArray
from scipy.optimize import Bounds, newton
from typing_extensions import override

from relife._plots import PlotSurvivalFunc
from relife.data import lifetime_data_factory
from relife.likelihood import maximum_likelihood_estimation
from relife.quadratures import gauss_legendre, quad_laguerre

from ._frozen import FrozenLifetimeModel
from ..likelihood.mle import FittingResults

if TYPE_CHECKING:
    from relife.data import LifetimeData
    from relife.model import FittableLifetimeModel

Args = TypeVarTuple("Args")


class BaseParametricModel:
    """
    Base class to create a parametric_model core.

    Any parametric_model core must inherit from `ParametricModel`.
    """

    def __init__(self):
        self.params_tree = ParamsTree()
        self.leaves_of_models = {}

    @property
    def params(self) -> NDArray[np.float64 | np.complex64]:
        """
        Parameters values.

        Returns
        -------
        ndarray
            Parameters values of the core

        Notes
        -----
        If parameter values are not set, they are encoded as `np.nan` value.

        Parameters can be by manually setting`params` through its setter, fitting the core if `fit` exists or
        by specifying all parameters values when the core object is initialized.
        """
        return np.array(self.params_tree.all_values)

    @params.setter
    def params(self, values: NDArray[np.float64 | np.complex64]):
        if values.ndim > 1:
            raise ValueError
        values: tuple[Optional[float], ...] = tuple(
            map(lambda x: x.item() if x != np.nan else None, values)
        )
        self.params_tree.all_values = values

    @property
    def params_names(self):
        """
        Parameters names.

        Returns
        -------
        list of str
            Parameters names

        Notes
        -----
        Parameters values can be requested (a.k.a. get) by their name at instance level.
        """
        return self.params_tree.all_keys

    @property
    def nb_params(self):
        """
        Number of parameters.

        Returns
        -------
        int
            Number of parameters.

        """
        return len(self.params_tree)

    def compose_with(self, **kwcomponents: Self):
        """Compose with new ``ParametricModel`` instance(s).

        This method must be seen as standard function composition exept that objects are not
        functions but group of functions (as object encapsulates functions). When you
        compose your ``ParametricModel`` instance with new one(s), the followings happen :

        - each new parameters are added to the current ``Parameters`` instance
        - each new `ParametricModel` instance is accessible as a standard attribute

        Like so, you can request new `ParametricModel` components in current `ParametricModel`
        instance while setting and getting all parameters. This is usefull when `ParametricModel`
        can be seen as a nested function (see `Regression`).

        Parameters
        ----------
        **kwcomponents : variadic named ``ParametricModel`` instance

            Instance names (keys) are followed by the instances themself (values).

        Notes
        -----
        If one wants to pass a `dict` of key-value, make sure to unpack the dict
        with `**` operator or you will get a nasty `TypeError`.
        """
        for name in kwcomponents.keys():
            if name in self.params_tree.data:
                raise ValueError(f"{name} already exists as param name")
            if name in self.leaves_of_models:
                raise ValueError(f"{name} already exists as leaf function")
        for name, model in kwcomponents.items():
            self.leaves_of_models[name] = model
            self.params_tree.set_leaf(f"{name}.params", model.params_tree)

    def set_params(self, **kwparams: Optional[float]):
        """Change local parameters structure.

        This method only affects **local** parameters. `ParametricModel` components are not
        affected. This is usefull when one wants to change core parameters for any reason. For
        instance `Regression` model use `new_params` to change number of regression coefficients
        depending on the number of covariates that are passed to the `fit` method.

        Parameters
        ----------
        **kwparams : variadic named floats corresponding to new parameters

            Float names (keys) are followed by float instances (values).

        Notes
        -----
        If one wants to pass a `dict` of key-value, make sure to unpack the dict
        with `**` operator or you will get a nasty `TypeError`.
        """

        for name in kwparams.keys():
            if name in self.leaves_of_models.keys():
                raise ValueError(f"{name} already exists as function name")
        self.params_tree.data = kwparams

    # def __getattribute__(self, item):
    #     if not item.startswith("_") and not item.startswith("__"):
    #         return super().__getattribute__(item)
    #     if item in (
    #         "new_params",
    #         "compose_with",
    #         "params",
    #         "params_names",
    #         "nb_params",
    #     ):
    #         return super().__getattribute__(item)
    #     if not self._all_params_set:
    #         raise ValueError(f"Can't call {item} if one parameter value is not set")
    #     return super().__getattribute__(item)

    def __getattr__(self, name: str):
        class_name = type(self).__name__
        if name in self.__dict__:
            return self.__dict__[name]
        if name in super().__getattribute__("params_tree"):
            return super().__getattribute__("params_tree")[name]
        if name in super().__getattribute__("leaves_of_models"):
            return super().__getattribute__("leaves_of_models")[name]
        raise AttributeError(f"{class_name} has no attribute named {name}")

    def __setattr__(self, name: str, value: Any):
        if name in ("params_tree", "leaves_of_models"):
            super().__setattr__(name, value)
        elif name in self.params_tree:
            self.params_tree[name] = value
        elif name in self.leaves_of_models:
            raise ValueError(
                "Can't modify leaf ParametricComponent. Recreate ParametricComponent instance instead"
            )
        else:
            super().__setattr__(name, value)


class BaseLifetimeModel(BaseParametricModel, Generic[*Args], ABC):
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
        self._fitting_results = None

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
            Upper bound(s) of integration. If lower bound(s) is infinite, use np.inf as value.
        *args : ndarray (max dim of 2)
            Other arguments needed by the lifetime core (eg. covariates)
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

        b = np.minimum(np.inf, b)
        a, b = np.atleast_2d(*np.broadcast_arrays(a, b))
        args_2d = np.atleast_2d(*args)  # type: ignore # Ts can't be bounded with current TypeVarTuple
        if isinstance(args_2d, np.ndarray):
            args_2d = (args_2d,)

        def integrand(x: NDArray[np.float64], *_: *Args) -> NDArray[np.float64]:
            return np.atleast_2d(func(x) * self.pdf(x, *_))

        if np.all(np.isinf(b)):
            b = np.atleast_2d(self.isf(np.array(1e-4), *args_2d))
            integration = gauss_legendre(
                integrand, a, b, *args_2d, ndim=2, deg=deg
            ) + quad_laguerre(integrand, b, *args_2d, ndim=2, deg=deg)
        else:
            integration = gauss_legendre(integrand, a, b, *args_2d, ndim=2, deg=deg)

        # if ndim is not None:
        #     if ndim > 2:
        #         raise ValueError("ndim can't be greater than 2")
        #     try:
        #         integration = np.reshape(
        #             integration, (-1,) + (1,) * (ndim - 1) if ndim > 0 else ()
        #         )
        #     except ValueError:
        #         raise ValueError("incompatible ndim value")

        return integration

        # if broadcast_to is not None:
        #     try:
        #         integration = np.broadcast_to(np.squeeze(integration), broadcast_to)
        #     except ValueError:
        #         raise ValueError("broadcast_to shape value is incompatible")

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
    ) -> FrozenLifetimeModel:
        return FrozenLifetimeModel(self, *args)

    @property
    def plot(self) -> PlotSurvivalFunc:
        """Plot"""
        return PlotSurvivalFunc(self)


# class BaseParametricLifetimeModel(ParametricModel, BaseLifetimeModel[*Args], ABC):
#     def __init__(self):
#         super().__init__()
#         self.fitting_results = None


NonParametricEstimation = NewType(
    "NonParametricEstimation",
    dict[
        str,
        tuple[NDArray[np.float64], NDArray[np.float64], Optional[NDArray[np.float64]]],
    ],
)


class BaseNonParametricLifetimeModel(ABC):
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


# type ParametricLifetimeModel[()]
class BaseDistribution(BaseLifetimeModel[()], ABC):
    """
    Base class for distribution model.
    """

    frozen: bool = True

    @property
    def fitting_results(self) -> Optional[FittingResults]:
        return self._fitting_results

    @fitting_results.setter
    def fitting_results(self, value : FittingResults):
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
    def freeze(self) -> FrozenLifetimeModel:
        return FrozenLifetimeModel(self)

    def fit(
        self,
        time: NDArray[np.float64],
        /,
        event: Optional[NDArray[np.bool_]] = None,
        entry: Optional[NDArray[np.float64]] = None,
        departure: Optional[NDArray[np.float64]] = None,
        **kwargs: Any,
    ) -> Self:
        # if update to 3.12 : maximum_likelihood_estimation[()](...), generic functions
        # Step 1: Prepare lifetime data
        lifetime_data = lifetime_data_factory(
            time,
            event,
            entry,
            departure,
        )
        optimized_model = maximum_likelihood_estimation(
            self,
            lifetime_data,
            **kwargs,
        )
        return optimized_model


class CovarEffect(BaseParametricModel):
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


# type ParametricLifetimeModel[float | NDArray[np.float64], *Args]
class BaseRegression(
    BaseLifetimeModel[float | NDArray[np.float64], *Args], ABC
):
    """
    Base class for regression model.
    """

    baseline: FittableLifetimeModel[*Args]
    covar_effect: CovarEffect

    def __init__(
        self,
        baseline: FittableLifetimeModel[*Args],
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
    ) -> FrozenLifetimeModel:
        return FrozenLifetimeModel(self, *(covar, *args))

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
        # if update to 3.12 : maximum_likelihood_estimation[float|NDArray[np.float64], *args](...), generic functions
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
        optimized_model = maximum_likelihood_estimation(
            self,
            lifetime_data,
            **kwargs,
        )
        return optimized_model


class ParamsTree:
    """
    Tree-structured parameters.

    Every ``Parametric`` are composed of ``Parameters`` instance.
    """

    def __init__(self):
        self.parent = None
        self._data = {}
        self.leaves = {}
        self._all_keys, self._all_values = (), ()
        self.dtype = float

    @property
    def data(self) -> dict[str, Optional[float | complex]]:
        """data of current node as dict"""
        return self._data

    @data.setter
    def data(self, mapping: dict[str, Optional[float | complex]]):
        self._data = mapping
        self.update()

    @property
    def all_keys(self) -> tuple[str, ...]:
        """keys of current and leaf nodes as list"""
        return self._all_keys

    @all_keys.setter
    def all_keys(self, keys: tuple[str, ...]):
        self.set_all_keys(*keys)
        self.update_parents()

    @property
    def all_values(self) -> tuple[Optional[float | complex], ...]:
        """values of current and leaf nodes as list"""
        return self._all_values

    @all_values.setter
    def all_values(self, values: tuple[Optional[float | complex], ...]):
        self.set_all_values(*values)
        self.update_parents()

    def set_all_values(self, *values: Optional[float | complex]):
        if len(values) != len(self):
            raise ValueError(f"values expects {len(self)} items but got {len(values)}")
        self._all_values = values
        pos = len(self._data)
        self._data.update(zip(self._data, values[:pos]))
        for leaf in self.leaves.values():
            leaf.set_all_values(*values[pos : pos + len(leaf)])
            pos += len(leaf)

    def set_all_keys(self, *keys: str):
        if len(keys) != len(self):
            raise ValueError(f"names expects {len(self)} items but got {len(keys)}")
        self._all_keys = keys
        pos = len(self._data)
        self._data = {keys[:pos][i]: v for i, v in self._data.values()}
        for leaf in self.leaves.values():
            leaf.set_all_keys(*keys[pos : pos + len(leaf)])
            pos += len(leaf)

    def __len__(self):
        return len(self._all_keys)

    def __contains__(self, item):
        """contains only applies on current node"""
        return item in self._data

    def __getitem__(self, item):
        return self._data[item]

    def __setitem__(self, key, value):
        self._data[key] = value
        self.update()

    def __delitem__(self, key):
        del self._data[key]
        self.update()

    def get_leaf(self, item):
        return self.leaves[item]

    def set_leaf(self, key, value):
        if key not in self.leaves:
            value.parent = self
        self.leaves[key] = value
        self.update()

    def del_leaf(self, key):
        del self.leaves[key]
        self.update()

    def items_walk(self) -> Iterator:
        """parallel walk through key value pairs"""
        yield list(self._data.items())
        for leaf in self.leaves.values():
            yield list(chain.from_iterable(leaf.items_walk()))

    def all_items(self) -> Iterator:
        return chain.from_iterable(self.items_walk())

    def update_items(self):
        """parallel iterations : faster than update_value followed by update_keys"""
        try:
            next(self.all_items())
            _k, _v = zip(*self.all_items())
            self._all_keys = list(_k)
            self._all_values = list(_v)
        except StopIteration:
            pass

    def update_parents(self):
        if self.parent is not None:
            self.parent.update()

    def update(self):
        """update names and values of current and parent nodes"""
        self.update_items()
        self.update_parents()
