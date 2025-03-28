from abc import ABC, abstractmethod
from typing import Generic, Optional, Callable, TypeVarTuple, NewType

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import newton

from relife.decorators import isbroadcastable
from relife.distributions.protocols import LifetimeDistribution
from relife.plots import PlotConstructor, PlotSurvivalFunc
from relife.quadratures import gauss_legendre, quad_laguerre

Z = TypeVarTuple("Z")
T = NewType("T", NDArray[np.floating] | NDArray[np.integer] | float | int)


class SurvivalABC(Generic[*Z], ABC):
    r"""A generic base class for lifetime distributions.

    This class defines the structure for creating lifetime distributions. It is s a blueprint
    for implementing lifetime distributions parametrized by a variadic set of arguments.
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

    univariate: bool = False

    @abstractmethod
    def hf(self, time: T, *z: *Z) -> NDArray[np.float64]:
        if hasattr(self, "pdf") and hasattr(self, "sf"):
            return self.pdf(time, *z) / self.sf(time, *z)
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
    def chf(self, time: T, *z: *Z) -> NDArray[np.float64]:
        if hasattr(self, "sf"):
            return -np.log(self.sf(time, *z))
        if hasattr(self, "pdf") and hasattr(self, "hf"):
            return -np.log(self.pdf(time, *z) / self.hf(time, *z))
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
    def sf(self, time: T, *z: *Z) -> NDArray[np.float64]:
        if hasattr(self, "chf"):
            return np.exp(
                -self.chf(
                    time,
                    *z,
                )
            )
        if hasattr(self, "pdf") and hasattr(self, "hf"):
            return self.pdf(time, *z) / self.hf(time, *z)

        class_name = type(self).__name__
        raise NotImplementedError(
            f"""
        {class_name} must implement concrete sf function
        """
        )

    @abstractmethod
    def pdf(self, time: T, *z: *Z) -> NDArray[np.float64]:
        try:
            return self.sf(time, *z) * self.hf(time, *z)
        except NotImplementedError as err:
            class_name = type(self).__name__
            raise NotImplementedError(
                f"""
            {class_name} must implement pdf or the above functions
            """
            ) from err

    def mrl(self, time: T, *z: *Z) -> NDArray[np.float64]:
        sf = self.sf(time, *z)
        ls = self.ls_integrate(lambda x: x - time, time, np.array(np.inf), *z)
        if sf.ndim < 2:  # 2d to 1d or 0d
            ls = np.squeeze(ls)
        return ls / sf

    def isf(
        self,
        probability: float | NDArray[np.float64],
        *z: *Z,
    ):
        """Inverse survival function.

        Parameters
        ----------
        probability : float or ndarray, shape (n, ) or (m, n)
        *z : variadic arguments required by the function

        Returns
        -------
        ndarray of shape (), (n, ) or (m, n)
            Complement quantile corresponding to probability.
        """
        return newton(
            lambda x: self.sf(x, *z) - probability,
            x0=np.zeros_like(probability),
            args=z,
        )

    def ichf(
        self,
        cumulative_hazard_rate: float | NDArray[np.float64],
        *z: *Z,
    ):
        return newton(
            lambda x: self.chf(x, *z) - cumulative_hazard_rate,
            x0=np.zeros_like(cumulative_hazard_rate),
        )

    def cdf(self, time: T, *z: *Z) -> NDArray[np.float64]:
        return 1 - self.sf(time, *z)

    def rvs(
        self,
        *z: *Z,
        size: int = 1,
        seed: Optional[int] = None,
    ) -> NDArray[np.float64]:
        """Random variable sampling.

        Parameters
        ----------
        *z : variadic arguments required by the function
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
        return self.isf(probability, *z)

    def ppf(
        self: LifetimeDistribution[*Z], probability: float | NDArray[np.float64], *z: *Z
    ) -> NDArray[np.float64]:
        return self.isf(1 - probability, *z)

    def ls_integrate(
        self: LifetimeDistribution[*Z],
        func: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        a: float | NDArray[np.float64],
        b: float | NDArray[np.float64],
        *z: *Z,
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
        *z : ndarray (max dim of 2)
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
        `ls_integrate` operations rely on arguments number of dimensions passed in `a`, `b`, `*z` or
        any other variable referenced in `func`. Because `func` callable is not easy to inspect, either one must specify
        the maximum number of dimensions used (0, 1 or 2), or `ls_integrate` converts all these objects to 2d-array.
        Currently, the second option is prefered. That's why, returns are always 2d-array.


        """

        b = np.minimum(np.inf, b)
        a, b = np.atleast_2d(*np.broadcast_arrays(a, b))
        z_2d = np.atleast_2d(*z)  # type: ignore # Ts can't be bounded with current TypeVarTuple
        if isinstance(z_2d, np.ndarray):
            z_2d = (z_2d,)

        def integrand(x: NDArray[np.float64], *_: *Z) -> NDArray[np.float64]:
            return np.atleast_2d(func(x) * self.pdf(x, *_))

        if np.all(np.isinf(b)):
            b = np.atleast_2d(self.isf(np.array(1e-4), *z_2d))
            integration = gauss_legendre(
                integrand, a, b, *z_2d, ndim=2, deg=deg
            ) + quad_laguerre(integrand, b, *z_2d, ndim=2, deg=deg)
        else:
            integration = gauss_legendre(integrand, a, b, *z_2d, ndim=2, deg=deg)

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

    def moment(self, n: int, *z: *Z) -> NDArray[np.float64]:
        """n-th order moment

        Parameters
        ----------
        n : order of the moment, at least 1.
        *z : variadic arguments required by the function

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
            *z,
        )
        ndim = max(map(np.ndim, *z), default=0)
        if ndim < 2:  # 2d to 1d or 0d
            ls = np.squeeze(ls)
        return ls

    def mean(self, *z: *Z) -> NDArray[np.float64]:
        return self.moment(1, *z)

    def var(self, *z: *Z) -> NDArray[np.float64]:
        return self.moment(2, *z) - self.moment(1, *z) ** 2

    def median(self: LifetimeDistribution[*Z], *z: *Z) -> NDArray[np.float64]:
        return self.ppf(np.array(0.5), *z)

    @property
    def plot(self) -> PlotConstructor:
        """Plot"""
        return PlotSurvivalFunc(self)

    def freeze_zvariables(
        self, *z: *Z
    ) -> LifetimeDistribution[()]:  # is equivalent to FrozenLifetimeModel[*Z]
        return FrozenLifetimeDistribution(self, *z)


def _reshape(*z: *Z):
    nb_assets = 1  # minimum value
    for arr in z:
        arr = np.asarray(arr)
        if arr.ndim > 2:
            raise ValueError("Number of dimension can't be higher than 2 in zvariables")
        if arr.size == 1:
            yield np.squeeze(arr).item()  # yield float
        else:
            arr = arr.reshape(-1, 1)
            if (
                nb_assets != 1 and arr.shape[0] != nb_assets
            ):  # test if nb assets changed
                raise ValueError("Different number of assets are given in zvariables")
            else:  # update nb_assets
                nb_assets = arr.shape[0]
            yield arr.reshape(-1, 1)


class FrozenLifetimeDistribution(Generic[*Z]):

    univariate: bool = True

    def __init__(
        self,
        baseline: LifetimeDistribution[*Z],
        *z: *Z,
    ):
        self.baseline = baseline
        self.z = tuple(_reshape(*z))
        self.nb_assets = max(
            map(lambda x: x.shape[0] if isinstance(x, np.ndarray) else 1, iter(self.z)),
            default=1,
        )

    @isbroadcastable("time")
    def hf(self, time: T) -> NDArray[np.float64]:
        return self.baseline.hf(time, *self.z)

    @isbroadcastable("time")
    def chf(self, time: T) -> NDArray[np.float64]:
        return self.baseline.chf(time, *self.z)

    @isbroadcastable("time")
    def sf(self, time: T) -> NDArray[np.float64]:
        return self.baseline.sf(time, *self.z)

    @isbroadcastable("time")
    def pdf(self, time: T) -> NDArray[np.float64]:
        return self.baseline.pdf(time, *self.z)

    @isbroadcastable("time")
    def mrl(self, time: T) -> NDArray[np.float64]:
        return self.baseline.mrl(time, *self.z)

    def moment(self, n: int) -> NDArray[np.float64]:
        return self.baseline.moment(n)

    def mean(self) -> NDArray[np.float64]:
        return self.baseline.moment(1, *self.z)

    def var(self) -> NDArray[np.float64]:
        return self.baseline.moment(2, *self.z) - self.baseline.moment(1, *self.z) ** 2

    @isbroadcastable("probability")
    def isf(self, probability: float | NDArray[np.float64]):
        return self.baseline.isf(probability, *self.z)

    @isbroadcastable("cumulative_hazard_rate")
    def ichf(self, cumulative_hazard_rate: float | NDArray[np.float64]):
        return self.baseline.ichf(cumulative_hazard_rate, *self.z)

    @isbroadcastable("time")
    def cdf(self, time: T) -> NDArray[np.float64]:
        return self.baseline.cdf(time, *self.z)

    def rvs(self, size: int = 1, seed: Optional[int] = None) -> NDArray[np.float64]:
        return self.baseline.rvs(*self.z, size=size, seed=seed)

    @isbroadcastable("probability")
    def ppf(self, probability: float | NDArray[np.float64]) -> NDArray[np.float64]:
        return self.baseline.ppf(probability, *self.z)

    def median(self) -> NDArray[np.float64]:
        return self.baseline.median(*self.z)

    def ls_integrate(
        self,
        func: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        a: float | NDArray[np.float64],
        b: float | NDArray[np.float64],
        deg: int = 100,
    ) -> NDArray[np.float64]:

        return self.baseline.ls_integrate(func, a, b, deg, *self.z)
