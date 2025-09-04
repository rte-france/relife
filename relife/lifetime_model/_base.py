from abc import ABC, abstractmethod

import numpy as np
from scipy.optimize import Bounds, newton

from relife import FrozenParametricModel, ParametricModel
from relife.data import LifetimeData
from relife.likelihood import LikelihoodFromLifetimes
from relife.quadrature import (
    check_and_broadcast_bounds,
    legendre_quadrature,
    unweighted_laguerre_quadrature,
)

from ._plot import PlotParametricLifetimeModel


class ParametricLifetimeModel(ParametricModel, ABC):
    r"""Base class for lifetime model.

    This class defines the structure for creating lifetime model. It is a blueprint
    for implementing lifetime model expecting a variadic set of arguments.
    It expects implemantation of hazard functions (``hf``), cumulative hazard functions (``chf``),
    probability density function (``pdf``) and survival function (``sf``).
    Other functions are implemented by default but can be overridden by derived classes.

    Note:
        The abstract methods also provides a default implementation. One may not have to implement
        ``hf``, ``chf``, ``pdf`` and ``sf`` and just call ``super()`` to access the base implementation.

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
    def sf(self, time, *args):
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
    def hf(self, time, *args):
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
    def chf(self, time, *args):
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
    def pdf(self, time, *args):
        try:
            return self.sf(time, *args) * self.hf(time, *args)
        except NotImplementedError as err:
            class_name = type(self).__name__
            raise NotImplementedError(
                f"""
            {class_name} must implement pdf or the above functions
            """
            ) from err

    def cdf(self, time, *args):
        return 1 - self.sf(time, *args)

    def ppf(self, probability, *args):
        return self.isf(1 - probability, *args)

    def median(self, *args):
        return self.ppf(0.5, *args)

    def isf(self, probability, *args):
        func = lambda x: self.sf(x, *args) - probability
        return newton(
            func,
            x0=np.zeros_like(probability),
            args=args,
        )

    def ichf(self, cumulative_hazard_rate, *args):
        func = lambda x: np.sum(self.chf(x, *args) - cumulative_hazard_rate)
        return newton(
            func,
            x0=np.zeros_like(cumulative_hazard_rate),
        )

    def rvs(self, size, *args, nb_assets=None, return_event=False, return_entry=False, seed=None):
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
    def plot(self):
        """Provides access to plotting functionnalities"""
        return PlotParametricLifetimeModel(self)

    def ls_integrate(self, func, a, b, *args, deg=10):
        def integrand(x):
            #  x.shape == (deg,), (deg, n) or (deg, m, n), ie points of quadratures
            # fx : (d_1, ..., d_i, deg), (d_1, ..., d_i, deg, n) or (d_1, ..., d_i, deg, m, n)
            x = np.asarray(x)
            fx = func(x)
            if fx.shape[-len(x.shape) :] != x.shape:
                raise ValueError(
                    f"""
                    func can't squeeze input dimensions. If x has shape (d_1, ..., d_i), func(x) must have shape (..., d_1, ..., d_i).
                    Ex : if x.shape == (m, n), func(x).shape == (..., m, n).
                    """
                )
            if x.ndim == 3:  # reshape because model.pdf is tested only for input ndim <= 2
                xdeg, m, n = x.shape
                x = np.rollaxis(x, 1).reshape(m, -1)  # (m, deg*n), roll on m because axis 0 must align with m of args
                pdf = self.pdf(x, *args)  # (m, deg*n)
                pdf = np.rollaxis(pdf.reshape(m, xdeg, n), 1, 0)  #  (deg, m, n)
            else:  # ndim == 1 | 2
                # reshape to (1, deg*n) or (1, deg), ie place 1 on axis 0 to allow broadcasting with m of args
                pdf = self.pdf(x.reshape(1, -1), *args)  # (1, deg*n) or (1, deg)
                pdf = pdf.reshape(x.shape)  # (deg, n) or (deg,)

            # (d_1, ..., d_i, deg) or (d_1, ..., d_i, deg, n) or (d_1, ..., d_i, deg, m, n)
            return fx * pdf

        arr_a, arr_b = check_and_broadcast_bounds(a, b)  # (), (n,) or (m, n)
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

    def moment(self, n, *args):
        if n < 1:
            raise ValueError("order of the moment must be at least 1")
        func = lambda x: np.power(x, n)
        return self.ls_integrate(
            func,
            0.0,
            np.inf,
            *args,
            deg=100,
        )  #  high degree of polynome to ensure high precision

    def mean(self, *args):
        return self.moment(1, *args)

    def var(self, *args):
        return self.moment(2, *args) - self.moment(1, *args) ** 2

    def mrl(self, time, *args):
        sf = self.sf(time, *args)
        func = lambda x: np.asarray(x) - time
        ls = self.ls_integrate(func, time, np.inf, *args)
        if sf.ndim < 2:  # 2d to 1d or 0d
            ls = np.squeeze(ls)
        return ls / sf


class FittableParametricLifetimeModel(ParametricLifetimeModel, ABC):

    def __init__(self, **kwparams):
        super().__init__(**kwparams)
        self.fitting_results = None

    @abstractmethod
    def _get_initial_params(self, time, *args, event=None, entry=None, departure=None): ...

    @abstractmethod
    def _get_params_bounds(self):
        return Bounds(
            np.full(self.nb_params, np.finfo(float).resolution),
            np.full(self.nb_params, np.inf),
        )

    @abstractmethod
    def dhf(self, time, *args): ...

    @abstractmethod
    def jac_hf(self, time, *args, asarray=True): ...

    @abstractmethod
    def jac_chf(self, time, *args, asarray=True): ...

    @abstractmethod
    def jac_sf(self, time, *args, asarray=True): ...

    @abstractmethod
    def jac_pdf(self, time, *args, asarray=True): ...

    def fit(
        self,
        time,
        *args,
        event=None,
        entry=None,
        departure=None,
        **options,
    ):
        lifetime_data = LifetimeData(time, event=event, entry=entry, departure=departure, args=args)
        self.params = self._get_initial_params(time, *args, event=event, entry=entry, departure=departure)
        likelihood = LikelihoodFromLifetimes(self, lifetime_data)
        if "bounds" not in options:
            options["bounds"] = self._get_params_bounds()
        fitting_results = likelihood.maximum_likelihood_estimation(**options)
        self.params = fitting_results.optimal_params
        self.fitting_results = fitting_results
        return self


class NonParametricLifetimeModel(ABC):

    @abstractmethod
    def fit(self, time, event=None, entry=None, departure=None): ...


class FrozenParametricLifetimeModel(FrozenParametricModel):

    def __init__(self, model, *args):
        super().__init__(model, *args)

    def hf(self, time):
        """
        The hazard function.

        Parameters
        ----------
        time : float or np.ndarray
            Elapsed time value(s) at which to compute the function.
            If ndarray, allowed shapes are ``()``, ``(n,)`` or ``(m, n)``.

        Returns
        -------
        np.float64 or np.ndarray
            Function values at each given time(s).
        """
        return self.unfrozen_model.hf(time, *self.args)

    def chf(self, time):
        """
        The cumulative hazard function.

        Parameters
        ----------
        time : float or np.ndarray
            Elapsed time value(s) at which to compute the function.
            If ndarray, allowed shapes are ``()``, ``(n,)`` or ``(m, n)``.

        Returns
        -------
        np.float64 or np.ndarray
            Function values at each given time(s).
        """
        return self.unfrozen_model.chf(time, *self.args)

    def sf(self, time):
        """
        The survival function.

        Parameters
        ----------
        time : float or np.ndarray
            Elapsed time value(s) at which to compute the function.
            If ndarray, allowed shapes are ``()``, ``(n,)`` or ``(m, n)``.

        Returns
        -------
        np.float64 or np.ndarray
            Function values at each given time(s).
        """
        return self.unfrozen_model.sf(time, *self.args)

    def pdf(self, time):
        """
        The probability density function.

        Parameters
        ----------
        time : float or np.ndarray
            Elapsed time value(s) at which to compute the function.
            If ndarray, allowed shapes are ``()``, ``(n,)`` or ``(m, n)``.

        Returns
        -------
        np.float64 or np.ndarray
            Function values at each given time(s).
        """
        return self.unfrozen_model.pdf(time, *self.args)

    def mrl(self, time):
        """
        The mean residual life function.

        Parameters
        ----------
        time : float or np.ndarray
            Elapsed time value(s) at which to compute the function.
            If ndarray, allowed shapes are ``()``, ``(n,)`` or ``(m, n)``.

        Returns
        -------
        np.float64 or np.ndarray
            Function values at each given time(s).
        """
        return self.unfrozen_model.mrl(time, *self.args)

    def moment(self, n: int):
        """
        n-th order moment.

        Parameters
        ----------
        n : int
            Order of the moment (at least 1)

        Returns
        -------
        np.float64 or np.ndarray
        """
        return self.unfrozen_model.moment(n, *self.args)

    def mean(self):
        """
        The mean.

        Returns
        -------
        np.float64 or np.ndarray
        """
        return self.unfrozen_model.moment(1, *self.args)

    def var(self):
        """
        The variance.

        Returns
        -------
        np.float64 or np.ndarray
        """
        return self.unfrozen_model.moment(2, *self.args) - self.unfrozen_model.moment(1, *self.args) ** 2

    def isf(self, probability):
        """
        The inverse survival function.

        Parameters
        ----------
        probability : float or np.ndarray
            Probability value(s) at which to compute the function.
            If ndarray, allowed shapes are ``()``, ``(n,)`` or ``(m, n)``.

        Returns
        -------
        np.float64 or np.ndarray
            Function values at each given probability value(s).
        """
        return self.unfrozen_model.isf(probability, *self.args)

    def ichf(self, cumulative_hazard_rate):
        """
        Inverse cumulative hazard function.

        Parameters
        ----------
        cumulative_hazard_rate : float or np.ndarray
            Cumulative hazard rate value(s) at which to compute the function.
            If ndarray, allowed shapes are (), (n,) or (m, n).

        Returns
        -------
        np.float64 or np.ndarray
            Function values at each given probability value(s).
        """
        return self.unfrozen_model.ichf(cumulative_hazard_rate, *self.args)

    def cdf(self, time):
        """
        The cumulative density function.

        Parameters
        ----------
        time : float or np.ndarray
            Elapsed time value(s) at which to compute the function.
            If ndarray, allowed shapes are ``()``, ``(n,)`` or ``(m, n)``.

        Returns
        -------
        np.float64 or np.ndarray
            Function values at each given time(s).
        """
        return self.unfrozen_model.cdf(time, *self.args)

    def rvs(self, size, nb_assets=None, return_event=False, return_entry=False, seed=None):
        """
        Random variable sampling.

        Parameters
        ----------
        size : int, (int,) or (int, int)
            Size of the generated sample. If size is ``n`` or ``(n,)``, n samples are generated. If size is ``(m,n)``, a
            2d array of samples is generated.
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
        return self.unfrozen_model.rvs(
            size, *self.args, nb_assets=nb_assets, return_event=return_event, return_entry=return_entry, seed=seed
        )

    def ppf(self, probability):
        """
        The percent point function.

        Parameters
        ----------
        probability : float or np.ndarray
            Probability value(s) at which to compute the function.
            If ndarray, allowed shapes are ``()``, ``(n,)`` or ``(m, n)``.

        Returns
        -------
        np.float64 or np.ndarray
            Function values at each given probability value(s).
        """
        return self.unfrozen_model.ppf(probability, *self.args)

    def median(self):
        """
        The median.

        Returns
        -------
        np.float64 or np.ndarray
        """
        return self.unfrozen_model.median(*self.args)

    def ls_integrate(self, func, a, b, deg=10):
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
        deg : int, default 10
            Degree of the polynomials interpolation

        Returns
        -------
        np.ndarray
            Lebesgue-Stieltjes integral of func from `a` to `b`.
        """
        return self.unfrozen_model.ls_integrate(func, a, b, *self.args, deg=deg)
