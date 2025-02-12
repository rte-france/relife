import copy
from abc import ABC, abstractmethod
from dataclasses import InitVar, asdict, dataclass, field
from functools import wraps
from itertools import chain
from typing import Any, Callable, Generic, Iterator, Optional, Self, Union, Protocol

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import Bounds, OptimizeResult, minimize, newton

from relife.data import LifetimeData, lifetime_data_factory
from .likelihoods import LikelihoodFromLifetimes, hessian_from_likelihood
from relife.plots import PlotSurvivalFunc
from .quadratures import gauss_legendre, quad_laguerre
from relife.types import VariadicArgs


class Parameters:
    """
    Tree-structured parameters.

    Every ``ParametricModel`` are composed of ``Parameters`` instance.
    """

    def __init__(self, **kwargs):
        self._node_data = {}
        if kwargs:
            self._node_data = kwargs
        self.parent = None
        self.leaves = {}
        self._names, self._values = [], []

    @property
    def node_data(self):
        """data of current node as dict"""
        return self._node_data

    @node_data.setter
    def node_data(self, new_values: dict[str, Any]):
        self._node_data = new_values
        self.update()

    @property
    def names(self):
        """keys of current and leaf nodes as list"""
        return self._names

    @names.setter
    def names(self, new_names: list[str]):
        self.set_names(new_names)
        self.update_parents()

    @property
    def values(self):
        """values of current and leaf nodes as list"""
        return self._values

    @values.setter
    def values(self, new_values: list[Any]):
        self.set_values(new_values)
        self.update_parents()

    def set_values(self, new_values: list[Any]):
        if len(new_values) != len(self):
            raise ValueError(
                f"values expects {len(self)} items but got {len(new_values)}"
            )
        self._values = new_values
        pos = len(self._node_data)
        self._node_data.update(zip(self._node_data, new_values[:pos]))
        for leaf in self.leaves.values():
            leaf.set_values(new_values[pos : pos + len(leaf)])
            pos += len(leaf)

    def set_names(self, new_names: list[str]):
        if len(new_names) != len(self):
            raise ValueError(
                f"names expects {len(self)} items but got {len(new_names)}"
            )
        self._names = new_names
        pos = len(self._node_data)
        self._node_data = {new_names[:pos][i]: v for i, v in self._node_data.values()}
        for leaf in self.leaves.values():
            leaf.set_names(new_names[pos : pos + len(leaf)])
            pos += len(leaf)

    def __len__(self):
        return len(self._names)

    def __contains__(self, item):
        """contains only applies on current node"""
        return item in self._node_data

    def __getitem__(self, item):
        return self._node_data[item]

    def __setitem__(self, key, value):
        self._node_data[key] = value
        self.update()

    def __delitem__(self, key):
        del self._node_data[key]
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
        yield list(self._node_data.items())
        for leaf in self.leaves.values():
            yield list(chain.from_iterable(leaf.items_walk()))

    def all_items(self) -> Iterator:
        return chain.from_iterable(self.items_walk())

    def update_items(self):
        """parallel iterations : faster than update_value followed by update_keys"""
        try:
            next(self.all_items())
            _k, _v = zip(*self.all_items())
            self._names = list(_k)
            self._values = list(_v)
        except StopIteration:
            pass

    def update_parents(self):
        if self.parent is not None:
            self.parent.update()

    def update(self):
        """update names and values of current and parent nodes"""
        self.update_items()
        self.update_parents()


class ParametricModel:
    """
    Base class to create a parametric core.

    Any parametric core must inherit from `ParametricModel`.
    """

    def __init__(self):
        self._params = Parameters()
        self.leaves = {}

    @property
    def params(self) -> NDArray[np.float64]:
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
        return np.array(self._params.values)

    @params.setter
    def params(self, new_values: NDArray[np.float64]):
        self._params.values = new_values

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
        return self._params.names

    @property
    def nb_params(self):
        """
        Number of parameters.

        Returns
        -------
        int
            Number of parameters.

        """
        return len(self._params)

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
            if name in self._params.node_data:
                raise ValueError(f"{name} already exists as param name")
            if name in self.leaves:
                raise ValueError(f"{name} already exists as leaf function")
        for name, module in kwcomponents.items():
            self.leaves[name] = module
            self._params.set_leaf(f"{name}.params", module._params)

    def new_params(self, **kwparams: float):
        """Change local parameters structure.

        This method only affects **local** parameters. `ParametricModel` components are not
        affected. This is usefull when one wants to change core parameters for any reason. For
        instance `Regression` models use `new_params` to change number of regression coefficients
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
            if name in self.leaves.keys():
                raise ValueError(f"{name} already exists as function name")
        self._params.node_data = kwparams

    def __getattr__(self, name: str):
        class_name = type(self).__name__
        if name in self.__dict__:
            return self.__dict__[name]
        if name in super().__getattribute__("_params"):
            return super().__getattribute__("_params")[name]
        if name in super().__getattribute__("leaves"):
            return super().__getattribute__("leaves")[name]
        raise AttributeError(f"{class_name} has no attribute named {name}")

    def __setattr__(self, name: str, value: Any):
        if name in ["_params", "leaves"]:
            super().__setattr__(name, value)
        elif name in self._params:
            self._params[name] = value
        elif name in self.leaves:
            raise ValueError(
                "Can't modify leaf ParametricComponent. Recreate ParametricComponent instance instead"
            )
        else:
            super().__setattr__(name, value)

    def copy(self):
        """
        Copy current instance.

        Returns
        -------
            An independant copied instance.
        """
        return copy.deepcopy(self)


class LifetimeModel(Generic[*VariadicArgs], ABC):
    """
    Base class to create a lifetime core.

    A lifetime core is an object that can answer to traditional lifetime probability
    functions (``sf``, ``hf`` etc.) and other common probabilitu functions (``pdf``,
    ``cdf``, etc.).
    """

    # def __init_subclass__(cls, **kwargs):
    #     """
    #     TODO : something to parse *args names and to fill args_names and nb_args
    #     see Descriptors
    #     """
    #
    # @property
    # def args_names(self):
    #     return self._args.names
    #
    # @property
    # def nb_args(self):
    #     return len(self._args)

    @abstractmethod
    def hf(
        self, time: float | NDArray[np.float64], *args: *VariadicArgs
    ) -> NDArray[np.float64]:
        """
        Hazard function.

        The hazard function of the distribution

        Parameters
        ----------
        time : float or ndarray, shape (n, ) or (m, n)
            Elapsed time.
        *args : variadic arguments required by the function

        Returns
        -------
        ndarray of shape (), (n, ) or (m, n)
            Hazard values at each given time.
        """
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
        self, time: float | NDArray[np.float64], *args: *VariadicArgs
    ) -> NDArray[np.float64]:
        """Cumulative hazard function.

        Parameters
        ----------
        time : float or ndarray, shape (n, ) or (m, n)
            Elapsed time.
        *args : variadic arguments required by the function

        Returns
        -------
        ndarray of shape (), (n, ) or (m, n)
            Cumulative hazard values at each given time.
        """

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
        self, time: float | NDArray[np.float64], *args: *VariadicArgs
    ) -> NDArray[np.float64]:
        """Survival function.

        Parameters
        ----------
        time : float or ndarray, shape (n, ) or (m, n)
            Elapsed time.
        *args : variadic arguments required by the function

        Returns
        -------
        ndarray of shape (), (n, ) or (m, n)
            Survival function values at each given time.
        """
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
        self, time: float | NDArray[np.float64], *args: *VariadicArgs
    ) -> NDArray[np.float64]:
        """Probability density function.

        Parameters
        ----------
        time : float or ndarray, shape (n, ) or (m, n)
            Elapsed time.
        *args : variadic arguments required by the function

        Returns
        -------
        ndarray of shape (), (n, ) or (m, n)
            The probability density function evaluated at each given time.
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

    def mrl(
        self, time: float | NDArray[np.float64], *args: *VariadicArgs
    ) -> NDArray[np.float64]:
        """Mean residual life.

        Parameters
        ----------
        time : float or ndarray, shape (n, ) or (m, n)
            Elapsed time.
        *args : variadic arguments required by the function

        Returns
        -------
        ndarray of shape (), (n, ) or (m, n)
            Mean residual life values.
        """
        sf = self.sf(time, *args)
        ls = self.ls_integrate(lambda x: x - time, time, np.array(np.inf), *args)
        if sf.ndim < 2:  # 2d to 1d or 0d
            ls = np.squeeze(ls)
        return ls / sf

        # masked_time: ma.MaskedArray = ma.MaskedArray(
        #     time, time >= self.support_upper_bound
        # )
        # upper_bound = np.broadcast_to(
        #     np.asarray(self.isf(np.array(1e-4), *args)),
        #     time.shape,
        # )
        # masked_upper_bound: ma.MaskedArray = ma.MaskedArray(
        #     upper_bound, time >= self.support_upper_bound
        # )
        #
        # def integrand(x):
        #     return (x - masked_time) * self.pdf(x, *args)
        #
        # integration = gauss_legendre(
        #     integrand,
        #     masked_time,
        #     masked_upper_bound,
        #     ndim=2,
        # ) + quad_laguerre(
        #     integrand,
        #     masked_upper_bound,
        #     ndim=2,
        # )
        # mrl_values = integration / self.sf(masked_time, *args)
        # return np.squeeze(ma.filled(mrl_values, 0.0))

    def moment(self, n: int, *args: *VariadicArgs) -> NDArray[np.float64]:
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
        ndim = max(map(np.ndim, args), default=0)
        if ndim < 2:  # 2d to 1d or 0d
            ls = np.squeeze(ls)
        return ls

        # if bool(args):
        #     return np.broadcast_to(ls, np.broadcast(*args).shape)
        # return ls

        # upper_bound = self.isf(np.array(1e-4), *args)
        #
        # def integrand(x):
        #     return x**n * self.pdf(x, *args)
        #
        # return np.squeeze(
        #     gauss_legendre(integrand, np.array(0.0), upper_bound, ndim=2)
        #     + quad_laguerre(integrand, upper_bound, ndim=2)
        # )

    def mean(self, *args: *VariadicArgs) -> NDArray[np.float64]:
        """Mean.

        Parameters
        ----------
        *args : variadic arguments required by the function

        Returns
        -------
        ndarray of shape (0,)
            Mean value.
        """
        return self.moment(1, *args)

    def var(self, *args: *VariadicArgs) -> NDArray[np.float64]:
        """Variance.

        Parameters
        ----------
        *args : variadic arguments required by the function

        Returns
        -------
        ndarray of shape (0,)
            Variance value.
        """
        return self.moment(2, *args) - self.moment(1, *args) ** 2

    def isf(
        self,
        probability: float | NDArray[np.float64],
        *args: *VariadicArgs,
    ):
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
        )

    def ichf(
        self,
        cumulative_hazard_rate: float | NDArray[np.float64],
        *args: *VariadicArgs,
    ):
        """Inverse cumulative hazard function.

        Parameters
        ----------
        Cumulative hazard rate : float or ndarray, shape (n, ) or (m, n)
        *args : variadic arguments required by the function

        Returns
        -------
        ndarray of shape (), (n, ) or (m, n)
            Inverse cumulative hazard values, i.e. time.
        """
        return newton(
            lambda x: self.chf(x, *args) - cumulative_hazard_rate,
            x0=np.zeros_like(cumulative_hazard_rate),
        )

    def cdf(
        self, time: float | NDArray[np.float64], *args: *VariadicArgs
    ) -> NDArray[np.float64]:
        """Cumulative distribution function.

        Parameters
        ----------
        time : float or ndarray, shape (n, ) or (m, n)
        *args : variadic arguments required by the function

        Returns
        -------
        ndarray of shape (), (n, ) or (m, n)
            Cumulative distribution function values at each given time.
        """
        return 1 - self.sf(time, *args)

    def rvs(
        self, *args: *VariadicArgs, size: int = 1, seed: Optional[int] = None
    ) -> NDArray[np.float64]:
        """Random variable sampling.

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
        self, probability: float | NDArray[np.float64], *args: *VariadicArgs
    ) -> NDArray[np.float64]:
        """Percent point function.

        Parameters
        ----------
        probability : float or ndarray, shape (n, ) or (m, n)
        *args : variadic arguments required by the function

        Returns
        -------
        ndarray of shape (), (n, ) or (m, n)
            Quantile corresponding to probability.
        """
        return self.isf(1 - probability, *args)

    def median(self, *args: *VariadicArgs) -> NDArray[np.float64]:
        """Median

        Parameters
        ----------
        *args : variadic arguments required by the function

        Returns
        -------
        ndarray of shape (0,)
            Median value.
        """
        return self.ppf(np.array(0.5), *args)

    def ls_integrate(
        self,
        func: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        a: float | NDArray[np.float64],
        b: float | NDArray[np.float64],
        *args: *VariadicArgs,
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
        args : ndarray (max dim of 2)
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

        def integrand(x: NDArray[np.float64], *_: *VariadicArgs) -> NDArray[np.float64]:
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

    @property
    def plot(self) -> PlotSurvivalFunc:
        """Plot"""
        return PlotSurvivalFunc(self)


@dataclass
class FittingResults:
    """Fitting results of the parametric core."""

    nb_samples: InitVar[int]  #: Number of observations (samples).

    opt: InitVar[OptimizeResult] = field(
        repr=False
    )  #: Optimization result (see scipy.optimize.OptimizeResult doc).
    var: Optional[NDArray[np.float64]] = field(
        repr=False, default=None
    )  #: Covariance matrix (computed as the inverse of the Hessian matrix)
    se: NDArray[np.float64] = field(
        init=False, repr=False
    )  #: Standard error, square root of the diagonal of the covariance matrix.

    nb_params: int = field(init=False)  #: Number of parameters.
    AIC: float = field(init=False)  #: Akaike Information Criterion.
    AICc: float = field(
        init=False
    )  #: Akaike Information Criterion with a correction for small sample sizes.
    BIC: float = field(init=False)  #: Bayesian Information Criterion.

    def __post_init__(self, nb_samples, opt):
        self.nb_params = opt.x.size
        self.AIC = 2 * self.nb_params + 2 * opt.fun
        self.AICc = self.AIC + 2 * self.nb_params * (self.nb_params + 1) / (
            nb_samples - self.nb_params - 1
        )
        self.BIC = np.log(nb_samples) * self.nb_params + 2 * opt.fun

        self.se = None
        if self.var is not None:
            self.se = np.sqrt(np.diag(self.var))

    def standard_error(self, jac_f: np.ndarray) -> np.ndarray:
        """Standard error estimation function.

        Parameters
        ----------
        jac_f : 1D array
            The Jacobian of a function f with respect to params.

        Returns
        -------
        1D array
            Standard error for f(params).

        References
        ----------
        .. [1] Meeker, W. Q., Escobar, L. A., & Pascual, F. G. (2022).
            Statistical methods for reliability data. John Wiley & Sons.
        """
        # [1] equation B.10 in Appendix
        return np.sqrt(np.einsum("ni,ij,nj->n", jac_f, self.var, jac_f))

    def asdict(self) -> dict:
        """converts FittingResult into a dictionary.

        Returns
        -------
        dict
            Returns the fitting result as a dictionary.
        """
        return asdict(self)


class ParametricLifetimeModel(LifetimeModel[*VariadicArgs], ParametricModel, ABC):

    fitting_results: FittingResults | None

    def __init__(self):
        super().__init__()
        self.fitting_results = None

    @property
    @abstractmethod
    def params_bounds(self) -> Bounds:
        """Bounds of the parameters

        Returns
        -------
        Bounds
            The lower and upper bounds for the parameters.
        """

    @abstractmethod
    def init_params(self, lifetime_data: LifetimeData, *args: *VariadicArgs) -> None:
        """Initialization of the parameters

        Initialize parameters values with respect to lifetime data.

        Parameters
        ----------
        lifetime_data : LifetimeData object
            lalal.
        args : tuple of numpy arrays
            lala.
        """

    def fit(
        self,
        time: NDArray[np.float64],
        event: Optional[NDArray[np.bool_]] = None,
        entry: Optional[NDArray[np.float64]] = None,
        departure: Optional[NDArray[np.float64]] = None,
        model_args: tuple[*VariadicArgs] = (),
        **kwargs: Any,
    ) -> Self:
        """
        Estimation of lifetime core parameters with respect to lifetime data.


        Parameters
        ----------
        time : ndarray (1d or 2d)
            Observed lifetime values.
        event : ndarray of boolean values (1d), default is None
            Boolean indicators tagging lifetime values as right censored or complete.
        entry : ndarray of float (1d), default is None
            Left truncations applied to lifetime values.
        departure : ndarray of float (1d), default is None
            Right truncations applied to lifetime values.
        model_args : tuple of ndarray, default is None
            Other arguments needed by the lifetime core.
        **kwargs
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
        ndarray of float
            Estimated parameters.

        Notes
        -----
        Supported lifetime observations format is either 1d-array or 2d-array. 2d-array is more advanced
        format that allows to pass other information as left-censored or interval-censored values. In this case,
        `event` is not needed as 2d-array encodes right-censored values by itself.

        """

        lifetime_data = lifetime_data_factory(
            time,
            event,
            entry,
            departure,
        )

        optimized_model = self.copy()
        optimized_model.init_params(lifetime_data, *model_args)
        # or just optimized_model.init_params(observed_lifetimes, *model_args)

        likelihood = LikelihoodFromLifetimes(
            optimized_model, lifetime_data, model_args=model_args
        )

        minimize_kwargs = {
            "method": kwargs.get("method", "L-BFGS-B"),
            "constraints": kwargs.get("constraints", ()),
            "tol": kwargs.get("tol", None),
            "callback": kwargs.get("callback", None),
            "options": kwargs.get("options", None),
            "bounds": kwargs.get("bounds", optimized_model.params_bounds),
            "x0": kwargs.get("x0", optimized_model.params),
        }

        optimizer = minimize(
            likelihood.negative_log,
            minimize_kwargs.pop("x0"),
            jac=None if not likelihood.hasjac else likelihood.jac_negative_log,
            **minimize_kwargs,
        )
        optimized_model.params = optimizer.x
        var = np.linalg.inv(likelihood.hessian())

        optimized_model.fitting_results = FittingResults(
            len(lifetime_data), optimizer, var
        )

        self.init_params(lifetime_data, *model_args)
        # or just self.init_params(observed_lifetimes, *model_args)
        self.params = optimized_model.params
        self.fitting_results = FittingResults(len(lifetime_data), optimizer, var)
        return self

    def __getattribute__(self, item):
        """control if params are set"""

        if (
            not item.startswith("_")
            and not item.startswith("__")
            and hasattr(LifetimeModel, item)
        ):
            if None in self.params:
                raise ValueError(
                    f"Can't call {item} if one param is None. Got {self.params} as params"
                )
        return super().__getattribute__(item)


@dataclass
class Estimates:
    """
    Stores the estimates for a non-parametric lifetime core.

    Parameters
    ----------
    timeline : np.ndarray of shape (n, )
        The timeline of the estimates.
    values : np.ndarray of shape (n, )
        The estimated values.
    se : np.ndarray of shape (n, ), optional
        The standard errors of the estimates. If not provided, defaults to an array of zeros.

    Raises
    ------
    ValueError
        If the shapes of `timeline`, `values`, and `se` are not compatible.
    """

    timeline: NDArray[np.float64]
    values: NDArray[np.float64]
    se: Optional[NDArray[np.float64]] = None

    def __post_init__(self):
        if self.se is None:
            self.se = np.zeros_like(
                self.values
            )  # garder None/Nan efaire le changement de valeur au niveau du plot

        if self.timeline.shape != self.values.shape != self.se:
            raise ValueError("Incompatible timeline, values and se in Estimates")

    def nearest_1dinterp(
        self, x: float | NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Returns x nearest interpolation based on timeline and values data points
        timeline has to be monotonically increasing

        Args:
            x (NDArray[np.float64]): 1d x coordinates to interpolate

        Returns:
            NDArray[np.float64]: interpolation values of x
        """
        spacing = np.diff(self.timeline) / 2
        xp = np.hstack([spacing, spacing[-1]]) + self.timeline
        values_p = np.concatenate([self.values, self.values[-1, None]])
        se_p = np.concatenate([self.se, self.se[-1, None]])
        return (
            values_p[np.searchsorted(xp, np.asarray(x))],
            se_p[np.searchsorted(xp, np.asarray(x))],
        )


class NonParametricModel(Protocol):
    """
    Non-parametric lifetime estimator.

    Attributes
    ----------
    estimates : Estimations
        The estimations produced when fitting the estimator.
    """

    estimates: dict[str, Optional[Estimates]]

    @property
    def plot(self):
        return PlotSurvivalFunc(self)
