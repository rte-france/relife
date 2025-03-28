from abc import ABC, abstractmethod
from itertools import chain
from typing import (
    Any,
    Iterator,
    Self,
    Generic,
    Optional,
    Callable,
    TypeVarTuple,
    NewType,
)

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import newton

from relife.decorators import isbroadcastable
from relife.descriptors import ShapedArgs
from relife.distributions.protocols import LifetimeDistribution
from relife.plots import PlotConstructor, PlotSurvivalFunc
from relife.quadratures import gauss_legendre, quad_laguerre

Z = TypeVarTuple("Z")
T = NewType("T", NDArray[np.floating] | NDArray[np.integer] | float | int)


class ParamsTree:
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


class ParametricMixin:
    """
    Base class to create a parametric core.

    Any parametric core must inherit from `ParametricModel`.
    """

    def __init__(self):
        self.params_tree = ParamsTree()
        self.leaf_models = {}

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
        return np.array(self.params_tree.values)

    @params.setter
    def params(self, new_values: NDArray[np.float64]):
        self.params_tree.values = new_values

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
        return self.params_tree.names

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
            if name in self.params_tree.node_data:
                raise ValueError(f"{name} already exists as param name")
            if name in self.leaf_models:
                raise ValueError(f"{name} already exists as leaf function")
        for name, module in kwcomponents.items():
            self.leaf_models[name] = module
            self.params_tree.set_leaf(f"{name}.params", module.params_tree)

    def new_params(self, **kwparams: float):
        """Change local parameters structure.

        This method only affects **local** parameters. `ParametricModel` components are not
        affected. This is usefull when one wants to change core parameters for any reason. For
        instance `Regression` distributions use `new_params` to change number of regression coefficients
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
            if name in self.leaf_models.keys():
                raise ValueError(f"{name} already exists as function name")
        self.params_tree.node_data = kwparams

    def __getattribute__(self, item):
        """
        Raises:
            ValueError: If any of the attributes in `params` is set to None when trying to
            access the attribute.
        """

        if (
            not item.startswith("_")
            and not item.startswith("__")
            and hasattr(self, item)
        ):
            if None in self.params:
                raise ValueError(
                    f"Can't call {item} if one param is None. Got {self.params} as params"
                )
        return super().__getattribute__(item)

    def __getattr__(self, name: str):
        class_name = type(self).__name__
        if name in self.__dict__:
            return self.__dict__[name]
        if name in super().__getattribute__("params_tree"):
            return super().__getattribute__("params_tree")[name]
        if name in super().__getattribute__("leaf_models"):
            return super().__getattribute__("leaf_models")[name]
        raise AttributeError(f"{class_name} has no attribute named {name}")

    def __setattr__(self, name: str, value: Any):
        if name in ["params_tree", "leaf_models"]:
            super().__setattr__(name, value)
        elif name in self.params_tree:
            self.params_tree[name] = value
        elif name in self.leaf_models:
            raise ValueError(
                "Can't modify leaf ParametricComponent. Recreate ParametricComponent instance instead"
            )
        else:
            super().__setattr__(name, value)


class LifetimeMixin(Generic[*Z], ABC):
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
        self, *z: *Z, size: int = 1, seed: Optional[int] = None
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
        self, probability: float | NDArray[np.float64], *z: *Z
    ) -> NDArray[np.float64]:
        return self.isf(1 - probability, *z)

    def median(self, *z: *Z) -> NDArray[np.float64]:
        return self.ppf(np.array(0.5), *z)

    def ls_integrate(
        self,
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

    @property
    def plot(self) -> PlotConstructor:
        """Plot"""
        return PlotSurvivalFunc(self)

    def freeze_zvariables(
        self, *z: *Z
    ) -> LifetimeDistribution[()]:  # is equivalent to FrozenLifetimeModel[*Z]
        return FrozenLifetimeDistribution(self, *z)


# generic function
def _get_nb_assets(*z: *Z) -> int:
    def as_2d():
        for x in z:
            if not isinstance(x, np.ndarray):
                x = np.asarray(x)
            if len(x.shape) > 2:
                raise ValueError
            yield np.atleast_2d(x)

    return max(map(lambda x: x.shape[0], as_2d()), default=1)


class FrozenLifetimeDistribution(Generic[*Z]):

    args = ShapedArgs(astuple=True)

    def __init__(
        self,
        baseline: LifetimeDistribution[*Z],
        *z: *Z,
    ):
        self.baseline = baseline
        self.nb_assets = _get_nb_assets(*z)
        self.z = z

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
