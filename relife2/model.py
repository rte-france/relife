import copy
import warnings
from abc import ABC, abstractmethod
from itertools import chain
from typing import Any, Callable, Generic, Iterator, Optional, TypeVarTuple, Union

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import Bounds, minimize, newton

from relife2.data import LifetimeSample, Truncations, lifetime_factory_template
from relife2.data.dataclass import Sample
from relife2.maths.integrations import gauss_legendre, quad_laguerre


class Composite:
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


class ParametricComponent:
    def __init__(self):
        self._params = Composite()
        self.leaves = {}

    @property
    def params(self):
        return np.array(self._params.values, dtype=np.float64)

    @params.setter
    def params(self, new_values):
        self._params.values = new_values

    @property
    def params_names(self):
        return self._params.names

    @property
    def nb_params(self):
        return len(self._params)

    @property
    def all_params_set(self):
        return np.isnan(self.params).any() and not np.isnan(self.params).all()

    def compose_with(self, **kwcomponents: "ParametricComponent"):
        """add functions that can be called from node"""
        for name in kwcomponents.keys():
            if name in self._params.node_data:
                raise ValueError(f"{name} already exists as param name")
            if name in self.leaves:
                raise ValueError(f"{name} already exists as leaf function")
        for name, module in kwcomponents.items():
            self.leaves[name] = module
            self._params.set_leaf(f"{name}.params", module._params)

    def new_params(self, **kwparams):
        """change local params structure (at node level)"""
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
                "Can't modify leaf function. Recreate Function instance instead"
            )
        else:
            super().__setattr__(name, value)

    def copy(self):
        return copy.deepcopy(self)


Args = TypeVarTuple("Args")


class LifetimeModel(Generic[*Args]):
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

    def hf(self, time: NDArray[np.float64], *args: *Args) -> NDArray[np.float64]:
        if "pdf" in self.__class__.__dict__ and "sf" in self.__class__.__dict__:
            return self.pdf(time, *args) / self.sf(time, *args)
        if "sf" in self.__class__.__dict__:
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
            {class_name} must implement hf function
            """
        )

    def chf(self, time: NDArray[np.float64], *args: *Args) -> NDArray[np.float64]:
        if "sf" in self.__class__.__dict__:
            return -np.log(self.sf(time, *args))
        if "pdf" in self.__class__.__dict__ and "hf" in self.__class__.__dict__:
            return -np.log(self.pdf(time, *args) / self.hf(time, *args))
        if "hf" in self.__class__.__dict__:
            return self.ls_integrate(
                lambda x: self.hf(x, *args), np.array(0.0), np.array(np.inf), *args
            )
        class_name = type(self).__name__
        raise NotImplementedError(
            f"""
        {class_name} must implement chf or at least hf function
        """
        )

    def ichf(self, cumulative_hazard_rate: NDArray[np.float64], *args: *Args):
        raise NotImplementedError

    def sf(self, time: NDArray[np.float64], *args: *Args) -> NDArray[np.float64]:
        if "chf" in self.__class__.__dict__:
            return np.exp(
                -self.chf(
                    time,
                    *args,
                )
            )
        if "pdf" in self.__class__.__dict__ and "hf" in self.__class__.__dict__:
            return self.pdf(time, *args) / self.hf(time, *args)

        class_name = type(self).__name__
        raise NotImplementedError(
            f"""
        {class_name} must implement sf function
        """
        )

    def pdf(self, time: NDArray[np.float64], *args: *Args) -> NDArray[np.float64]:
        try:
            return self.sf(time, *args) * self.hf(time, *args)
        except NotImplementedError as err:
            class_name = type(self).__name__
            raise NotImplementedError(
                f"""
            {class_name} must implement pdf or the above functions
            """
            ) from err

    def mrl(self, time: NDArray[np.float64], *args: *Args) -> NDArray[np.float64]:

        return self.ls_integrate(
            lambda x: x - time, time, np.array(np.inf), *args
        ) / self.sf(time, *args)

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

    def moment(self, n: int, *args: *Args) -> NDArray[np.float64]:
        return self.ls_integrate(
            lambda x: x**n,
            np.array(0.0),
            np.array(np.inf),
            *args,
        )

        # upper_bound = self.isf(np.array(1e-4), *args)
        #
        # def integrand(x):
        #     return x**n * self.pdf(x, *args)
        #
        # return np.squeeze(
        #     gauss_legendre(integrand, np.array(0.0), upper_bound, ndim=2)
        #     + quad_laguerre(integrand, upper_bound, ndim=2)
        # )

    def mean(self, *args: *Args) -> NDArray[np.float64]:
        return self.moment(1, *args)

    def var(self, *args: *Args) -> NDArray[np.float64]:
        return self.moment(2, *args) - self.moment(1, *args) ** 2

    def isf(
        self,
        probability: NDArray[np.float64],
        *args: *Args,
    ):
        return newton(
            lambda x: self.sf(x, *args) - probability,
            x0=np.zeros_like(probability),
        )

    def cdf(self, time: NDArray[np.float64], *args: *Args) -> NDArray[np.float64]:
        return 1 - self.sf(time, *args)

    def rvs(
        self, *args: *Args, size: Optional[int] = 1, seed: Optional[int] = None
    ) -> NDArray[np.float64]:
        generator = np.random.RandomState(seed=seed)
        probability = generator.uniform(size=size)
        return self.isf(probability, *args)

    def ppf(
        self, probability: NDArray[np.float64], *args: *Args
    ) -> NDArray[np.float64]:
        return self.isf(1 - probability, *args)

    def median(self, *args: *Args) -> NDArray[np.float64]:
        return self.ppf(np.array(0.5), *args)

    def ls_integrate(
        self,
        func: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        a: NDArray[np.float64],
        b: NDArray[np.float64],
        *args: *Args,
        deg: int = 100,
    ) -> NDArray[np.float64]:

        b = np.minimum(np.inf, b)
        a, b = np.atleast_2d(*np.broadcast_arrays(a, b))
        args_2d = np.atleast_2d(*args)  # type: ignore # Ts can't be bounded with current TypeVarTuple
        if isinstance(args_2d, np.ndarray):
            args_2d = (args_2d,)

        def integrand(x: NDArray[np.float64], *_: *Args) -> NDArray[np.float64]:
            return np.atleast_2d(func(x) * self.pdf(x, *_))

        if np.all(np.isinf(b)):
            b = np.atleast_2d(self.isf(np.array(1e-4), *args_2d))
            return np.squeeze(
                gauss_legendre(integrand, a, b, *args_2d, ndim=2, deg=deg)
                + quad_laguerre(integrand, b, *args_2d, ndim=2, deg=deg)
            )
        else:
            return np.squeeze(
                gauss_legendre(integrand, a, b, *args_2d, ndim=2, deg=deg)
            )


class ParametricModel(ParametricComponent, ABC):
    @abstractmethod
    def init_params(self, *args: Any) -> None: ...

    @property
    @abstractmethod
    def params_bounds(self) -> Bounds:
        """BLABLABLA"""

    @abstractmethod
    def fit(self, *args: Any, **kwargs: Any) -> NDArray[np.float64]:
        """
        Args:
            *args ():
            **kwargs ():

        Returns:
        """


class Likelihood(ParametricComponent, ABC):
    def __init__(self, model: ParametricModel):
        super().__init__()
        self.compose_with(model=model)
        self.hasjac = False
        if hasattr(self.model, "jac_hf") and hasattr(self.model, "jac_chf"):
            self.hasjac = True

    @abstractmethod
    def negative_log(self, params: NDArray[np.float64]) -> float:
        """
        Args:
            params ():

        Returns:
            Negative log likelihood value given a set a parameters values
        """


class LikelihoodFromLifetimes(Likelihood):
    def __init__(
        self,
        model: "ParametricLifetimeModel",
        observed_lifetimes: LifetimeSample,
        truncations: Truncations,
    ):
        super().__init__(model)
        self.observed_lifetimes = observed_lifetimes
        self.truncations = truncations

    def _complete_contribs(self, lifetimes: Sample) -> float:
        return -np.sum(np.log(self.model.hf(lifetimes.values, *lifetimes.args)))

    def _right_censored_contribs(self, lifetimes: Sample) -> float:
        return np.sum(
            self.model.chf(lifetimes.values, *lifetimes.args), dtype=np.float64
        )

    def _left_censored_contribs(self, lifetimes: Sample) -> float:
        return -np.sum(
            np.log(-np.expm1(-self.model.chf(lifetimes.values, *lifetimes.args)))
        )

    def _left_truncations_contribs(self, lifetimes: Sample) -> float:
        return -np.sum(
            self.model.chf(lifetimes.values, *lifetimes.args), dtype=np.float64
        )

    def _jac_complete_contribs(self, lifetimes: Sample) -> NDArray[np.float64]:
        return -np.sum(
            self.model.jac_hf(lifetimes.values, *lifetimes.args)
            / self.model.hf(lifetimes.values, *lifetimes.args),
            axis=0,
        )

    def _jac_right_censored_contribs(self, lifetimes: Sample) -> NDArray[np.float64]:
        return np.sum(
            self.model.jac_chf(lifetimes.values, *lifetimes.args),
            axis=0,
        )

    def _jac_left_censored_contribs(self, lifetimes: Sample) -> NDArray[np.float64]:
        return -np.sum(
            self.model.jac_chf(lifetimes.values, *lifetimes.args)
            / np.expm1(self.model.chf(lifetimes.values, *lifetimes.args)),
            axis=0,
        )

    def _jac_left_truncations_contribs(self, lifetimes: Sample) -> NDArray[np.float64]:
        return -np.sum(
            self.model.jac_chf(lifetimes.values, *lifetimes.args),
            axis=0,
        )

    def negative_log(
        self,
        params: NDArray[np.float64],
    ) -> float:
        self.params = params
        return (
            self._complete_contribs(self.observed_lifetimes.complete)
            + self._right_censored_contribs(self.observed_lifetimes.rc)
            + self._left_censored_contribs(self.observed_lifetimes.left_censored)
            + self._left_truncations_contribs(self.truncations.left)
        )

    def jac_negative_log(
        self,
        params: NDArray[np.float64],
    ) -> Union[None, NDArray[np.float64]]:
        """

        Args:
            params ():

        Returns:

        """
        if not self.hasjac:
            warnings.warn("Model does not support jac negative likelihood natively")
            return None
        self.params = params
        return (
            self._jac_complete_contribs(self.observed_lifetimes.complete)
            + self._jac_right_censored_contribs(self.observed_lifetimes.rc)
            + self._jac_left_censored_contribs(self.observed_lifetimes.left_censored)
            + self._jac_left_truncations_contribs(self.truncations.left)
        )


class ParametricLifetimeModel(LifetimeModel[*Args], ParametricModel, ABC):
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
    def init_params(self, lifetimes: LifetimeSample) -> None:
        """"""

    def fit(
        self,
        time: NDArray[np.float64],
        event: Optional[NDArray[np.bool_]] = None,
        entry: Optional[NDArray[np.float64]] = None,
        departure: Optional[NDArray[np.float64]] = None,
        args: tuple[NDArray[np.float64], ...] | tuple[()] = (),
        inplace: bool = True,
        **kwargs: Any,
    ) -> NDArray[np.float64]:
        """
        Args:
            time ():
            event ():
            entry ():
            departure ():
            args ():
            inplace ():
            **kwargs ():

        Returns:
        """
        observed_lifetimes, truncations = lifetime_factory_template(
            time,
            event,
            entry,
            departure,
            args,
        )

        optimized_model = self.copy()
        optimized_model.init_params(observed_lifetimes)

        likelihood = LikelihoodFromLifetimes(
            optimized_model,
            observed_lifetimes,
            truncations,
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

        if inplace:
            self.init_params(observed_lifetimes)
            self.params = likelihood.params

        return optimizer.x

    def __getattribute__(self, item):
        """control if params are set"""

        if (
            not item.startswith("_")
            and not not item.startswith("__")
            and hasattr(LifetimeModel, item)
        ):
            if not self.all_params_set:
                raise ValueError(
                    f"Can't call {item} if one model params is not set. Instanciate fully parametrized model or fit it"
                )
        return super().__getattribute__(item)
