"""
This module defines fundamental types of statistical models used in survival analysis

Copyright (c) 2022, RTE (https://www.rte-france.com)
See AUTHORS.txt
SPDX-License-Identifier: Apache-2.0 (see LICENSE.txt)
"""

import copy
import functools
import inspect
import warnings
from abc import ABC, abstractmethod
from itertools import chain
from typing import Any, Iterator, Union, Optional, Sequence

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import Bounds, minimize

from relife2.data import LifetimeSample, Truncations, lifetime_factory_template
from relife2.data.dataclass import Sample
from relife2.io import array_factory, preprocess_lifetime_data
from relife2.protocols import LifetimeProtocol, ParametricModelProtocol


class Composite:
    """
    Composite pattern used to structure dictionaries of values in a tree structure
    (used to store models' parameters)
    """

    def __init__(self, **kwargs):
        self._node_data: dict[str, Any] = {}
        if kwargs:
            self._node_data = kwargs
        self.parent: Union["Composite", None] = None
        self.leaves: dict[str, "Composite"] = {}
        self._names, self._values = [], []

    @property
    def node_data(self):
        """data of current node as dict"""
        return self._node_data

    @property
    def names(self):
        """keys of current and leaf nodes as list"""
        return self._names

    @property
    def values(self):
        """values of current and leaf nodes as list"""
        return self._values

    @values.setter
    def values(self, new_values: list[Any]):
        self._set_values(new_values)
        self.update_parents()

    def _set_values(self, new_values: list[Any]):
        if len(new_values) != len(self):
            raise ValueError(
                f"values expects {len(self)} items but got {len(new_values)}"
            )
        self._values = new_values
        pos = len(self._node_data)
        self._node_data.update(zip(self._node_data, new_values[:pos]))
        for leaf in self.leaves.values():
            leaf._set_values(new_values[pos : pos + len(leaf)])
            pos += len(leaf)

    @names.setter
    def names(self, new_names: list[str]):
        self._set_names(new_names)
        self.update_parents()

    def _set_names(self, new_names: list[str]):
        if len(new_names) != len(self):
            raise ValueError(
                f"names expects {len(self)} items but got {len(new_names)}"
            )
        self._names = new_names
        pos = len(self._node_data)
        self._node_data = {new_names[:pos][i]: v for i, v in self._node_data.values()}
        for leaf in self.leaves.values():
            leaf._set_names(new_names[pos : pos + len(leaf)])
            pos += len(leaf)

    @node_data.setter
    def node_data(self, new_values: dict[str, Any]):
        self._node_data = new_values
        self.update()

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

    def _items_walk(self) -> Iterator:
        """parallel walk through key value pairs"""
        yield list(self._node_data.items())
        for leaf in self.leaves.values():
            yield list(chain.from_iterable(leaf._items_walk()))

    def _all_items(self) -> Iterator:
        return chain.from_iterable(self._items_walk())

    def update_items(self):
        """parallel iterations : faster than update_value followed by update_keys"""
        try:
            next(self._all_items())
            _k, _v = zip(*self._all_items())
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


class ParametricModule:
    """
    Base class of all parametric functions grouped in one class. It makes easier to compose functions
    and to tune parametric structure of functions.

    main methods are : new_params and add_functions
    """

    def __init__(self):
        self._params = Composite()
        self.leaves: dict[str, "ParametricModule"] = {}

    @property
    def all_params_set(self):
        return (
            np.isnan(self._params.values).any()
            and not np.isnan(self._params.values).all()
        )

    @property
    def params(self):
        return np.array(self._params.values, dtype=np.float64)

    @property
    def params_names(self):
        return self._params.names

    @params.setter
    def params(self, new_values):
        self._params.values = new_values

    @property
    def nb_params(self):
        return len(self._params)

    def compose_with(self, **kwmodule: "ParametricModule"):
        """add functions that can be called from node"""
        for name in kwmodule.keys():
            if name in self._params.node_data:
                raise ValueError(f"{name} already exists as param name")
            if name in self.leaves:
                raise ValueError(f"{name} already exists as leaf function")
        for name, module in kwmodule.items():
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
        """
        Returns:
            ParametricModule : a copy of the ParametricFunctions object
        """
        return copy.deepcopy(self)


class LifetimeModel(ABC):
    """
    Base class controling that subclass interface are composed of expected probability functions
    """

    def __init_subclass__(cls, concrete=True, **kwargs):
        if concrete and not inspect.isabstract(cls):
            if not issubclass(cls, LifetimeProtocol):
                raise NotImplementedError(
                    f"{cls} must implement protocols.LifetimeProtocol, at least one method is missing"
                )
        super().__init_subclass__(**kwargs)

    def __getattribute__(self, item):
        if hasattr(LifetimeProtocol, item):
            attr = super().__getattribute__(item)
            if inspect.ismethod(attr):

                @functools.wraps(attr)
                def wrapper(*args, **kwargs):
                    if "time" in inspect.signature(attr).parameters:
                        time_pos = list(inspect.signature(attr).parameters).index(
                            "time"
                        )
                        time = args[time_pos]
                        time = array_factory(time)
                        args = [
                            array_factory(arg, nb_units=time.shape[0])
                            for i, arg in enumerate(args)
                        ]
                    if getattr(attr, "decorated_method", False):
                        return attr(*args, **kwargs)
                    else:
                        return np.squeeze(attr(*args, **kwargs))

                return wrapper
            else:
                return attr
        return super().__getattribute__(item)

    @property
    @abstractmethod
    def support_upper_bound(self): ...

    @property
    @abstractmethod
    def support_lower_bound(self): ...


class ParametricModel(ParametricModule, ABC):
    def __init_subclass__(cls, concrete=True, **kwargs):
        if concrete and not inspect.isabstract(cls):
            if not issubclass(cls, ParametricModelProtocol):
                raise NotImplementedError(
                    f"{cls} must implement protocols.ParametricModelProtocol, at least one method is missing"
                )
        super().__init_subclass__(**kwargs)

    @abstractmethod
    def init_params(self, *args): ...

    @property
    @abstractmethod
    def params_bounds(self) -> Bounds:
        """BLABLABLA"""


class Likelihood(ParametricModule, ABC):
    """
    Class that instanciates likelihood base having finite number of parameters related to
    one parametric functions
    """

    def __init__(self, model: ParametricModel):
        super().__init__()
        self.compose_with(model=model)
        self.hasjac = False
        if hasattr(self.function, "jac_hf") and hasattr(self.function, "jac_chf"):
            self.hasjac = True

    @abstractmethod
    def negative_log(self, params: np.ndarray) -> float:
        """
        Args:
            params ():

        Returns:
            Negative log likelihood value given a set a parameters values
        """


class LikelihoodFromLifetimes(Likelihood):
    """
    BLABLABLA
    """

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
            np.log(-np.expm1(-self.function.chf(lifetimes.values, *lifetimes.args)))
        )

    def _left_truncations_contribs(self, lifetimes: Sample) -> float:
        return -np.sum(
            self.model.chf(lifetimes.values, *lifetimes.args), dtype=np.float64
        )

    def _jac_complete_contribs(self, lifetimes: Sample) -> np.ndarray:
        return -np.sum(
            self.model.jac_hf(lifetimes.values, *lifetimes.args)
            / self.model.hf(lifetimes.values, *lifetimes.args),
            axis=0,
        )

    def _jac_right_censored_contribs(self, lifetimes: Sample) -> np.ndarray:
        return np.sum(
            self.model.jac_chf(lifetimes.values, *lifetimes.args),
            axis=0,
        )

    def _jac_left_censored_contribs(self, lifetimes: Sample) -> np.ndarray:
        return -np.sum(
            self.model.jac_chf(lifetimes.values, *lifetimes.args)
            / np.expm1(self.model.chf(lifetimes.values, *lifetimes.args)),
            axis=0,
        )

    def _jac_left_truncations_contribs(self, lifetimes: Sample) -> np.ndarray:
        return -np.sum(
            self.model.jac_chf(lifetimes.values, *lifetimes.args),
            axis=0,
        )

    def negative_log(
        self,
        params: np.ndarray,
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
        params: np.ndarray,
    ) -> Union[None, np.ndarray]:
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


class ParametricLifetimeModel(LifetimeModel, ParametricModel, ABC):  # concrete=False):
    """
    Extended interface of LifetimeModel whose params can be estimated with fit method
    """

    def fit(
        self,
        time: ArrayLike,
        event: Optional[ArrayLike] = None,
        entry: Optional[ArrayLike] = None,
        departure: Optional[ArrayLike] = None,
        args: Optional[Sequence[ArrayLike] | ArrayLike] = (),
        inplace: bool = True,
        **kwargs,
    ) -> np.ndarray:
        """
        BLABLABLABLA
        Args:
            time (ArrayLike):
            event (Optional[ArrayLike]):
            entry (Optional[ArrayLike]):
            departure (Optional[ArrayLike]):
            args (Optional[tuple[ArrayLike]]):
            inplace (bool): (default is True)

        Returns:
            Parameters: optimum parameters found
        """
        time, event, entry, departure, args = preprocess_lifetime_data(
            time, event, entry, departure, args
        )
        observed_lifetimes, truncations = lifetime_factory_template(
            time,
            event,
            entry,
            departure,
            args,
        )

        optimized_model = self.copy()
        optimized_model.init_params(observed_lifetimes.rlc, *args)
        param0 = optimized_model.params

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
            "x0": kwargs.get("x0", param0),
        }

        optimizer = minimize(
            likelihood.negative_log,
            minimize_kwargs.pop("x0"),
            jac=None if not likelihood.hasjac else likelihood.jac_negative_log,
            **minimize_kwargs,
        )

        if inplace:
            self.params = likelihood.params

        return optimizer.x
