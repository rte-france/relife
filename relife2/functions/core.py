"""
This module defines fundamental types of statistical models used in survival analysis

Copyright (c) 2022, RTE (https://www.rte-france.com)
See AUTHORS.txt
SPDX-License-Identifier: Apache-2.0 (see LICENSE.txt)
"""

import copy
from abc import ABC, abstractmethod
from itertools import chain
from typing import Any, Iterator, Optional, Union

import numpy as np
from numpy import ma
from scipy.optimize import Bounds, newton

from relife2.functions.maths.integrations import gauss_legendre, quad_laguerre


class Composite:
    def __init__(self, **kwargs):
        self._data: dict[str, Any] = {}
        if kwargs:
            self._data = kwargs
        self.parent: Union["Composite", None] = None
        self.leaves: dict[str, "Composite"] = {}
        self._names, self._values = [], []

    @property
    def data(self):
        """data of current node as dict"""
        return self._data

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
        pos = len(self._data)
        self._data.update(zip(self._data, new_values[:pos]))
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
        pos = len(self._data)
        self._data = {new_names[:pos][i]: v for i, v in self._data.values()}
        for leaf in self.leaves.values():
            leaf._set_names(new_names[pos : pos + len(leaf)])
            pos += len(leaf)

    @data.setter
    def data(self, new_data: dict[str, Any]):
        self._data = new_data
        self.update()

    def __len__(self):
        return len(self._names)

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

    def _items_walk(self) -> Iterator:
        """parallel walk through key value pairs"""
        yield list(self._data.items())
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


class ParametricFunction(ABC):
    def __init__(self):
        self._params = Composite()
        self._args = Composite()
        self.leaves: dict[str, "Composite"] = {}

    @property
    def params(self):
        return np.array(self._params.values, dtype=np.float64)

    @property
    def args(self):
        return self._args.values

    @property
    def params_names(self):
        return self._params.names

    @property
    def args_names(self):
        return self._args.names

    @params.setter
    def params(self, new_values):
        self._params.values = new_values

    @args.setter
    def args(self, new_values):
        self._args.values = new_values

    @property
    def nb_params(self):
        return len(self._params)

    @property
    def nb_args(self):
        return len(self._args)

    @abstractmethod
    def init_params(self, *args: Any) -> np.ndarray:
        """initialization of params values (usefull before fit)"""

    @property
    @abstractmethod
    def params_bounds(self) -> Bounds:
        """BLABLABLA"""

    def add_functions(self, **kwfunctions: "ParametricFunction"):
        """add functions that can be called from node"""
        for name in kwfunctions.keys():
            if name in self._args.data:
                raise ValueError(f"{name} already exists as arg name")
            if name in self._params.data:
                raise ValueError(f"{name} already exists as param name")
            if name in self.leaves:
                raise ValueError(f"{name} already exists as leaf function")
        for name, function in kwfunctions.items():
            self.leaves[name] = function
            self._params.set_leaf(f"{name}.params", function._params)
            self._args.set_leaf(f"{name}.args", function._args)

    def new_args(self, **kwargs: np.ndarray):
        """change local args (at node level)"""
        for name in kwargs.keys():
            if name in self._params.data:
                raise ValueError(f"{name} already exists as param name")
            if name in self.leaves.keys():
                raise ValueError(f"{name} already exists as function name")
        self._args.data = kwargs

    def new_params(self, **kwparams):
        """change local params structure (at node level)"""
        for name in kwparams.keys():
            if name in self._args.data:
                raise ValueError(f"{name} already exists as arg name")
            if name in self.leaves.keys():
                raise ValueError(f"{name} already exists as function name")
        self._params.data = kwparams

    def __getattr__(self, name: str):
        class_name = type(self).__name__
        if name in self.__dict__:
            return self.__dict__[name]
        if name in super().__getattribute__("_params"):
            return super().__getattribute__("_params")[name]
        if name in super().__getattribute__("_args"):
            return super().__getattribute__("_args")[name]
        if name in super().__getattribute__("leaves"):
            return super().__getattribute__("leaves")[name]
        raise AttributeError(f"{class_name} has no attribute named {name}")

    def __setattr__(self, name: str, value: Any):
        if name in ["_args", "_params", "leaves"]:
            super().__setattr__(name, value)
        elif name in self._params:
            self._params[name] = value
        elif name in self._args:
            self._args[name] = value
        elif name in self.leaves:
            raise ValueError(
                "Can't modify leaf function. Recreate Function instance instead"
            )
        else:
            super().__setattr__(name, value)

    def copy(self):
        """
        Returns:
            ParametricFunctions : a copy of the ParametricFunctions object
        """
        return copy.deepcopy(self)


class ParametricLifetimeFunction(ParametricFunction, ABC):
    """
    BLABLABLA
    """

    def __init__(
        self,
        **kwparams: Union[float, None],
    ):
        super().__init__()
        self.new_params(**kwparams)
        self._base_functions: list[str] = []
        for name in ["sf", "hf", "chf", "pdf"]:
            if name in self.__class__.__dict__:
                self._base_functions.append(name)

    @property
    @abstractmethod
    def support_upper_bound(self):
        """
        Returns:
            BLABLABLABLA
        """

    @property
    @abstractmethod
    def support_lower_bound(self):
        """
        Returns:
            BLABLABLABLA
        """

    def hf(self, time: np.ndarray) -> np.ndarray:
        """
        BLABLABLABLA
        Args:
            time (np.ndarray): BLABLABLABLA

        Returns:
            np.ndarray: BLABLABLABLA
        """
        if "pdf" in self._base_functions and "sf" in self._base_functions:
            return self.pdf(time) / self.sf(time)
        if "sf" in self._base_functions:
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

    def chf(self, time: np.ndarray) -> np.ndarray:
        """
        BLABLABLABLA
        Args:
            time (np.ndarray): BLABLABLABLA

        Returns:
            np.ndarray: BLABLABLABLA
        """
        if "sf" in self._base_functions:
            return -np.log(self.sf(time))
        if "pdf" in self._base_functions and "hf" in self._base_functions:
            return -np.log(self.pdf(time) / self.hf(time))
        if "hf" in self._base_functions:
            lower_bound = np.zeros_like(time)
            upper_bound = np.broadcast_to(
                np.asarray(self.isf(np.array(1e-4))), time.shape
            )
            masked_upper_bound: ma.MaskedArray = ma.MaskedArray(
                upper_bound, time >= self.support_upper_bound
            )
            masked_lower_bound: ma.MaskedArray = ma.MaskedArray(
                lower_bound, time >= self.support_upper_bound
            )

            integration = gauss_legendre(
                self.hf,
                masked_lower_bound,
                masked_upper_bound,
                ndim=2,
            ) + quad_laguerre(
                self.hf,
                masked_upper_bound,
                ndim=2,
            )
            return ma.filled(integration, 1.0)

        class_name = type(self).__name__
        raise NotImplementedError(
            f"""
        {class_name} must implement chf or at least hf function
        """
        )

    def sf(self, time: np.ndarray) -> np.ndarray:
        """
        BLABLABLABLA
        Args:
            time (np.ndarray): BLABLABLABLA

        Returns:
            np.ndarray: BLABLABLABLA
        """
        if "chf" in self._base_functions:
            return np.exp(-self.chf(time))
        if "pdf" in self._base_functions and "hf" in self._base_functions:
            return self.pdf(time) / self.hf(time)

        class_name = type(self).__name__
        raise NotImplementedError(
            f"""
        {class_name} must implement sf function
        """
        )

    def pdf(self, time: np.ndarray) -> np.ndarray:
        """

        Args:
            time ():

        Returns:

        """
        try:
            return self.sf(time) * self.hf(time)
        except NotImplementedError as err:
            class_name = type(self).__name__
            raise NotImplementedError(
                f"""
            {class_name} must implement pdf or the above functions
            """
            ) from err

    def mrl(self, time: np.ndarray) -> np.ndarray:
        """
        BLABLABLABLA
        Args:
            time (np.ndarray): BLABLABLABLA

        Returns:
            np.ndarray: BLABLABLABLA
        """
        masked_time: ma.MaskedArray = ma.MaskedArray(
            time, time >= self.support_upper_bound
        )
        upper_bound = np.broadcast_to(np.asarray(self.isf(np.array(1e-4))), time.shape)
        masked_upper_bound: ma.MaskedArray = ma.MaskedArray(
            upper_bound, time >= self.support_upper_bound
        )

        def integrand(x):
            return (x - masked_time) * self.pdf(x)

        integration = gauss_legendre(
            integrand,
            masked_time,
            masked_upper_bound,
            ndim=2,
        ) + quad_laguerre(
            integrand,
            masked_upper_bound,
            ndim=2,
        )
        mrl = integration / self.sf(masked_time)
        return ma.filled(mrl, 0.0)

    def moment(self, n: int) -> np.ndarray:
        """
        BLABLABLA
        Args:
            n (int): BLABLABLA

        Returns:
            BLABLABLA
        """
        upper_bound = self.isf(np.array(1e-4))

        def integrand(x):
            return x**n * self.pdf(x)

        return gauss_legendre(
            integrand, np.array(0.0), upper_bound, ndim=2
        ) + quad_laguerre(integrand, upper_bound, ndim=2)

    def mean(self) -> np.ndarray:
        """
        BLABLABLABLA
        Returns:
            float: BLABLABLABLA
        """
        return self.moment(1)

    def var(self) -> Union[float | np.ndarray]:
        """
        BLABLABLABLA

        Returns:
            float: BLABLABLABLA
        """
        return self.moment(2) - self.moment(1) ** 2

    def isf(
        self,
        probability: np.ndarray,
    ) -> np.ndarray:
        """
        BLABLABLABLA
        Args:
            probability (np.ndarray): BLABLABLABLA

        Returns:
            np.ndarray: BLABLABLABLA
        """
        return newton(
            lambda x: self.sf(x) - probability,
            x0=np.zeros_like(probability),
        )

    def cdf(self, time: np.ndarray) -> np.ndarray:
        """
        BLABLABLABLA
        Args:
            time (np.ndarray): BLABLABLABLA

        Returns:
            np.ndarray: BLABLABLABLA
        """
        return 1 - self.sf(time)

    def rvs(self, size: Optional[int] = 1, seed: Optional[int] = None) -> np.ndarray:
        """
        BLABLABLABLA
        Args:
            size (Optional[int]): BLABLABLABLA
            seed (Optional[int]): BLABLABLABLA

        Returns:
            np.ndarray: BLABLABLABLA
        """
        generator = np.random.RandomState(seed=seed)
        probability = generator.uniform(size=size)
        return self.isf(probability)

    def ppf(
        self,
        probability: np.ndarray,
    ) -> np.ndarray:
        """
        BLABLABLABLA
        Args:
            probability (np.ndarray): BLABLABLABLA

        Returns:
            np.ndarray: BLABLABLABLA
        """
        return self.isf(1 - probability)

    def median(self) -> np.ndarray:
        """
        BLABLABLABLA

        Returns:
            float: BLABLABLABLA
        """
        return self.ppf(np.array(0.5))
