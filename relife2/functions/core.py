"""
This module defines fundamental types of statistical models used in survival analysis

Copyright (c) 2022, RTE (https://www.rte-france.com)
See AUTHORS.txt
SPDX-License-Identifier: Apache-2.0 (see LICENSE.txt)
"""

import copy
from abc import ABC, abstractmethod
from typing import Any, Optional, Union

import numpy as np
from numpy import ma
from numpy.typing import ArrayLike
from scipy.optimize import Bounds, newton

from relife2.functions.maths.integrations import gauss_legendre, quad_laguerre


class ParametricFunctions(ABC):
    """
    Base class of objects implementing parametric functions.
    ParametricFunctions have a tree structure of parameters allowing to compose functions with add_functions method.
    Parameters can be called and set by their name.

    Examples:
        >>> # ParametricFunctions types must implement init_params and params_bounds methods
        ... # They are passed in this example just to show how main ParametricFunctions methods work
        ... class MyFunctions(ParametricFunctions):
        ...   def init_params(self):
        ...       pass
        ...   def params_bounds(self):
        ...       pass
        >>> f = MyFunctions(w1=1.0, w2=2.0) # a parametric functions made of two parameters
        >>> print(f.params) # parameters are returned as np.ndarray
        ... # doctest: +NORMALIZE_WHITESPACE
        [1. 2.]
        >>> g = MyFunctions(w3=3.0, w4=4, w5=5) # other parametric functions
        >>> h = MyFunctions(w6=6)
        >>> y = MyFunctions(w7=7.0)
        >>> z = MyFunctions(w8=8.0)
        >>> k = MyFunctions(w9=9.0)
        >>> f.add_functions("g", g)
        >>> h.add_functions("y", y) # compose functions
        >>> h.add_functions("z", z)
        >>> f.add_functions("h", h)
        >>> f.add_functions("k", k)
        >>> print(f) # functions are like tree in their structure of parameters
        ... # doctest: +NORMALIZE_WHITESPACE
        MyFunctions {'w1': 1.0, 'w2': 2.0},
        |____g {'w3': 3.0, 'w4': 4.0, 'w5': 5.0},
        |____h {'w6': 6.0},
        |________y {'w7': 7.0},
        |________z {'w8': 8.0},
        |____k {'w9': 9.0},
        >>> (f.params == np.array([1., 2., 3., 4., 5., 6., 7., 8., 9.])).all()
        ... # but parameters are always seen as one np.ndarray object
        ... # read from top to bottom and left to right
        True
        >>> f.w5 = 10.0 # parameters can be set by their name
        >>> (f.params == np.array([1., 2., 3., 4., 10., 6., 7., 8., 9.])).all()
        True
        >>> print(f.leaves_functions["g"]) # in the backend functions are stored in a dict
        MyFunctions {'w3': 3.0, 'w4': 4.0, 'w5': 10.0},
        >>> print(hasattr(f, "h")) # functions are seen as attribute and can be resquested
        True
        >>> print(hasattr(f, "w5")) # even parameters are seen as attribute and can be requested
        True
    """

    def __init__(self, **kwparams: Union[float, None]):

        self.root_params: dict[str, Union[float, None]] = {}
        for k, v in kwparams.items():
            if v is not None:
                self.root_params[k] = float(v)
            else:
                self.root_params[k] = v
        self.leaves_functions: dict[str, "ParametricFunctions"] = {}
        self.all_params: dict[str, Union[float, None]] = self.root_params.copy()

    @abstractmethod
    def init_params(self, *args: Any) -> np.ndarray:
        """initialization of params values (usefull before fit)"""

    @property
    @abstractmethod
    def params_bounds(self) -> Bounds:
        """BLABLABLA"""

    @property
    def params_names(self) -> tuple[str, ...]:
        """
        Returns:
            tuple[str, ...]: param names
        """
        return tuple(self.all_params.keys())

    @property
    def nb_params(self) -> int:
        """
        Returns:
            int: nb of parameters (alias of len)
        """
        return len(self.all_params)

    @property
    def params(self) -> np.ndarray:
        """
        Returns:
            (np.ndarray) : parameters values
        """
        return np.array(tuple(self.all_params.values()), dtype=np.float64)

    @params.setter
    def params(self, values: ArrayLike) -> None:
        """
        Affects new values to Parameters attributes
        Args:
            values (Union[float, ArrayLike]):
        """
        values = np.asarray(values, dtype=np.float64).reshape(
            -1,
        )
        if values.size != self.nb_params:
            raise ValueError(
                f"Can't set different number of params, expected {self.nb_params} param values, got {values.size}"
            )

        self.all_params.update(zip(self.all_params, values))
        pos = len(self.root_params)
        self.root_params.update(zip(self.root_params, values[:pos]))
        for functions in self.leaves_functions.values():
            functions.params = values[pos : pos + functions.nb_params]
            pos += functions.nb_params

    def add_params(self, **kwparams: Union[float, None]) -> None:
        """
        Args:
            **kwparams ():

        Returns:
        """
        if set(self.params_names) & set(kwparams.keys()):
            raise ValueError("Can't add params when param names are already used")
        pos = len(self.root_params)
        root_params_items = list(self.root_params.items())
        all_params_items = list(self.all_params.items())
        items_to_insert = list(kwparams.items())
        self.root_params = dict(
            root_params_items[:pos] + items_to_insert + root_params_items[pos:]
        )
        self.all_params = dict(
            all_params_items[:pos] + items_to_insert + all_params_items[pos:]
        )

    def add_functions(self, name: str, functions: "ParametricFunctions") -> None:
        """
        add leaf functions to tree
        Appends another Parameters object to itself
        Args:
            name (str):
            functions (ParametricFunctions): ParametricFunctions object to append
        """
        if set(self.params_names) & set(functions.params_names):
            raise ValueError("Can't add functions having common param names")
        self.leaves_functions[name] = functions
        self.all_params.update(functions.all_params)

    def _get_leaf_functions(self, name: str) -> Union[None, "ParametricFunctions"]:
        leaves = []
        queue = [self]
        found_functions = None
        while queue:
            current_node = queue.pop(0)
            if current_node is None:
                continue
            if not current_node.leaves_functions:
                leaves.append(current_node)
                continue
            for leaf_name, leaf in current_node.leaves_functions.items():
                if leaf_name == name:
                    found_functions = leaf
                    break
                queue.append(leaf)
        return found_functions

    # def _set_leaf_functions(self, name: str, functions: "ParametricFunctions") -> None:
    #     leaves = []
    #     queue = [self]
    #     pos = len(self.root_params)
    #     functions_set = False
    #     while queue:
    #         current_node = queue.pop(0)
    #         if current_node is None:
    #             continue
    #         if not current_node.leaves_functions:
    #             leaves.append(current_node)
    #             continue
    #         for leaf_name, leaf in current_node.leaves_functions.items():
    #             if leaf_name == name:
    #                 all_params_items = list(self.all_params.items())
    #                 items_to_insert = list(functions.all_params.items())
    #
    #                 all_params_items = (
    #                     all_params_items[:pos]
    #                     + items_to_insert
    #                     + all_params_items[pos + leaf.nb_params :]
    #                 )
    #
    #                 all_params_dict = dict(all_params_items)
    #                 if len(all_params_dict) < len(all_params_items):
    #                     raise ValueError(
    #                         "Can't set functions having common param names with other functions"
    #                     )
    #                 self.all_params = all_params_dict
    #                 current_node.leaves_functions[name] = functions
    #                 queue = []
    #                 functions_set = True
    #                 break
    #             queue.append(leaf)
    #             pos += len(leaf.root_params)
    #     if not functions_set:
    #         raise ValueError(f"No functions named {name} was found")

    def __getattr__(self, name: str):
        class_name = type(self).__name__
        if name in self.__dict__:
            return self.__dict__[name]
        if name in super().__getattribute__("all_params"):
            return super().__getattribute__("all_params")[name]
        functions = self._get_leaf_functions(name)
        if functions is None:
            raise AttributeError(f"{class_name} has no attribute named {name}")
        return functions

    def __setattr__(self, name: str, value: Any):
        if name in ["all_params", "leaves_functions", "root_params"]:
            super().__setattr__(name, value)
        elif name in self.all_params:
            self.all_params[name] = float(value)
            if name in self.root_params:
                self.root_params[name] = float(value)
            stop = False
            while not stop:
                for functions in self.leaves_functions.values():
                    if name in functions.all_params:
                        setattr(functions, name, float(value))
                stop = True
        # elif isinstance(value, ParametricFunctions):
        #     self._set_leaf_functions(name, value)
        elif isinstance(value, ParametricFunctions):
            raise AttributeError(
                "Can't set a functions. Recreate a Function object instead"
            )
        else:
            super().__setattr__(name, value)

    def _get_functions_names(self, node):
        leaf_names = {
            key: type(value).__name__ for key, value in node.leaves_functions.items()
        }
        if not node.leaves_functions:
            return leaf_names
        if node.leaves_functions:
            for leaf in node.leaves_functions.values():
                leaf_names.update(self._get_functions_names(leaf))
        return leaf_names

    # def __repr__(self):
    #     class_name = type(self).__name__
    #     return f"{class_name}({self.all_params})"

    def __str__(self):
        class_name = type(self).__name__
        return (
            f"{class_name}(\n"
            f" params = {self.all_params}\n"
            f" functions = {self._get_functions_names(self)}\n)"
        )

    def copy(self):
        """
        Returns:
            ParametricFunctions : a copy of the ParametricFunctions object
        """
        return copy.deepcopy(self)


class ParametricLifetimeFunctions(ParametricFunctions, ABC):
    """
    BLABLABLA
    """

    def __init__(
        self,
        **kwparams: Union[float, None],
    ):
        super().__init__(**kwparams)
        self._base_functions: list[str] = []
        for name in ["sf", "hf", "chf", "pdf"]:
            if name in self.__class__.__dict__:
                self._base_functions.append(name)

        # any other values than time used as inputs in prob functions
        # ex : covar
        self.extravars: dict[str, Any] = {}

    @property
    def extravars_names(self):
        return list(self.extravars.keys())

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
