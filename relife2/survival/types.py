"""
This module defines fundamental types used in regression package

Copyright (c) 2022, RTE (https://www.rte-france.com)
See AUTHORS.txt
SPDX-License-Identifier: Apache-2.0 (see LICENSE.txt)
"""

from abc import ABC, abstractmethod
from typing import Optional, Any, Union

import numpy as np
from numpy.typing import ArrayLike
from numpy.typing import NDArray
from scipy.optimize import Bounds

from relife2.survival.data import ObservedLifetimes, Truncations, LifetimeData
from relife2.survival.parameters import Parameters

IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]
FloatArray = NDArray[np.float64]


class Functions(ABC):
    def __init__(self, params: Parameters):
        self._params = params

    @property
    def params(self):
        """BLABLABLA"""
        return self._params

    @params.setter
    def params(self, values: Union[FloatArray, Parameters]) -> None:
        """BLABLABLA"""
        if isinstance(values, Parameters):
            values = values.values
        self._params.values = values

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

    @abstractmethod
    def initial_params(self, *lifetimes: LifetimeData) -> FloatArray:
        """initialization of params values given observed lifetimes"""

    @property
    @abstractmethod
    def params_bounds(self) -> Bounds:
        """BLABLABLA"""

    def __getattr__(self, name: str):
        class_name = type(self).__name__
        if name in self.__dict__:
            value = self.__dict__[name]
        elif name in ["values", "size"]:
            raise AttributeError(
                f"""
            {class_name} has no attribute named {name}. Maybe you meant functions.params.{name}
            """
            )
        else:
            if not hasattr(self.params, name):
                raise AttributeError(f"{class_name} has no attribute named {name}")
            value = getattr(self.params, name)
        return value

    def __setattr__(self, name: str, value: Any):
        if name == "_params":
            super().__setattr__(name, value)
        elif hasattr(self.params, name):
            setattr(self.params, name, value)
        else:
            super().__setattr__(name, value)

    def __repr__(self):
        class_name = type(self).__name__
        return f"{class_name}({self.params.__repr__()})"


class CompositionFunctions(Functions, ABC):

    def __init__(self, **kwfunctions: Functions):
        self.composites = kwfunctions
        params = Parameters()
        for functions in kwfunctions.values():
            params.append(functions.params)
        super().__init__(params)

    @property
    def params(self):
        """BLABLABLA"""
        return self._params

    @params.setter
    def params(self, values: Union[FloatArray, Parameters]) -> None:
        """BLABLABLA"""
        if isinstance(values, Parameters):
            values = values.values
        self._params.values = values
        pos = 0
        for functions in self.composites.values():
            functions.params = values[pos : pos + functions.params.size]
            pos += functions.params.size

    def __getattr__(self, name: str):
        value = None
        class_name = type(self).__name__
        if name in self.__dict__:
            value = self.__dict__[name]
        elif name in self.composites:
            value = self.composites[name]
        elif name in ["values", "size"]:
            raise AttributeError(
                f"""
                {class_name} has no attribute named {name}. Maybe you meant functions.params.{name}
                """
            )
        else:
            for functions in self.composites.values():
                if hasattr(functions, name):
                    value = getattr(functions, name)
                    break
        if value is None:
            raise AttributeError(f"{class_name} has no attribute named {name}")
        return value

    def __setattr__(self, name: str, value: Any):
        if name == "composites":
            self.__dict__[name] = value
        elif hasattr(self.params, name):
            setattr(self.params, name, value)
            for functions in self.composites.values():
                if hasattr(functions, name):
                    setattr(functions, name, value)
                    break
        else:
            super().__setattr__(name, value)

    def __repr__(self):
        class_name = type(self).__name__
        functions_repr = "".join(
            (
                f"    {name} : {value.__repr__()},\n"
                for name, value in self.composites.items()
            )
        )
        return f"{class_name}(\n{functions_repr})"


class Likelihood(ABC):
    def __init__(
        self,
        functions: Functions,
        observed_lifetimes: ObservedLifetimes,
        truncations: Truncations,
    ):
        self.functions = functions
        self.observed_lifetimes = observed_lifetimes
        self.truncations = truncations

    def initial_params(self, *lifetimes: LifetimeData):
        return self.functions.initial_params(*lifetimes)

    @property
    def params(self):
        return self.functions.params

    @params.setter
    def params(self, values: Union[FloatArray, Parameters]):
        self.functions.params = values

    @abstractmethod
    def negative_log_likelihood(self):
        """BLABLABLA"""

    def __getattr__(self, name: str):
        class_name = type(self).__name__
        if name in self.__dict__:
            value = self.__dict__[name]
        else:
            if not hasattr(self.functions, name):
                raise AttributeError(f"{class_name} has no attribute named {name}")
            value = getattr(self.functions, name)
        return value

    def __setattr__(self, name: str, value: Any):
        if name == "functions":
            super().__setattr__(name, value)
        elif hasattr(self.functions, name):
            setattr(self.functions, name, value)
        else:
            super().__setattr__(name, value)


class Model(ABC):
    def __init__(self, functions: Functions):
        self.functions = functions

    @property
    def params(self):
        return self.functions.params

    @params.setter
    def params(self, values: Union[FloatArray, Parameters]):
        self.functions.params = values

    @abstractmethod
    def fit(
        self,
        time: ArrayLike,
        entry: Optional[ArrayLike] = None,
        departure: Optional[ArrayLike] = None,
        lc_indicators: Optional[ArrayLike] = None,
        rc_indicators: Optional[ArrayLike] = None,
        inplace: bool = True,
        **kwargs: Any,
    ) -> Parameters:
        """
        BLABLABLABLA
        Args:
            time (ArrayLike):
            entry (Optional[ArrayLike]):
            departure (Optional[ArrayLike]):
            lc_indicators (Optional[ArrayLike]):
            rc_indicators (Optional[ArrayLike]):
            inplace (bool): (default is True)

        Returns:
            Parameters: optimum parameters found
        """

    def __getattr__(self, name: str):
        class_name = type(self).__name__
        if name in self.__dict__:
            value = self.__dict__[name]
        else:
            if not hasattr(self.functions, name):
                raise AttributeError(f"{class_name} has no attribute named {name}")
            value = getattr(self.functions, name)
        return value

    def __setattr__(self, name: str, value: Any):
        if name == "functions":
            super().__setattr__(name, value)
        elif hasattr(self.functions, name):
            setattr(self.functions, name, value)
        else:
            super().__setattr__(name, value)

    def __repr__(self):
        class_name = type(self).__name__
        return f"{class_name}({self.params.__repr__()})"
