"""
This module defines fundamental types used in regression package

Copyright (c) 2022, RTE (https://www.rte-france.com)
See AUTHORS.txt
SPDX-License-Identifier: Apache-2.0 (see LICENSE.txt)
"""

from abc import ABC, abstractmethod
from types import EllipsisType
from typing import Optional, Any, Union, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import Bounds

IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]
FloatArray = NDArray[np.float64]
Index = Union[EllipsisType, int, slice, tuple[EllipsisType, int, slice, ...]]


class Parameters:
    """
    Object that encapsulates model parameters names and values
    It emulated a 1D np.array with named values

    Examples:
        >>> params = Parameters(rate=1, scale=2)
        >>> params.rate
        1.0
        >>> params.values
        array([1., 2.])
        >>> other_params = Parameters(w0=None, w1=None, w2=None)
        >>> other_params.values = (3., 5., 6.)
        >>> other_params.values
        array([3., 5., 6.])
        >>> params.append(other_params)
        >>> params.values
        array([1., 2., 3., 5., 6.])
        >>> params.w1
        5.0
        >>> params[2]
        3.0
        >>> params[2:]
        array([3., 5., 6.])
        >>> params.update(w1=8, rate=2.)
        >>> params.values
        array([2., 2., 3., 8., 6.])
        >>> params[1:3] = [10., 11.]
        >>> params.values
        array([ 2., 10., 11.,  8.,  6.])
    """

    def __init__(self, **kwparams: Union[float, None]):

        self.name_to_indice = {}
        values = []
        for i, (k, v) in enumerate(kwparams.items()):
            self.name_to_indice[k] = i
            if v is None:
                values.append(np.random.random())
            else:
                values.append(float(v))
        self._values = np.array(values, dtype=np.float64)

    def __len__(self):
        return len(self.name_to_indice)

    @property
    def names(self) -> tuple[str, ...]:
        """
        Returns:
            tuple[str, ...]: param names
        """
        return tuple(self.name_to_indice.keys())

    @property
    def size(self) -> int:
        """
        Returns:
            int: nb of parameters (alias of len)
        """
        return len(self)

    @property
    def values(self):
        return self._values

    @values.setter
    def values(self, new_values: ArrayLike) -> None:
        """
        Affects new values to Parameters attributes
        Args:
            new_values (Union[float, ArrayLike]):
        """
        new_values = np.asarray(new_values, dtype=np.float64).reshape(
            -1,
        )
        nb_of_params = self.size
        if new_values.size != nb_of_params:
            raise ValueError(
                f"Can't set different number of params, expected {nb_of_params} param values, got {new_values.size}"
            )
        self._values = new_values

    def __getitem__(self, index: Index) -> Union[float, FloatArray]:
        try:
            val = self.values[index]
            if isinstance(val, np.float64):
                val = val.item()
            return val
        except IndexError as exc:
            raise IndexError(
                f"Invalid index : params indices are 1d from 0 to {len(self) - 1}"
            ) from exc

    def __setitem__(
        self,
        index: Index,
        new_values: ArrayLike,
    ) -> None:
        new_values = np.asarray(new_values, dtype=np.float64).reshape(
            -1,
        )
        try:
            self._values[index] = new_values
        except IndexError as exc:
            raise IndexError(
                f"Invalid index : params indices are 1d from 0 to {len(self) - 1}"
            ) from exc

    def __getattr__(self, name: str):
        if name in self.__dict__:
            value = self.__dict__[name]
        elif name in self.name_to_indice:
            value = self._values[self.name_to_indice[name]].item()
        else:
            raise AttributeError(f"Parameters has no attribute named {name}")
        return value

    def __setattr__(self, name: str, value: Any):
        if name == "name_to_indice":
            super().__setattr__(name, value)
        elif name in self.name_to_indice:
            self._values[self.name_to_indice[name]] = float(value)
        else:
            super().__setattr__(name, value)

    def append(self, params: Self) -> None:
        """
        Appends another Parameters object to itself
        Args:
            params (Parameters): Parameters object to append
        """
        if set(self.names) & set(params.names):
            raise ValueError("Can't append Parameters object having common param names")
        self._values = np.concatenate((self._values, params.values))
        self.name_to_indice.update(
            {name: indice + len(self) for name, indice in params.name_to_indice.items()}
        )

    def update(self, **kparams_names: float) -> None:
        """
        Args:
            **kparams_names (float): new parameters values
        Returns:
            Update param values of specified param names
        """
        for name, value in kparams_names.items():
            if hasattr(self, name):
                setattr(self, name, value)
            else:
                raise AttributeError(f"Parameters has no attribute {name}")

    def copy(self) -> Self:
        names = self.names
        return Parameters(**{names[i]: value for i, value in enumerate(self._values)})

    def __repr__(self):
        class_name = type(self).__name__
        attributes = ", ".join([f"{name}={getattr(self, name)}" for name in self.names])
        return f"{class_name}({attributes})"

    def __str__(self):
        attributes = "\n".join(
            [f"\t{name}: {getattr(self, name)}," for name in self.names]
        )
        return f"Parameters(\n{attributes}\n)"


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
    def init_params(self, *args: Any) -> FloatArray:
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
    """
    BLABLABLA
    """

    def __init__(
        self,
        functions: Functions,
    ):
        self.functions = functions

    @property
    def params(self):
        """
        Returns:
        """
        return self.functions.params

    @params.setter
    def params(self, values: Union[FloatArray, Parameters]):
        """
        Args:
            values ():

        Returns:
        """
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


class JacLikelihood(Likelihood, ABC):
    """
    BLABLABLA
    """

    @abstractmethod
    def jac_negative_log_likelihood(self):
        """"""


class Model(ABC):
    """
    BLABLABLA
    """

    def __init__(self, functions: Functions):
        self.functions = functions

    @property
    def params(self):
        """
        Returns:
        """
        return self.functions.params

    @params.setter
    def params(self, values: Union[FloatArray, Parameters]):
        """
        Args:
            values ():

        Returns:
        """
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
