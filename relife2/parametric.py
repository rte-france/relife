"""
This module defines fundamental types of statistical models used in survival analysis

Copyright (c) 2022, RTE (https://www.rte-france.com)
See AUTHORS.txt
SPDX-License-Identifier: Apache-2.0 (see LICENSE.txt)
"""

from abc import ABC, abstractmethod
from types import EllipsisType
from typing import Any, Optional, Self, Union

import numpy as np
from numpy import ma
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import Bounds, root_scalar

from relife2.utils.integrations import gauss_legendre, quad_laguerre

IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]
FloatArray = NDArray[np.float64]
Index = Union[EllipsisType, int, slice, tuple[EllipsisType, int, slice, ...]]


class Functions(ABC):
    """
    Class that instanciates parametric functions having finite number of parameters.
    They are encapsulated in params attribute and can be set and call
    by their names.

    Examples:
        >>> func = Functions(rate=1, scale=2)
        >>> func.rate
        1.0
        >>> func.params
        array([1., 2.])
        >>> func.params[1:] = [11.]
        >>> func.params
        array([ 1., 11.])
        >>> func.scale = 4.
        >>> func.params
        array([ 1., 4.])
    """

    def __init__(self, **kwparams: Union[float, None]):

        self.params_names_indices = {}
        params_values = []
        for i, (k, v) in enumerate(kwparams.items()):
            self.params_names_indices[k] = i
            if v is None:
                params_values.append(np.random.random())
            else:
                params_values.append(float(v))
        self._params = np.array(params_values, dtype=np.float64)

    @abstractmethod
    def init_params(self, *args: Any) -> FloatArray:
        """initialization of params values given observed lifetimes"""

    @property
    @abstractmethod
    def params_bounds(self) -> Bounds:
        """BLABLABLA"""

    @property
    def params(self) -> FloatArray:
        """BLABLABLA"""
        return self._params

    @params.setter
    def params(self, values: ArrayLike) -> None:
        """
        Args:
            values ():

        Returns:

        """
        self._set_params(values)

    def _set_params(self, values: ArrayLike) -> None:
        """BLABLABLA"""
        values = np.asarray(values, dtype=np.float64).reshape(
            -1,
        )
        nb_of_params = values.size
        expected_nb_of_params = self.params.size
        if nb_of_params != expected_nb_of_params:
            raise ValueError(
                f"""
                Can't set different number of params, expected {expected_nb_of_params} param values, got {nb_of_params}
                """
            )
        self._params = values

    @property
    def params_names(self) -> tuple[str, ...]:
        """
        Returns:
            tuple[str, ...]: param names
        """
        return tuple(self.params_names_indices.keys())

    def copy(self) -> Self:
        """
        Returns:
            A ParamtricFunctions object copied from current instance
        """
        return self.__class__(
            **{self.params_names[i]: value for i, value in enumerate(self._params)}
        )

    def __getattr__(self, name: str):
        class_name = type(self).__name__
        if name in self.__dict__:
            value = self.__dict__[name]
        elif name in self.params_names_indices:
            value = self._params[self.params_names_indices[name]].item()
        else:
            raise AttributeError(f"{class_name} has no attribute named {name}")
        return value

    def __setattr__(self, name: str, value: Any):
        if name == "params_names_indices":
            super().__setattr__(name, value)
        elif name in self.params_names_indices:
            self._params[self.params_names_indices[name]] = float(value)
        else:
            super().__setattr__(name, value)

    def __repr__(self):
        class_name = type(self).__name__
        params_repr = ", ".join(
            [f"{name}: {value}" for name, value in zip(self.params_names, self.params)]
        )
        return f"{class_name}({params_repr})"


class LifetimeFunctions(Functions, ABC):
    """
    BLABLABLA
    """

    def __init__(
        self,
        extra_args: Optional[tuple[str, ...]] = None,
        **kwparams: Union[float, None],
    ):
        super().__init__(**kwparams)
        self.extra_args = list(extra_args) if extra_args else None
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

    def hf(self, time: FloatArray) -> FloatArray:
        """
        BLABLABLABLA
        Args:
            time (FloatArray): BLABLABLABLA

        Returns:
            FloatArray: BLABLABLABLA
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

    def chf(self, time: FloatArray) -> FloatArray:
        """
        BLABLABLABLA
        Args:
            time (FloatArray): BLABLABLABLA

        Returns:
            FloatArray: BLABLABLABLA
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

    def sf(self, time: FloatArray) -> FloatArray:
        """
        BLABLABLABLA
        Args:
            time (FloatArray): BLABLABLABLA

        Returns:
            FloatArray: BLABLABLABLA
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

    def pdf(self, time: FloatArray) -> FloatArray:
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

    def mrl(self, time: FloatArray) -> FloatArray:
        """
        BLABLABLABLA
        Args:
            time (FloatArray): BLABLABLABLA

        Returns:
            FloatArray: BLABLABLABLA
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

    def moment(self, n: int) -> FloatArray:
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

    def mean(self) -> Union[float | FloatArray]:
        """
        BLABLABLABLA
        Returns:
            float: BLABLABLABLA
        """
        return np.squeeze(self.moment(1))[()]

    def var(self) -> Union[float | FloatArray]:
        """
        BLABLABLABLA

        Returns:
            float: BLABLABLABLA
        """
        return np.squeeze(self.moment(2) - self.moment(1) ** 2)[()]

    def isf(
        self,
        probability: FloatArray,
    ) -> FloatArray:
        """
        BLABLABLABLA
        Args:
            probability (FloatArray): BLABLABLABLA

        Returns:
            FloatArray: BLABLABLABLA
        """
        return root_scalar(
            lambda x: self.sf(x) - probability,
            method="newton",
            x0=0.0,
        ).root

    def cdf(self, time: FloatArray) -> FloatArray:
        """
        BLABLABLABLA
        Args:
            time (FloatArray): BLABLABLABLA

        Returns:
            FloatArray: BLABLABLABLA
        """
        return 1 - self.sf(time)

    def rvs(self, size: Optional[int] = 1, seed: Optional[int] = None) -> FloatArray:
        """
        BLABLABLABLA
        Args:
            size (Optional[int]): BLABLABLABLA
            seed (Optional[int]): BLABLABLABLA

        Returns:
            FloatArray: BLABLABLABLA
        """
        generator = np.random.RandomState(seed=seed)
        probability = generator.uniform(size=size)
        return self.isf(probability)

    def ppf(
        self,
        probability: FloatArray,
    ) -> FloatArray:
        """
        BLABLABLABLA
        Args:
            probability (FloatArray): BLABLABLABLA

        Returns:
            FloatArray: BLABLABLABLA
        """
        return self.isf(1 - probability)

    def median(self) -> Union[float | FloatArray]:
        """
        BLABLABLABLA

        Returns:
            float: BLABLABLABLA
        """
        return np.squeeze(self.ppf(np.array(0.5)))[()]

    def copy(self) -> Self:
        """
        Returns:
            A ParamtricFunctions object copied from current instance
        """
        return self.__class__(
            extra_args=tuple(self.extra_args) if self.extra_args else None,
            **{self.params_names[i]: value for i, value in enumerate(self._params)},
        )


class CompositeLifetimeFunctions(LifetimeFunctions, ABC):
    """
    Class that instanciates objects whose probability functions are defined
    from hazard function and constructed from several parametric functions
    """

    def __init__(
        self, extra_args: Optional[tuple[str, ...]] = None, **kwfunctions: Functions
    ):
        self.components = kwfunctions
        kwparams: dict[str, float] = {}
        for functions in kwfunctions.values():
            if set(kwparams.keys()) & set(functions.params_names):
                raise ValueError("Can't compose functions with common param names")
            kwparams.update(dict(zip(functions.params_names, functions.params)))
        super().__init__(extra_args=extra_args, **kwparams)

    @property
    def params(self):
        """BLABLABLA"""
        return self._params

    @params.setter
    def params(self, values: FloatArray) -> None:
        """BLABLABLA"""
        pos = 0
        super()._set_params(values)
        for functions in self.components.values():
            functions.params = values[pos : pos + functions.params.size]
            pos += functions.params.size

    def copy(self) -> Self:
        """
        Returns:
            A CompositeHazard object copied from current instance
        """
        return self.__class__(
            extra_args=tuple(self.extra_args) if self.extra_args else None,
            **{k: v.copy() for k, v in self.components.items()},
        )

    def __getattr__(self, name: str):
        value = None
        class_name = type(self).__name__
        if name in self.__dict__:
            value = self.__dict__[name]
        elif name in self.components:
            value = self.components[name]
        else:
            for functions in self.components.values():
                if hasattr(functions, name):
                    value = getattr(functions, name)
                    break
        if value is None:
            raise AttributeError(f"{class_name} has no attribute named {name}")
        return value

    def __setattr__(self, name: str, value: Any):
        if name == "components":
            self.__dict__[name] = value
        elif name in self.params_names_indices:
            self._params[self.params_names_indices[name]] = float(value)
            for functions in self.components.values():
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
                for name, value in self.components.items()
            )
        )
        return f"{class_name}(\n{functions_repr})"


class LifetimeFunctionsBridge:
    """
    Bridge class to functions implementor that can be extended.
    The bridge pattern allows to decouple varying functions from varying interfaces using them
    """

    def __init__(self, functions: LifetimeFunctions):
        self.functions = functions
        class_name = self.functions.__class__.__name__
        if self.extra_args:
            for arg in self.extra_args:
                if not hasattr(functions, arg):
                    raise AttributeError(
                        f"""
                        {class_name} expected extra arg {arg} but has no attribute called {arg}.
                        Set {arg} has attribute of {class_name} or remove {arg} from extra_args
                        """
                    )

    @property
    def extra_args(self):
        """
        Returns:

        """
        return self.functions.extra_args

    @property
    def params(self):
        """
        Returns:
        """
        return self.functions.params

    @params.setter
    def params(self, values: FloatArray):
        """
        Args:
            values ():

        Returns:
        """
        self.functions.params = values

    def control_extra_args(self, **kwargs: Any) -> None:
        """

        Args:
            **kwargs ():

        Returns:

        """
        if self.extra_args:
            if set(kwargs.keys()) != set(self.extra_args):
                class_name = self.__class__.__name__
                raise ValueError(
                    f"kwargs must contain values for {self.extra_args} to work with {class_name}"
                )

    def set_extra_args(self, **kwargs: Any) -> None:
        """

        Args:
            **kwargs ():

        Returns:

        """
        self.control_extra_args(**kwargs)
        for name, value in kwargs.items():
            setattr(self.functions, name, value)

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
