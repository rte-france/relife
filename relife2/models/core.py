"""
This module defines classes that instanciate facade objects used to create statistical models

Copyright (c) 2022, RTE (https://www.rte-france.com)
See AUTHORS.txt
SPDX-License-Identifier: Apache-2.0 (see LICENSE.txt)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import wraps
from typing import Any, Optional

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import minimize

from relife2.data import ObservedLifetimes, Truncations, lifetime_factory_template
from relife2.functions.core import ParametricFunctions
from relife2.functions.likelihoods import LikelihoodFromLifetimes
from relife2.typing import FloatArray

_LIFETIME_FUNCTIONS_NAMES = [
    "sf",
    "isf",
    "hf",
    "chf",
    "cdf",
    "cdf",
    "pdf",
    "ppf",
    "mrl",
    "ichf",
    "rvs",
    "mean",
    "var",
    "median",
]


def are_params_set(functions: ParametricFunctions):
    """
    Args:
        functions ():

    Returns:
    """
    if None in functions.all_params.values():
        params_to_set = " ".join(
            [name for name, value in functions.all_params.items() if value is None]
        )
        raise ValueError(
            f"Params {params_to_set} unset. Please set them first or fit the model."
        )


def squeeze(method):
    """
    Args:
        method ():

    Returns:
    """

    @wraps(method)
    def _impl(self, *method_args, **method_kwargs):
        method_output = method(self, *method_args, **method_kwargs)
        return np.squeeze(method_output)[()]

    return _impl


class ParametricModel:
    """Façade like Model type"""

    def __init__(self, functions: ParametricFunctions):
        self.functions = functions

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

    # def __repr__(self):
    #     class_name = type(self).__name__
    #     return f"{class_name}(\n" f" params = {self.functions.all_params}\n"

    def __str__(self):
        class_name = type(self).__name__
        return f"{class_name}(\n" f" params = {self.functions.all_params}\n"


class ParametricLifetimeModel(ParametricModel, ABC):
    """
    Façade class for lifetime model (where functions is a LifetimeFunctions)
    """

    @abstractmethod
    def _init_likelihood(
        self,
        observed_lifetimes: ObservedLifetimes,
        truncations: Truncations,
        **kwargs: Any,
    ) -> LikelihoodFromLifetimes:
        """"""

    def __getattribute__(self, item):
        if item in _LIFETIME_FUNCTIONS_NAMES:
            are_params_set(self.functions)
        return super().__getattribute__(item)

    def fit(
        self,
        time: ArrayLike,
        event: Optional[ArrayLike] = None,
        entry: Optional[ArrayLike] = None,
        departure: Optional[ArrayLike] = None,
        inplace: bool = True,
        **kwargs: Any,
    ) -> FloatArray:
        """
        BLABLABLABLA
        Args:
            time (ArrayLike):
            event (Optional[ArrayLike]):
            entry (Optional[ArrayLike]):
            departure (Optional[ArrayLike]):
            inplace (bool): (default is True)

        Returns:
            Parameters: optimum parameters found
        """
        observed_lifetimes, truncations = lifetime_factory_template(
            time,
            event,
            entry,
            departure,
        )

        minimize_kwargs = {
            "method": kwargs.pop("method", "L-BFGS-B"),
            "constraints": kwargs.pop("constraints", ()),
            "tol": kwargs.pop("tol", None),
            "callback": kwargs.pop("callback", None),
            "options": kwargs.pop("options", None),
        }

        likelihood = self._init_likelihood(observed_lifetimes, truncations, **kwargs)
        param0 = kwargs.pop(
            "x0", likelihood.functions.init_params(observed_lifetimes.rlc)
        )
        minimize_kwargs.update(
            {"bounds": kwargs.pop("bounds", likelihood.functions.params_bounds)}
        )

        optimizer = minimize(
            likelihood.negative_log,
            param0,
            jac=None if not likelihood.hasjac else likelihood.jac_negative_log,
            **minimize_kwargs,
        )

        if inplace:
            self.functions = likelihood.functions.copy()
        return optimizer.x


@dataclass
class Estimates:
    """
    BLABLABLABLA
    """

    timeline: FloatArray
    values: FloatArray
    se: Optional[FloatArray] = None

    def __post_init__(self):
        if self.se is None:
            self.se = np.zeros_like(
                self.values
            )  # garder None/Nan efaire le changement de valeur au niveau du plot

        if self.timeline.shape != self.values.shape != self.se:
            raise ValueError("Incompatible timeline, values and se in Estimates")


class NonParametricEstimators(ABC):
    """_summary_"""

    def __init__(
        self,
    ):
        self.estimations = {}

    @abstractmethod
    def estimate(
        self,
        time: ArrayLike,
        event: Optional[ArrayLike] = None,
        entry: Optional[ArrayLike] = None,
        departure: Optional[ArrayLike] = None,
    ) -> None:
        """_summary_

        Returns:
            Tuple[Estimates]: description
        """
