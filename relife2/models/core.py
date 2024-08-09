"""
This module defines classes that instanciate facade objects used to create statistical models

Copyright (c) 2022, RTE (https://www.rte-france.com)
See AUTHORS.txt
SPDX-License-Identifier: Apache-2.0 (see LICENSE.txt)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import wraps
from typing import Any, Optional, Union

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import minimize

from relife2.data import (
    array_factory,
    lifetime_factory_template,
)
from relife2.functions import LikelihoodFromLifetimes
from relife2.functions.core import ParametricFunctions, ParametricLifetimeFunctions
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

    functions: ParametricLifetimeFunctions

    def __getattribute__(self, item):
        if item in _LIFETIME_FUNCTIONS_NAMES:
            are_params_set(self.functions)
        return super().__getattribute__(item)

    def get_likelihood(
        self,
        time: ArrayLike,
        event: Optional[ArrayLike] = None,
        entry: Optional[ArrayLike] = None,
        departure: Optional[ArrayLike] = None,
        **extravars,
    ) -> LikelihoodFromLifetimes:
        observed_lifetimes, truncations = lifetime_factory_template(
            time,
            event,
            entry,
            departure,
            **extravars,
        )
        return LikelihoodFromLifetimes(
            self.functions.copy(),
            observed_lifetimes,
            truncations,
        )

    def _check_extravars(self, **extravars: FloatArray):
        given_extravars = set(extravars.keys())
        expected_extravars = set(self.extravars_names)
        common_extravars = given_extravars & expected_extravars
        if common_extravars != set(expected_extravars):
            raise ValueError(
                f"Method expects {expected_extravars} but got only {common_extravars}"
            )

    def _set_extravars(self, **extravars: FloatArray):
        try:
            self._check_extravars(**extravars)
        except ValueError as err:
            raise ValueError("Missing variables") from err
        for name in self.extravars_names:
            setattr(self.functions, name, extravars[name])

    @squeeze
    def sf(self, time: ArrayLike, **extravars: FloatArray) -> Union[float, FloatArray]:
        """
        Args:
            time ():

        Returns:

        """
        time = array_factory(time)
        self._set_extravars(**extravars)
        return self.functions.sf(time)

    @squeeze
    def isf(self, probability: ArrayLike, **extravars) -> Union[float, FloatArray]:
        """

        Args:
            probability ():

        Returns:

        """
        probability = array_factory(probability)
        self._set_extravars(**extravars)
        return self.functions.isf(array_factory(probability))

    @squeeze
    def hf(self, time: ArrayLike, **extravars) -> Union[float, FloatArray]:
        """

        Args:
            time ():

        Returns:

        """
        time = array_factory(time)
        self._set_extravars(**extravars)
        return self.functions.hf(time)

    @squeeze
    def chf(self, time: ArrayLike, **extravars) -> Union[float, FloatArray]:
        """

        Args:
            time ():

        Returns:

        """
        time = array_factory(time)
        self._set_extravars(**extravars)
        return self.functions.chf(time)

    @squeeze
    def cdf(self, time: ArrayLike, **extravars) -> Union[float, FloatArray]:
        """

        Args:
            time ():

        Returns:

        """
        time = array_factory(time)
        self._set_extravars(**extravars)
        return self.functions.cdf(time)

    @squeeze
    def pdf(self, probability: ArrayLike, **extravars) -> Union[float, FloatArray]:
        """

        Args:
            probability ():

        Returns:

        """
        probability = array_factory(probability)
        self._set_extravars(**extravars)
        return self.functions.pdf(probability)

    @squeeze
    def ppf(self, time: ArrayLike, **extravars) -> Union[float, FloatArray]:
        """

        Args:
            time ():

        Returns:

        """
        time = array_factory(time)
        self._set_extravars(**extravars)
        return self.functions.ppf(time)

    @squeeze
    def mrl(self, time: ArrayLike, **extravars) -> Union[float, FloatArray]:
        """

        Args:
            time ():


        Returns:

        """
        time = array_factory(time)
        self._set_extravars(**extravars)
        return self.functions.mrl(time)

    @squeeze
    def ichf(
        self, cumulative_hazard_rate: ArrayLike, **extravars
    ) -> Union[float, FloatArray]:
        """

        Args:
            cumulative_hazard_rate ():

        Returns:

        """
        cumulative_hazard_rate = array_factory(cumulative_hazard_rate)
        self._set_extravars(**extravars)
        return self.functions.ichf(cumulative_hazard_rate)

    @squeeze
    def rvs(
        self, size: Optional[int] = 1, seed: Optional[int] = None, **extravars
    ) -> Union[float, FloatArray]:
        """

        Args:
            size ():
            seed ():

        Returns:

        """
        self._set_extravars(**extravars)
        return self.functions.rvs(size=size, seed=seed)

    @squeeze
    def mean(self, **extravars) -> Union[float, FloatArray]:
        """

        Returns:

        """
        self._set_extravars(**extravars)
        return self.functions.mean()

    @squeeze
    def var(self, **extravars) -> Union[float, FloatArray]:
        """

        Returns:

        """
        self._set_extravars(**extravars)
        return self.functions.var()

    @squeeze
    def median(self, **extravars) -> Union[float, FloatArray]:
        """

        Returns:

        """
        self._set_extravars(**extravars)
        return self.functions.median()

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
        self._check_extravars(**kwargs)
        extravars = {k: kwargs[k] for k in self.extravars_names}

        likelihood = self.get_likelihood(
            time,
            event,
            entry,
            departure,
            **extravars,
        )
        minimize_kwargs = {
            "method": kwargs.get("method", "L-BFGS-B"),
            "constraints": kwargs.get("constraints", ()),
            "tol": kwargs.get("tol", None),
            "callback": kwargs.get("callback", None),
            "options": kwargs.get("options", None),
        }
        param0 = kwargs.get(
            "x0", likelihood.functions.init_params(likelihood.observed_lifetimes.rlc)
        )
        minimize_kwargs.update(
            {"bounds": kwargs.get("bounds", likelihood.functions.params_bounds)}
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
