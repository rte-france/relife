"""
This module defines classes that instanciate facade objects used to create statistical models

Copyright (c) 2022, RTE (https://www.rte-france.com)
See AUTHORS.txt
SPDX-License-Identifier: Apache-2.0 (see LICENSE.txt)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Union, Sequence

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import minimize

from relife2.data.factories import lifetime_factory_template
from relife2.functions import LikelihoodFromLifetimes
from relife2.functions.core import ParametricFunction, ParametricLifetimeFunction
from relife2.models.io import (
    are_all_args_given,
    array_factory,
    preprocess_lifetime_data,
)

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


class ParametricModel:
    """Façade like Model type"""

    def __init__(self, function: ParametricFunction):

        self.params_set = False
        if np.isnan(function.params).any() and not np.isnan(function.params).all():
            raise ValueError(
                "Can't instanciate partially initialized model. Set all params or instanciate empty model and fit it"
            )
        else:
            self.params_set = True

        self.function = function

    @property
    def params(self):
        """
        Returns:
        """
        return self.function.params

    @params.setter
    def params(self, values: np.ndarray):
        """
        Args:
            values ():

        Returns:
        """
        self.function.params = values

    def __getattr__(self, name: str):
        class_name = type(self).__name__
        if name in self.__dict__:
            value = self.__dict__[name]
        else:
            if not hasattr(self.function, name):
                raise AttributeError(f"{class_name} has no attribute named {name}")
            value = getattr(self.function, name)
        return value

    # def __setattr__(self, name: str, value: Any):
    #     if name in ("function", "params_set"):
    #         super().__setattr__(name, value)
    #     elif hasattr(self.function, name):
    #         setattr(self.function, name, value)
    #     else:
    #         super().__setattr__(name, value)

    # def __repr__(self):
    #     class_name = type(self).__name__
    #     return f"{class_name}(\n" f" params = {self.functions.all_params}\n"

    def __str__(self):
        class_name = type(self).__name__
        return f"{class_name}(\n" f" params = {self.function.all_params}\n"


class ParametricLifetimeModel(ParametricModel, ABC):
    """
    Façade class for lifetime model (where functions is a LifetimeFunctions)
    """

    function: ParametricLifetimeFunction

    def __getattribute__(self, item):
        if item in _LIFETIME_FUNCTIONS_NAMES:
            if not self.params_set:
                raise ValueError(
                    "Model parameters are empty, fit model first or instanciate new model with parameters"
                )
        return super().__getattribute__(item)

    def sf(self, time: ArrayLike, *args: ArrayLike) -> Union[float, np.ndarray]:
        """
        Args:
            time ():

        Returns:

        """
        time = array_factory(time)
        args = [array_factory(arg, nb_units=time.shape[0]) for arg in args]
        are_all_args_given(self.function, *args)
        self.function.args = args
        return np.squeeze(self.function.sf(time))

    def isf(self, probability: ArrayLike, *args: ArrayLike) -> Union[float, np.ndarray]:
        """

        Args:
            probability ():

        Returns:

        """
        probability = array_factory(probability)
        args = [array_factory(arg, nb_units=probability.shape[0]) for arg in args]
        are_all_args_given(self.function, *args)
        self.function.args = args
        return np.squeeze(self.function.isf(probability))

    def hf(self, time: ArrayLike, *args: ArrayLike) -> Union[float, np.ndarray]:
        """

        Args:
            time ():

        Returns:

        """
        time = array_factory(time)
        args = [array_factory(arg, nb_units=time.shape[0]) for arg in args]
        are_all_args_given(self.function, *args)
        self.function.args = args
        return np.squeeze(self.function.hf(time))

    def chf(self, time: ArrayLike, *args: ArrayLike) -> Union[float, np.ndarray]:
        """

        Args:
            time ():

        Returns:

        """
        time = array_factory(time)
        args = [array_factory(arg, nb_units=time.shape[0]) for arg in args]
        are_all_args_given(self.function, *args)
        self.function.args = args
        return np.squeeze(self.function.chf(time))

    def cdf(self, time: ArrayLike, *args: ArrayLike) -> Union[float, np.ndarray]:
        """

        Args:
            time ():

        Returns:

        """
        time = array_factory(time)
        args = [array_factory(arg, nb_units=time.shape[0]) for arg in args]
        are_all_args_given(self.function, *args)
        self.function.args = args
        return np.squeeze(self.function.cdf(time))

    def pdf(self, probability: ArrayLike, *args: ArrayLike) -> Union[float, np.ndarray]:
        """

        Args:
            probability ():

        Returns:

        """
        probability = array_factory(probability)
        args = [array_factory(arg, nb_units=probability.shape[0]) for arg in args]
        are_all_args_given(self.function, *args)
        self.function.args = args
        return np.squeeze(self.function.pdf(probability))

    def ppf(self, time: ArrayLike, *args: ArrayLike) -> Union[float, np.ndarray]:
        """

        Args:
            time ():

        Returns:

        """
        time = array_factory(time)
        args = [array_factory(arg, nb_units=time.shape[0]) for arg in args]
        are_all_args_given(self.function, *args)
        self.function.args = args
        return np.squeeze(self.function.ppf(time))

    def mrl(self, time: ArrayLike, *args: ArrayLike) -> Union[float, np.ndarray]:
        """

        Args:
            time ():


        Returns:

        """
        time = array_factory(time)
        args = [array_factory(arg, nb_units=time.shape[0]) for arg in args]
        are_all_args_given(self.function, *args)
        self.function.args = args
        return np.squeeze(self.function.mrl(time))

    def ichf(
        self, cumulative_hazard_rate: ArrayLike, *args: ArrayLike
    ) -> Union[float, np.ndarray]:
        """

        Args:
            cumulative_hazard_rate ():

        Returns:

        """
        cumulative_hazard_rate = array_factory(cumulative_hazard_rate)
        args = [
            array_factory(arg, nb_units=cumulative_hazard_rate.shape[0]) for arg in args
        ]
        are_all_args_given(self.function, *args)
        self.function.args = args
        return np.squeeze(self.function.ichf(cumulative_hazard_rate))

    def rvs(
        self,
        size: Optional[int] = 1,
        seed: Optional[int] = None,
        args: Optional[tuple[ArrayLike]] = (),
    ) -> Union[float, np.ndarray]:
        """

        Args:
            size ():
            seed ():
            args ():

        Returns:

        """
        args = [array_factory(arg) for arg in args]
        are_all_args_given(self.function, *args)
        self.function.args = args
        return np.squeeze(self.function.rvs(size=size, seed=seed))

    def mean(self, *args: ArrayLike) -> Union[float, np.ndarray]:
        """

        Returns:

        """
        args = [array_factory(arg) for arg in args]
        are_all_args_given(self.function, *args)
        self.function.args = args
        return np.squeeze(self.function.mean())

    def var(self, *args: ArrayLike) -> Union[float, np.ndarray]:
        """

        Returns:

        """
        args = [array_factory(arg) for arg in args]
        are_all_args_given(self.function, *args)
        self.function.args = args
        return np.squeeze(self.function.var())

    def median(self, *args: ArrayLike) -> Union[float, np.ndarray]:
        """

        Returns:

        """
        args = [array_factory(arg) for arg in args]
        are_all_args_given(self.function, *args)
        self.function.args = args
        return np.squeeze(self.function.median())

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

        optimized_function = self.function.copy()
        optimized_function.args = [
            np.empty_like(arg) for arg in args
        ]  # used for init_params if it depends on args
        optimized_function.init_params(observed_lifetimes.rlc)
        param0 = optimized_function.params

        likelihood = LikelihoodFromLifetimes(
            optimized_function,
            observed_lifetimes,
            truncations,
        )

        minimize_kwargs = {
            "method": kwargs.get("method", "L-BFGS-B"),
            "constraints": kwargs.get("constraints", ()),
            "tol": kwargs.get("tol", None),
            "callback": kwargs.get("callback", None),
            "options": kwargs.get("options", None),
            "bounds": kwargs.get("bounds", optimized_function.params_bounds),
            "x0": kwargs.get("x0", param0),
        }

        optimizer = minimize(
            likelihood.negative_log,
            minimize_kwargs.pop("x0"),
            jac=None if not likelihood.hasjac else likelihood.jac_negative_log,
            **minimize_kwargs,
        )

        if inplace:
            self.function = likelihood.function.copy()
            self.params_set = True

        return optimizer.x


@dataclass
class Estimates:
    """
    BLABLABLABLA
    """

    timeline: np.ndarray
    values: np.ndarray
    se: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.se is None:
            self.se = np.zeros_like(
                self.values
            )  # garder None/Nan efaire le changement de valeur au niveau du plot

        if self.timeline.shape != self.values.shape != self.se:
            raise ValueError("Incompatible timeline, values and se in Estimates")


class NonParametricLifetimeEstimators(ABC):
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
