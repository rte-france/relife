"""
This module defines classes that instanciate facade objects used to create statistical models

Copyright (c) 2022, RTE (https://www.rte-france.com)
See AUTHORS.txt
SPDX-License-Identifier: Apache-2.0 (see LICENSE.txt)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import minimize

from relife2.data import lifetime_factory_template
from relife2.functions import LikelihoodFromLifetimes
from relife2.functions.core import ParametricFunctions, ParametricLifetimeFunctions
from relife2.models.io import preprocess_vars

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
    def params(self, values: np.ndarray):
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

    def _set_extravars(self, **extravars: np.ndarray):
        for name, extravar in extravars.items():
            setattr(self.functions, name, extravar)

    def get_likelihood(
        self,
        time: ArrayLike,
        event: Optional[ArrayLike] = None,
        entry: Optional[ArrayLike] = None,
        departure: Optional[ArrayLike] = None,
        **extravars,
    ) -> LikelihoodFromLifetimes:

        time, event, entry, departure, extravars = preprocess_vars(
            self.functions, time, event, entry, departure, **extravars
        )
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

    def sf(self, time: ArrayLike, **extravars: ArrayLike) -> Union[float, np.ndarray]:
        """
        Args:
            time ():

        Returns:

        """
        time, extravars = preprocess_vars(self.functions, time, **extravars)
        self._set_extravars(**extravars)
        return np.squeeze(self.functions.sf(time))

    def isf(
        self, probability: ArrayLike, **extravars: ArrayLike
    ) -> Union[float, np.ndarray]:
        """

        Args:
            probability ():

        Returns:

        """
        probability, extravars = preprocess_vars(
            self.functions, probability, **extravars
        )
        self._set_extravars(**extravars)
        return np.squeeze(self.functions.isf(probability))

    def hf(self, time: ArrayLike, **extravars: ArrayLike) -> Union[float, np.ndarray]:
        """

        Args:
            time ():

        Returns:

        """
        time, extravars = preprocess_vars(self.functions, time, **extravars)
        self._set_extravars(**extravars)
        return np.squeeze(self.functions.hf(time))

    def chf(self, time: ArrayLike, **extravars: ArrayLike) -> Union[float, np.ndarray]:
        """

        Args:
            time ():

        Returns:

        """
        time, extravars = preprocess_vars(self.functions, time, **extravars)
        self._set_extravars(**extravars)
        return np.squeeze(self.functions.chf(time))

    def cdf(self, time: ArrayLike, **extravars: ArrayLike) -> Union[float, np.ndarray]:
        """

        Args:
            time ():

        Returns:

        """
        time, extravars = preprocess_vars(self.functions, time, **extravars)
        self._set_extravars(**extravars)
        return np.squeeze(self.functions.cdf(time))

    def pdf(
        self, probability: ArrayLike, **extravars: ArrayLike
    ) -> Union[float, np.ndarray]:
        """

        Args:
            probability ():

        Returns:

        """
        probability, extravars = preprocess_vars(
            self.functions, probability, **extravars
        )
        self._set_extravars(**extravars)
        return np.squeeze(self.functions.pdf(probability))

    def ppf(self, time: ArrayLike, **extravars: ArrayLike) -> Union[float, np.ndarray]:
        """

        Args:
            time ():

        Returns:

        """
        time, extravars = preprocess_vars(self.functions, time, **extravars)
        self._set_extravars(**extravars)
        return np.squeeze(self.functions.ppf(time))

    def mrl(self, time: ArrayLike, **extravars: ArrayLike) -> Union[float, np.ndarray]:
        """

        Args:
            time ():


        Returns:

        """
        time, extravars = preprocess_vars(self.functions, time, **extravars)
        self._set_extravars(**extravars)
        return np.squeeze(self.functions.mrl(time))

    def ichf(
        self, cumulative_hazard_rate: ArrayLike, **extravars: ArrayLike
    ) -> Union[float, np.ndarray]:
        """

        Args:
            cumulative_hazard_rate ():

        Returns:

        """
        cumulative_hazard_rate, extravars = preprocess_vars(
            self.functions, cumulative_hazard_rate, **extravars
        )
        self._set_extravars(**extravars)
        return np.squeeze(self.functions.ichf(cumulative_hazard_rate))

    def rvs(
        self,
        size: Optional[int] = 1,
        seed: Optional[int] = None,
        **extravars: ArrayLike,
    ) -> Union[float, np.ndarray]:
        """

        Args:
            size ():
            seed ():

        Returns:

        """
        extravars = preprocess_vars(self.functions, **extravars)
        self._set_extravars(**extravars)
        return np.squeeze(self.functions.rvs(size=size, seed=seed))

    def mean(self, **extravars: ArrayLike) -> Union[float, np.ndarray]:
        """

        Returns:

        """
        extravars = preprocess_vars(self.functions, **extravars)
        self._set_extravars(**extravars)
        print(np.squeeze(self.functions.mean()))
        return np.squeeze(self.functions.mean())

    def var(self, **extravars: ArrayLike) -> Union[float, np.ndarray]:
        """

        Returns:

        """
        extravars = preprocess_vars(self.functions, **extravars)
        self._set_extravars(**extravars)
        return np.squeeze(self.functions.var())

    def median(self, **extravars: ArrayLike) -> Union[float, np.ndarray]:
        """

        Returns:

        """
        extravars = preprocess_vars(self.functions, **extravars)
        self._set_extravars(**extravars)
        return np.squeeze(self.functions.median())

    def fit(
        self,
        time: ArrayLike,
        event: Optional[ArrayLike] = None,
        entry: Optional[ArrayLike] = None,
        departure: Optional[ArrayLike] = None,
        inplace: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
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
            "bounds": kwargs.get("bounds", self.functions.params_bounds),
        }
        param0 = kwargs.get(
            "x0", likelihood.functions.init_params(likelihood.observed_lifetimes.rlc)
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
