"""
This module defines probability functions used in regression

Copyright (c) 2022, RTE (https://www.rte-france.com)
See AUTHORS.txt
SPDX-License-Identifier: Apache-2.0 (see LICENSE.txt)
"""

import warnings
from abc import abstractmethod
from typing import Union, Any, Optional

import numpy as np
from scipy.optimize import Bounds
from scipy.stats import gamma

from relife2 import parametric
from relife2.data import Lifetimes, ObservedLifetimes, Truncations
from relife2.data.dataclass import Deteriorations
from relife2.gammaprocess import GPFunctions
from relife2.types import FloatArray


# Likelihood(FunctionsBridge)
class Likelihood(parametric.Functions):
    """
    Class that instanciates likelihood base having finite number of parameters related to
    one parametric functions
    """

    hasjac: bool = False

    def __init__(self, functions: parametric.Functions):
        super().__init__()
        self.add_functions("functions", functions)

    def init_params(self, *args: Any) -> FloatArray:
        return self.functions.init_params()

    @property
    def params_bounds(self) -> Bounds:
        return self.functions.params_bounds

    @abstractmethod
    def negative_log(self, params: FloatArray) -> float:
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
        functions: parametric.LifetimeFunctions,
        observed_lifetimes: ObservedLifetimes,
        truncations: Truncations,
        **kwdata: FloatArray,
    ):
        super().__init__(functions)
        self.observed_lifetimes = observed_lifetimes
        self.truncations = truncations
        self.kwdata = kwdata

        if hasattr(self.functions, "jac_hf") and hasattr(self.functions, "jac_chf"):
            self.hasjac = True

    def d_contrib(self, lifetimes: Lifetimes) -> float:
        """

        Args:
            lifetimes ():

        Returns:

        """
        for name, data in self.kwdata.items():
            setattr(self.functions, name, data[lifetimes.index])
        return -np.sum(np.log(self.functions.hf(lifetimes.values)))

    def rc_contrib(self, lifetimes: Lifetimes) -> float:
        """

        Args:
            lifetimes ():

        Returns:

        """
        for name, data in self.kwdata.items():
            setattr(self.functions, name, data[lifetimes.index])
        return np.sum(self.functions.chf(lifetimes.values), dtype=np.float64)

    def lc_contrib(self, lifetimes: Lifetimes) -> float:
        """

        Args:
            lifetimes ():

        Returns:

        """
        for name, data in self.kwdata.items():
            setattr(self.functions, name, data[lifetimes.index])
        return -np.sum(
            np.log(
                -np.expm1(
                    -self.functions.chf(
                        lifetimes.values,
                    )
                )
            )
        )

    def lt_contrib(self, lifetimes: Lifetimes) -> float:
        """

        Args:
            lifetimes ():

        Returns:

        """
        for name, data in self.kwdata.items():
            setattr(self.functions, name, data[lifetimes.index])
        return -np.sum(self.functions.chf(lifetimes.values), dtype=np.float64)

    def jac_d_contrib(self, lifetimes: Lifetimes) -> FloatArray:
        """

        Args:
            lifetimes ():

        Returns:

        """
        for name, data in self.kwdata.items():
            setattr(self.functions, name, data[lifetimes.index])
        return -np.sum(
            self.functions.jac_hf(lifetimes.values)
            / self.functions.hf(lifetimes.values),
            axis=0,
        )

    def jac_rc_contrib(self, lifetimes: Lifetimes) -> FloatArray:
        """

        Args:
            lifetimes ():

        Returns:

        """
        for name, data in self.kwdata.items():
            setattr(self.functions, name, data[lifetimes.index])
        return np.sum(
            self.functions.jac_chf(lifetimes.values),
            axis=0,
        )

    def jac_lc_contrib(self, lifetimes: Lifetimes) -> FloatArray:
        """

        Args:
            lifetimes ():

        Returns:

        """
        for name, data in self.kwdata.items():
            setattr(self.functions, name, data[lifetimes.index])
        return -np.sum(
            self.functions.jac_chf(lifetimes.values)
            / np.expm1(self.functions.chf(lifetimes.values)),
            axis=0,
        )

    def jac_lt_contrib(self, lifetimes: Lifetimes) -> FloatArray:
        """

        Args:
            lifetimes ():

        Returns:

        """
        for name, data in self.kwdata.items():
            setattr(self.functions, name, data[lifetimes.index])
        return -np.sum(
            self.functions.jac_chf(lifetimes.values),
            axis=0,
        )

    def negative_log(
        self,
        params: FloatArray,
    ) -> float:
        self.params = params
        return (
            self.d_contrib(self.observed_lifetimes.complete)
            + self.rc_contrib(self.observed_lifetimes.rc)
            + self.lc_contrib(self.observed_lifetimes.left_censored)
            + self.lt_contrib(self.truncations.left)
        )

    def jac_negative_log(
        self,
        params: FloatArray,
    ) -> Union[None, FloatArray]:
        """

        Args:
            params ():

        Returns:

        """
        if not self.hasjac:
            warnings.warn("Functions does not support jac negative likelihood natively")
            return None
        self.params = params
        return (
            self.jac_d_contrib(self.observed_lifetimes.complete)
            + self.jac_rc_contrib(self.observed_lifetimes.rc)
            + self.jac_lc_contrib(self.observed_lifetimes.left_censored)
            + self.jac_lt_contrib(self.truncations.left)
        )


class LikelihoodFromDeteriorations(Likelihood):
    """BLABLABLA"""

    def __init__(
        self,
        functions: GPFunctions,
        deterioration_data: Deteriorations,
        first_increment_uncertainty: Optional[tuple] = None,
        measurement_tol: float = np.finfo(float).resolution,
    ):
        super().__init__(functions)
        self.deterioration_data = deterioration_data
        self.first_increment_uncertainty = first_increment_uncertainty
        self.measurement_tol = measurement_tol

    def negative_log(self, params: FloatArray) -> float:
        self.params = params

        delta_shape = np.diff(
            self.functions.shape_function.nu(self.deterioration_data.times), axis=1
        )

        print(self.rate)
        contributions = (
            delta_shape * np.log(self.rate)
            + (self.deterioration_data.increments - 1)
            * np.log(
                self.deterioration_data.increments, where=~self.deterioration_data.event
            )
            - self.rate * self.deterioration_data.increments
            - np.log(
                gamma(self.deterioration_data.increments),
                where=~self.deterioration_data.event,
            )
        )
        censored_contributions = -np.sum(
            np.log(
                gamma.cdf(
                    self.deterioration_data.increments + self.measurement_tol,
                    a=np.diff(
                        self.functions.shape_function.nu(self.deterioration_data.times)
                    ),
                    scale=1 / self.rate,
                )
                - gamma.cdf(
                    self.deterioration_data.increments - self.measurement_tol,
                    a=np.diff(
                        self.functions.shape_function.nu(self.deterioration_data.times)
                    ),
                    scale=1 / self.rate,
                )
            )
        )

        contributions = np.where(self.event, censored_contributions, contributions)

        if self.first_increment_uncertainty is not None:

            first_inspections = self.deterioration_data.times[:, 0]
            self.functions.shape_function.nu(first_inspections)
            first_increment_contribution = np.log(
                gamma.cdf(
                    self.first_increment_uncertainty[1] - contributions[:, 0],
                    a=self.shape_function.nu(first_inspections),
                    scale=1 / self.rate,
                )
                - gamma.cdf(
                    self.first_increment_uncertainty[0] - contributions[:, 0],
                    a=self.functions.shape_function.nu(first_inspections),
                    scale=1 / self.rate,
                )
            )
            contributions[:, 0] = first_increment_contribution[:, None]

        return np.sum(
            contributions,
            axis=None,
        )
