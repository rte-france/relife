"""
This module defines probability functions used in regression

Copyright (c) 2022, RTE (https://www.rte-france.com)
See AUTHORS.txt
SPDX-License-Identifier: Apache-2.0 (see LICENSE.txt)
"""

import warnings
from abc import abstractmethod
from typing import Any, Optional, Union

import numpy as np
from scipy.optimize import Bounds
from scipy.special import gamma as gamma_function
from scipy.stats import gamma

from relife2.data.dataclass import Deteriorations, LifetimeSample, Truncations

from ..data.dataclass import Sample
from .core import ParametricFunction, ParametricLifetimeFunction


class Likelihood(ParametricFunction):
    """
    Class that instanciates likelihood base having finite number of parameters related to
    one parametric functions
    """

    hasjac: bool = False

    def __init__(self, function: ParametricFunction):
        super().__init__()
        self.add_functions(function=function)

    def init_params(self, *args: Any) -> np.ndarray:
        return self.function.init_params()

    @property
    def params_bounds(self) -> Bounds:
        return self.function.params_bounds

    @abstractmethod
    def negative_log(self, params: np.ndarray) -> float:
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
        function: ParametricLifetimeFunction,
        observed_lifetimes: LifetimeSample,
        truncations: Truncations,
    ):
        super().__init__(function)
        self.observed_lifetimes = observed_lifetimes
        self.truncations = truncations

        if hasattr(self.function, "jac_hf") and hasattr(self.function, "jac_chf"):
            self.hasjac = True

    def complete_contribs(self, lifetimes: Sample) -> float:
        """

        Args:
            lifetimes ():

        Returns:

        """
        self.function.args = lifetimes.args
        return -np.sum(np.log(self.function.hf(lifetimes.values)))

    def right_censored_contribs(self, lifetimes: Sample) -> float:
        """

        Args:
            lifetimes ():

        Returns:

        """
        self.function.args = lifetimes.args
        return np.sum(self.function.chf(lifetimes.values), dtype=np.float64)

    def left_censored_contribs(self, lifetimes: Sample) -> float:
        """

        Args:
            lifetimes ():

        Returns:

        """
        self.function.args = lifetimes.args
        return -np.sum(
            np.log(
                -np.expm1(
                    -self.function.chf(
                        lifetimes.values,
                    )
                )
            )
        )

    def left_truncations_contribs(self, lifetimes: Sample) -> float:
        """

        Args:
            lifetimes ():

        Returns:

        """
        self.function.args = lifetimes.args
        return -np.sum(self.function.chf(lifetimes.values), dtype=np.float64)

    def jac_complete_contribs(self, lifetimes: Sample) -> np.ndarray:
        """

        Args:
            lifetimes ():

        Returns:

        """
        self.function.args = lifetimes.args
        return -np.sum(
            self.function.jac_hf(lifetimes.values) / self.function.hf(lifetimes.values),
            axis=0,
        )

    def jac_right_censored_contribs(self, lifetimes: Sample) -> np.ndarray:
        """

        Args:
            lifetimes ():

        Returns:

        """
        self.function.args = lifetimes.args
        return np.sum(
            self.function.jac_chf(lifetimes.values),
            axis=0,
        )

    def jac_left_censored_contribs(self, lifetimes: Sample) -> np.ndarray:
        """

        Args:
            lifetimes ():

        Returns:

        """
        self.function.args = lifetimes.args
        return -np.sum(
            self.function.jac_chf(lifetimes.values)
            / np.expm1(self.function.chf(lifetimes.values)),
            axis=0,
        )

    def jac_left_truncations_contribs(self, lifetimes: Sample) -> np.ndarray:
        """

        Args:
            lifetimes ():

        Returns:

        """
        self.function.args = lifetimes.args
        return -np.sum(
            self.function.jac_chf(lifetimes.values),
            axis=0,
        )

    def negative_log(
        self,
        params: np.ndarray,
    ) -> float:
        self.params = params
        print(self.complete_contribs(self.observed_lifetimes.complete))
        return (
            self.complete_contribs(self.observed_lifetimes.complete)
            + self.right_censored_contribs(self.observed_lifetimes.rc)
            + self.left_censored_contribs(self.observed_lifetimes.left_censored)
            + self.left_truncations_contribs(self.truncations.left)
        )

    def jac_negative_log(
        self,
        params: np.ndarray,
    ) -> Union[None, np.ndarray]:
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
            self.jac_complete_contribs(self.observed_lifetimes.complete)
            + self.jac_right_censored_contribs(self.observed_lifetimes.rc)
            + self.jac_left_censored_contribs(self.observed_lifetimes.left_censored)
            + self.jac_left_truncations_contribs(self.truncations.left)
        )


class LikelihoodFromDeteriorations(Likelihood):
    """BLABLABLA"""

    def __init__(
        self,
        functions: ParametricFunction,
        deterioration_data: Deteriorations,
        first_increment_uncertainty: Optional[tuple] = None,
        measurement_tol: np.floating[Any] = np.finfo(float).resolution,
    ):
        super().__init__(functions)
        self.deterioration_data = deterioration_data
        self.first_increment_uncertainty = first_increment_uncertainty
        self.measurement_tol = measurement_tol

    def negative_log(self, params: np.ndarray) -> float:
        """
        All deteriorations have R0 in first column
        All times have 0 in first column
        """
        self.params = params

        delta_shape = np.diff(
            self.function.shape_function.nu(self.deterioration_data.times),
            axis=1,
        )

        contributions = -(
            delta_shape * np.log(self.rate)
            + (delta_shape - 1)
            * np.log(
                self.deterioration_data.increments,
                where=~self.deterioration_data.event,
                out=np.zeros_like(delta_shape),
            )
            - self.rate * self.deterioration_data.increments
            - np.log(
                gamma_function(delta_shape),
                where=~self.deterioration_data.event,
                out=np.zeros_like(delta_shape),
            )
        )

        censored_contributions = -np.log(
            gamma.cdf(
                self.deterioration_data.increments + self.measurement_tol,
                a=np.diff(
                    self.function.shape_function.nu(self.deterioration_data.times)
                ),
                scale=1 / self.rate,
            )
            - gamma.cdf(
                self.deterioration_data.increments - self.measurement_tol,
                a=np.diff(
                    self.function.shape_function.nu(self.deterioration_data.times)
                ),
                scale=1 / self.rate,
            ),
            where=self.deterioration_data.event,
            out=np.zeros_like(delta_shape),
        )

        contributions = np.where(
            self.deterioration_data.event, censored_contributions, contributions
        )

        if self.first_increment_uncertainty is not None:

            first_inspections = self.deterioration_data.times[:, 1]
            a = self.function.shape_function.nu(first_inspections)
            first_increment_contribution = -np.log(
                gamma.cdf(
                    self.first_increment_uncertainty[1]
                    - self.deterioration_data.values[:, 1],
                    a=a,
                    scale=1 / self.rate,
                )
                - gamma.cdf(
                    self.first_increment_uncertainty[0]
                    - self.deterioration_data.values[:, 1],
                    a=a,
                    scale=1 / self.rate,
                )
            )
            contributions[:, 0] = first_increment_contribution[:, None]

        # print(
        #     "neg_log",
        #     np.sum(
        #         contributions[~np.isnan(contributions)],
        #     ),
        #     "\n ===================================",
        # )
        # print(
        #     "sum contributions exactes :",
        #     np.sum(
        #         contributions[~np.isnan(contributions)][
        #             ~self.deterioration_data.event[~np.isnan(contributions)]
        #         ]
        #     ),
        # )
        # print(
        #     "sum contributions censures :",
        #     np.sum(
        #         contributions[~np.isnan(contributions)][
        #             self.deterioration_data.event[~np.isnan(contributions)]
        #         ]
        #     ),
        # )

        return np.sum(contributions[~np.isnan(contributions)])
