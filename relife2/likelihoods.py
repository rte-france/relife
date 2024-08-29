"""
This module defines probability functions used in regression

Copyright (c) 2022, RTE (https://www.rte-france.com)
See AUTHORS.txt
SPDX-License-Identifier: Apache-2.0 (see LICENSE.txt)
"""

import warnings
from typing import Any, Optional, Union

import numpy as np
from scipy.special import gamma as gamma_function
from scipy.stats import gamma

from relife2.core import LifetimeModel
from relife2.data.dataclass import Deteriorations, LifetimeSample, Truncations
from relife2.data.dataclass import Sample
from relife2.functions import ParametricFunctions, Likelihood


class LikelihoodFromLifetimes(Likelihood):
    """
    BLABLABLA
    """

    def __init__(
        self,
        function: LifetimeModel,
        observed_lifetimes: LifetimeSample,
        truncations: Truncations,
    ):
        super().__init__(function)
        self.observed_lifetimes = observed_lifetimes
        self.truncations = truncations

        if hasattr(self.function, "jac_hf") and hasattr(self.function, "jac_chf"):
            self.hasjac = True

    def _complete_contribs(self, lifetimes: Sample) -> float:
        return -np.sum(np.log(self.function.hf(lifetimes.values, *lifetimes.args)))

    def _right_censored_contribs(self, lifetimes: Sample) -> float:
        return np.sum(
            self.function.chf(lifetimes.values, *lifetimes.args), dtype=np.float64
        )

    def _left_censored_contribs(self, lifetimes: Sample) -> float:
        return -np.sum(
            np.log(-np.expm1(-self.function.chf(lifetimes.values, *lifetimes.args)))
        )

    def _left_truncations_contribs(self, lifetimes: Sample) -> float:
        return -np.sum(
            self.function.chf(lifetimes.values, *lifetimes.args), dtype=np.float64
        )

    def _jac_complete_contribs(self, lifetimes: Sample) -> np.ndarray:
        self.function.args = lifetimes.args
        return -np.sum(
            self.function.jac_hf(lifetimes.values, *lifetimes.args)
            / self.function.hf(lifetimes.values, *lifetimes.args),
            axis=0,
        )

    def _jac_right_censored_contribs(self, lifetimes: Sample) -> np.ndarray:
        return np.sum(
            self.function.jac_chf(lifetimes.values, *lifetimes.args),
            axis=0,
        )

    def _jac_left_censored_contribs(self, lifetimes: Sample) -> np.ndarray:
        return -np.sum(
            self.function.jac_chf(lifetimes.values, *lifetimes.args)
            / np.expm1(self.function.chf(lifetimes.values, *lifetimes.args)),
            axis=0,
        )

    def _jac_left_truncations_contribs(self, lifetimes: Sample) -> np.ndarray:
        return -np.sum(
            self.function.jac_chf(lifetimes.values, *lifetimes.args),
            axis=0,
        )

    def negative_log(
        self,
        params: np.ndarray,
    ) -> float:
        self.params = params
        return (
            self._complete_contribs(self.observed_lifetimes.complete)
            + self._right_censored_contribs(self.observed_lifetimes.rc)
            + self._left_censored_contribs(self.observed_lifetimes.left_censored)
            + self._left_truncations_contribs(self.truncations.left)
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
            self._jac_complete_contribs(self.observed_lifetimes.complete)
            + self._jac_right_censored_contribs(self.observed_lifetimes.rc)
            + self._jac_left_censored_contribs(self.observed_lifetimes.left_censored)
            + self._jac_left_truncations_contribs(self.truncations.left)
        )


class LikelihoodFromDeteriorations(Likelihood):
    """BLABLABLA"""

    def __init__(
        self,
        functions: ParametricFunctions,
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
