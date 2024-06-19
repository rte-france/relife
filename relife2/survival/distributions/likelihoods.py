"""
This module defines likelihoods used in distributions

Copyright (c) 2022, RTE (https://www.rte-france.com)
See AUTHORS.txt
SPDX-License-Identifier: Apache-2.0 (see LICENSE.txt)
"""

import numpy as np
from numpy.typing import NDArray

from relife2.survival.data import ObservedLifetimes, Truncations
from relife2.survival.distributions.types import DistributionFunctions
from relife2.survival.types import JacLikelihood

IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]
FloatArray = NDArray[np.float64]


class GenericDistributionLikelihood(JacLikelihood):
    """
    BLABLABLABLA
    """

    def __init__(
        self,
        functions: DistributionFunctions,
        observed_lifetimes: ObservedLifetimes,
        truncations: Truncations,
    ):
        super().__init__(functions)
        self.observed_lifetimes = observed_lifetimes
        self.truncations = truncations

    def negative_log_likelihood(
        self,
    ) -> float:

        d_contrib = -np.sum(
            np.log(self.functions.hf(self.observed_lifetimes.complete.values))
        )
        rc_contrib = np.sum(self.functions.chf(self.observed_lifetimes.rc.values))
        lc_contrib = -np.sum(
            np.log(
                -np.expm1(
                    -self.functions.chf(
                        self.observed_lifetimes.left_censored.values,
                    )
                )
            )
        )
        lt_contrib = -np.sum(self.functions.chf(self.truncations.left.values))
        return d_contrib + rc_contrib + lc_contrib + lt_contrib

    # relife/parametric.ParametricHazardFunction
    def jac_negative_log_likelihood(
        self,
    ) -> FloatArray:

        jac_d_contrib = -np.sum(
            self.functions.jac_hf(self.observed_lifetimes.complete.values)
            / self.functions.hf(self.observed_lifetimes.complete.values),
            axis=0,
        )

        jac_rc_contrib = np.sum(
            self.functions.jac_chf(self.observed_lifetimes.rc.values),
            axis=0,
        )

        jac_lc_contrib = -np.sum(
            self.functions.jac_chf(self.observed_lifetimes.left_censored.values)
            / np.expm1(
                self.functions.chf(self.observed_lifetimes.left_censored.values)
            ),
            axis=0,
        )

        jac_lt_contrib = -np.sum(
            self.functions.jac_chf(self.truncations.left.values),
            axis=0,
        )

        return jac_d_contrib + jac_rc_contrib + jac_lc_contrib + jac_lt_contrib
