"""
This module defines likelihoods used in regressions

Copyright (c) 2022, RTE (https://www.rte-france.com)
See AUTHORS.txt
SPDX-License-Identifier: Apache-2.0 (see LICENSE.txt)
"""

import numpy as np

from relife2.survival.data import ObservedLifetimes, Truncations
from relife2.survival.regressions.types import FloatArray, RegressionFunctions
from relife2.survival.types import JacLikelihood


class GenericRegressionLikelihood(JacLikelihood):
    """BLABLABLABLA"""

    def __init__(
        self,
        functions: RegressionFunctions,
        observed_lifetimes: ObservedLifetimes,
        truncations: Truncations,
        covar: FloatArray,
    ):
        super().__init__(functions)
        self.observed_lifetimes = observed_lifetimes
        self.truncations = truncations
        self.covar = covar

    def negative_log_likelihood(
        self,
    ) -> float:

        d_contrib = -np.sum(
            np.log(
                self.functions.hf(
                    self.observed_lifetimes.complete.values,
                    self.covar[self.observed_lifetimes.complete.index],
                )
            )
        )
        rc_contrib = np.sum(
            self.functions.chf(
                self.observed_lifetimes.rc.values,
                self.covar[self.observed_lifetimes.rc.index],
            )
        )
        lc_contrib = -np.sum(
            np.log(
                -np.expm1(
                    -self.functions.chf(
                        self.observed_lifetimes.left_censored.values,
                        self.covar[self.observed_lifetimes.left_censored.index],
                    )
                )
            )
        )
        lt_contrib = -np.sum(
            self.functions.chf(
                self.truncations.left.values, self.covar[self.truncations.left.index]
            )
        )
        return d_contrib + rc_contrib + lc_contrib + lt_contrib

    # relife/parametric.ParametricHazardFunction
    def jac_negative_log_likelihood(
        self,
    ) -> FloatArray:

        jac_d_contrib = -np.sum(
            self.functions.jac_hf(
                self.observed_lifetimes.complete.values,
                self.covar[self.observed_lifetimes.complete.index],
            )
            / self.functions.hf(
                self.observed_lifetimes.complete.values,
                self.covar[self.observed_lifetimes.complete.index],
            ),
            axis=0,
        )

        jac_rc_contrib = np.sum(
            self.functions.jac_chf(
                self.observed_lifetimes.rc.values,
                self.covar[self.observed_lifetimes.rc.index],
            ),
            axis=0,
        )

        jac_lc_contrib = -np.sum(
            self.functions.jac_chf(
                self.observed_lifetimes.left_censored.values,
                self.covar[self.observed_lifetimes.left_censored.index],
            )
            / np.expm1(
                self.functions.chf(
                    self.observed_lifetimes.left_censored.values,
                    self.covar[self.observed_lifetimes.left_censored.index],
                )
            ),
            axis=0,
        )

        jac_lt_contrib = -np.sum(
            self.functions.jac_chf(
                self.truncations.left.values, self.covar[self.truncations.left.index]
            ),
            axis=0,
        )

        return jac_d_contrib + jac_rc_contrib + jac_lc_contrib + jac_lt_contrib
