"""
This module defines likelihoods used in regressions

Copyright (c) 2022, RTE (https://www.rte-france.com)
See AUTHORS.txt
SPDX-License-Identifier: Apache-2.0 (see LICENSE.txt)
"""

import numpy as np

from relife2.survival.regressions.types import FloatArray, RegressionLikelihood


class GenericRegressionLikelihood(RegressionLikelihood):
    """BLABLABLABLA"""

    def negative_log_likelihood(
        self,
    ) -> float:

        d_contrib = -np.sum(
            np.log(
                self.functions.hf(self.observed_lifetimes.complete.values, self.covar)
            )
        )
        rc_contrib = np.sum(
            self.functions.chf(self.observed_lifetimes.rc.values, self.covar)
        )
        lc_contrib = -np.sum(
            np.log(
                -np.expm1(
                    -self.functions.chf(
                        self.observed_lifetimes.left_censored.values, self.covar
                    )
                )
            )
        )
        lt_contrib = -np.sum(
            self.functions.chf(self.truncations.left.values, self.covar)
        )
        return d_contrib + rc_contrib + lc_contrib + lt_contrib

    # relife/parametric.ParametricHazardFunction
    def jac_negative_log_likelihood(
        self,
    ) -> FloatArray:

        jac_d_contrib = -np.sum(
            self.functions.jac_hf(self.observed_lifetimes.complete.values, self.covar)
            / self.functions.hf(self.observed_lifetimes.complete.values, self.covar),
            axis=0,
        )

        jac_rc_contrib = np.sum(
            self.functions.jac_chf(self.observed_lifetimes.rc.values, self.covar),
            axis=0,
        )

        jac_lc_contrib = -np.sum(
            self.functions.jac_chf(
                self.observed_lifetimes.left_censored.values, self.covar
            )
            / np.expm1(
                self.functions.chf(
                    self.observed_lifetimes.left_censored.values, self.covar
                )
            ),
            axis=0,
        )

        jac_lt_contrib = -np.sum(
            self.functions.jac_chf(self.truncations.left.values, self.covar),
            axis=0,
        )

        return jac_d_contrib + jac_rc_contrib + jac_lc_contrib + jac_lt_contrib
