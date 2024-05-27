"""
This module defines likelihoods used in distributions

Copyright (c) 2022, RTE (https://www.rte-france.com)
See AUTHORS.txt
SPDX-License-Identifier: Apache-2.0 (see LICENSE.txt)
"""

from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import approx_fprime

from relife2.survival.distributions.types import DistributionLikelihood

IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]
FloatArray = NDArray[np.float64]


class GenericDistributionLikelihood(DistributionLikelihood):
    """
    BLABLABLABLA
    """

    def negative_log_likelihood(
        self,
    ) -> float:

        d_contrib = -np.sum(np.log(self.functions.hf(self.complete_lifetimes.values)))
        rc_contrib = np.sum(
            self.functions.chf(
                np.concatenate(
                    (
                        self.complete_lifetimes.values,
                        self.right_censorships.values,
                    )
                ),
            )
        )
        lc_contrib = -np.sum(
            np.log(
                -np.expm1(
                    -self.functions.chf(
                        self.left_censorships.values,
                    )
                )
            )
        )
        lt_contrib = -np.sum(self.functions.chf(self.left_truncations.values))
        return d_contrib + rc_contrib + lc_contrib + lt_contrib

    # relife/parametric.ParametricHazardFunction
    def jac_negative_log_likelihood(
        self,
    ) -> FloatArray:

        jac_d_contrib = -np.sum(
            self.functions.jac_hf(np.squeeze(self.complete_lifetimes.values))
            / self.functions.hf(self.complete_lifetimes.values),
            axis=0,
        )

        jac_rc_contrib = np.sum(
            self.functions.jac_chf(
                np.concatenate(
                    (
                        np.squeeze(self.complete_lifetimes.values),
                        np.squeeze(self.right_censorships.values),
                    )
                )
            ),
            axis=0,
        )

        jac_lc_contrib = -np.sum(
            self.functions.jac_chf(np.squeeze(self.left_censorships.values))
            / np.expm1(self.functions.chf(self.left_censorships.values)),
            axis=0,
        )

        jac_lt_contrib = -np.sum(
            self.functions.jac_chf(np.squeeze(self.left_truncations.values)),
            axis=0,
        )

        return jac_d_contrib + jac_rc_contrib + jac_lc_contrib + jac_lt_contrib

    def hess_negative_log_likelihood(
        self,
        eps: float = 1e-6,
        scheme: Optional[str] = None,
    ) -> FloatArray:

        size = self.functions.params.size
        # print(size)
        hess = np.empty((size, size))
        params_values = self.functions.params.values

        if scheme is None:
            scheme = self.default_hess_scheme

        if scheme == "cs":
            u = eps * 1j * np.eye(size)
            for i in range(size):
                for j in range(i, size):
                    # print(type(u[i]))
                    # print(u[i])
                    # print(self.functions.params.values)
                    # print(self.functions.params.values + u[i])
                    self.functions.params.values = self.functions.params.values + u[i]
                    # print(self.jac_negative_log_likelihood(self.functions))

                    hess[i, j] = np.imag(self.jac_negative_log_likelihood()[j]) / eps
                    self.functions.params.values = params_values
                    if i != j:
                        hess[j, i] = hess[i, j]

        elif scheme == "2-point":

            for i in range(size):
                hess[i] = approx_fprime(
                    self.functions.params.values,
                    self.jac_negative_log_likelihood()[i],
                    eps,
                )
        else:
            raise ValueError("scheme argument must be 'cs' or '2-point'")

        return hess
