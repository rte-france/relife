"""
This module defines likelihoods used in distributions

Copyright (c) 2022, RTE (https://www.rte-france.com)
See AUTHORS.txt
SPDX-License-Identifier: Apache-2.0 (see LICENSE.txt)
"""

from typing import Optional, Union

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import approx_fprime

from relife2.survival.distributions.types import (
    DistributionFunctions,
    DistributionLikelihood,
)

IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]
FloatArray = NDArray[np.float64]


class GenericDistributionLikelihood(DistributionLikelihood):
    """
    BLABLABLABLA
    """

    def __init__(
        self,
        functions: DistributionFunctions,
        time: FloatArray,
        entry: Optional[FloatArray] = None,
        departure: Optional[FloatArray] = None,
        **indicators: Union[IntArray, BoolArray],
    ):
        super().__init__(functions, time, entry, departure, **indicators)

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
            self.functions.jac_hf(
                np.squeeze(self.complete_lifetimes.values), self.functions
            )
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
                ),
                self.functions,
            ),
            axis=0,
        )

        jac_lc_contrib = -np.sum(
            self.functions.jac_chf(
                np.squeeze(self.left_censorships.values), self.functions
            )
            / np.expm1(self.functions.chf(self.left_censorships.values)),
            axis=0,
        )

        jac_lt_contrib = -np.sum(
            self.functions.jac_chf(
                np.squeeze(self.left_truncations.values), self.functions
            ),
            axis=0,
        )

        return jac_d_contrib + jac_rc_contrib + jac_lc_contrib + jac_lt_contrib

    def hess_negative_log_likelihood(
        self,
        eps: float = 1e-6,
        scheme: str = None,
    ) -> FloatArray:

        size = np.size(self.functions.params.values)
        # print(size)
        hess = np.empty((size, size))
        params_values = self.functions.params.values

        if scheme is None:
            scheme = self._default_hess_scheme

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

            def f(xk):
                self.functions.params.values = xk
                return self.jac_negative_log_likelihood()

            xk = self.functions.params.values

            for i in range(size):
                hess[i] = approx_fprime(
                    xk,
                    lambda x: f(x)[i],
                    eps,
                )
        else:
            raise ValueError("scheme argument must be 'cs' or '2-point'")

        return hess
