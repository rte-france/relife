"""
This module defines optimizers of likelihoods used in distributions

Copyright (c) 2022, RTE (https://www.rte-france.com)
See AUTHORS.txt
SPDX-License-Identifier: Apache-2.0 (see LICENSE.txt)
"""

import numpy as np
from numpy.typing import NDArray

from relife2.survival.optimizers import LikelihoodOptimizer

FloatArray = NDArray[np.float64]


class DistributionLikelihoodOptimizer(LikelihoodOptimizer):
    """
    BLABLABLABLA
    """

    def update_params(self, new_values: FloatArray) -> None:
        self.likelihood.functions.params.values = new_values


class GompertzLikelihoodOptimizer(LikelihoodOptimizer):
    """
    BLABLABLABLA
    """

    def update_params(self, new_values: FloatArray) -> None:
        self.likelihood.functions.params.values = new_values

    def init_params(self) -> FloatArray:
        nb_params = self.likelihood.functions.params.size
        param0 = np.empty(nb_params, dtype=np.float64)
        rate = np.pi / (
            np.sqrt(6)
            * np.std(
                np.concatenate(
                    (
                        self.likelihood.observed_lifetimes.complete.values,
                        self.likelihood.observed_lifetimes.left_censored.values,
                        self.likelihood.observed_lifetimes.right_censored.values,
                    ),
                    axis=0,
                )
            )
        )

        shape = np.exp(
            -rate
            * np.mean(
                np.concatenate(
                    (
                        self.likelihood.observed_lifetimes.complete.values,
                        self.likelihood.observed_lifetimes.left_censored.values,
                        self.likelihood.observed_lifetimes.right_censored.values,
                    ),
                    axis=0,
                )
            )
        )

        param0[0] = shape
        param0[1] = rate

        return param0
