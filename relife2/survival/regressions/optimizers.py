"""
This module defines optimizers of likelihoods used in regressions

Copyright (c) 2022, RTE (https://www.rte-france.com)
See AUTHORS.txt
SPDX-License-Identifier: Apache-2.0 (see LICENSE.txt)
"""

from relife2.survival.optimizers import LikelihoodOptimizer
from relife2.survival.regressions.types import FloatArray


class RegressionLikelihoodOptimizer(LikelihoodOptimizer):
    """BLABLABLA"""

    def update_params(self, new_values: FloatArray) -> None:
        self.likelihood.functions.params.values = new_values
        self.likelihood.functions.baseline.params.values = new_values[
            : self.likelihood.functions.baseline.params.size
        ]
        self.likelihood.functions.covar_effect.params.values = new_values[
            self.likelihood.functions.covar_effect.params.size - 1 :
        ]
