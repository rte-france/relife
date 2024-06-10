"""
This module defines probability functions used in regression

Copyright (c) 2022, RTE (https://www.rte-france.com)
See AUTHORS.txt
SPDX-License-Identifier: Apache-2.0 (see LICENSE.txt)
"""

from typing import Union

import numpy as np

from relife2.survival.regressions.types import (
    CovarEffect,
    FloatArray,
    RegressionFunctions,
)


class ProportionalHazardEffect(CovarEffect):
    """
    BLABLABLABLA
    """

    def g(self, covar: FloatArray) -> Union[float, FloatArray]:
        return np.exp(np.sum(self.params.values * covar, axis=1, keepdims=True))

    def jac_g(self, covar: FloatArray) -> Union[float, FloatArray]:
        return covar * self.g(covar)


class ProportionalHazardFunctions(RegressionFunctions):
    """
    BLABLABLABLA
    """

    def hf(self, time: FloatArray, covar: FloatArray) -> Union[float, FloatArray]:
        return self.covar_effect.g(covar) * self.baseline.hf(time)

    def chf(self, time: FloatArray, covar: FloatArray) -> Union[float, FloatArray]:
        return self.covar_effect.g(covar) * self.baseline.chf(time)

    def ichf(
        self, cumulative_hazard_rate: FloatArray, covar: FloatArray
    ) -> Union[float, FloatArray]:
        return self.baseline.ichf(cumulative_hazard_rate / self.covar_effect.g(covar))

    def jac_hf(self, time: FloatArray, covar: FloatArray) -> Union[float, FloatArray]:
        return np.column_stack(
            (
                self.covar_effect.jac_g(covar) * self.baseline.hf(time),
                self.covar_effect.g(covar) * self.baseline.jac_hf(time),
            )
        )

    def jac_chf(self, time: FloatArray, covar: FloatArray) -> Union[float, FloatArray]:
        return np.column_stack(
            (
                self.covar_effect.jac_g(covar) * self.baseline.chf(time),
                self.covar_effect.g(covar) * self.baseline.jac_chf(time),
            )
        )

    def dhf(self, time: FloatArray, covar: FloatArray) -> Union[float, FloatArray]:
        return self.covar_effect.g(covar) * self.baseline.dhf(time)


class AFTEffect(CovarEffect):
    """
    BLABLABLABLA
    """

    def g(self, covar: FloatArray) -> Union[float, FloatArray]:
        return np.exp(np.sum(self.params.values * covar, axis=1, keepdims=True))

    def jac_g(self, covar: FloatArray) -> Union[float, FloatArray]:
        return covar * self.g(covar)


class AFTFunctions(RegressionFunctions):
    """
    BLABLABLABLA
    """

    def hf(self, time: FloatArray, covar: FloatArray) -> Union[float, FloatArray]:
        t0 = time / self.covar_effect.g(covar)
        return self.baseline.hf(t0) / self.covar_effect.g(covar)

    def chf(self, time: FloatArray, covar: FloatArray) -> Union[float, FloatArray]:
        t0 = time / self.covar_effect.g(covar)
        return self.baseline.chf(t0)

    def ichf(
        self, cumulative_hazard_rate: FloatArray, covar: FloatArray
    ) -> Union[float, FloatArray]:
        return self.covar_effect.g(covar) * self.baseline.ichf(cumulative_hazard_rate)

    def jac_hf(self, time: FloatArray, covar: FloatArray) -> Union[float, FloatArray]:
        t0 = time / self.covar_effect.g(covar)
        return np.column_stack(
            (
                -self.covar_effect.jac_g(covar)
                / self.covar_effect.g(covar) ** 2
                * (self.baseline.hf(t0) + t0 * self.baseline.dhf(t0)),
                self.baseline.jac_hf(t0) / self.covar_effect.g(covar),
            )
        )

    def jac_chf(self, time: FloatArray, covar: FloatArray) -> Union[float, FloatArray]:
        t0 = time / self.covar_effect.g(covar)
        return np.column_stack(
            (
                -self.covar_effect.jac_g(covar)
                / self.covar_effect.g(covar)
                * t0
                * self.baseline.hf(t0),
                self.baseline.jac_chf(t0),
            )
        )

    def dhf(self, time: FloatArray, covar: FloatArray) -> Union[float, FloatArray]:
        t0 = time / self.covar_effect.g(covar)
        return self.baseline.dhf(t0) / self.covar_effect.g(covar) ** 2
