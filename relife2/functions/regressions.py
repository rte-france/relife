"""
This module defines probability functions used in regression

Copyright (c) 2022, RTE (https://www.rte-france.com)
See AUTHORS.txt
SPDX-License-Identifier: Apache-2.0 (see LICENSE.txt)
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from scipy.optimize import Bounds

from relife2.functions.core import ParametricFunctions, ParametricLifetimeFunctions
from relife2.typing import FloatArray

from .distributions import DistributionFunctions


class CovarEffect(ParametricFunctions):
    """
    Object that computes covariates effect functions
    """

    @property
    def params_bounds(self) -> Bounds:
        """BLABLABLA"""
        return Bounds(
            np.full(self.nb_params, -np.inf),
            np.full(self.nb_params, np.inf),
        )

    def init_params(self, *args: Any) -> FloatArray:
        return np.zeros(self.nb_params)

    def g(self, covar: FloatArray) -> FloatArray:
        """
        BLABLABLABLA
        Args:
            covar (FloatArray): BLABLABLABLA

        Returns:
            FloatArray: BLABLABLABLA
        """
        return np.exp(np.sum(self.params * covar, axis=1, keepdims=True))

    def jac_g(self, covar: FloatArray) -> FloatArray:
        """
        Args:
            covar ():

        Returns:
        """
        return covar * self.g(covar)


class RegressionFunctions(ParametricLifetimeFunctions, ABC):
    """
    Object that computes every probability functions of a regression model
    """

    def __init__(
        self,
        covar_effect: CovarEffect,
        baseline: DistributionFunctions,
    ):
        super().__init__()
        self.add_functions("covar_effect", covar_effect)
        self.add_functions("baseline", baseline)
        self._covar = np.empty((0, 0), dtype=np.float64)

    @property
    def nb_covar(self) -> int:
        """BLABLA"""
        return self.covar_effect.nb_params

    @property
    def covar(self):
        """
        Returns:
            Covar extra arguments used in functions
        """
        return self._covar

    @covar.setter
    def covar(self, values: FloatArray) -> None:
        nb_covar = values.shape[-1]
        if values.shape[-1] != self.nb_covar:
            raise ValueError(
                f"Invalid number of covar : expected {self.nb_covar}, got {nb_covar}"
            )
        self._covar = values

    @property
    def support_lower_bound(self):
        """
        Returns:
            BLABLABLABLA
        """
        return 0.0

    @property
    def support_upper_bound(self):
        """
        Returns:
            BLABLABLABLA
        """
        return np.inf

    def init_params(self, *args: Any) -> FloatArray:
        return np.concatenate(
            (
                self.covar_effect.init_params(*args),
                self.baseline.init_params(*args),
            )
        )

    @property
    def params_bounds(self) -> Bounds:
        """BLABLABLA"""
        lb = np.concatenate(
            (
                self.covar_effect.params_bounds.lb,
                self.baseline.params_bounds.lb,
            )
        )
        ub = np.concatenate(
            (
                self.covar_effect.params_bounds.ub,
                self.baseline.params_bounds.ub,
            )
        )
        return Bounds(lb, ub)

    @abstractmethod
    def jac_hf(self, time: FloatArray) -> FloatArray:
        """
        BLABLABLABLA
        Args:
            time (FloatArray): BLABLABLABLA

        Returns:
            FloatArray: BLABLABLABLA
        """

    @abstractmethod
    def jac_chf(self, time: FloatArray) -> FloatArray:
        """
        BLABLABLABLA
        Args:
            time (FloatArray): BLABLABLABLA

        Returns:
            FloatArray: BLABLABLABLA
        """

    @abstractmethod
    def dhf(self, time: FloatArray) -> FloatArray:
        """
        BLABLABLABLA
        Args:
            time (FloatArray): BLABLABLABLA

        Returns:
            FloatArray: BLABLABLABLA
        """

    @abstractmethod
    def ichf(
        self,
        cumulative_hazard_rate: FloatArray,
    ) -> FloatArray:
        """
        BLABLABLABLA
        Args:
            cumulative_hazard_rate (Union[int, float, ArrayLike, FloatArray]): BLABLABLABLA
        Returns:
            FloatArray: BLABLABLABLA
        """

    def isf(self, probability: FloatArray) -> FloatArray:
        """
        BLABLABLABLA
        Args:
            probability (FloatArray): BLABLABLABLA
        Returns:
            FloatArray: BLABLABLABLA
        """
        cumulative_hazard_rate = -np.log(probability)
        return self.ichf(cumulative_hazard_rate)


class ProportionalHazardFunctions(RegressionFunctions):
    """
    BLABLABLABLA
    """

    def hf(
        self,
        time: FloatArray,
    ) -> FloatArray:
        return self.covar_effect.g(self.covar) * self.baseline.hf(time)

    def chf(self, time: FloatArray) -> FloatArray:
        return self.covar_effect.g(self.covar) * self.baseline.chf(time)

    def ichf(self, cumulative_hazard_rate: FloatArray) -> FloatArray:
        return self.baseline.ichf(
            cumulative_hazard_rate / self.covar_effect.g(self.covar)
        )

    def jac_hf(self, time: FloatArray) -> FloatArray:
        return np.column_stack(
            (
                self.covar_effect.jac_g(self.covar) * self.baseline.hf(time),
                self.covar_effect.g(self.covar) * self.baseline.jac_hf(time),
            )
        )

    def jac_chf(self, time: FloatArray) -> FloatArray:
        return np.column_stack(
            (
                self.covar_effect.jac_g(self.covar) * self.baseline.chf(time),
                self.covar_effect.g(self.covar) * self.baseline.jac_chf(time),
            )
        )

    def dhf(self, time: FloatArray) -> FloatArray:
        return self.covar_effect.g(self.covar) * self.baseline.dhf(time)


class AFTFunctions(RegressionFunctions):
    """
    BLABLABLABLA
    """

    def hf(self, time: FloatArray) -> FloatArray:
        t0 = time / self.covar_effect.g(self.covar)
        return self.baseline.hf(t0) / self.covar_effect.g(self.covar)

    def chf(self, time: FloatArray) -> FloatArray:
        t0 = time / self.covar_effect.g(self.covar)
        return self.baseline.chf(t0)

    def ichf(self, cumulative_hazard_rate: FloatArray) -> FloatArray:
        return self.covar_effect.g(self.covar) * self.baseline.ichf(
            cumulative_hazard_rate
        )

    def jac_hf(self, time: FloatArray) -> FloatArray:
        t0 = time / self.covar_effect.g(self.covar)
        return np.column_stack(
            (
                -self.covar_effect.jac_g(self.covar)
                / self.covar_effect.g(self.covar) ** 2
                * (self.baseline.hf(t0) + t0 * self.baseline.dhf(t0)),
                self.baseline.jac_hf(t0) / self.covar_effect.g(self.covar),
            )
        )

    def jac_chf(self, time: FloatArray) -> FloatArray:
        t0 = time / self.covar_effect.g(self.covar)
        return np.column_stack(
            (
                -self.covar_effect.jac_g(self.covar)
                / self.covar_effect.g(self.covar)
                * t0
                * self.baseline.hf(t0),
                self.baseline.jac_chf(t0),
            )
        )

    def dhf(self, time: FloatArray) -> FloatArray:
        t0 = time / self.covar_effect.g(self.covar)
        return self.baseline.dhf(t0) / self.covar_effect.g(self.covar) ** 2
