"""
This module defines probability functions used in regression

Copyright (c) 2022, RTE (https://www.rte-france.com)
See AUTHORS.txt
SPDX-License-Identifier: Apache-2.0 (see LICENSE.txt)
"""

from abc import ABC, abstractmethod
from typing import Any, Sequence, Union

import numpy as np
from scipy.optimize import Bounds

from relife2.functions.core import ParametricFunction, ParametricLifetimeFunction


class CovarEffect(ParametricFunction):
    """
    Object that computes covariates effect functions
    """

    def __init__(self, coef: Sequence[float | None]):
        super().__init__()
        self.new_params(**{f"coef_{i}": v for i, v in enumerate(coef)})
        self.new_args(covar=np.empty((0, len(coef)), dtype=np.float64))

    @property
    def params_bounds(self) -> Bounds:
        """BLABLABLA"""
        return Bounds(
            np.full(self.nb_params, -np.inf),
            np.full(self.nb_params, np.inf),
        )

    def init_params(self, *args):
        self.new_params(**{f"coef_{i}": 0.0 for i in range(self.covar.shape[-1])})

    def g(self) -> np.ndarray:
        """
        BLABLABLABLA
        Returns:
            np.ndarray: BLABLABLABLA
        """
        if self.covar.shape[-1] != self.nb_params:
            raise ValueError(
                f"Invalid number of covar : expected {self.nb_params}, got {self.covar.shape[-1]}"
            )
        return np.exp(np.sum(self.params * self.covar, axis=1, keepdims=True))

    def jac_g(self) -> np.ndarray:
        """
        Returns:
        """
        return self.covar * self.g()


class RegressionFunction(ParametricLifetimeFunction, ABC):
    """
    Object that computes every probability functions of a regression model
    """

    def __init__(
        self,
        baseline: ParametricLifetimeFunction,
        coef: Sequence[float | None],
    ):
        super().__init__()
        self.add_functions(
            covar_effect=CovarEffect(coef),
            baseline=baseline,
        )

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

    def init_params(self, *args: Any) -> np.ndarray:
        self.baseline.init_params(*args)
        self.covar_effect.init_params(*args)

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
    def jac_hf(self, time: np.ndarray) -> np.ndarray:
        """
        BLABLABLABLA
        Args:
            time (np.ndarray): BLABLABLABLA

        Returns:
            np.ndarray: BLABLABLABLA
        """

    @abstractmethod
    def jac_chf(self, time: np.ndarray) -> np.ndarray:
        """
        BLABLABLABLA
        Args:
            time (np.ndarray): BLABLABLABLA

        Returns:
            np.ndarray: BLABLABLABLA
        """

    @abstractmethod
    def dhf(self, time: np.ndarray) -> np.ndarray:
        """
        BLABLABLABLA
        Args:
            time (np.ndarray): BLABLABLABLA

        Returns:
            np.ndarray: BLABLABLABLA
        """

    @abstractmethod
    def ichf(
        self,
        cumulative_hazard_rate: np.ndarray,
    ) -> np.ndarray:
        """
        BLABLABLABLA
        Args:
            cumulative_hazard_rate (Union[int, float, ArrayLike, np.ndarray]): BLABLABLABLA
        Returns:
            np.ndarray: BLABLABLABLA
        """

    def isf(self, probability: np.ndarray) -> np.ndarray:
        """
        BLABLABLABLA
        Args:
            probability (np.ndarray): BLABLABLABLA
        Returns:
            np.ndarray: BLABLABLABLA
        """
        cumulative_hazard_rate = -np.log(probability)
        return self.ichf(cumulative_hazard_rate)


class ProportionalHazardFunction(RegressionFunction):
    """
    BLABLABLABLA
    """

    def hf(
        self,
        time: np.ndarray,
    ) -> np.ndarray:
        return self.covar_effect.g() * self.baseline.hf(time)

    def chf(self, time: np.ndarray) -> np.ndarray:
        return self.covar_effect.g() * self.baseline.chf(time)

    def ichf(self, cumulative_hazard_rate: np.ndarray) -> np.ndarray:
        return self.baseline.ichf(cumulative_hazard_rate / self.covar_effect.g())

    def jac_hf(self, time: np.ndarray) -> np.ndarray:
        return np.column_stack(
            (
                self.covar_effect.jac_g() * self.baseline.hf(time),
                self.covar_effect.g() * self.baseline.jac_hf(time),
            )
        )

    def jac_chf(self, time: np.ndarray) -> np.ndarray:
        return np.column_stack(
            (
                self.covar_effect.jac_g() * self.baseline.chf(time),
                self.covar_effect.g() * self.baseline.jac_chf(time),
            )
        )

    def dhf(self, time: np.ndarray) -> np.ndarray:
        return self.covar_effect.g() * self.baseline.dhf(time)


class AFTFunction(RegressionFunction):
    """
    BLABLABLABLA
    """

    def hf(self, time: np.ndarray) -> np.ndarray:
        t0 = time / self.covar_effect.g()
        return self.baseline.hf(t0) / self.covar_effect.g()

    def chf(self, time: np.ndarray) -> np.ndarray:
        t0 = time / self.covar_effect.g()
        return self.baseline.chf(t0)

    def ichf(self, cumulative_hazard_rate: np.ndarray) -> np.ndarray:
        return self.covar_effect.g() * self.baseline.ichf(cumulative_hazard_rate)

    def jac_hf(self, time: np.ndarray) -> np.ndarray:
        t0 = time / self.covar_effect.g()
        return np.column_stack(
            (
                -self.covar_effect.jac_g()
                / self.covar_effect.g() ** 2
                * (self.baseline.hf(t0) + t0 * self.baseline.dhf(t0)),
                self.baseline.jac_hf(t0) / self.covar_effect.g(),
            )
        )

    def jac_chf(self, time: np.ndarray) -> np.ndarray:
        t0 = time / self.covar_effect.g()
        return np.column_stack(
            (
                -self.covar_effect.jac_g()
                / self.covar_effect.g()
                * t0
                * self.baseline.hf(t0),
                self.baseline.jac_chf(t0),
            )
        )

    def dhf(self, time: np.ndarray) -> np.ndarray:
        t0 = time / self.covar_effect.g()
        return self.baseline.dhf(t0) / self.covar_effect.g() ** 2
