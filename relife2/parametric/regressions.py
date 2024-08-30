"""
This module defines probability functions used in regression

Copyright (c) 2022, RTE (https://www.rte-france.com)
See AUTHORS.txt
SPDX-License-Identifier: Apache-2.0 (see LICENSE.txt)
"""

from abc import ABC, abstractmethod
from typing import Any, Sequence, Optional

import numpy as np
from scipy.optimize import Bounds

from relife2.core import ParametricFunctions
from relife2.probabilities import default
from .base import ParametricLifetimeModel


class CovarEffect(ParametricFunctions):
    """
    Object that computes covariates effect functions
    """

    def __init__(self, coef: Sequence[float | None]):
        super().__init__()
        self.new_params(**{f"coef_{i}": v for i, v in enumerate(coef)})

    @property
    def params_bounds(self) -> Bounds:
        """BLABLABLA"""
        return Bounds(
            np.full(self.nb_params, -np.inf),
            np.full(self.nb_params, np.inf),
        )

    def init_params(self, *args):
        self.new_params(**{f"coef_{i}": 0.0 for i in range(self.covar.shape[-1])})

    def g(self, covar: np.ndarray) -> np.ndarray:
        """
        BLABLABLABLA
        Returns:
            np.ndarray: BLABLABLABLA
        """
        if covar.shape[-1] != self.nb_params:
            raise ValueError(
                f"Invalid number of covar : expected {self.nb_params}, got {covar.shape[-1]}"
            )
        return np.exp(np.sum(self.params * covar, axis=1, keepdims=True))

    def jac_g(self, covar: np.ndarray) -> np.ndarray:
        """
        Returns:
        """
        return covar * self.g(covar)


class Regression(ParametricLifetimeModel, ABC):
    """
    Object that computes every probability functions of a regression model
    """

    def __init__(
        self,
        baseline: ParametricLifetimeModel,
        coef: Sequence[float | None],
    ):
        super().__init__()
        self.add_functions(
            covar_effect=CovarEffect(coef),
            baseline=baseline,
        )

    @default
    def sf(self, time: np.ndarray, covar: np.ndarray, *args: np.ndarray) -> np.ndarray:
        """
        Args:
            time ():
            covar ():
            *args ():

        Returns:

        """

    def isf(
        self, probability: np.ndarray, covar: np.ndarray, *args: np.ndarray
    ) -> np.ndarray:
        """
        Args:
            probability ():
            covar ():
            *args ():
        Returns:
        """
        cumulative_hazard_rate = -np.log(probability)
        return self.ichf(cumulative_hazard_rate, covar, *args)

    @abstractmethod
    def hf(self, time: np.ndarray, covar: np.ndarray, *args: np.ndarray) -> np.ndarray:
        """
        Args:
            time ():
            covar ():
            *args ():

        Returns:

        """

    @abstractmethod
    def chf(self, time: np.ndarray, covar: np.ndarray, *args: np.ndarray) -> np.ndarray:
        """

        Args:
            time ():
            covar ():
            *args ():

        Returns:

        """

    @default
    def cdf(self, time: np.ndarray, covar: np.ndarray, *args: np.ndarray) -> np.ndarray:
        """

        Args:
            time ():
            covar ():
            *args ():

        Returns:

        """

    @default
    def pdf(self, time: np.ndarray, covar: np.ndarray, *args: np.ndarray) -> np.ndarray:
        """
        Args:
            time ():
            covar ():
            *args ():

        Returns:

        """

    @default
    def ppf(
        self, probability: np.ndarray, covar: np.ndarray, *args: np.ndarray
    ) -> np.ndarray:
        """
        Args:
            probability ():
            covar ():
            *args ():

        Returns:

        """

    @default
    def mrl(self, time: np.ndarray, covar: np.ndarray, *args: np.ndarray) -> np.ndarray:
        """
        Args:
            time ():
            covar ():
            *args ():

        Returns:
        """

    @default
    def rvs(
        self,
        covar: np.ndarray,
        *args: np.ndarray,
        size: Optional[int] = 1,
        seed: Optional[int] = None,
    ):
        """
        Args:
            covar ():
            *args ():
            size ():
            seed ():

        Returns:
        """

    @default
    def mean(self, covar: np.ndarray, *args: np.ndarray) -> np.ndarray:
        """
        Args:
            covar ():
            *args ():

        Returns:

        """

    @default
    def var(self, covar: np.ndarray, *args: np.ndarray) -> np.ndarray:
        """

        Args:
            covar ():
            *args ():

        Returns:

        """

    @default
    def median(self, covar: np.ndarray, *args: np.ndarray):
        """
        Args:
            covar ():
            *args ():

        Returns:

        """

    @default
    @abstractmethod
    def ichf(
        self, cumulative_hazard_rate: np.ndarray, covar: np.ndarray, *args: np.ndarray
    ) -> np.ndarray:
        """
        Args:
            cumulative_hazard_rate ():
            covar ():
            *args ():

        Returns:

        """

    @abstractmethod
    def jac_hf(
        self,
        time: np.ndarray,
        covar: np.ndarray,
        *args: np.ndarray,
    ) -> np.ndarray:
        """
        Args:
            time ():
            covar ():
            *args ():

        Returns:

        """

    @abstractmethod
    def jac_chf(
        self,
        time: np.ndarray,
        covar: np.ndarray,
        *args: np.ndarray,
    ) -> np.ndarray:
        """
        Args:
            time ():
            covar ():
            *args ():

        Returns:
        """

    @abstractmethod
    def dhf(
        self,
        time: np.ndarray,
        covar: np.ndarray,
        *args: np.ndarray,
    ) -> np.ndarray:
        """
        Args:
            time ():
            covar ():
            *args ():

        Returns:
        """

    def init_params(self, *args: Any):
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


class ProportionalHazard(Regression):
    """
    BLABLABLABLA
    """

    def hf(
        self,
        time: np.ndarray,
        covar: np.ndarray,
        *args: np.ndarray,
    ) -> np.ndarray:
        return self.covar_effect.g(covar) * self.baseline.hf(time, *args)

    def chf(self, time: np.ndarray, covar: np.ndarray, *args: np.ndarray) -> np.ndarray:
        return self.covar_effect.g(covar) * self.baseline.chf(time, *args)

    def ichf(
        self, cumulative_hazard_rate: np.ndarray, covar: np.ndarray, *args: np.ndarray
    ) -> np.ndarray:
        return self.baseline.ichf(
            cumulative_hazard_rate / self.covar_effect.g(covar), *args
        )

    def jac_hf(
        self, time: np.ndarray, covar: np.ndarray, *args: np.ndarray
    ) -> np.ndarray:
        return np.column_stack(
            (
                self.covar_effect.jac_g(covar) * self.baseline.hf(time, *args),
                self.covar_effect.g(covar) * self.baseline.jac_hf(time, *args),
            )
        )

    def jac_chf(
        self, time: np.ndarray, covar: np.ndarray, *args: np.ndarray
    ) -> np.ndarray:
        return np.column_stack(
            (
                self.covar_effect.jac_g(covar) * self.baseline.chf(time, *args),
                self.covar_effect.g(covar) * self.baseline.jac_chf(time, *args),
            )
        )

    def dhf(self, time: np.ndarray, covar: np.ndarray, *args: np.ndarray) -> np.ndarray:
        return self.covar_effect.g(covar) * self.baseline.dhf(time, *args)


class AFT(Regression):
    """
    BLABLABLABLA
    """

    def hf(self, time: np.ndarray, covar: np.ndarray, *args: np.ndarray) -> np.ndarray:
        t0 = time / self.covar_effect.g(covar)
        return self.baseline.hf(t0, *args) / self.covar_effect.g(covar)

    def chf(self, time: np.ndarray, covar: np.ndarray, *args: np.ndarray) -> np.ndarray:
        t0 = time / self.covar_effect.g(covar)
        return self.baseline.chf(t0, *args)

    def ichf(
        self, cumulative_hazard_rate: np.ndarray, covar: np.ndarray, *args: np.ndarray
    ) -> np.ndarray:
        return self.covar_effect.g(covar) * self.baseline.ichf(
            cumulative_hazard_rate, *args
        )

    def jac_hf(
        self, time: np.ndarray, covar: np.ndarray, *args: np.ndarray
    ) -> np.ndarray:
        t0 = time / self.covar_effect.g(covar)
        return np.column_stack(
            (
                -self.covar_effect.jac_g(covar)
                / self.covar_effect.g(covar) ** 2
                * (self.baseline.hf(t0, *args) + t0 * self.baseline.dhf(t0, *args)),
                self.baseline.jac_hf(t0, *args) / self.covar_effect.g(covar),
            )
        )

    def jac_chf(
        self, time: np.ndarray, covar: np.ndarray, *args: np.ndarray
    ) -> np.ndarray:
        t0 = time / self.covar_effect.g(covar)
        return np.column_stack(
            (
                -self.covar_effect.jac_g(covar)
                / self.covar_effect.g(covar)
                * t0
                * self.baseline.hf(t0, *args),
                self.baseline.jac_chf(t0, *args),
            )
        )

    def dhf(self, time: np.ndarray, covar: np.ndarray, *args: np.ndarray) -> np.ndarray:
        t0 = time / self.covar_effect.g(covar)
        return self.baseline.dhf(t0, *args) / self.covar_effect.g(covar) ** 2
