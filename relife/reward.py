"""Rewards for renewal reward processes."""

# Copyright (c) 2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# SPDX-License-Identifier: Apache-2.0 (see LICENSE.txt)
# This file is part of ReLife, an open source Python library for asset
# management based on reliability theory and lifetime data analysis.

from abc import ABC, abstractmethod
import numpy as np


class Reward(ABC):
    """Generic reward class."""

    @abstractmethod
    def conditional_expectation(
        self, x: np.ndarray, *reward_args: np.ndarray
    ) -> np.ndarray:
        """Conditional expected reward.

        Parameters
        ----------
        x : ndarray
            Duration.
        *reward_args : ndarray
            Extra arguments of the reward random variable.

        Returns
        -------
        ndarray
            The conditional expected reward with respect to the duration.
        """
        pass

    def sample(self, x: np.ndarray, *reward_args: np.ndarray) -> np.ndarray:
        """Reward conditional sampling.

        Parameters
        ----------
        x : ndarray
            Duration.
        *reward_args : ndarray
            Extra arguments of the reward random variable.

        Returns
        -------
        ndarray
            Random drawing of a reward with respect to the duration.

        """
        return self.conditional_expectation(x, *reward_args)


class FailureCost(Reward):
    """Run-to-failure costs.

    The replacements occur upon failures with costs `cf`.
    """

    def conditional_expectation(self, x: np.ndarray, cf: np.ndarray) -> np.ndarray:
        return cf


class AgeReplacementCost(Reward):
    """Age replacement costs.

    The replacements occur at a fixed age `ar` with preventive costs `cp` or
    upon failure with failure costs `cf` if earlier.
    """

    def conditional_expectation(
        self, x: np.ndarray, ar: np.ndarray, cf: np.ndarray, cp: np.ndarray
    ) -> np.ndarray:
        return np.where(x < ar, cf, cp)
