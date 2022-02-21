"""Discount functions."""

# Copyright (c) 2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# SPDX-License-Identifier: Apache-2.0 (see LICENSE.txt)
# This file is part of ReLife, an open source Python library for asset
# management based on reliability theory and lifetime data analysis.

from abc import ABC, abstractmethod
import numpy as np


class Discount(ABC):
    """Generic discount function.

    References
    ----------
    .. [1] Van der Weide, J. A. M., & Van Noortwijk, J. M. (2008). Renewal
        theory with exponential and hyperbolic discounting. Probability in the
        Engineering and Informational Sciences, 22(1), 53-74.
    """

    @classmethod
    @abstractmethod
    def factor(cls, t: np.ndarray, *discount_args: np.ndarray) -> np.ndarray:
        r"""The discount factor.

        Parameters
        ----------
        t : ndarray
            Time.
        *discount_args : ndarray
            Extra arguments required by the discount model.

        Returns
        -------
        ndarray
            The discount factor evaluated at `t`.

        Notes
        -----
        The discount factor evaluated at :math:`t` that multiplies the reward to get the
        discounted reward :math:`D(t) \cdot y`.
        """
        pass

    @classmethod
    @abstractmethod
    def rate(cls, t: np.ndarray, *discount_args: np.ndarray) -> np.ndarray:
        r"""The discount rate.

        Parameters
        ----------
        t : ndarray
            Time.
        *discount_args : ndarray
            Extra arguments required by the discount model.

        Returns
        -------
        ndarray
            The discount rate evaluated at `t`.

        Notes
        -----
        The discount rate evaluated at :math:`t` is defined by:

        .. math::

            r(t) = -\dfrac{D'(t)}{D(t)}

        where :math:`D` is the discount factor.
        """
        pass

    @classmethod
    @abstractmethod
    def annuity_factor(cls, t: np.ndarray, *discount_args: np.ndarray) -> np.ndarray:
        r"""The annuity factor.

        Parameters
        ----------
        t : ndarray
            Time.
        *discount_args : ndarray
            Extra arguments required by the discount model.

        Returns
        -------
        ndarray
            The annuity factor evaluated at `t`.

        Notes
        -----

        The annuity factor at time :math:`t` is defined by:

        .. math::

            AF(t) = \int_0^t D(x) \mathrm{d}x

        where :math:`D` is the discount factor.

        It is used to compute the equivalent annual cost of continuous
        discounted cash flows over the period :math:`[0, t]`.
        """
        pass


class ExponentialDiscounting(Discount):
    r"""Exponential discounting model.

    The exponetial discount factor is:

    .. math::

        D(x) = e^{-\delta x}

    where :math:`\delta` is the `rate`.
    """

    @classmethod
    def factor(cls, t: np.ndarray, rate: np.ndarray = 0) -> np.ndarray:
        return np.exp(-rate * t)

    @classmethod
    def rate(cls, t: np.ndarray, rate: np.ndarray = 0) -> np.ndarray:
        return rate * np.ones_like(t)

    @classmethod
    def annuity_factor(cls, t: np.ndarray, rate: np.ndarray = 0) -> np.ndarray:
        mask = rate == 0
        rate = np.ma.MaskedArray(rate, mask)
        return np.where(mask, t, (1 - np.exp(-rate * t)) / rate)


class HyperbolicDiscounting(Discount):
    r"""Hyperbolic discounting model.

    The hyperbolic discount factor is:

    .. math::

        D(x) = \dfrac{1}{1 + \beta x}

    where :math:`\beta>0`.
    """

    @classmethod
    def factor(cls, t: np.ndarray, beta: np.ndarray = 0) -> np.ndarray:
        return 1 / (1 + beta * t)

    @classmethod
    def rate(cls, t: np.ndarray, beta: np.ndarray = 0) -> np.ndarray:
        return beta / (1 + beta * t)

    @classmethod
    def annuity_factor(cls, t: np.ndarray, beta: np.ndarray = 0) -> np.ndarray:
        mask = beta == 0
        beta = np.ma.MaskedArray(beta, mask)
        return np.where(mask, t, np.log1p(beta * t) / beta)


class GeneralizedHyperbolicDiscounting(Discount):
    r"""Generalized hyperbolic discounting model.

    The generalized hyperbolic discount factor is:

    .. math::

        D(x) = \dfrac{1}{(1 + \beta x)^\eta}

    where :math:`\beta,\eta >0`.
    """

    @classmethod
    def factor(
        cls, t: np.ndarray, beta: np.ndarray = 0, eta: np.ndarray = 1
    ) -> np.ndarray:
        return 1 / (1 + beta * t) ** eta

    @classmethod
    def rate(
        cls, t: np.ndarray, beta: np.ndarray = 0, eta: np.ndarray = 1
    ) -> np.ndarray:
        return beta * eta / (1 + beta * t)

    @classmethod
    def annuity_factor(
        cls, t: np.ndarray, beta: np.ndarray = 0, eta: np.ndarray = 1
    ) -> np.ndarray:
        mask_beta = beta == 0
        mask_eta = eta == 1
        beta = np.ma.MaskedArray(beta, mask_beta)
        eta = np.ma.MaskedArray(eta, mask_eta)
        return np.where(
            mask_eta,
            HyperbolicDiscounting.annuity_factor(t, beta),
            np.where(
                mask_beta, t, ((1 + beta * t) ** (1 - eta) - 1) / (beta * (1 - eta))
            ),
        )
