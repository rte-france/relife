from __future__ import annotations

from typing import Optional, Self

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import newton
from typing_extensions import override

from relife import freeze
from relife.economic import AgeReplacementReward, ExponentialDiscounting
from relife.lifetime_model import (
    AgeReplacementModel,
    FrozenAgeReplacementModel,
    FrozenLeftTruncatedModel,
    FrozenLifetimeRegression,
    LifetimeDistribution,
)
from relife.quadrature import legendre_quadrature

from ._base import AgeRenewalPolicy, OneCycleAgeRenewalPolicy


class OneCycleAgeReplacementPolicy(OneCycleAgeRenewalPolicy):
    r"""One-cyle age replacement policy.

    The asset is disposed at a fixed age :math:`a_r` with costs :math:`c_p` or upon failure
    with costs :math:`c_f` if earlier.

    References
    ----------
    .. [1] Coolen-Schrijner, P., & Coolen, F. P. A. (2006). On optimality
        criteria for age replacement. Proceedings of the Institution of
        Mechanical Engineers, Part O: Journal of Risk and Reliability,
        220(1), 21-29
    """

    lifetime_model: FrozenAgeReplacementModel
    reward: AgeReplacementReward

    def __init__(
        self,
        lifetime_model: LifetimeDistribution | FrozenLifetimeRegression | FrozenLeftTruncatedModel,
        cf: float | NDArray[np.float64],
        cp: float | NDArray[np.float64],
        discounting_rate: float = 0.0,
        period_before_discounting: float = 1.0,
        ar: float | NDArray[np.float64] = np.nan,
    ) -> None:
        lifetime_model: FrozenAgeReplacementModel = freeze(AgeReplacementModel(lifetime_model), ar)
        reward = AgeReplacementReward(cf, cp, ar)
        super().__init__(lifetime_model, reward, discounting_rate, period_before_discounting)

    @property
    def cf(self) -> NDArray[np.float64]:
        return self.reward.cf

    @property
    def cp(self) -> NDArray[np.float64]:
        return self.reward.cp

    @property
    def ar(self) -> float | NDArray[np.float64]:
        return self.lifetime_model.ar

    @ar.setter
    def ar(self, value: float | NDArray[np.float64]) -> None:
        self.lifetime_model.ar = value
        self.reward.ar = value

    @override
    def expected_total_cost(
        self,
        tf: float,
        nb_steps: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        if np.isnan(self.ar):
            raise ValueError
        return super().expected_total_cost(tf, nb_steps)

    @override
    def asymptotic_expected_total_cost(self) -> NDArray[np.float64]:
        if np.isnan(self.ar):
            raise ValueError
        return super().asymptotic_expected_total_cost()

    @override
    def expected_equivalent_annual_cost(
        self, tf: float, nb_steps: int
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        if np.isnan(self.ar):
            raise ValueError
        return super().expected_equivalent_annual_cost(tf, nb_steps)

    @override
    def asymptotic_expected_equivalent_annual_cost(self) -> NDArray[np.float64]:
        if np.isnan(self.ar):
            raise ValueError
        return super().asymptotic_expected_equivalent_annual_cost()

    def optimize(
        self,
    ) -> OneCycleAgeReplacementPolicy:
        """
        Computes the optimal age(s) of replacement and updates the internal ``ar`̀` value(s) and, optionally ``ar1``.

        Returns
        -------
        Self
             Same instance with optimized ``ar`` (optionnaly ``ar1``).
        """

        cf_3d, cp_3d = np.array(self.cf, ndmin=3), np.array(self.cp, ndmin=3)
        x0 = np.minimum(np.sum(cp_3d, axis=0) / np.sum(cf_3d - cp_3d, axis=0), 1)

        def hf(t: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]:
            return self.lifetime_model.unfrozen_model.baseline.hf(t)  # unconditional_model

        def eq(a):
            return np.sum(
                self.discounting.factor(a)
                / self.discounting.annuity_factor(a)
                * ((cf_3d - cp_3d) * hf(a) - cp_3d / self.discounting.annuity_factor(a)),
                axis=0,
            )

        ar = np.asarray(newton(eq, x0), dtype=np.float64)
        self.ar = ar
        return self


class AgeReplacementPolicy(AgeRenewalPolicy):
    r"""Time based replacement policy.

    Renewal reward stochastic_process where assets are replaced at a fixed age :math:`a_r`
    with costs :math:`c_p` or upon failure with costs :math:`c_f` if earlier [1]_.

    .. [1] Mazzuchi, T. A., Van Noortwijk, J. M., & Kallen, M. J. (2007).
        Maintenance optimization. Encyclopedia of Statistics in Quality and
        Reliability, 1000-1008.

    Attributes
    ----------
    ar : np.ndarray or None
        Times until preventive replacements. This parameter can be optimized
        with ``optimize``
    ar1 : np.ndarray or None
        Times until preventive replacements for the first cycle. This parameter can be optimized
        with ``optimize``
    """

    lifetime_model: FrozenAgeReplacementModel
    first_lifetime_model: Optional[FrozenAgeReplacementModel]
    reward: AgeReplacementReward
    first_reward: Optional[AgeReplacementReward]

    def __init__(
        self,
        lifetime_model: LifetimeDistribution | FrozenLifetimeRegression | FrozenLeftTruncatedModel,
        cf: float | NDArray[np.float64],
        cp: float | NDArray[np.float64],
        discounting_rate: float = 0.0,
        first_lifetime_model: Optional[
            LifetimeDistribution | FrozenLifetimeRegression | FrozenLeftTruncatedModel
        ] = None,
        ar: float | NDArray[np.float64] = np.nan,
        ar1: float | NDArray[np.float64] = np.nan,
    ) -> None:
        self.lifetime_model: FrozenAgeReplacementModel = freeze(AgeReplacementModel(lifetime_model), ar)
        reward = AgeReplacementReward(cf, cp, ar)
        first_reward = None
        if first_lifetime_model is not None:
            first_lifetime_model: FrozenAgeReplacementModel = freeze(AgeReplacementModel(lifetime_model), ar1)
            first_reward = AgeReplacementReward(cf, cp, ar1)
        super().__init__(lifetime_model, reward, discounting_rate, first_lifetime_model, first_reward)

    @property
    def ar(self) -> float | NDArray[np.float64]:
        return self.lifetime_model.ar

    @ar.setter
    def ar(self, value: float | NDArray[np.float64]) -> None:
        self.lifetime_model.ar = value
        self.reward.ar = value

    @property
    def ar1(self) -> Optional[float | NDArray[np.float64]]:
        if self.first_lifetime_model is not None:
            return self.lifetime_model.ar
        return None

    @ar1.setter
    def ar1(self, value: float | NDArray[np.float64]) -> None:
        if self.first_lifetime_model is not None:
            self.first_lifetime_model.ar = value
            self.first_reward.ar = value
        else:
            raise ValueError

    @property
    def discounting_rate(self) -> float:
        return self.discounting.rate

    @property
    def cp(self) -> float | NDArray[np.float64]:
        return self.reward.cp

    @property
    def cf(self) -> float | NDArray[np.float64]:
        return self.reward.cf

    def expected_total_cost(
        self,
        tf: float,
        nb_steps: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        if np.isnan(self.ar):
            raise ValueError
        if self.ar1 is not None and np.isnan(self.ar):
            raise ValueError
        return self.underlying_process.expected_total_reward(tf, nb_steps)

    def expected_equivalent_annual_cost(
        self,
        tf: float,
        nb_steps: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        if np.isnan(self.ar):
            raise ValueError
        if self.ar1 is not None and np.isnan(self.ar):
            raise ValueError
        return self.underlying_process.expected_equivalent_annual_worth(tf, nb_steps)

    def asymptotic_expected_total_cost(self) -> NDArray[np.float64]:
        if np.isnan(self.ar):
            raise ValueError
        if self.ar1 is not None and np.isnan(self.ar):
            raise ValueError
        return self.underlying_process.asymptotic_expected_total_reward()

    def asymptotic_expected_equivalent_annual_cost(self) -> NDArray[np.float64]:
        if np.isnan(self.ar):
            raise ValueError
        if self.ar1 is not None and np.isnan(self.ar):
            raise ValueError
        return self.underlying_process.asymptotic_expected_equivalent_annual_worth()

    def optimize(
        self,
    ) -> AgeReplacementPolicy:
        """
        Computes the optimal age(s) of replacement and updates the internal ``ar`̀` value(s) and, optionally ``ar1``.

        Returns
        -------
        Self
             Same instance with optimized ``ar`` (optionnaly ``ar1``).
        """
        cf_3d, cp_3d = np.array(self.cf, ndmin=3), np.array(self.cp, ndmin=3)
        x0 = np.minimum(np.sum(cp_3d, axis=0) / np.sum(cf_3d - cp_3d, axis=0), 1)

        def eq(a):
            f = legendre_quadrature(
                lambda x: self.discounting.factor(x) * self.lifetime_model.unfrozen_model.baseline.sf(x),
                0,
                a,
            )
            g = legendre_quadrature(
                lambda x: self.discounting.factor(x) * self.lifetime_model.unfrozen_model.baseline.pdf(x),
                0,
                a,
            )
            return np.sum(
                self.discounting.factor(a)
                * ((cf_3d - cp_3d) * (self.lifetime_model.unfrozen_model.baseline.hf(a) * f - g) - cp_3d)
                / f**2,
                axis=0,
            )

        self.ar = newton(eq, x0)
        if self.ar1 is not None:
            one_cycle_ar_policy = OneCycleAgeReplacementPolicy(
                self.lifetime_model, self.cf, self.cp, self.discounting
            ).optimize()
            self.ar1 = one_cycle_ar_policy.ar
        return self


# class NonHomogeneousPoissonAgeReplacementPolicy(AgeRenewalPolicy):
#     """
#     Implements a Non-Homogeneous Poisson Process (NHPP) age-replacement policy..
#
#     Attributes
#     ----------
#     ar : np.ndarray or None
#         Optimized replacement age (optimized policy parameter).
#     cp : np.ndarray
#         The cost of failure for each asset.
#     cr : np.ndarray
#         The cost of repair for each asset.
#     """
#
#     def __init__(
#         self,
#         process: NonHomogeneousPoissonProcess[()],
#         cp: float | NDArray[np.float64],
#         cr: float | NDArray[np.float64],
#         *,
#         discounting_rate: Optional[float] = None,
#         ar: Optional[float | NDArray[np.float64]] = None,
#     ) -> None:
#         super().__init__(process.model, discounting_rate=discounting_rate)
#         self.ar = ar
#         self.cp = cp
#         self.cr = cr
#         if not isinstance(process, FrozenNonHomogeneousPoissonProcess):
#             raise ValueError
#         self._process = process
#
#     @property
#     def underlying_process(self) -> FrozenNonHomogeneousPoissonProcess:
#         return self._process
#
#     @property
#     def discounting_rate(self):
#         return self.discounting.rate
#
#     def expected_total_cost(
#         self,
#         timeline: NDArray[np.float64],
#         ar: Optional[float | NDArray[np.float64]] = None,
#     ) -> NDArray[np.float64]:
#         pass
#
#     def asymptotic_expected_total_cost(self, ar: Optional[float | NDArray[np.float64]] = None) -> NDArray[np.float64]:
#         pass
#
#     def expected_equivalent_annual_cost(
#         self,
#         timeline: NDArray[np.float64],
#         ar: Optional[float | NDArray[np.float64]] = None,
#     ) -> NDArray[np.float64]:
#         pass
#
#     def number_of_replacements(self, timeline: NDArray[np.float64]) -> NDArray[np.float64]:
#         pass
#
#     def expected_number_of_repairs(self, timeline: NDArray[np.float64]) -> NDArray[np.float64]:
#         pass
#
#     def asymptotic_expected_equivalent_annual_cost(self) -> NDArray[np.float64]:
#         pass
#
#     def optimize(
#         self,
#     ) -> NDArray[np.float64]:
#
#         x0 = self.model.mean()
#
#         if self.discounting.rate != 0:
#
#             def dcost(a):
#                 a = np.atleast_2d(a)
#                 return (
#                     (1 - self.discounting.factor(a)) / self.discounting.rate * self.underlying_process.intensity(a)
#                     - legendre_quadrature(
#                         lambda t: self.discounting.factor(t) * self.underlying_process.intensity(t),
#                         np.array(0.0),
#                         a,
#                         ndim=2,
#                     )
#                     - self.cp / self.cr
#                 )
#
#         else:
#
#             def dcost(a):
#                 a = np.atleast_2d(a)
#                 return (
#                     a * self.underlying_process.intensity(a)
#                     - self.underlying_process.cumulative_intensity(a)
#                     - self.cp / self.cr
#                 )
#
#         ar = np.squeeze(newton(dcost, x0))
#         self.ar = ar
#         return ar


from ._docstring import (
    ASYMPTOTIC_EEAC_DOCSTRING,
    ASYMPTOTIC_ETC_DOCSTRING,
    EEAC_DOCSTRING,
    ETC_DOCSTRING,
)

WARNING = r"""

.. warning::

    This method requires the ``ar`` attribute to be set either at initialization
    or with the ``optimize`` method.
"""

OneCycleAgeReplacementPolicy.expected_total_cost.__doc__ = ETC_DOCSTRING + WARNING
OneCycleAgeReplacementPolicy.expected_equivalent_annual_cost.__doc__ = EEAC_DOCSTRING + WARNING
OneCycleAgeReplacementPolicy.asymptotic_expected_total_cost.__doc__ = ASYMPTOTIC_ETC_DOCSTRING + WARNING
OneCycleAgeReplacementPolicy.asymptotic_expected_equivalent_annual_cost.__doc__ = ASYMPTOTIC_EEAC_DOCSTRING + WARNING
AgeReplacementPolicy.expected_total_cost.__doc__ = ETC_DOCSTRING + WARNING
AgeReplacementPolicy.expected_equivalent_annual_cost.__doc__ = EEAC_DOCSTRING + WARNING
AgeReplacementPolicy.asymptotic_expected_total_cost.__doc__ = ASYMPTOTIC_ETC_DOCSTRING + WARNING
AgeReplacementPolicy.asymptotic_expected_equivalent_annual_cost.__doc__ = ASYMPTOTIC_EEAC_DOCSTRING + WARNING
