from __future__ import annotations

import copy
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
    LeftTruncatedModel,
    LifetimeDistribution,
)
from relife.quadrature import legendre_quadrature

from ..stochastic_process import RenewalRewardProcess
from ._base import BaseAgeReplacementPolicy, BaseOneCycleAgeReplacementPolicy


class OneCycleAgeReplacementPolicy(BaseOneCycleAgeReplacementPolicy[FrozenAgeReplacementModel, AgeReplacementReward]):
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
        a0: Optional[float | NDArray[np.float64]] = None,
        ar: Optional[float | NDArray[np.float64]] = None,
    ) -> None:
        if a0 is not None:
            lifetime_model: FrozenAgeReplacementModel = freeze(
                AgeReplacementModel(freeze(LeftTruncatedModel(lifetime_model), a0=a0)),
                ar=ar if ar is not None else np.nan,
            )
        else:
            lifetime_model: FrozenAgeReplacementModel = freeze(
                AgeReplacementModel(lifetime_model), ar=ar if ar is not None else np.nan
            )
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
        if np.any(np.isnan(self.ar)):
            raise ValueError
        return super().expected_total_cost(tf, nb_steps)

    @override
    def asymptotic_expected_total_cost(self) -> NDArray[np.float64]:
        if np.any(np.isnan(self.ar)):
            raise ValueError
        return super().asymptotic_expected_total_cost()

    @override
    def expected_equivalent_annual_cost(
        self, tf: float, nb_steps: int
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        if np.any(np.isnan(self.ar)):
            raise ValueError
        # because b = np.minimum(ar, b) in ls_integrate, b can be lower than a depending on period before discounting
        if np.any(self.period_before_discounting >= self.ar):
            raise ValueError("The period before discounting must be lower than ar values")
        return super().expected_equivalent_annual_cost(tf, nb_steps)

    @override
    def asymptotic_expected_equivalent_annual_cost(self) -> NDArray[np.float64]:
        if np.any(np.isnan(self.ar)):
            raise ValueError
        # because b = np.minimum(ar, b) in ls_integrate, b can be lower than a depending on period before discounting
        if np.any(self.period_before_discounting >= self.ar):
            raise ValueError("The period before discounting must be lower than ar values")
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

        x0 = np.minimum(self.cp / (self.cf - self.cp), 1)  # ()
        hf = self.lifetime_model.unfrozen_model.baseline.hf
        x0 = np.broadcast_to(x0, hf(x0).shape).copy()  # () or (m, 1)

        def eq(a):  # () or (m, 1)
            return (
                self.discounting.factor(a)
                / self.discounting.annuity_factor(a)
                * ((self.cf - self.cp) * hf(a) - self.cp / self.discounting.annuity_factor(a))
            )
            # return ((self.cf - self.cp) / self.cp) * hf(a) * self.discounting.annuity_factor(a)

        self.ar = newton(eq, x0)  # () or (m, 1)
        return self


class AgeReplacementPolicy(BaseAgeReplacementPolicy[FrozenAgeReplacementModel, AgeReplacementReward]):
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

    stochastic_process: RenewalRewardProcess[FrozenAgeReplacementModel, AgeReplacementReward]

    def __init__(
        self,
        lifetime_model: LifetimeDistribution | FrozenLifetimeRegression,
        cf: float | NDArray[np.float64],
        cp: float | NDArray[np.float64],
        discounting_rate: float = 0.0,
        first_lifetime_model: Optional[
            LifetimeDistribution | FrozenLifetimeRegression | FrozenLeftTruncatedModel
        ] = None,
        a0: Optional[float | NDArray[np.float64]] = None,
        ar: Optional[float | NDArray[np.float64]] = None,
        ar1: Optional[float | NDArray[np.float64]] = None,
    ) -> None:
        first_reward: Optional[AgeReplacementReward] = None
        if first_lifetime_model is not None:
            if a0 is not None:
                first_lifetime_model: FrozenAgeReplacementModel = freeze(
                    AgeReplacementModel(freeze(LeftTruncatedModel(first_lifetime_model), a0=a0)),
                    ar=ar1 if ar1 is not None else np.nan,
                )
            else:
                first_lifetime_model: FrozenAgeReplacementModel = freeze(
                    AgeReplacementModel(first_lifetime_model), ar=ar1 if ar1 is not None else np.nan
                )
            first_reward = AgeReplacementReward(cf, cp, ar1 if ar1 is not None else np.nan)
        elif a0 is not None:
            first_lifetime_model: FrozenAgeReplacementModel = freeze(
                AgeReplacementModel(freeze(LeftTruncatedModel(lifetime_model), a0=a0)),
                ar=ar1 if ar1 is not None else np.nan,
            )
        first_lifetime_model: Optional[FrozenAgeReplacementModel]
        lifetime_model: FrozenAgeReplacementModel = freeze(
            AgeReplacementModel(lifetime_model), ar=ar if ar is not None else np.nan
        )
        reward = AgeReplacementReward(cf, cp, ar if ar is not None else np.nan)

        stochastic_process = RenewalRewardProcess(
            lifetime_model,
            reward,
            discounting_rate,
            first_lifetime_model=first_lifetime_model,
            first_reward=first_reward,
        )
        super().__init__(stochastic_process)

    @property
    def ar(self) -> float | NDArray[np.float64]:
        return np.squeeze(self.stochastic_process.lifetime_model.ar)  # may be (m, 1) so squeeze it

    @ar.setter
    def ar(self, value: float | NDArray[np.float64]) -> None:
        self.stochastic_process.lifetime_model.ar = value
        self.stochastic_process.reward.ar = value

    @property
    def ar1(self) -> Optional[float | NDArray[np.float64]]:
        if self.stochastic_process.first_lifetime_model is not None:
            return np.squeeze(self.stochastic_process.first_lifetime_model.ar)  # may be (m, 1) so squeeze it
        return None

    @ar1.setter
    def ar1(self, value: float | NDArray[np.float64]) -> None:
        if self.stochastic_process.first_lifetime_model is not None:
            self.stochastic_process.first_lifetime_model.ar = value
            self.stochastic_process.first_reward.ar = value
        else:
            raise ValueError

    @property
    def cf(self) -> float | NDArray[np.float64]:
        return self.stochastic_process.reward.cf

    @cf.setter
    def cf(self, value: float | NDArray[np.float64]) -> None:
        self.stochastic_process.reward.cf = value
        self.stochastic_process.first_reward.cf = value

    @property
    def cp(self) -> float | NDArray[np.float64]:
        return self.stochastic_process.reward.cp

    @cp.setter
    def cp(self, value: float | NDArray[np.float64]) -> None:
        self.stochastic_process.reward.cp = value
        self.stochastic_process.first_reward.cp = value

    def expected_total_cost(
        self,
        tf: float,
        nb_steps: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        if np.any(np.isnan(self.ar)):
            raise ValueError
        if self.ar1 is not None and np.any(np.isnan(self.ar1)):
            raise ValueError
        return self.stochastic_process.expected_total_reward(tf, nb_steps)

    def expected_equivalent_annual_cost(
        self,
        tf: float,
        nb_steps: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        if np.any(np.isnan(self.ar)):
            raise ValueError
        if self.ar1 is not None and np.any(np.isnan(self.ar1)):
            raise ValueError
        return self.stochastic_process.expected_equivalent_annual_worth(tf, nb_steps)

    def asymptotic_expected_total_cost(self) -> NDArray[np.float64]:
        if np.any(np.isnan(self.ar)):
            raise ValueError
        if self.ar1 is not None and np.any(np.isnan(self.ar1)):
            raise ValueError
        return self.stochastic_process.asymptotic_expected_total_reward()

    def asymptotic_expected_equivalent_annual_cost(self) -> NDArray[np.float64]:
        if np.any(np.isnan(self.ar)):
            raise ValueError
        if self.ar1 is not None and np.any(np.isnan(self.ar1)):
            raise ValueError
        return self.stochastic_process.asymptotic_expected_equivalent_annual_worth()

    def annual_number_of_replacements(
        self,
        nb_years: int,
        upon_failure: bool = False,
        total: bool = True,
    ):
        if np.any(np.isnan(self.ar)):
            raise ValueError
        if self.ar1 is not None and np.any(np.isnan(self.ar1)):
            raise ValueError

        copied_policy = copy.deepcopy(self)
        copied_policy.cf = 1.0
        copied_policy.cp = 1.0
        copied_policy.discounting_rate = 0.0
        timeline, total_cost = copied_policy.expected_total_cost(nb_years, nb_years + 1)  # equiv to np.arange
        if total:
            mt = np.sum(np.atleast_2d(total_cost), axis=0)
        else:
            mt = total_cost
        nb_replacements = np.diff(mt)
        if upon_failure:
            copied_policy.cp = 0.0
            _, total_cost = copied_policy.expected_total_cost(nb_years, nb_years + 1)  # equiv to np.arange
            if total:
                mf = np.sum(np.atleast_2d(total_cost), axis=0)
            else:
                mf = total_cost
            nb_failures = np.diff(mf)
            return timeline[1:], nb_replacements, nb_failures
        return timeline[1:], nb_replacements

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
        x0 = np.minimum(self.cp / (self.cf - self.cp), 1)  # ()
        discounting = ExponentialDiscounting(self.stochastic_process.discounting_rate)
        sf = self.stochastic_process.lifetime_model.unfrozen_model.baseline.sf
        pdf = self.stochastic_process.lifetime_model.unfrozen_model.baseline.pdf
        hf = self.stochastic_process.lifetime_model.unfrozen_model.baseline.hf
        x0 = np.broadcast_to(x0, hf(x0).shape).copy()  # () or (m, 1)

        def eq(a):  # () or (m, 1)
            f = legendre_quadrature(
                lambda x: discounting.factor(x) * sf(x),
                0,
                a,
            )
            g = legendre_quadrature(
                lambda x: discounting.factor(x) * pdf(x),
                0,
                a,
            )
            return discounting.factor(a) * ((self.cf - self.cp) * (hf(a) * f - g) - self.cp) / f**2

        self.ar = newton(eq, x0)
        if self.ar1 is not None:
            one_cycle_ar_policy = OneCycleAgeReplacementPolicy(
                self.stochastic_process.first_lifetime_model.unfrozen_model.baseline,
                self.cf,
                self.cp,
                self.stochastic_process.discounting.rate,
            ).optimize()
            self.ar1 = one_cycle_ar_policy.ar
            self.ar = np.broadcast_to(self.ar, self.ar1.shape).copy()
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
