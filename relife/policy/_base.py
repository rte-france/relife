from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
from numpy.typing import NDArray

from relife.economic import ExponentialDiscounting, Reward, cost
from relife.stochastic_process import RenewalRewardProcess

if TYPE_CHECKING:
    from relife.lifetime_model import (
        FrozenParametricLifetimeModel,
        LifetimeDistribution,
    )


class OneCycleAgeRenewalPolicy:

    def __init__(
        self,
        lifetime_model: LifetimeDistribution | FrozenParametricLifetimeModel[*tuple[float | NDArray[np.float64], ...]],
        reward: Reward,
        discounting_rate: float = 0.0,
        period_before_discounting: float = 1.0,
    ) -> None:
        self.cost = cost
        self.reward = reward
        self.discounting = ExponentialDiscounting(discounting_rate)
        if period_before_discounting <= 0:
            raise ValueError("The period_before_discounting must be greater than 0")
        self.period_before_discounting = period_before_discounting
        self.lifetime_model = lifetime_model

    @property
    def discounting_rate(self):
        return self.discounting.rate

    def expected_total_cost(self, tf: float, nb_steps: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        timeline = np.linspace(0, tf, nb_steps, dtype=np.float64)
        # timeline : () or (nb_steps,)
        # like in stochastic_process.renewal_process::RenewalRewardProcess::expected_total_reward, timeline must be
        # reshaped if reward as more than on value
        if self.reward.cost_array.size > 1:
            timeline = np.tile(timeline, (self.reward.cost_array.size, 1))
        # reward partial expectation
        return timeline, self.lifetime_model.ls_integrate(
            lambda x: self.reward.sample(x) * self.discounting.factor(x), np.zeros_like(timeline), timeline, deg=15
        )

    def asymptotic_expected_total_cost(self) -> NDArray[np.float64]:
        # reward partial expectation
        return self.lifetime_model.ls_integrate(
            lambda x: self.reward.sample(x) * self.discounting.factor(x), 0.0, np.inf, deg=15
        )

    def _expected_equivalent_annual_cost(self, timeline: NDArray[np.float64]) -> NDArray[np.float64]:
        # timeline : () or (nb_steps,)
        # like in stochastic_process.renewal_process::RenewalRewardProcess::expected_total_reward, timeline must be
        # reshaped if reward as more than on value
        if self.reward.cost_array.size > 1:
            timeline = np.tile(timeline, (self.reward.cost_array.size, 1))

        # timeline : (), (nb_steps,) or (m, nb_steps)

        def f(x: float | NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
            return (
                self.reward.conditional_expectation(x) * self.discounting.factor(x) / self.discounting.annuity_factor(x)
            )

        q0 = self.lifetime_model.cdf(self.period_before_discounting) * f(self.period_before_discounting)  # () or (m, 1)
        a = np.full_like(timeline, self.period_before_discounting)  # (nb_steps,)
        # change first value of lower bound to compute the integral
        a[timeline < self.period_before_discounting] = 0. # (nb_steps,)
        # a = np.where(timeline < self.period_before_discounting, 0., a)  # (nb_steps,)
        integral = self.lifetime_model.ls_integrate(f, a, timeline, deg=100)  # (nb_steps,) if q0: (), or (m, nb_steps) if q0 : (m, 1)
        mask = np.broadcast_to(timeline < self.period_before_discounting, integral.shape) # (), (nb_steps,) or (m, nb_steps)
        q0 = np.broadcast_to(q0, integral.shape)  # (nb_steps,) or (m, nb_steps)
        integral = np.where(mask, q0, q0 + integral)
        return timeline, integral

    def expected_equivalent_annual_cost(
        self, tf: float, nb_steps: int
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        timeline = np.linspace(0, tf, nb_steps, dtype=np.float64)  # (nb_steps,)
        return self._expected_equivalent_annual_cost(timeline)

    def asymptotic_expected_equivalent_annual_cost(
        self,
    ) -> NDArray[np.float64]:
        return self._expected_equivalent_annual_cost(np.array(np.inf))[-1]


class AgeRenewalPolicy:

    def __init__(
        self,
        lifetime_model: LifetimeDistribution | FrozenParametricLifetimeModel[*tuple[float | NDArray[np.float64], ...]],
        reward: Reward,
        discounting_rate: float = 0.0,
        first_lifetime_model: Optional[
            LifetimeDistribution | FrozenParametricLifetimeModel[*tuple[float | NDArray[np.float64], ...]]
        ] = None,
        first_reward: Optional[Reward] = None,
    ) -> None:
        self.lifetime_model = lifetime_model
        self.first_lifetime_model = first_lifetime_model
        self.reward = reward
        self.first_reward = first_reward
        self.discounting_rate = discounting_rate

    @property
    def underlying_process(self) -> RenewalRewardProcess:
        return RenewalRewardProcess(
            self.lifetime_model,
            self.reward,
            self.discounting_rate,
            first_lifetime_model=self.first_lifetime_model,
            first_reward=self.first_reward,
        )

    def expected_nb_replacements(self, tf: float, nb_steps: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return self.underlying_process.renewal_function(tf, nb_steps)

    def expected_total_cost(self, tf: float, nb_steps: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return self.underlying_process.expected_total_reward(tf, nb_steps)

    def asymptotic_expected_total_cost(self) -> NDArray[np.float64]:
        return self.underlying_process.asymptotic_expected_total_reward()

    def expected_equivalent_annual_cost(
        self, tf: float, nb_steps: int
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return self.underlying_process.expected_equivalent_annual_worth(tf, nb_steps)

    def asymptotic_expected_equivalent_annual_cost(self) -> NDArray[np.float64]:
        return self.underlying_process.asymptotic_expected_equivalent_annual_worth()


# @overload
# def age_replacement_policy(
#     lifetime_model: LifetimeDistribution | FrozenLifetimeRegression,
#     cost: Cost,
#     one_cycle = True,
#     a0: Optional[float | NDArray[np.float64]] = None,
#     ar: Optional[float | NDArray[np.float64]] = None,
#     ar1 = None,
#     discounting_rate: float = 0.0,
# ) -> OneCycleAgeReplacementPolicy: ...
#
#
# @overload
# def age_replacement_policy(
#     lifetime_model: LifetimeDistribution | FrozenLifetimeRegression,
#     cost: Cost,
#     one_cycle = False,
#     a0: Optional[float | NDArray[np.float64]] = None,
#     ar: Optional[float | NDArray[np.float64]] = None,
#     ar1: Optional[float | NDArray[np.float64]] = None,
#     discounting_rate: float = 0.0,
# ) -> AgeReplacementPolicy: ...
#
#
# def age_replacement_policy(
#     lifetime_model: LifetimeDistribution | FrozenLifetimeRegression,
#     cost: Cost,
#     one_cycle: bool = False,
#     a0: float | NDArray[np.float64] = None,
#     ar: Optional[float | NDArray[np.float64]] = None,
#     ar1: Optional[float | NDArray[np.float64]] = None,
#     discounting_rate: float = 0.0,
# ) -> OneCycleAgeRenewalPolicy | AgeReplacementPolicy:
#
#     ar = np.nan if ar is None else ar
#     if not one_cycle:
#         first_lifetime_model = None
#         if a0 is not None and ar1 is not None:
#             first_lifetime_model = freeze(AgeReplacementModel(LeftTruncatedModel(lifetime_model)), ar1, a0)
#         elif ar1 is not None:
#             first_lifetime_model = freeze(AgeReplacementModel(lifetime_model), ar1)
#         elif a0 is not None:
#             first_lifetime_model = freeze(AgeReplacementModel(LeftTruncatedModel(lifetime_model)), ar, a0)
#         return AgeReplacementPolicy(
#             freeze(AgeReplacementModel(lifetime_model), ar),
#             cost,
#             ExponentialDiscounting(discounting_rate),
#             first_lifetime_model=first_lifetime_model
#         )
#     else:
#         if ar1 is not None:
#             raise ValueError
#         if a0 is not None and
#
#
#
#     return DefaultAgeReplacementPolicy(
#         model,
#         cost["cf"],
#         cost["cp"],
#         discounting_rate=discounting_rate,
#         ar=ar,
#         ar1=ar1,
#         a0=a0,
#         model1=model1,
#     )
#
