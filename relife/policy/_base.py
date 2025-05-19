from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Optional, overload

import numpy as np
from numpy.typing import NDArray

from relife import freeze
from relife.economic import Cost, ExponentialDiscounting, cost, Reward, Discounting
from relife.lifetime_model import LeftTruncatedModel, FrozenLifetimeRegression, LifetimeDistribution, FrozenLeftTruncatedModel, AgeReplacementModel
from ..stochastic_process import RenewalRewardProcess

if TYPE_CHECKING:
    from relife.lifetime_model import (
        FrozenParametricLifetimeModel,
        LifetimeDistribution,
    )
    from .age_replacement import (
        AgeReplacementPolicy,
        OneCycleAgeReplacementPolicy,
    )
    from .run_to_failure import RunToFailurePolicy


class OneCycleAgeRenewalPolicy:

    def __init__(
        self,
        lifetime_model: LifetimeDistribution | FrozenParametricLifetimeModel[*tuple[float| NDArray[np.float64], ...]],
        reward : Reward,
        discounting: Discounting,
        period_before_discounting: float = 1.0,
    ) -> None:
        self.cost = cost
        self.reward = reward
        self.discounting = discounting
        if period_before_discounting <= 0:
            raise ValueError("The period_before_discounting must be greater than 0")
        self.period_before_discounting = period_before_discounting
        self.lifetime_model = lifetime_model

    @property
    def discounting_rate(self):
        return self.discounting.rate

    def expected_total_cost(self, tf: float, nb_steps: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        timeline = np.linspace(0, tf, nb_steps, dtype=np.float64)
        # reward partial expectation
        return timeline, self.lifetime_model.ls_integrate(
            lambda x: self.reward.sample(x) * self.discounting.factor(x), np.zeros_like(timeline), timeline, deg=10
        )

    def asymptotic_expected_total_cost(self) -> NDArray[np.float64]:
        # reward partial expectation
        return self.lifetime_model.ls_integrate(
            lambda x: self.reward.sample(x) * self.discounting.factor(x), 0.0, np.inf, deg=10
        )

    def _expected_equivalent_annual_cost(self, timeline: NDArray[np.float64]) -> NDArray[np.float64]:
        def f(x: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]:
            return (
                self.reward.conditional_expectation(x) * self.discounting.factor(x) / self.discounting.annuity_factor(x)
            )

        q0 = self.lifetime_model.cdf(self.period_before_discounting) * f(self.period_before_discounting)  # () or (m, 1)
        a = np.full_like(timeline, self.period_before_discounting)  # (nb_steps,)
        # change first value of lower bound to compute the integral
        a = np.where(timeline < self.period_before_discounting, timeline, a)  # (nb_steps,)
        integral = np.atleast_2d(
            self.lifetime_model.ls_integrate(f, a, timeline)
        )  # (nb_steps,) if q0: (), or (m, nb_steps) if q0 : (m, 1)
        mask = np.broadcast_to(
            timeline < self.period_before_discounting, integral.shape
        )  # (nb_steps,) or (m, nb_steps)
        q0 = np.broadcast_to(q0, integral.shape)  # (nb_steps,) or (m, nb_steps)
        integral = np.where(mask, q0, integral)
        return integral

    def expected_equivalent_annual_cost(self, tf: float, nb_steps: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        timeline = np.linspace(0, tf, nb_steps, dtype=np.float64)  # (nb_steps,)
        return timeline, self._expected_equivalent_annual_cost(timeline)

    def asymptotic_expected_equivalent_annual_cost(
        self,
    ) -> NDArray[np.float64]:
        return self._expected_equivalent_annual_cost(np.array(np.inf))


class AgeRenewalPolicy:

    def __init__(
        self,
        lifetime_model: LifetimeDistribution | FrozenParametricLifetimeModel[*tuple[float| NDArray[np.float64], ...]],
        reward: Reward,
        discounting: Discounting,
        first_lifetime_model : Optional[LifetimeDistribution | FrozenParametricLifetimeModel[*tuple[float| NDArray[np.float64], ...]]] = None,
        first_reward : Optional[Reward] = None,
    ) -> None:
        self.lifetime_model = lifetime_model
        self.first_lifetime_model = first_lifetime_model
        self.reward = reward
        self.first_reward = first_reward
        self.discounting = discounting

    @property
    def underlying_process(self) -> RenewalRewardProcess:
        return RenewalRewardProcess(
            self.lifetime_model,
            self.reward,
            self.discounting,
            first_lifetime_model=self.first_lifetime_model,
            first_reward=self.first_reward
        )

    @property
    def discounting_rate(self):
        return self.discounting.rate

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


@overload
def run_to_failure_policy(
    lifetime_model: LifetimeDistribution | FrozenLifetimeRegression,
    cost: Cost,
    a0: Optional[float | NDArray[np.float64]] = None,
    one_cycle: Literal[True] = True,
    discounting_rate: float = 0.0,
) -> OneCycleRunToFailurePolicy: ...


@overload
def run_to_failure_policy(
    lifetime_model: LifetimeDistribution | FrozenLifetimeRegression,
    cost: Cost,
    a0: Optional[float | NDArray[np.float64]] = None,
    one_cycle: Literal[False] = False,
    discounting_rate: float = 0.0,
) -> RunToFailurePolicy: ...


def run_to_failure_policy(
    lifetime_model: LifetimeDistribution | FrozenLifetimeRegression,
    cost: Cost,
    a0: Optional[float | NDArray[np.float64]] = None,
    one_cycle: bool = False,
    discounting_rate: float = 0.0,
) -> RunToFailurePolicy | OneCycleRunToFailurePolicy:
    from .run_to_failure import RunToFailurePolicy, OneCycleRunToFailurePolicy

    discounting = ExponentialDiscounting(discounting_rate)
    if not one_cycle:
        first_lifetime_model = None
        if a0 is not None:
            first_lifetime_model : FrozenLeftTruncatedModel = freeze(LeftTruncatedModel(lifetime_model), a0)
        return RunToFailurePolicy(
            lifetime_model,
            cost,
            discounting,
            first_lifetime_model = first_lifetime_model,
        )
    if a0 is not None:
        lifetime_model : FrozenLeftTruncatedModel = freeze(LeftTruncatedModel(lifetime_model), a0)
    return OneCycleRunToFailurePolicy(
        lifetime_model,
        cost,
        discounting,
    )

@overload
def age_replacement_policy(
    lifetime_model: LifetimeDistribution | FrozenLifetimeRegression,
    cost: Cost,
    one_cycle = True,
    a0: Optional[float | NDArray[np.float64]] = None,
    ar: Optional[float | NDArray[np.float64]] = None,
    ar1 = None,
    discounting_rate: float = 0.0,
) -> OneCycleAgeReplacementPolicy: ...


@overload
def age_replacement_policy(
    lifetime_model: LifetimeDistribution | FrozenLifetimeRegression,
    cost: Cost,
    one_cycle = False,
    a0: Optional[float | NDArray[np.float64]] = None,
    ar: Optional[float | NDArray[np.float64]] = None,
    ar1: Optional[float | NDArray[np.float64]] = None,
    discounting_rate: float = 0.0,
) -> AgeReplacementPolicy: ...


def age_replacement_policy(
    lifetime_model: LifetimeDistribution | FrozenLifetimeRegression,
    cost: Cost,
    one_cycle: bool = False,
    a0: float | NDArray[np.float64] = None,
    ar: Optional[float | NDArray[np.float64]] = None,
    ar1: Optional[float | NDArray[np.float64]] = None,
    discounting_rate: float = 0.0,
) -> OneCycleRunToFailurePolicy | RunToFailurePolicy:

    ar = np.nan if ar is None else ar
    if not one_cycle:
        first_lifetime_model = None
        if a0 is not None and ar1 is not None:
            first_lifetime_model = freeze(AgeReplacementModel(LeftTruncatedModel(lifetime_model)), ar1, a0)
        elif ar1 is not None:
            first_lifetime_model = freeze(AgeReplacementModel(lifetime_model), ar1)
        elif a0 is not None:
            first_lifetime_model = freeze(AgeReplacementModel(LeftTruncatedModel(lifetime_model)), ar, a0)
        return AgeReplacementPolicy(
            freeze(AgeReplacementModel(lifetime_model), ar),
            cost,
            ExponentialDiscounting(discounting_rate),
            first_lifetime_model=first_lifetime_model
        )
    else:
        if ar1 is not None:
            raise ValueError
        if a0 is not None and



    return DefaultAgeReplacementPolicy(
        model,
        cost["cf"],
        cost["cp"],
        discounting_rate=discounting_rate,
        ar=ar,
        ar1=ar1,
        a0=a0,
        model1=model1,
    )


@overload
def run_to_failure_policy(
    model: LifetimeDistribution | FrozenParametricLifetimeModel,
    cost: Cost,
    one_cycle: Literal[False] = False,
    discounting_rate: Optional[float] = None,
    model1: Optional[LifetimeDistribution | FrozenParametricLifetimeModel] = None,
    a0: Optional[float | NDArray[np.float64]] = None,
) -> RunToFailurePolicy: ...



#
# def make_renewal_policy(
#     model: ParametricLifetimeModel[()] | NonHomogeneousPoissonProcess,
#     cost_structure: Cost,
#     one_cycle: bool = False,
#     run_to_failure: bool = False,
#     discounting_rate: Optional[float] = None,
#     model1: Optional[ParametricLifetimeModel[()] | NonHomogeneousPoissonProcess] = None,
#     a0: Optional[float | NDArray[np.float64]] = None,
#     ar: Optional[float | NDArray[np.float64]] = None,
#     ar1: Optional[float | NDArray[np.float64]] = None,
# ) -> RenewalPolicy:
#     """
#     Parameters
#     ----------
#     model : Parametric
#     cost_structure : dict of np.ndarray
#     one_cycle : bool, default False
#     run_to_failure : bool, default False
#     discounting_rate : float
#     ar1
#     ar
#     a0
#     model1
#     """
#
#     from relife.stochastic_process import NonHomogeneousPoissonProcess
#
#     from .age_replacement import (
#         DefaultAgeReplacementPolicy,
#         NonHomogeneousPoissonAgeReplacementPolicy,
#         OneCycleAgeReplacementPolicy,
#     )
#     from .run_to_failure import DefaultRunToFailurePolicy, OneCycleRunToFailurePolicy
#
#     if isinstance(model, NonHomogeneousPoissonProcess):
#         try:
#             cp, cr = (
#                 cost_structure["cp"],
#                 cost_structure["cr"],
#             )
#         except KeyError:
#             raise ValueError("Costs must contain cf and cr")
#         return NonHomogeneousPoissonAgeReplacementPolicy(
#             model,
#             cp,
#             cr,
#             discounting_rate=discounting_rate,
#             ar=ar,
#         )
#
#     if run_to_failure:
#         if not one_cycle:
#             try:
#                 cf = cost_structure["cf"]
#             except KeyError:
#                 raise ValueError("Costs must only contain cf")
#             return DefaultRunToFailurePolicy(
#                 model,
#                 cf,
#                 discounting_rate=discounting_rate,
#                 a0=a0,
#                 model1=model1,
#             )
#         else:
#             try:
#                 cf = cost_structure["cf"]
#             except KeyError:
#                 raise ValueError("Costs must only contain cf")
#             return OneCycleRunToFailurePolicy(
#                 model,
#                 cf,
#                 discounting_rate=discounting_rate,
#                 a0=a0,
#             )
#     else:
#         if not one_cycle:
#             try:
#                 cf, cp = (
#                     cost_structure["cf"],
#                     cost_structure["cp"],
#                 )
#             except KeyError:
#                 raise ValueError("Costs must contain cf and cp")
#             return DefaultAgeReplacementPolicy(
#                 model,
#                 cf,
#                 cp,
#                 discounting_rate=discounting_rate,
#                 ar=ar,
#                 ar1=ar1,
#                 a0=a0,
#                 model1=model1,
#             )
#         else:
#             try:
#                 cf, cp = (
#                     cost_structure["cf"],
#                     cost_structure["cp"],
#                 )
#             except KeyError:
#                 raise ValueError("Costs must contain cf and cp")
#             return OneCycleAgeReplacementPolicy(
#                 model,
#                 cf,
#                 cp,
#                 discounting_rate=discounting_rate,
#                 ar=ar,
#                 a0=a0,
#             )
