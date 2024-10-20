from typing import TypeVarTuple, Optional, Protocol, TypeVar, TypedDict

import numpy as np
from numpy.typing import NDArray

from relife2.fiability.addons import LeftTruncated
from relife2.model import LifetimeModel
from relife2.renewal.discountings import exponential_discounting
from relife2.renewal.process import reward_partial_expectation, RenewalRewardProcess
from relife2.renewal.rewards import (
    run_to_failure_cost,
    age_replacement_cost,
)

M = TypeVar("M", tuple[NDArray[np.float64], ...], tuple[()])
M1 = TypeVar("M1", tuple[NDArray[np.float64], ...], tuple[()])
R = TypeVar("R", tuple[NDArray[np.float64], ...], tuple[()])
R1 = TypeVar("R1", tuple[NDArray[np.float64], ...], tuple[()])
D = TypeVar("D", tuple[NDArray[np.float64], ...], tuple[()])

T = TypeVarTuple("T")


class PolicyArgs(TypedDict):
    model_args: M
    delayed_model_args: M1
    reward_args: R
    delayed_reward_args: R1


# policy class are just facade class of one RenewalRewardProcess with more "financial" methods names
# and ergonomic parametrization (named parameters instead of generic tuple)
class Policy(Protocol[M, M1, R, R1]):

    args: PolicyArgs

    def expected_total_cost(
        self, timeline, rate: NDArray[np.float64], *args: *T
    ) -> NDArray[np.float64]: ...

    def asymptotic_expected_total_cost(
        self, rate: NDArray[np.float64], *args: *T
    ) -> NDArray[np.float64]: ...

    def expected_equivalent_annual_cost(
        self, timeline, rate: NDArray[np.float64], *args: *T
    ) -> NDArray[np.float64]: ...

    def asymptotic_expected_equivalent_annual_cost(
        self, rate: NDArray[np.float64], *args: *T
    ) -> NDArray[np.float64]: ...


class OneCycleRunToFailure:
    """One cyle run-to-failure policy."""

    def __init__(
        self,
        model: LifetimeModel[*M],
        model_args: M = (),
        nb_assets: int = 1,
        a0: Optional[NDArray[np.float64]] = None,
    ) -> None:
        if a0 is not None:
            model = LeftTruncated(model)
            model_args = (a0, *model_args)
        self.model = model
        self.model_args = model_args
        self.nb_assets = nb_assets

    def expected_total_cost(
        self,
        timeline: NDArray[np.float64],
        cf: NDArray[np.float64],
        rate: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        return reward_partial_expectation(
            timeline,
            self.model,
            run_to_failure_cost,
            exponential_discounting,
            model_args=self.model_args,
            reward_args=(cf,),
            discounting_args=(rate,),
        )

    def asymptotic_expected_total_cost(
        self,
        cf: NDArray[np.float64],
        rate: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        return self.expected_total_cost(np.array(np.inf), cf, rate)

    def expected_equivalent_annual_cost(
        self,
        timeline: NDArray[np.float64],
        cf: NDArray[np.float64],
        rate: NDArray[np.float64],
        dt: float = 1.0,
    ) -> NDArray[np.float64]:
        f = (
            lambda x: run_to_failure_cost(x, *(cf,))
            * exponential_discounting.factor(x, *(rate,))
            / exponential_discounting.annuity_factor(x, *(rate,))
        )
        mask = timeline < dt
        q0 = self.model.cdf(dt, *self.model_args) * f(dt)
        return q0 + np.where(
            mask, 0, self.model.ls_integrate(f, dt, timeline, *self.model_args)
        )

    def asymptotic_expected_equivalent_annual_cost(
        self,
        cf: NDArray[np.float64],
        rate: NDArray[np.float64],
        dt: float = 1.0,
    ) -> NDArray[np.float64]:
        return self.expected_equivalent_annual_cost(np.inf, cf, rate, dt)

    # def sample(
    #     self,
    #     nb_samples: int,
    #     cf: NDArray[np.float64],
    #     rate: NDArray[np.float64],
    # ) -> ReplacementPolicyData:
    #
    #     generator = lifetimes_rewards_generator(
    #         self.model,
    #         self.reward,
    #         self.discounting,
    #         nb_samples,
    #         self.nb_assets,
    #         np.inf,
    #         model_args=self.model_args,
    #         reward_args=(cf,),
    #         discount_args=(rate,),
    #     )
    #
    #     lifetimes, event_times, total_rewards, events, still_valid = next(generator)
    #     assets, samples = np.where(still_valid)
    #     assets = np.astype(assets, np.int64)
    #     samples = np.astype(samples, np.int64)
    #
    #     return ReplacementPolicyData


class OneCycleAgeReplacementPolicy:

    def __init__(
        self,
        model: LifetimeModel[*M],
        model_args: M = (),
        nb_assets: int = 1,
        a0: Optional[NDArray[np.float64]] = None,
    ) -> None:
        if a0 is not None:
            model = LeftTruncated(model)
            model_args = (a0, *model_args)
        self.model = model
        self.model_args = model_args
        self.nb_assets = nb_assets

    def expected_total_cost(
        self,
        timeline: NDArray[np.float64],
        ar: NDArray[np.float64],
        cf: NDArray[np.float64],
        cp: NDArray[np.float64],
        rate: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        return reward_partial_expectation(
            timeline,
            self.model,
            age_replacement_cost,
            exponential_discounting,
            model_args=(ar, *self.model_args),
            reward_args=(cp, cf),
            discounting_args=(rate,),
        )

    def asymptotic_expected_total_cost(
        self,
        ar: NDArray[np.float64],
        cf: NDArray[np.float64],
        cp: NDArray[np.float64],
        rate: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        return self.expected_total_cost(np.array(np.inf), ar, cf, cp, rate)

    def expected_equivalent_annual_cost(
        self,
        timeline: NDArray[np.float64],
        ar: NDArray[np.float64],
        cf: NDArray[np.float64],
        cp: NDArray[np.float64],
        rate: NDArray[np.float64],
        dt: float = 1.0,
    ) -> NDArray[np.float64]:
        f = (
            lambda x: age_replacement_cost(x, *(ar, cf, cp))
            * exponential_discounting.factor(x, *(rate,))
            / exponential_discounting.annuity_factor(x, *(rate,))
        )
        mask = timeline < dt
        q0 = self.model.cdf(dt, *self.model_args) * f(dt)
        return q0 + np.where(
            mask,
            0,
            self.model.ls_integrate(f, np.array(dt), timeline, *self.model_args),
        )

    def asymptotic_expected_equivalent_annual_cost(
        self,
        ar: NDArray[np.float64],
        cf: NDArray[np.float64],
        cp: NDArray[np.float64],
        rate: NDArray[np.float64],
        dt: float = 1.0,
    ) -> NDArray[np.float64]:
        return self.expected_equivalent_annual_cost(
            np.array(np.inf), ar, cf, cp, rate, dt
        )


class RunToFailure:

    def __init__(
        self,
        model: LifetimeModel[*M],
        model_args: M = (),
        nb_assets: int = 1,
        a0: Optional[NDArray[np.float64]] = None,
    ) -> None:

        if a0 is not None:
            model = LeftTruncated(model)
            model_args = (a0, *model_args)
        self.model = model
        self.model_args = model_args
        self.nb_assets = nb_assets

        self.delayed_model = None
        self.delayed_model_args = ()

    def add_delayed_model(
        self,
        model: LifetimeModel[*M1],
        model_args: M1 = (),
        a0: Optional[NDArray[np.float64]] = None,
    ):
        if a0 is not None:
            model = LeftTruncated(model)
            model_args = (a0, *model_args)
        self.delayed_model = model
        self.delayed_model_args = model_args

    @property
    def rrp(self):
        return RenewalRewardProcess(
            self.model,
            run_to_failure_cost,
            nb_assets=self.nb_assets,
            delayed_model=self.delayed_model,
            delayed_reward=run_to_failure_cost,
        )

    def expected_total_cost(
        self,
        timeline: NDArray[np.float64],
        cf: NDArray[np.float64],
        rate: NDArray[np.float64],
        *,
        cf1: Optional[NDArray[np.float64]] = None,
    ) -> NDArray[np.float64]:
        if not cf1:
            cf1 = cf1

        return self.rrp.expected_total_reward(
            timeline,
            discounting_rate=rate,
            model_args=self.model_args,
            reward_args=(cf,),
            delayed_model_args=self.delayed_model_args,
            delayed_reward_args=(cf1,),
        )

    def asymptotic_expected_total_cost(
        self,
        cf: NDArray[np.float64],
        rate: NDArray[np.float64],
        cf1: Optional[NDArray[np.float64]] = None,
    ) -> NDArray[np.float64]:
        if not cf1:
            cf1 = cf1

        return self.rrp.asymptotic_expected_total_reward(
            discounting_rate=rate,
            model_args=self.model_args,
            reward_args=(cf,),
            delayed_model_args=self.delayed_model_args,
            delayed_reward_args=(cf1,),
        )

    def expected_equivalent_annual_cost(
        self,
        timeline: NDArray[np.float64],
        cf: NDArray[np.float64],
        rate: NDArray[np.float64],
        cf1: Optional[NDArray[np.float64]] = None,
    ) -> NDArray[np.float64]:
        if not cf1:
            cf1 = cf1

        return self.rrp.expected_equivalent_annual_worth(
            timeline,
            discounting_rate=rate,
            model_args=self.model_args,
            reward_args=(cf,),
            delayed_model_args=self.delayed_model_args,
            delayed_reward_args=(cf1,),
        )

    def asymptotic_expected_equivalent_annual_cost(
        self,
        cf: NDArray[np.float64],
        rate: NDArray[np.float64],
        cf1: Optional[NDArray[np.float64]] = None,
    ) -> NDArray[np.float64]:
        if not cf1:
            cf1 = cf1

        return self.rrp.asymptotic_expected_equivalent_annual_worth(
            discounting_rate=rate,
            model_args=self.model_args,
            reward_args=(cf,),
            delayed_model_args=self.delayed_model_args,
            delayed_reward_args=(cf1,),
        )
