from typing import Optional, Protocol

import numpy as np
from numpy.typing import NDArray

from relife2.data import RenewalRewardData
from relife2.fiability.addon import LeftTruncatedModel
from relife2.fiability.model import LifetimeModel
from relife2.renewal.discount import Discount, exponential_discount
from relife2.renewal.process import RenewalRewardProcess, reward_partial_expectation
from relife2.renewal.reward import Reward, age_replacement_cost, run_to_failure_cost
from relife2.renewal.sampling import lifetimes_rewards_generator
from relife2.typing import (
    DiscountArgs,
    Model1Args,
    ModelArgs,
    PolicyArgs,
    Reward1Args,
    RewardArgs,
)


# policy class are just facade class of one RenewalRewardProcess with more "financial" methods names
# and ergonomic parametrization (named parameters instead of generic tuple)
class Policy(Protocol):

    args: PolicyArgs
    model: LifetimeModel[*ModelArgs]
    reward: Reward[*RewardArgs]
    discount: Discount[*DiscountArgs]
    model1: Optional[LifetimeModel[*Model1Args]] = None
    reward1: Optional[Reward[*Reward1Args]] = None
    nb_assets: int = 1

    def expected_total_cost(
        self, timeline: NDArray[np.float64]
    ) -> NDArray[np.float64]: ...

    def asymptotic_expected_total_cost(self) -> NDArray[np.float64]: ...

    def expected_equivalent_annual_cost(
        self, timeline: NDArray[np.float64]
    ) -> NDArray[np.float64]: ...

    def asymptotic_expected_equivalent_annual_cost(self) -> NDArray[np.float64]: ...


class OneCycleRunToFailure:
    """One cyle run-to-failure policy."""

    args: PolicyArgs
    model: LifetimeModel[*ModelArgs]
    reward = run_to_failure_cost
    discount = exponential_discount
    model1 = None
    reward1 = None
    nb_assets: int = 1

    def __init__(
        self,
        model: LifetimeModel[*ModelArgs],
        cf: NDArray[np.float64],
        rate: NDArray[np.float64],
        model_args: ModelArgs = (),
        nb_assets: int = 1,
        a0: Optional[NDArray[np.float64]] = None,
    ) -> None:
        if a0 is not None:
            model = LeftTruncatedModel(model)
            model_args = (a0, *model_args)
        self.model = model
        self.model_args = model_args
        self.nb_assets = nb_assets
        self.args["model"] = model_args
        self.args["reward"] = (cf,)
        self.args["discount"] = (rate,)

    def expected_total_cost(self, timeline: NDArray[np.float64]) -> NDArray[np.float64]:
        return reward_partial_expectation(
            timeline,
            self.model,
            run_to_failure_cost,
            exponential_discount,
            model_args=self.model_args,
            reward_args=self.args["reward"],
            discount_args=self.args["discount"],
        )

    def asymptotic_expected_total_cost(self) -> NDArray[np.float64]:
        return self.expected_total_cost(np.array(np.inf))

    def expected_equivalent_annual_cost(
        self, timeline: NDArray[np.float64], dt: float = 1.0
    ) -> NDArray[np.float64]:
        f = (
            lambda x: run_to_failure_cost(x, *self.args["reward"])
            * exponential_discount.factor(x, *self.args["discount"])
            / exponential_discount.annuity_factor(x, *self.args["discount"])
        )
        mask = timeline < dt
        q0 = self.model.cdf(dt, *self.model_args) * f(dt)
        return q0 + np.where(
            mask, 0, self.model.ls_integrate(f, dt, timeline, *self.model_args)
        )

    def asymptotic_expected_equivalent_annual_cost(
        self, dt: float = 1.0
    ) -> NDArray[np.float64]:
        return self.expected_equivalent_annual_cost(np.array(np.inf), dt)

    def sample(
        self,
        nb_samples: int,
    ) -> RenewalRewardData:
        generator = lifetimes_rewards_generator(
            self.model,
            self.reward,
            self.discount,
            nb_samples,
            self.nb_assets,
            np.inf,
            model_args=self.args["model"],
            reward_args=self.args["reward"],
            discount_args=self.args["discount"],
        )

        *gen_data, still_valid = next(generator)
        assets, samples = np.where(still_valid)
        assets.astype(np.int64)
        samples.astype(np.int64)
        lifetimes = gen_data[0][still_valid]
        event_times = gen_data[1][still_valid]
        total_rewards = gen_data[2][still_valid]
        events = gen_data[3][still_valid]
        order = np.zeros_like(lifetimes)

        return RenewalRewardData(
            samples,
            assets,
            order,
            event_times,
            lifetimes,
            events,
            total_rewards,
        )


class OneCycleAgeReplacementPolicy:

    args: PolicyArgs
    model: LifetimeModel[*ModelArgs]
    reward = age_replacement_cost
    discount = exponential_discount
    model1 = None
    reward1 = None
    nb_assets: int = 1

    def __init__(
        self,
        model: LifetimeModel[*ModelArgs],
        ar: NDArray[np.float64],
        cf: NDArray[np.float64],
        cp: NDArray[np.float64],
        rate: NDArray[np.float64],
        model_args: ModelArgs = (),
        nb_assets: int = 1,
        a0: Optional[NDArray[np.float64]] = None,
    ) -> None:
        if a0 is not None:
            model = LeftTruncatedModel(model)
            model_args = (a0, *model_args)
        self.model = model
        self.nb_assets = nb_assets
        self.args["model"] = (ar, *model_args)
        self.args["reward"] = (ar, cf, cp)
        self.args["discount"] = (rate,)

    def expected_total_cost(self, timeline: NDArray[np.float64]) -> NDArray[np.float64]:
        return reward_partial_expectation(
            timeline,
            self.model,
            self.reward,
            self.discount,
            model_args=self.args["model"],
            reward_args=self.args["reward"],
            discount_args=self.args["discount"],
        )

    def asymptotic_expected_total_cost(self) -> NDArray[np.float64]:
        return self.expected_total_cost(np.array(np.inf))

    def expected_equivalent_annual_cost(
        self, timeline: NDArray[np.float64], dt: float = 1.0
    ) -> NDArray[np.float64]:
        f = (
            lambda x: age_replacement_cost(x, *self.args["reward"])
            * exponential_discount.factor(x, *self.args["discount"])
            / exponential_discount.annuity_factor(x, *self.args["discount"])
        )
        mask = timeline < dt
        q0 = self.model.cdf(dt, *self.args["model"]) * f(dt)
        return q0 + np.where(
            mask,
            0,
            self.model.ls_integrate(f, np.array(dt), timeline, *self.args["model"]),
        )

    def asymptotic_expected_equivalent_annual_cost(
        self, dt: float = 1.0
    ) -> NDArray[np.float64]:
        return self.expected_equivalent_annual_cost(np.array(np.inf), dt)


class RunToFailure:

    args: PolicyArgs
    model: LifetimeModel[*ModelArgs]
    reward = run_to_failure_cost
    discount = exponential_discount
    model1: Optional[LifetimeModel[*Model1Args]] = None
    reward1 = run_to_failure_cost
    nb_assets: int = 1

    def __init__(
        self,
        model: LifetimeModel[*ModelArgs],
        cf: NDArray[np.float64],
        rate: NDArray[np.float64],
        model_args: ModelArgs = (),
        nb_assets: int = 1,
        a0: Optional[NDArray[np.float64]] = None,
        model1: Optional[LifetimeModel[*Model1Args]] = None,
        model1_args: Model1Args = (),
        cf1: Optional[NDArray[np.float64]] = None,
    ) -> None:

        if a0 is not None:
            if model1 is not None:
                model1 = LeftTruncatedModel(model1)
                model1_args = (a0, *model1_args)
            else:
                model = LeftTruncatedModel(model)
                model_args = (a0, *model_args)
        self.model = model
        self.model1 = model1
        self.args["model"] = model_args
        self.args["reward"] = (cf,)
        self.args["discount"] = (rate,)
        if model1_args is not None:
            self.args["model1"] = model1_args
        if cf1 is not None:
            self.args["reward1"] = (cf1,)
        self.nb_assets = nb_assets

        self.rrp = RenewalRewardProcess(
            self.model,
            run_to_failure_cost,
            nb_assets=self.nb_assets,
            model_args=self.args["model"],
            reward_args=self.args["reward"],
            discount_rate=rate,
            model1=self.model1,
            model1_args=self.args["model1"],
            reward1=run_to_failure_cost,
            reward1_args=self.args["reward1"],
        )

    def expected_total_cost(self, timeline: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.rrp.expected_total_reward(timeline)

    def asymptotic_expected_total_cost(self) -> NDArray[np.float64]:
        return self.rrp.asymptotic_expected_total_reward()

    def expected_equivalent_annual_cost(
        self, timeline: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return self.rrp.expected_equivalent_annual_worth(timeline)

    def asymptotic_expected_equivalent_annual_cost(self) -> NDArray[np.float64]:
        return self.rrp.asymptotic_expected_equivalent_annual_worth()
