from functools import partial, wraps
from typing import Callable, Optional, ParamSpec, TypeVar

import numpy as np
from numpy.typing import NDArray

from relife.model import LifetimeModel
from relife.renewal import (
    delayed_renewal_equation_solver,
    renewal_equation_solver,
)
from relife.generator import lifetimes_generator, lifetimes_rewards_generator
from relife.data.renewal import RenewalData, RenewalRewardData
from relife.discountings import exponential_discounting, Discounting
from relife.rewards import Reward
from relife.typing import (
    DiscountingArgs,
    Model1Args,
    ModelArgs,
    RenewalProcessArgs,
    RenewalRewardProcessArgs,
    Reward1Args,
    RewardArgs,
)

P = ParamSpec("P")
T = TypeVar("T")


def argscheck(method: Callable[P, T]) -> Callable[P, T]:
    @wraps(method)
    def wrapper(self, *args: P.args, **kwargs: P.kwargs) -> T:
        for key, value in kwargs.items():
            if "args" in key and value is not None:
                if not isinstance(value, tuple):
                    raise ValueError(
                        f"args must parameters must be filled in tuple : got {type(value)} for {key}"
                    )
                if self.nb_assets > 1:
                    for array in value:
                        if not isinstance(array, np.ndarray):
                            raise ValueError(
                                f"args values must be ndarray only : got {type(array)} for {key}"
                            )
                        if array.ndim != 2:
                            raise ValueError(
                                f"If nb_assets is {self.nb_assets}, args array must have 2 dimensions"
                            )
                        if array.shape[0] != self.nb_assets:
                            raise ValueError(
                                f"Expected {self.nb_assets} nb assets but got {array.shape[0]} in {key}"
                            )
                else:
                    for array in value:
                        if array.ndim > 1:
                            raise ValueError(
                                f"If nb_assets is 1, args array cannot have more than 1 dimension"
                            )
        return method(self, *args, **kwargs)

    return wrapper


class RenewalProcess:
    args: RenewalProcessArgs

    def __init__(
        self,
        model: LifetimeModel[*ModelArgs],
        *,
        nb_assets: int = 1,
        model_args: ModelArgs = (),
        model1: Optional[LifetimeModel[*Model1Args]] = None,
        model1_args: Model1Args = (),
    ):
        self.model = model
        self.model1 = model1
        self.nb_assets = nb_assets
        self.args = {"model": model_args, "model1": model1_args}

    def renewal_function(self, timeline: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.model1 is None:
            evaluated_func = partial(
                lambda time, args: self.model.cdf(time, *args),
                args=self.args["model"],
            )
        else:
            evaluated_func = partial(
                lambda time, args: self.model1.cdf(time, *args),
                args=self.args["model1"],
            )
        return renewal_equation_solver(
            timeline,
            self.model,
            evaluated_func,
            model_args=self.args["model"],
        )

    def renewal_density(self, timeline: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.model1 is None:
            evaluated_func = partial(
                lambda time, args: self.model.pdf(time, *args),
                args=self.args["model"],
            )
        else:
            evaluated_func = partial(
                lambda time, args: self.model1.pdf(time, *args),
                args=self.args["model1"],
            )

        return renewal_equation_solver(
            timeline,
            self.model,
            evaluated_func,
            model_args=self.args["model"],
        )

    def sample(
        self,
        nb_samples: int,
        end_time: float,
        seed: Optional[int] = None,
    ) -> RenewalData:

        lifetimes = np.array([], dtype=np.float64)
        event_times = np.array([], dtype=np.float64)
        events = np.array([], dtype=np.bool_)
        samples_index = np.array([], dtype=np.int64)
        assets_index = np.array([], dtype=np.int64)
        order = np.array([], dtype=np.int64)

        for i, (_lifetimes, _event_times, _events, still_valid) in enumerate(
            lifetimes_generator(
                self.model,
                nb_samples,
                self.nb_assets,
                end_time,
                model_args=self.args["model"],
                model1=self.model1,
                model1_args=self.args["model1"],
                seed=seed,
            )
        ):
            lifetimes = np.concatenate((lifetimes, _lifetimes[still_valid]))
            event_times = np.concatenate((event_times, _event_times[still_valid]))
            events = np.concatenate((events, _events[still_valid]))
            order = np.concatenate((order, np.ones_like(_lifetimes[still_valid]) * i))
            _assets_index, _samples_index = np.where(still_valid)
            samples_index = np.concatenate((samples_index, _samples_index))
            assets_index = np.concatenate((assets_index, _assets_index))

        return RenewalData(
            samples_index,
            assets_index,
            order,
            event_times,
            lifetimes,
            events,
            model_args=self.args["model"],
            with_model1=self.model1 is not None,
        )


def reward_partial_expectation(
    timeline: NDArray[np.float64],
    model: LifetimeModel[*ModelArgs],
    reward: Reward[*RewardArgs],
    discounting: Discounting[*DiscountingArgs],
    *,
    model_args: ModelArgs = (),
    reward_args: RewardArgs = (),
    discounting_args: DiscountingArgs = (),
) -> np.ndarray:
    def func(x):
        return reward(x, *reward_args) * discounting.factor(x, *discounting_args)

    ls = model.ls_integrate(func, np.zeros_like(timeline), timeline, *model_args)
    # reshape 2d -> final_dim
    ndim = max(map(np.ndim, (timeline, *model_args, *reward_args)), default=0)
    if ndim < 2:
        ls = np.squeeze(ls)
    return ls


class RenewalRewardProcess(RenewalProcess):
    args: RenewalRewardProcessArgs
    discounting: Discounting[float] = exponential_discounting

    def __init__(
        self,
        model: LifetimeModel[*ModelArgs],
        reward: Reward[*RewardArgs],
        *,
        nb_assets: int = 1,
        model_args: ModelArgs = (),
        reward_args: RewardArgs = (),
        discounting_rate: float = 0.0,
        model1: Optional[LifetimeModel[*Model1Args]] = None,
        model1_args: Model1Args = (),
        reward1: Optional[Reward[*Reward1Args]] = None,
        reward1_args: Reward1Args = (),
    ):
        super().__init__(model, nb_assets=nb_assets, model1=model1)
        self.reward = reward
        self.reward1 = reward1
        self.args = {
            "model": model_args,
            "model1": model1_args,
            "reward": reward_args,
            "discounting": (discounting_rate,),
            "reward1": reward1_args,
        }

    def expected_total_reward(
        self, timeline: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        z = renewal_equation_solver(
            timeline,
            self.model,
            partial(
                reward_partial_expectation,
                model=self.model,
                reward=self.reward,
                discounting=self.discounting,
                model_args=self.args["model"],
                reward_args=self.args["reward"],
                discounting_args=self.args["discounting"],
            ),
            model_args=self.args["model"],
            discounting=self.discounting,
            discounting_args=self.args["discounting"],
        )

        if self.model1 is None:
            return z
        else:
            return delayed_renewal_equation_solver(
                timeline,
                z,
                self.model1,
                partial(
                    reward_partial_expectation,
                    model=self.model1,
                    reward=self.reward1,
                    discounting=self.discounting,
                    model_args=self.args["model1"],
                    reward_args=self.args["reward1"],
                    discounting_args=self.args["discounting"],
                ),
                model1_args=self.args["model1"],
                discounting=self.discounting,
                discounting_args=self.args["discounting"],
            )

    def asymptotic_expected_total_reward(
        self,
    ) -> NDArray[np.float64]:
        mask = self.args["discounting"][0] <= 0
        rate = np.ma.MaskedArray(self.args["discounting"][0], mask)

        def f(x):
            return self.discounting.factor(x, rate)

        def y(x):
            return self.discounting.factor(x, rate) * self.reward(
                x, *self.args["reward"]
            )

        lf = self.model.ls_integrate(
            f, np.array(0.0), np.array(np.inf), *self.args["model"]
        )
        ly = self.model.ls_integrate(
            y, np.array(0.0), np.array(np.inf), *self.args["model"]
        )
        z = ly / (1 - lf)

        if self.model1 is not None:

            def y1(x):
                return self.discounting.factor(x, rate) * self.reward1(
                    x, *self.args["reward1"]
                )

            lf1 = self.model1.ls_integrate(
                f, np.array(0.0), np.array(np.inf), *self.args["model1"]
            )
            ly1 = self.model1.ls_integrate(
                y1, np.array(0.0), np.array(np.inf), *self.args["model1"]
            )
            z = ly1 + z * lf1
        return np.where(mask, np.inf, z)

    def expected_equivalent_annual_cost(
        self, timeline: NDArray[np.float64]
    ) -> NDArray[np.float64]:

        z = self.expected_total_reward(timeline)
        af = self.discounting.annuity_factor(timeline, *self.args["discounting"])
        mask = af == 0.0
        af = np.ma.masked_where(mask, af)
        q = z / af
        if self.model1 is None:
            q0 = self.reward(np.array(0.0), *self.args["reward"]) * self.model.pdf(
                np.array(0.0), *self.args["model"]
            )
        else:
            q0 = self.reward1(np.array(0.0), *self.args["reward1"]) * self.model1.pdf(
                np.array(0.0), *self.args["model1"]
            )
        return np.where(mask, q0, q)

    def asymptotic_expected_equivalent_annual_cost(self) -> NDArray[np.float64]:
        mask = self.args["discounting"][0] <= 0
        discounting_rate = np.ma.MaskedArray(self.args["discounting"][0], mask)

        q = discounting_rate * self.asymptotic_expected_total_reward()

        q0 = self.model.ls_integrate(
            lambda x: self.reward(x, *self.args["reward"]),
            np.array(0.0),
            np.array(np.inf),
            *self.args["model"],
        ) / self.model.mean(*self.args["model"])
        return np.where(mask, q0, q)

    def sample(
        self,
        nb_samples: int,
        end_time: float,
        seed: Optional[int] = None,
    ) -> RenewalRewardData:
        lifetimes = np.array([], dtype=np.float64)
        event_times = np.array([], dtype=np.float64)
        total_rewards = np.array([], dtype=np.float64)
        events = np.array([], dtype=np.bool_)
        samples_index = np.array([], dtype=np.int64)
        assets_index = np.array([], dtype=np.int64)
        order = np.array([], dtype=np.int64)

        for i, (
            _lifetimes,
            _event_times,
            _total_rewards,
            _events,
            still_valid,
        ) in enumerate(
            lifetimes_rewards_generator(
                self.model,
                self.reward,
                self.discounting,
                nb_samples,
                self.nb_assets,
                end_time=end_time,
                model_args=self.args["model"],
                reward_args=self.args["reward"],
                discounting_args=self.args["discounting"],
                model1=self.model1,
                model1_args=self.args["model1"],
                reward1=self.reward1,
                reward1_args=self.args["reward1"],
                seed=seed,
            )
        ):
            lifetimes = np.concatenate((lifetimes, _lifetimes[still_valid]))
            event_times = np.concatenate((event_times, _event_times[still_valid]))
            total_rewards = np.concatenate((total_rewards, _total_rewards[still_valid]))
            events = np.concatenate((events, _events[still_valid]))
            order = np.concatenate((order, np.ones_like(_lifetimes[still_valid]) * i))
            _assets_index, _samples_index = np.where(still_valid)
            samples_index = np.concatenate((samples_index, _samples_index))
            assets_index = np.concatenate((assets_index, _assets_index))

        return RenewalRewardData(
            samples_index,
            assets_index,
            order,
            event_times,
            lifetimes,
            events,
            model_args=self.args["model"],
            with_model1=self.model1 is not None,
            total_rewards=total_rewards,
        )
