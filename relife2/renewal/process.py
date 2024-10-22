from functools import partial, wraps
from typing import Callable, Generic, Optional, ParamSpec, TypedDict, TypeVar

import numpy as np
from numpy.typing import NDArray

from relife2.model import LifetimeModel
from relife2.renewal.discounts import Discount, exponential_discount
from relife2.renewal.equations import (
    delayed_renewal_equation_solver,
    renewal_equation_solver,
)
from relife2.renewal.rewards import Reward
from relife2.renewal.sampling import (
    RenewalData,
    RenewalRewardData,
    lifetimes_generator,
    lifetimes_rewards_generator,
)

M = TypeVar("M", tuple[NDArray[np.float64], ...], tuple[()])
M1 = TypeVar("M1", tuple[NDArray[np.float64], ...], tuple[()])
R = TypeVar("R", tuple[NDArray[np.float64], ...], tuple[()])
R1 = TypeVar("R1", tuple[NDArray[np.float64], ...], tuple[()])
D = TypeVar("D", tuple[NDArray[np.float64], ...], tuple[()])

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


class RenewalProcessArgs(TypedDict, Generic[M, M1]):
    model: M
    model1: M1


# make partial in method only : do not split in multiple functions
class RenewalProcess(Generic[M, M1]):

    args: RenewalProcessArgs

    def __init__(
        self,
        model: LifetimeModel[*M],
        *,
        nb_assets: int = 1,
        model_args: M = (),
        model1: Optional[LifetimeModel[*M1]] = None,
        delayed_model_args: M1 = (),
    ):
        self.model = model
        self.model1 = model1
        self.nb_assets = nb_assets
        self.args["model"] = model_args
        self.args["model1"] = delayed_model_args

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
    ) -> RenewalData[M, M1]:

        lifetimes = np.array([], dtype=np.float64)
        event_times = np.array([], dtype=np.float64)
        events = np.array([], dtype=np.bool_)
        samples = np.array([], dtype=np.int64)
        assets = np.array([], dtype=np.int64)
        order = np.array([], dtype=np.int64)

        for i, (*gen_data, still_valid) in enumerate(
            lifetimes_generator(
                self.model,
                nb_samples,
                self.nb_assets,
                end_time,
                model_args=self.args["model"],
                model1=self.model1,
                model1_args=self.args["model1"],
            )
        ):
            lifetimes = np.concatenate((lifetimes, gen_data[0][still_valid]))
            event_times = np.concatenate((event_times, gen_data[1][still_valid]))
            events = np.concatenate((events, gen_data[2][still_valid]))
            order = np.ones_like(lifetimes) * i
            _assets, _samples = np.where(still_valid)
            samples = np.concatenate((samples, _samples))
            assets = np.concatenate((assets, _assets))

        return RenewalData(
            samples,
            assets,
            order,
            event_times,
            lifetimes,
            events,
            self.args["model"],
            self.args["model1"],
        )


def reward_partial_expectation(
    timeline: NDArray[np.float64],
    model: LifetimeModel[*M],
    reward: Reward[*R],
    discount: Discount[*D],
    *,
    model_args: M = (),
    reward_args: R = (),
    discount_args: D = (),
) -> np.ndarray:

    def func(x):
        return reward(x, *reward_args) * discount.factor(x, *discount_args)

    return model.ls_integrate(func, np.zeros_like(timeline), timeline, *model_args)


class RenewalRewardProcessArgs(TypedDict, Generic[M, M1, R, R1, D]):
    model: M
    model1: M1
    discount: D
    reward: R
    reward1: R1


class RenewalRewardProcess(RenewalProcess[M, M1], Generic[M, M1, R, R1]):

    args: RenewalRewardProcessArgs

    def __init__(
        self,
        model: LifetimeModel[*M],
        reward: Reward[*R],
        *,
        nb_assets: int = 1,
        model_args: M = (),
        reward_args: R = (),
        discount_rate: float | NDArray[np.float64] = 0.0,
        model1: Optional[LifetimeModel[*M1]] = None,
        model1_args: M1 = (),
        reward1: Optional[Reward[*R1]] = None,
        reward1_args: R1 = (),
    ):
        super().__init__(model, nb_assets=nb_assets, model1=model1)
        self.reward = reward
        self.reward1 = reward1
        self.args["model"] = model_args
        self.args["model1"] = model1_args
        self.args["reward"] = reward_args
        self.args["discount"] = (discount_rate,)
        self.args["reward1"] = reward1_args

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
                discount=exponential_discount,
                model_args=self.args["model"],
                reward_args=self.args["reward"],
                disounting_args=self.args["discount"],
            ),
            model_args=self.args["model"],
            discount=exponential_discount,
            discount_args=self.args["discount"],
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
                    discount=exponential_discount,
                    model_args=self.args["model1"],
                    reward_args=self.args["reward1"],
                    disount_args=self.args["discount"],
                ),
                delayed_model_args=self.args["model1"],
                discount=exponential_discount,
                discount_args=self.args["discount"],
            )

    def asymptotic_expected_total_reward(
        self,
    ) -> NDArray[np.float64]:
        mask = self.args["discount"][0] <= 0
        rate = np.ma.MaskedArray(self.args["discount"][0], mask)

        def f(x):
            return exponential_discount.factor(x, rate)

        def y(x):
            return exponential_discount.factor(x, rate) * self.reward(
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
                return exponential_discount.factor(x, rate) * self.reward(
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

    def expected_equivalent_annual_worth(
        self, timeline: NDArray[np.float64]
    ) -> NDArray[np.float64]:

        z = self.expected_total_reward(timeline)
        af = exponential_discount.annuity_factor(timeline, *self.args["discount"])
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

    def asymptotic_expected_equivalent_annual_worth(self) -> NDArray[np.float64]:
        mask = self.args["discount"][0] <= 0
        discount_rate = np.ma.MaskedArray(self.args["discount"][0], mask)

        q = discount_rate * self.asymptotic_expected_total_reward()
        q0 = self.model.ls_integrate(
            lambda x: self.reward(x, *self.args["reward"]),
            np.array(0.0),
            np.array(np.inf),
            *self.args["model"],
        ) / self.model.mean(*self.args["model"])
        return np.where(mask, q0, q)

    def sample(self, nb_samples: int, end_time: float) -> RenewalRewardData[M, M1]:
        lifetimes = np.array([], dtype=np.float64)
        event_times = np.array([], dtype=np.float64)
        total_rewards = np.array([], dtype=np.float64)
        events = np.array([], dtype=np.bool_)
        samples = np.array([], dtype=np.int64)
        assets = np.array([], dtype=np.int64)
        order = np.array([], dtype=np.int64)

        for i, (
            *gen_data,
            still_valid,
        ) in enumerate(
            lifetimes_rewards_generator(
                self.model,
                self.reward,
                exponential_discount,
                nb_samples,
                self.nb_assets,
                end_time=end_time,
                model_args=self.args["model"],
                reward_args=self.args["reward"],
                discount_args=self.args["discount"],
                model1=self.model1,
                model1_args=self.args["model1"],
                reward1=self.reward1,
                reward1_args=self.args["reward1"],
            )
        ):
            lifetimes = np.concatenate((lifetimes, gen_data[0][still_valid]))
            event_times = np.concatenate((event_times, gen_data[1][still_valid]))
            total_rewards = np.concatenate((total_rewards, gen_data[2][still_valid]))
            events = np.concatenate((events, gen_data[3][still_valid]))
            order = np.concatenate((order, np.ones_like(lifetimes) * i))
            _assets, _samples = np.where(still_valid)
            samples = np.concatenate((samples, _samples))
            assets = np.concatenate((assets, _assets))

        return RenewalRewardData(
            samples,
            assets,
            order,
            event_times,
            lifetimes,
            events,
            self.args["model"],
            self.args["model1"],
            total_rewards,
        )
