from functools import partial
from typing import Optional, Callable

import numpy as np
from numpy.typing import NDArray

from relife.data import RenewalData, RenewalRewardData
from relife.descriptors import ShapedArgs
from relife.discounting import exponential_discounting
from relife.generator import lifetimes_generator, lifetimes_rewards_generator
from relife.model import LifetimeModel
from relife.types import Model1Args, ModelArgs, Reward1Args, RewardArgs, Reward


def renewal_equation_solver(
    timeline: NDArray[np.float64],
    model: LifetimeModel[*ModelArgs],
    evaluated_func: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    *,
    model_args: ModelArgs = (),
    discounting_rate: float = 0.0,
) -> NDArray[np.float64]:

    tm = 0.5 * (timeline[1:] + timeline[:-1])
    f = model.cdf(timeline, *model_args)
    fm = model.cdf(tm, *model_args)
    y = evaluated_func(timeline)
    if discounting_rate != 0.0:
        d = exponential_discounting.factor(timeline, discounting_rate)
    else:
        d = np.ones_like(f)
    z = np.empty(y.shape)
    u = d * np.insert(f[..., 1:] - fm, 0, 1, axis=-1)
    v = d[..., :-1] * np.insert(np.diff(fm), 0, 1, axis=-1)
    q0 = 1 / (1 - d[..., 0] * fm[..., 0])
    z[..., 0] = y[..., 0]
    z[..., 1] = q0 * (y[..., 1] + z[..., 0] * u[..., 1])
    for n in range(2, f.shape[-1]):
        z[..., n] = q0 * (
            y[..., n]
            + z[..., 0] * u[..., n]
            + np.sum(z[..., 1:n][..., ::-1] * v[..., 1:n], axis=-1)
        )
    return z


def delayed_renewal_equation_solver(
    timeline: NDArray[np.float64],
    z: NDArray[np.float64],
    model1: LifetimeModel[*Model1Args],
    evaluated_func: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    model1_args: Model1Args = (),
    discounting_rate: float = 0.0,
) -> NDArray[np.float64]:

    tm = 0.5 * (timeline[1:] + timeline[:-1])
    f1 = model1.cdf(timeline, *model1_args)
    f1m = model1.cdf(tm, *model1_args)
    y1 = evaluated_func(timeline)

    if discounting_rate != 0.0:
        d = exponential_discounting.factor(timeline, discounting_rate)
    else:
        d = np.ones_like(f1)
    z1 = np.empty(y1.shape)
    u1 = d * np.insert(f1[..., 1:] - f1m, 0, 1, axis=-1)
    v1 = d[..., :-1] * np.insert(np.diff(f1m), 0, 1, axis=-1)
    z1[..., 0] = y1[..., 0]
    z1[..., 1] = (
        y1[..., 1] + z[..., 0] * u1[..., 1] + z[..., 1] * d[..., 0] * f1m[..., 0]
    )
    for n in range(2, f1.shape[-1]):
        z1[..., n] = (
            y1[..., n]
            + z[..., 0] * u1[..., n]
            + z[..., n] * d[..., 0] * f1m[..., 0]
            + np.sum(z[..., 1:n][..., ::-1] * v1[..., 1:n], axis=-1)
        )
    return z1


class RenewalProcess:
    model_args = ShapedArgs(astuple=True)
    model1_args = ShapedArgs(astuple=True)

    def __init__(
        self,
        model: LifetimeModel[*ModelArgs],
        *,
        nb_assets: int = 1,
        model_args: ModelArgs = (),
        model1: Optional[LifetimeModel[*Model1Args]] = None,
        model1_args: Model1Args = (),
    ):
        self.nb_assets = nb_assets
        self.model = model
        self.model1 = model1
        self.model_args = model_args
        self.model1_args = model1_args

    def renewal_function(self, timeline: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.model1 is None:
            evaluated_func = partial(
                lambda time, args: self.model.cdf(time, *args),
                args=self.model_args,
            )
        else:
            evaluated_func = partial(
                lambda time, args: self.model1.cdf(time, *args),
                args=self.model1_args,
            )
        return renewal_equation_solver(
            timeline,
            self.model,
            evaluated_func,
            model_args=self.model_args,
        )

    def renewal_density(self, timeline: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.model1 is None:
            evaluated_func = partial(
                lambda time, args: self.model.pdf(time, *args),
                args=self.model_args,
            )
        else:
            evaluated_func = partial(
                lambda time, args: self.model1.pdf(time, *args),
                args=self.model1_args,
            )

        return renewal_equation_solver(
            timeline,
            self.model,
            evaluated_func,
            model_args=self.model_args,
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
                model_args=self.model_args,
                model1=self.model1,
                model1_args=self.model1_args,
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
            model_args=self.model_args,
            with_model1=self.model1 is not None,
        )


def reward_partial_expectation(
    timeline: NDArray[np.float64],
    model: LifetimeModel[*ModelArgs],
    reward: Reward,
    *,
    model_args: ModelArgs = (),
    reward_args: RewardArgs = (),
    discounting_rate: float = 0.0,
) -> np.ndarray:
    def func(x):
        return reward(x, *reward_args) * exponential_discounting.factor(
            x, discounting_rate
        )

    ls = model.ls_integrate(func, np.zeros_like(timeline), timeline, *model_args)
    # reshape 2d -> final_dim
    ndim = max(map(np.ndim, (timeline, *model_args, *reward_args)), default=0)
    if ndim < 2:
        ls = np.squeeze(ls)
    return ls


class RenewalRewardProcess(RenewalProcess):
    model_args = ShapedArgs(astuple=True)
    model1_args = ShapedArgs(astuple=True)
    reward_args = ShapedArgs(astuple=True)
    reward1_args = ShapedArgs(astuple=True)

    def __init__(
        self,
        model: LifetimeModel[*ModelArgs],
        reward: Reward,
        *,
        nb_assets: int = 1,
        model_args: ModelArgs = (),
        reward_args: RewardArgs = (),
        discounting_rate: float = 0.0,
        model1: Optional[LifetimeModel[*Model1Args]] = None,
        model1_args: Model1Args = (),
        reward1: Optional[Reward] = None,
        reward1_args: Reward1Args = (),
    ):
        super().__init__(
            model,
            nb_assets=nb_assets,
            model1=model1,
            model_args=model_args,
            model1_args=model1_args,
        )
        self.reward = reward
        self.reward1 = reward1
        self.reward_args = reward_args
        self.discounting_rate = discounting_rate
        self.reward1_args = reward1_args

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
                model_args=self.model_args,
                reward_args=self.reward_args,
                discounting_rate=self.discounting_rate,
            ),
            model_args=self.model_args,
            discounting_rate=self.discounting_rate,
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
                    model_args=self.model1_args,
                    reward_args=self.reward1_args,
                    discounting_rate=self.discounting_rate,
                ),
                model1_args=self.model1_args,
                discounting_rate=self.discounting_rate,
            )

    def asymptotic_expected_total_reward(
        self,
    ) -> NDArray[np.float64]:
        mask = self.discounting_rate <= 0
        rate = np.ma.MaskedArray(self.discounting_rate, mask)

        def f(x):
            return exponential_discounting.factor(x, rate)

        def y(x):
            return exponential_discounting.factor(x, rate) * self.reward(
                x, *self.reward_args
            )

        lf = self.model.ls_integrate(
            f, np.array(0.0), np.array(np.inf), *self.model_args
        )
        ly = self.model.ls_integrate(
            y, np.array(0.0), np.array(np.inf), *self.model_args
        )
        z = ly / (1 - lf)

        if self.model1 is not None:

            def y1(x):
                return exponential_discounting.factor(x, rate) * self.reward1(
                    x, *self.reward1_args
                )

            lf1 = self.model1.ls_integrate(
                f, np.array(0.0), np.array(np.inf), *self.model1_args
            )
            ly1 = self.model1.ls_integrate(
                y1, np.array(0.0), np.array(np.inf), *self.model1_args
            )
            z = ly1 + z * lf1
        return np.where(mask, np.inf, z)

    def expected_equivalent_annual_cost(
        self, timeline: NDArray[np.float64]
    ) -> NDArray[np.float64]:

        z = self.expected_total_reward(timeline)
        af = exponential_discounting.annuity_factor(timeline, self.discounting_rate)
        mask = af == 0.0
        af = np.ma.masked_where(mask, af)
        q = z / af
        if self.model1 is None:
            q0 = self.reward(np.array(0.0), *self.reward_args) * self.model.pdf(
                np.array(0.0), *self.model_args
            )
        else:
            q0 = self.reward1(np.array(0.0), *self.reward1_args) * self.model1.pdf(
                np.array(0.0), *self.model1_args
            )
        return np.where(mask, q0, q)

    def asymptotic_expected_equivalent_annual_cost(self) -> NDArray[np.float64]:
        mask = self.discounting_rate <= 0
        discounting_rate = np.ma.MaskedArray(self.discounting_rate, mask)

        q = discounting_rate * self.asymptotic_expected_total_reward()

        q0 = self.model.ls_integrate(
            lambda x: self.reward(x, *self.reward_args),
            np.array(0.0),
            np.array(np.inf),
            *self.model_args,
        ) / self.model.mean(*self.model_args)
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
                nb_samples,
                self.nb_assets,
                end_time=end_time,
                model_args=self.model_args,
                reward_args=self.reward_args,
                discounting_rate=self.discounting_rate,
                model1=self.model1,
                model1_args=self.model1_args,
                reward1=self.reward1,
                reward1_args=self.reward1_args,
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
            model_args=self.model_args,
            with_model1=self.model1 is not None,
            total_rewards=total_rewards,
        )
