from typing import Optional

import numpy as np
from numpy.typing import NDArray

from relife2.model import LifetimeModel
from relife2.renewal.discounting import Discounting
from relife2.renewal.reward import Reward


# lifetime ~ durations
# time ~ times


def model_rvs(
    model,
    size,
    args=(),
):
    return model.rvs(*args, size=size)


def rvs_size(
    nb_samples,
    nb_assets,
    model_args: tuple[NDArray[np.float64], ...] = (),
):
    if bool(model_args) and model_args[0].ndim == 2:
        size = nb_samples  # rvs size
    else:
        size = nb_samples * nb_assets
    return size


def compute_rewards(
    reward,
    lifetimes,
    args=(),
):
    return reward(lifetimes, *args)


def lifetimes_generator(
    model,
    nb_samples,
    nb_assets,
    end_time,
    *,
    model_args: tuple[NDArray[np.float64], ...] = (),
    delayed_model: Optional[LifetimeModel] = None,
    delayed_model_args: tuple[NDArray[np.float64], ...] = (),
):
    failure_times = np.zeros((nb_assets, nb_samples))
    still_valid = failure_times < end_time

    def sample_routine(target_model, args):
        nonlocal failure_times  # modify these variables
        lifetimes = model_rvs(target_model, size, args=args).reshape(
            (nb_assets, nb_samples)
        )
        failure_times += lifetimes
        return lifetimes, failure_times, still_valid

    if delayed_model:
        size = rvs_size(nb_samples, nb_assets, delayed_model_args)
        yield sample_routine(delayed_model, delayed_model_args)
        still_valid = failure_times < end_time

    size = rvs_size(nb_samples, nb_assets, model_args)
    while np.any(still_valid):
        yield sample_routine(model, model_args)
        still_valid = failure_times < end_time


def lifetimes_rewards_generator(
    model: LifetimeModel,
    reward: Reward,
    discount: Discounting,
    nb_samples: int,
    nb_assets: int,
    end_time: float,
    *,
    model_args: tuple[NDArray[np.float64], ...] = (),
    reward_args: tuple[NDArray[np.float64], ...] = (),
    discount_args: tuple[NDArray[np.float64], ...] = (),
    delayed_model: Optional[LifetimeModel] = None,
    delayed_model_args: tuple[NDArray[np.float64], ...] = (),
    delayed_reward: Optional[Reward] = None,
    delayed_reward_args: tuple[NDArray[np.float64], ...] = (),
):
    failure_times = np.zeros((nb_assets, nb_samples))
    total_rewards = np.zeros((nb_assets, nb_samples))
    still_valid = failure_times < end_time

    lifetimes_gen = lifetimes_generator(
        model,
        nb_samples,
        nb_assets,
        end_time,
        model_args=model_args,
        delayed_model=delayed_model,
        delayed_model_args=delayed_model_args,
    )

    def sample_routine(target_reward, args):
        nonlocal failure_times, total_rewards, still_valid  # modify these variables
        lifetimes, failure_times, still_valid = next(lifetimes_gen)
        rewards = target_reward(lifetimes, *args)
        discounts = discount.factor(failure_times, *discount_args)
        total_rewards += rewards * discounts
        return lifetimes, failure_times, total_rewards, still_valid

    if delayed_reward is None:
        delayed_reward = reward
        delayed_reward_args = reward_args

    if delayed_model:
        try:
            yield sample_routine(delayed_reward, delayed_reward_args)
        except StopIteration:
            return

    while np.any(still_valid):
        try:
            yield sample_routine(reward, reward_args)
        except StopIteration:
            return


class DataIterable:

    def __init__(
        self,
        samples: NDArray[np.int64],
        assets: NDArray[np.int64],
        /,
        *data: NDArray[np.float64],
    ):
        self.samples = samples
        self.assets = assets
        self.data = data

        self.samples_index = np.unique(self.samples)
        self.assets_index = np.unique(self.assets)
        self.nb_samples = len(self.samples_index)
        self.nb_assets = len(self.assets_index)

    def __len__(self):
        return self.nb_samples

    def __iter__(self) -> dict[str, list[NDArray[np.float64]]]:
        for sample in self.samples_index:
            res = []
            for v in self.data:
                res.append(
                    tuple(
                        v[self.samples == sample][
                            self.assets[self.samples == sample] == asset
                        ]
                        for asset in self.assets_index
                    )
                )

            yield tuple(res)

    def __getitem__(
        self, key: int | slice | tuple[int, int] | tuple[slice, slice]
    ) -> dict[str, list[NDArray[np.float64] | NDArray[np.float64]]]:
        if not isinstance(key, tuple):
            key = (key, None)
        if len(key) > 2:
            raise IndexError(
                f"{self.__class__.__name__} getter has a maximum of 2 index (samples and assets) but got {len(key)}"
            )

        if isinstance(key[0], slice):
            samples_mask = np.isin(
                self.samples,
                range(
                    key[0].start if key[0].start else 0,
                    key[0].stop if key[0].stop else self.nb_samples,
                    key[0].step if key[0].step else 1,
                ),
            )
        else:
            if key[0] not in self.samples_index:
                raise IndexError(
                    f"index {key[0]} is out of bounds for {self.__class__.__name__} with {self.nb_samples} nb samples"
                )
            samples_mask = self.samples == key[0]

        res = []
        for v in self.data:
            res.append(
                tuple(
                    v[samples_mask][self.assets[samples_mask] == asset]
                    for asset in self.assets_index
                )
            )

        if key[1] is not None:
            try:
                res = [v[key[1]] for v in res]
            except IndexError:
                raise IndexError(
                    f"index {key[1]} is out of bounds for {self.__class__.__name__} with {self.nb_assets} nb assets"
                )

        return tuple(res)


class LifetimesIterable(DataIterable):
    def __init__(
        self,
        model,
        nb_samples,
        nb_assets,
        end_time,
        *,
        model_args: tuple[NDArray[np.float64], ...] = (),
        delayed_model: Optional[LifetimeModel] = None,
        delayed_model_args: tuple[NDArray[np.float64], ...] = (),
    ):
        lifetimes = np.array([], dtype=np.float64)
        failure_times = np.array([], dtype=np.float64)
        samples = np.array([], dtype=np.int64)
        assets = np.array([], dtype=np.int64)

        for _lifetimes, _failure_times, still_valid in lifetimes_generator(
            model,
            nb_samples,
            nb_assets,
            end_time,
            model_args=model_args,
            delayed_model=delayed_model,
            delayed_model_args=delayed_model_args,
        ):
            lifetimes = np.concatenate((lifetimes, _lifetimes[still_valid].reshape(-1)))
            failure_times = np.concatenate(
                (failure_times, _failure_times[still_valid].reshape(-1))
            )
            _assets, _samples = np.where(still_valid)
            samples = np.concatenate((samples, _samples))
            assets = np.concatenate((assets, _assets))

        sorted_index = np.lexsort((assets, samples))
        lifetimes = lifetimes[sorted_index]
        failure_times = failure_times[sorted_index]
        samples = samples[sorted_index]
        assets = assets[sorted_index]

        super().__init__(samples, assets, lifetimes, failure_times)


class RewardedLifetimesIterable(DataIterable):
    def __init__(
        self,
        model: LifetimeModel,
        reward: Reward,
        discount: Discounting,
        nb_samples: int,
        nb_assets: int,
        end_time: float,
        *,
        model_args: tuple[NDArray[np.float64], ...] = (),
        reward_args: tuple[NDArray[np.float64], ...] = (),
        discount_args: tuple[NDArray[np.float64], ...] = (),
        delayed_model: Optional[LifetimeModel] = None,
        delayed_model_args: tuple[NDArray[np.float64], ...] = (),
        delayed_reward: Optional[Reward] = None,
        delayed_reward_args: tuple[NDArray[np.float64], ...] = (),
    ):

        failure_times = np.array([], dtype=np.float64)
        lifetimes = np.array([], dtype=np.float64)
        total_rewards = np.array([], dtype=np.float64)
        samples = np.array([], dtype=np.int64)
        assets = np.array([], dtype=np.int64)

        for (
            _lifetimes,
            _failure_times,
            _total_rewards,
            still_valid,
        ) in lifetimes_rewards_generator(
            model,
            reward,
            discount,
            nb_samples,
            nb_assets,
            end_time,
            model_args=model_args,
            reward_args=reward_args,
            discount_args=discount_args,
            delayed_model=delayed_model,
            delayed_model_args=delayed_model_args,
            delayed_reward=delayed_reward,
            delayed_reward_args=delayed_reward_args,
        ):
            lifetimes = np.concatenate((lifetimes, _lifetimes[still_valid].reshape(-1)))
            failure_times = np.concatenate(
                (failure_times, _failure_times[still_valid].reshape(-1))
            )
            total_rewards = np.concatenate(
                (total_rewards, _total_rewards[still_valid].reshape(-1))
            )
            _assets, _samples = np.where(still_valid)
            samples = np.concatenate((samples, _samples))
            assets = np.concatenate((assets, _assets))

        sorted_index = np.lexsort((assets, samples))
        lifetimes = lifetimes[sorted_index]
        failure_times = failure_times[sorted_index]
        total_rewards = total_rewards[sorted_index]
        samples = samples[sorted_index]
        assets = assets[sorted_index]

        super().__init__(
            samples,
            assets,
            lifetimes,
            failure_times,
            total_rewards,
        )
