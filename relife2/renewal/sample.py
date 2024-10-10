from dataclasses import dataclass, field, fields, replace
from typing import Coroutine, Iterator, Self

import numpy as np
from numpy.typing import NDArray

from relife2.discount import Discount
from relife2.model import LifetimeModel
from relife2.reward import Reward


# lifetime ~ durations
# time ~ times


@dataclass
class GeneratedData:
    samples: NDArray[np.int64] = field(repr=False, default_factory=np.ndarray([]))
    assets: NDArray[np.int64] = field(repr=False, default_factory=np.ndarray([]))
    nb_samples: int = field(init=False)
    nb_assets: int = field(init=False)
    samples_index: int = field(init=False, repr=False)
    assets_index: int = field(init=False, repr=False)
    closed: bool = field(init=False, repr=False, default=False)

    def update(self, *field_values):
        i = 0
        for _field in fields(self):
            if _field.init:
                v = getattr(self, _field.name)
                setattr(self, _field.name, np.concatenate((v, field_values[i])))
                i += 1

    def close(self):
        self.samples_index = np.unique(self.samples)
        self.assets_index = np.unique(self.assets)
        self.nb_samples = len(self.samples_index)
        self.nb_assets = len(self.assets_index)

        sorted_index = np.lexsort((self.assets, self.samples))
        for _field in fields(self):
            if _field.init:
                v = getattr(self, _field.name)
                setattr(self, _field.name, v[sorted_index])

        self.closed = True

    def __len__(self):
        if not self.closed:
            raise ValueError(
                f"Can't call len on unclosed {self.__class__.__name__} object"
            )
        return self.nb_samples * self.nb_assets

    def split(self, sample_only=False, asset_only=False) -> Iterator[Self]:
        if not self.closed:
            raise ValueError(
                f"Can't call split on unclosed {self.__class__.__name__} object"
            )
        if sample_only and asset_only:
            raise ValueError(
                "sample_only and asset_only can't be true at the same time"
            )
        current_fields = {
            _field.name: getattr(self, _field.name)
            for _field in fields(self)
            if _field.init
        }
        if sample_only:
            for index in self.samples_index:
                yield replace(
                    self,
                    **{k: v[self.samples == index] for k, v in current_fields.items()},
                )
        if asset_only:
            for index in self.assets_index:
                yield replace(
                    self,
                    **{k: v[self.assets == index] for k, v in current_fields.items()},
                )

    def __getitem__(
        self, key: int | slice | tuple[int, int] | tuple[slice, slice]
    ) -> Self:
        if not self.closed:
            raise ValueError(
                f"Can't slice on unclosed {self.__class__.__name__} object"
            )
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
        changes = {
            _field.name: getattr(self, _field.name)[samples_mask]
            for _field in fields(self)
            if _field.init
        }

        if key[1] is not None:
            if isinstance(key[0], slice):
                assets_mask = np.isin(
                    changes["assets"],
                    range(
                        key[1].start if key[1].start else 0,
                        key[1].stop if key[1].stop else self.nb_assets,
                        key[1].step if key[1].step else 1,
                    ),
                )
            else:
                if key[1] not in self.assets_index:
                    raise IndexError(
                        f"index {key[1]} is out of bounds for {self.__class__.__name__} with {self.nb_assets} nb assets"
                    )
                assets_mask = changes["assets"] == key[1]
            changes = {k: v[assets_mask] for k, v in changes.items()}

        return replace(self, **changes)


@dataclass
class GeneratedLifetime(GeneratedData):
    time: NDArray[np.float64] = field(repr=False, default_factory=np.ndarray([]))
    lifetime: NDArray[np.float64] = field(repr=False, default_factory=np.ndarray([]))

    @property
    def nb_events(self) -> int:
        return np.sum(np.unique(self.samples, return_counts=True)[-1])

    @property
    def mean_number_of_events(self) -> float:
        return np.mean(np.unique(self.samples, return_counts=True)[-1])


@dataclass
class GeneratedRewardLifetime(GeneratedLifetime):
    reward: NDArray[np.float64] = field(repr=False, default_factory=np.ndarray([]))

    def cumulative_reward(self) -> float:
        return np.insert(self.reward.cumsum(), 0, 0)

    def mean_cumulative_reward(self) -> float:
        return np.insert(self.reward.cumsum(), 0, 0) / self.nb_samples


def coroutine(func):
    def starter(*args, **kwargs):
        gen = func(*args, **kwargs)
        next(gen)
        return gen

    return starter


@coroutine
def lifetimes_sampler(
    nb_samples,
    nb_assets,
    model,
    args: tuple[NDArray[np.float64, ...]] = (),
) -> Coroutine[
    tuple[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64], None
]:
    """
    advantage compared to simple function returning times and lifetimes :
    avoid computing rvs_size and passing nb_samples, nb_assets, model and args each time

    why not using partial function instead ?

    generator
        receiving : times
        yielding : incremented times and lifetimes
    """

    times = None
    lifetimes = None

    if bool(args) and args[0].ndim == 2:
        rvs_size = nb_samples  # rvs size
    else:
        rvs_size = nb_samples * nb_assets

    while True:
        times = (
            yield times,
            lifetimes,
        )  # return times and lifetimes when receiving times
        lifetimes = model.rvs(*args, size=rvs_size).reshape(-1)
        times += lifetimes


@coroutine
def lifetimes_rewards_sampler(
    nb_samples,
    nb_assets,
    end_time,
    model: LifetimeModel,
    reward: Reward,
    discount: Discount,
    model_args: tuple[NDArray[np.float64, ...]] = (),
    reward_args: tuple[NDArray[np.float64, ...]] = (),
    discount_args: tuple[NDArray[np.float64, ...]] = (),
) -> Coroutine[
    tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
    NDArray[np.float64],
    None,
]:  # YieldType, SendType, ReturnType
    """
    generator
        receiving : times
        yielding : incremented times, lifetimes and rewards
    """

    # block A
    times = None
    lifetimes = None
    rewards = None

    lifetimes_generator = lifetimes_sampler(
        nb_samples, nb_assets, end_time, model, model_args
    )

    while True:
        times = yield times, lifetimes, rewards
        times, lifetimes = lifetimes_generator.send(times)
        rewards = (
            np.array(
                reward(
                    lifetimes.reshape(-1, 1),
                    *reward_args,
                ).swapaxes(-2, -1)
                * discount.factor(times, *discount_args),
                ndmin=3,
            )
            .sum(axis=0)
            .ravel()
        )
