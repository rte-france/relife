from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field, fields
from typing import Generic, Optional, TypeVar

import numpy as np
from numpy.typing import NDArray

from relife2.discount import Discount, ExponentialDiscounting
from relife2.model import LifetimeModel, ParametricLifetimeModel, ParametricModel
from relife2.renewal.args import ArgsDict, argscheck
from relife2.reward import Reward


# lifetime ~ durations
# time ~ times


@dataclass
class SampleData:
    samples: NDArray[np.int64] = field(repr=False)
    assets: NDArray[np.int64] = field(repr=False)
    nb_samples: int = field(init=False)
    nb_assets: int = field(init=False)
    samples_index: int = field(init=False, repr=False)
    assets_index: int = field(init=False, repr=False)

    def __len__(self):
        return self.nb_samples * self.nb_assets

    @property
    def data(self):
        for _field in fields(self):
            if _field.name not in (
                "samples",
                "assets",
                "nb_samples",
                "nb_assets",
                "samples_index",
                "assets_index",
            ):
                yield _field.name, getattr(self, _field.name)

    @property
    def split_by_samples(self):
        split_pos = np.where(self.samples[:-1] != self.samples[1:])[0] + 1
        for k, v in self.data:
            yield k, np.split(v, split_pos)

    @property
    def split_by_assets(self):
        if self.nb_samples != 1:
            raise AttributeError(
                f"{self.__class__.__name__} can't be splitted by assets because nb samples is greater than 1"
            )
        split_pos = np.where(self.assets[:-1] != self.assets[1:])[0] + 1
        for k, v in self.data:
            yield k, np.split(v, split_pos)

    def __post_init__(self):
        self.samples_index = np.unique(self.samples)
        self.assets_index = np.unique(self.assets)
        self.nb_samples = len(self.samples_index)
        self.nb_assets = len(self.assets_index)

        sorted_index = np.lexsort((self.assets, self.samples))
        self.samples = self.samples[sorted_index]
        self.assets = self.assets[sorted_index]
        for k, v in self.data:
            setattr(self, k, v[sorted_index])


@dataclass
class LifetimeSample(SampleData):
    time: NDArray[np.float64] = field(repr=False)
    lifetime: NDArray[np.float64] = field(repr=False)

    @property
    def nb_events(self) -> int:
        return np.sum(np.unique(self.samples, return_counts=True)[-1])

    @property
    def mean_number_of_events(self) -> float:
        return np.mean(np.unique(self.samples, return_counts=True)[-1])


@dataclass
class LifetimeRewardSample(LifetimeSample):
    reward: NDArray[np.float64] = field(repr=False)

    def cumulative_reward(self) -> float:
        return np.insert(self.reward.cumsum(), 0, 0)

    def mean_cumulative_reward(self) -> float:
        return np.insert(self.reward.cumsum(), 0, 0) / self.nb_samples


Model = TypeVar("Model", LifetimeModel, ParametricModel, ParametricLifetimeModel)
Data = TypeVar("Data", bound=SampleData)


class DataSampler(ABC, Generic[Model, Data]):
    def __init__(
        self,
        nb_samples: int,
        nb_assets: int,
        model: Model,
        *,
        args: Optional[ArgsDict] = None,
        initmodel: Optional[Model] = None,
    ):
        """
        nb_assets is mandotory to control all model's information coherence
        # and allow model with ndim 2 args and initmodel without args
        # see how it is used in lifetime_rvs
        """

        self.nb_assets = nb_assets
        self.nb_samples = nb_samples
        self.model = model
        if args is None:
            args = {}
        self.args = args
        self.initmodel = initmodel
        self.container = self.sample()

    def __len__(self):
        if self.nb_samples is None:
            return 0
        if self.nb_assets == 1:
            return self.nb_samples
        else:
            return self.nb_assets

    def __getattr__(self, item):
        if hasattr(self.container, item):
            return getattr(self.container, item)
        raise AttributeError(f"{self.__class__.__name__} has no attribute {item}")

    def __getitem__(
        self, key: int | slice | tuple[int, int] | tuple[slice, slice]
    ) -> Data:
        if not isinstance(key, tuple):
            key = (key, None)
        if len(key) > 2:
            raise IndexError(
                f"{self.__class__.__name__} getter has a maximum of 2 index (samples and assets) but got {len(key)}"
            )

        if isinstance(key[0], slice):
            samples_mask = np.isin(
                self.container.samples,
                range(
                    key[0].start if key[0].start else 0,
                    key[0].stop if key[0].stop else self.container.nb_samples,
                    key[0].step if key[0].step else 1,
                ),
            )
        else:
            if key[0] not in self.container.samples_index:
                raise IndexError(
                    f"index {key[0]} is out of bounds for {self.__class__.__name__} with {self.container.nb_samples} nb samples"
                )
            samples_mask = self.container.samples == key[0]
        values = (v[samples_mask] for _, v in self.container.data)
        samples = self.container.samples[samples_mask]
        assets = self.container.assets[samples_mask]

        if key[1] is not None:
            if isinstance(key[0], slice):
                assets_mask = np.isin(
                    assets,
                    range(
                        key[1].start if key[1].start else 0,
                        key[1].stop if key[1].stop else self.container.nb_assets,
                        key[1].step if key[1].step else 1,
                    ),
                )
            else:
                if key[1] not in self.container.assets_index:
                    raise IndexError(
                        f"index {key[1]} is out of bounds for {self.__class__.__name__} with {self.container.nb_assets} nb assets"
                    )
                assets_mask = assets == key[1]

            values = (v[assets_mask] for v in values)
            samples = samples[assets_mask]
            assets = assets[assets_mask]

        return type(self.container)(samples, assets, *tuple(values))

    @abstractmethod
    def sample(
        self,
    ) -> Data: ...


def lifetime_rvs(
    model: LifetimeModel,
    nb_samples: int,
    nb_assets: int,
    args: tuple[NDArray[np.float64], ...],
) -> Iterator[NDArray[np.float64]]:
    if bool(args) and args[0].ndim == 2:
        rvs_size = nb_samples  # rvs size
    else:
        rvs_size = nb_samples * nb_assets
    yield model.rvs(*args, size=rvs_size)


def lifetime_generator(
    model,
    nb_samples,
    nb_assets,
    args: ArgsDict,
    initmodel=None,
) -> Iterator[NDArray[np.float64]]:
    if initmodel is not None:
        yield from lifetime_rvs(
            initmodel, nb_samples, nb_assets, args.get("initmodel", ())
        )
    while True:
        yield from lifetime_rvs(model, nb_samples, nb_assets, args.get("model", ()))


class LifetimeSampler(DataSampler[LifetimeModel, LifetimeSample]):

    def __init__(
        self,
        nb_samples: int,
        nb_assets: int,
        model: Model,
        end_time: float,
        *,
        args: Optional[ArgsDict] = None,
        initmodel: Optional[Model] = None,
    ):
        self.end_time = end_time
        super().__init__(nb_samples, nb_assets, model, args=args, initmodel=initmodel)

    def sample(self) -> LifetimeSample:

        all_samples, all_assets = np.unravel_index(
            np.arange(self.nb_samples * self.nb_assets),
            (self.nb_samples, self.nb_assets),
        )

        generator = lifetime_generator(
            self.model,
            self.nb_samples,
            self.nb_assets,
            self.args,
            initmodel=self.initmodel,
        )
        spent_time = np.zeros(self.nb_samples * self.nb_assets)
        time = np.array([], dtype=np.float64)
        lifetime = np.array([], dtype=np.float64)
        samples = np.array([], dtype=np.int64)
        assets = np.array([], dtype=np.int64)

        still_valid = spent_time < self.end_time
        while still_valid.any():
            new_lifetime = next(generator).reshape(-1)
            spent_time += new_lifetime
            time = np.concatenate((time, spent_time[still_valid]))
            lifetime = np.concatenate((lifetime, new_lifetime[still_valid]))
            samples = np.concatenate((samples, all_samples[still_valid]))
            assets = np.concatenate((assets, all_assets[still_valid]))
            still_valid = spent_time < self.end_time

        return LifetimeSample(samples, assets, time, lifetime)


class LifetimeRewardSampler(DataSampler[LifetimeModel, LifetimeRewardSample]):
    def __init__(
        self,
        nb_samples: int,
        nb_assets: int,
        model: Model,
        reward: Reward,
        end_time: float,
        discount: Discount = ExponentialDiscounting(),
        *,
        args: Optional[ArgsDict] = None,
        initmodel: Optional[Model] = None,
        initreward: Optional[Reward] = None,
    ):
        super().__init__(nb_samples, nb_assets, model, args=args, initmodel=initmodel)
        self.end_time = end_time
        self.reward = reward
        self.discount = discount
        self.initreward = initreward

    @argscheck
    def sample(
        self,
    ) -> LifetimeRewardSample:

        all_samples_index, all_assets_index = np.unravel_index(
            np.arange(self.nb_samples * self.nb_assets),
            (self.nb_samples, self.nb_assets),
        )

        generator = lifetime_generator(
            self.model,
            self.nb_samples,
            self.nb_assets,
            self.args,
            initmodel=self.initmodel,
        )
        time = np.zeros(self.nb_samples * self.nb_assets)
        lifetime = np.array([], dtype=np.float64)
        reward = np.array([], dtype=np.float64)
        samples = np.array([], dtype=np.int64)
        assets = np.array([], dtype=np.int64)

        still_valid = time < self.end_time
        while still_valid.any():
            new_lifetime = next(generator).reshape(-1)
            time[still_valid] += new_lifetime[still_valid]

            # init reward
            new_reward = (
                np.array(
                    self.reward(
                        new_lifetime.reshape(-1, 1),
                        *self.args.get("reward", ()),
                    ).swapaxes(-2, -1)
                    * self.discount.factor(time, *self.args.get("discount", ())),
                    ndmin=3,
                )
                .sum(axis=0)
                .ravel()
            )

            time = np.c_[time, time[still_valid]]
            lifetime = np.c_[lifetime, new_lifetime[still_valid]]
            reward = np.c_[reward, new_reward[still_valid]]
            samples = np.c_[samples, all_samples_index[still_valid]]
            assets = np.c_[assets, all_assets_index[still_valid]]
            still_valid = time < self.end_time

        return LifetimeRewardSample(samples, assets, time, lifetime, reward)
