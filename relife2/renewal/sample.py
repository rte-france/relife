from dataclasses import dataclass, field, fields, replace
from typing import Optional, Self, Iterator

import numpy as np
from numpy.typing import NDArray

from relife2.model import LifetimeModel
from relife2.renewal.args import ArgsDict, argscheck


# lifetime ~ durations
# time ~ times


@dataclass
class GeneratedData:
    samples: NDArray[np.int64] = field(repr=False)
    assets: NDArray[np.int64] = field(repr=False)
    nb_samples: int = field(init=False)
    nb_assets: int = field(init=False)
    samples_index: int = field(init=False, repr=False)
    assets_index: int = field(init=False, repr=False)

    def __post_init__(self):
        self.samples_index = np.unique(self.samples)
        self.assets_index = np.unique(self.assets)
        self.nb_samples = len(self.samples_index)
        self.nb_assets = len(self.assets_index)

        sorted_index = np.lexsort((self.assets, self.samples))
        for _field in fields(self):
            if _field.init:
                v = getattr(self, _field.name)
                setattr(self, _field.name, v[sorted_index])

    def __len__(self):
        return self.nb_samples * self.nb_assets

    def split(self, sample_only=False, asset_only=False) -> Iterator[Self]:
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
    time: NDArray[np.float64] = field(repr=False)
    lifetime: NDArray[np.float64] = field(repr=False)

    @property
    def nb_events(self) -> int:
        return np.sum(np.unique(self.samples, return_counts=True)[-1])

    @property
    def mean_number_of_events(self) -> float:
        return np.mean(np.unique(self.samples, return_counts=True)[-1])


@dataclass
class LifetimeRewardSample(GeneratedLifetime):
    reward: NDArray[np.float64] = field(repr=False)

    def cumulative_reward(self) -> float:
        return np.insert(self.reward.cumsum(), 0, 0)

    def mean_cumulative_reward(self) -> float:
        return np.insert(self.reward.cumsum(), 0, 0) / self.nb_samples


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


# method of renewal process directly
@argscheck
def sample(
    nb_samples: int,
    nb_assets: int,
    model: LifetimeModel,
    end_time: float,
    *,
    args: Optional[ArgsDict] = None,
    initmodel: Optional[LifetimeModel] = None,
) -> GeneratedLifetime:
    if args is None:
        args = {}

    all_samples, all_assets = np.unravel_index(
        np.arange(nb_samples * nb_assets),
        (nb_samples, nb_assets),
    )

    generator = lifetime_generator(
        model,
        nb_samples,
        nb_assets,
        args,
        initmodel=initmodel,
    )
    spent_time = np.zeros(nb_samples * nb_assets)
    time = np.array([], dtype=np.float64)
    lifetime = np.array([], dtype=np.float64)
    samples = np.array([], dtype=np.int64)
    assets = np.array([], dtype=np.int64)

    still_valid = spent_time < end_time
    while still_valid.any():
        new_lifetime = next(generator).reshape(-1)
        spent_time += new_lifetime
        time = np.concatenate((time, spent_time[still_valid]))
        lifetime = np.concatenate((lifetime, new_lifetime[still_valid]))
        samples = np.concatenate((samples, all_samples[still_valid]))
        assets = np.concatenate((assets, all_assets[still_valid]))
        still_valid = spent_time < end_time

    return GeneratedLifetime(samples, assets, time, lifetime)


#
# class LifetimeRewardSampler(DataSampler[LifetimeModel, LifetimeRewardSample]):
#     def __init__(
#         self,
#         nb_samples: int,
#         nb_assets: int,
#         model: Model,
#         reward: Reward,
#         end_time: float,
#         discount: Discount = ExponentialDiscounting(),
#         *,
#         args: Optional[ArgsDict] = None,
#         initmodel: Optional[Model] = None,
#         initreward: Optional[Reward] = None,
#     ):
#         super().__init__(nb_samples, nb_assets, model, args=args, initmodel=initmodel)
#         self.end_time = end_time
#         self.reward = reward
#         self.discount = discount
#         self.initreward = initreward
#
#     @argscheck
#     def sample(
#         self,
#     ) -> LifetimeRewardSample:
#
#         all_samples_index, all_assets_index = np.unravel_index(
#             np.arange(self.nb_samples * self.nb_assets),
#             (self.nb_samples, self.nb_assets),
#         )
#
#         generator = lifetime_generator(
#             self.model,
#             self.nb_samples,
#             self.nb_assets,
#             self.args,
#             initmodel=self.initmodel,
#         )
#         time = np.zeros(self.nb_samples * self.nb_assets)
#         lifetime = np.array([], dtype=np.float64)
#         reward = np.array([], dtype=np.float64)
#         samples = np.array([], dtype=np.int64)
#         assets = np.array([], dtype=np.int64)
#
#         still_valid = time < self.end_time
#         while still_valid.any():
#             new_lifetime = next(generator).reshape(-1)
#             time[still_valid] += new_lifetime[still_valid]
#
#             # init reward
#             new_reward = (
#                 np.array(
#                     self.reward(
#                         new_lifetime.reshape(-1, 1),
#                         *self.args.get("reward", ()),
#                     ).swapaxes(-2, -1)
#                     * self.discount.factor(time, *self.args.get("discount", ())),
#                     ndmin=3,
#                 )
#                 .sum(axis=0)
#                 .ravel()
#             )
#
#             time = np.c_[time, time[still_valid]]
#             lifetime = np.c_[lifetime, new_lifetime[still_valid]]
#             reward = np.c_[reward, new_reward[still_valid]]
#             samples = np.c_[samples, all_samples_index[still_valid]]
#             assets = np.c_[assets, all_assets_index[still_valid]]
#             still_valid = time < self.end_time
#
#         return LifetimeRewardSample(samples, assets, time, lifetime, reward)
