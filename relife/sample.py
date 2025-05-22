from __future__ import annotations

from abc import ABC
from collections.abc import Iterator
from typing import TYPE_CHECKING, NamedTuple, Optional, Self, TypeVarTuple

import numpy as np
from numpy.lib import recfunctions as rfn
from numpy.typing import ArrayLike, DTypeLike, NDArray
from typing_extensions import override

from relife.economic import ExponentialDiscounting, Reward
from relife.lifetime_model import (
    FrozenAgeReplacementModel,
    FrozenLeftTruncatedModel,
    FrozenLifetimeRegression,
    LifetimeDistribution,
)


if TYPE_CHECKING:
    from relife.stochastic_process.renewal_process import (
        RenewalProcess,
        RenewalRewardProcess,
    )
    from relife.policy import BaseOneCycleAgeReplacementPolicy

Args = TypeVarTuple("Args")


class CountData(NamedTuple):
    t0: float
    tf: float
    struct_array: NDArray[DTypeLike]


def concatenate_count_data(
    iterator: CountDataIterator,
    maxsample: int = 1e5,
) -> CountData:
    struct_array = next(iterator)
    for arr in iterator:
        if len(arr) > maxsample:
            raise RuntimeError("Max number of sample reached")
        struct_array = np.concatenate((struct_array, arr))
    struct_array = np.sort(struct_array, order=("sample_id", "asset_id", "timeline"))
    return CountData(iterator.t0, iterator.tf, struct_array)


class CountDataIterator(Iterator[NDArray[DTypeLike]], ABC):
    timeline: Optional[NDArray[np.float64]]
    start_counter: Optional[NDArray[np.int64]]
    end_counter: Optional[NDArray[np.int64]]

    def __init__(
        self,
        size: int | tuple[int] | tuple[int, int],
        window: tuple[float, float],
        seed: Optional[int] = None,
    ):
        self.size = size
        self.t0, self.tf = window
        self.seed = seed
        self.timeline = None
        self.stop_counter = None
        self.start_counter = None
        self.cycle = 0

    @property
    def stop(self) -> Optional[bool]:
        if self.stop_counter is not None:
            return np.all(self.stop_counter > 0)

    @property
    def just_crossed_t0(self) -> Optional[NDArray[np.bool_]]:
        if self.start_counter is not None:
            return self.start_counter == 1

    @property
    def just_crossed_tf(self) -> Optional[NDArray[np.bool_]]:
        if self.stop_counter is not None:
            return self.stop_counter == 1

    @property
    def selection(self) -> Optional[NDArray[np.bool_]]:
        if self.start_counter is not None and self.stop_counter is not None:
            return np.logical_and(self.start_counter >= 1, self.stop_counter <= 1)

    def base_structarray(self) -> Optional[NDArray[DTypeLike]]:
        selection = self.selection
        if selection is None:
            return None
        asset_id, sample_id = np.where(np.atleast_2d(selection))
        struct_array = np.zeros(
            sample_id.size,
            dtype=np.dtype(
                [
                    ("timeline", np.float64),
                    ("sample_id", np.uint32),  #  unsigned 32bit integer
                    ("asset_id", np.uint32),  #  unsigned 32bit integer
                ]
            ),
        )
        struct_array["sample_id"] = sample_id.astype(np.uint32)
        struct_array["asset_id"] = asset_id.astype(np.uint32)
        struct_array["timeline"] = self.timeline[selection]
        return struct_array


class RenewalProcessIterator(CountDataIterator):

    def __init__(
        self,
        process: RenewalProcess[
            LifetimeDistribution | FrozenLifetimeRegression | FrozenAgeReplacementModel | FrozenLeftTruncatedModel
        ],
        size: int | tuple[int] | tuple[int, int],
        window: tuple[float, float],
        seed: Optional[int] = None,
    ):
        super().__init__(size, window, seed=seed)
        self.process = process

    @property
    def model(
        self,
    ) -> LifetimeDistribution | FrozenLifetimeRegression | FrozenAgeReplacementModel | FrozenLeftTruncatedModel:
        if self.cycle == 0 and self.process.first_lifetime_model is not None:
            return self.process.first_lifetime_model
        return self.process.lifetime_model

    def step(self):
        time, event, entry = self.model.rvs(size=self.size, return_event=True, return_entry=True, seed=self.seed)
        if self.cycle == 0:
            # initialize timeline, stop_counter and start_counter
            self.timeline = np.zeros_like(time)  #  ensure broadcasting
            self.stop_counter = np.zeros_like(self.timeline, dtype=np.int64)
            self.start_counter = np.zeros_like(self.timeline, dtype=np.int64)
        else:  # no model entry after the first cycle
            entry.fill(0.0)

        # update timeline
        self.timeline += time

        # update start and stop counter
        self.start_counter[self.timeline > self.t0] += 1
        self.stop_counter[self.timeline > self.tf] += 1

        # tf right censorings
        time = np.where(self.just_crossed_tf, time - (self.timeline - self.tf), time)
        self.timeline[self.just_crossed_tf] = self.tf
        event[self.just_crossed_tf] = False

        # t0 entry (entry is 0 after the first cycle)
        entry = np.where(self.just_crossed_t0, self.t0 + entry, entry)

        # update seed to avoid having the same rvs result
        if self.seed is not None:
            self.seed += 1
        # update cycle
        self.cycle += 1

        struct_arr = rfn.append_fields(  #  works on structured_array too
            self.base_structarray(),  # add to struct of timeline, sample_id, asset_id
            ("time", "event", "entry"),
            (time[self.selection], event[self.selection], entry[self.selection]),
            (np.float64, np.bool_, np.float64),
            usemask=False,
            asrecarray=False,
        )
        return struct_arr

    def __next__(self) -> NDArray[DTypeLike]:
        while not self.stop:
            struct_arr = self.step()
            while struct_arr.size == 0 and not self.stop:  # skip cycles while arrays are empty (if t0 != 0.)
                struct_arr = self.step()
            return struct_arr
        raise StopIteration


class RenewalRewardProcessIterator(RenewalProcessIterator):
    reward: Reward
    discounting: ExponentialDiscounting

    def __init__(
        self,
        process: RenewalRewardProcess[
            LifetimeDistribution | FrozenLifetimeRegression | FrozenAgeReplacementModel | FrozenLeftTruncatedModel,
            Reward,
        ],
        size: int | tuple[int] | tuple[int, int],
        window: tuple[float, float],
        seed: Optional[int] = None,
    ):
        super().__init__(process, size, window, seed)
        self.reward = self.process.reward
        self.discounting = self.process.discounting

    @override
    def __next__(self) -> NDArray[DTypeLike]:
        for struct_array in super().__iter__():
            return rfn.append_fields(
                struct_array,
                "reward",
                self.reward.sample(struct_array["time"]) * self.discounting.factor(struct_array["timeline"]),
                np.float64,
                usemask=False,
                asrecarray=False,
            )


# class NonHomogeneousPoissonProcessIterator(SampleIterator):
#
#     rewards: Optional[Reward]
#     discounting: Optional[Discounting]
#     hpp_timeline: Optional[NDArray[np.float64]]
#     failure_times: Optional[NDArray[np.float64]]
#     ages: Optional[NDArray[np.float64]]
#     ar: Optional[NDArray[np.float64]]
#     is_new_asset: Optional[NDArray[np.bool_]]
#     entries: Optional[NDArray[np.float64]]
#     renewals_ids: Optional[NDArray[np.int64]]
#
#     def __init__(
#         self,
#         size: int,
#         tf: float,  # calendar end time
#         t0: float = 0.0,  # calendar beginning time
#         *,
#         seed: Optional[int] = None,
#         keep_last: bool = True,
#     ):
#         super().__init__(size, tf, t0, seed=seed, keep_last=keep_last)
#
#         self.rewards = None
#         self.discounting = None
#
#         self.hpp_timeline = None  # exposed attribute (set/get)
#         self.failure_times = None
#         self.ages = None
#         # self._assets_ids = None
#         self.ar = None
#         self.is_new_asset = None
#         self.entries = None
#         self.renewals_ids = None
#         self.exponential_dist = Exponential(1.0)
#
#     def compute_rewards(
#         self, timeline: NDArray[np.float64], ages: NDArray[np.float64]
#     ) -> NDArray[np.float64]:
#         rewards = np.zeros_like(ages)
#         if self.rewards and self.discounting:
#             rewards = self.rewards(ages) * self.discounting.factor(timeline)
#         if self.rewards and not self.discounting:
#             rewards = self.rewards(ages)
#         return rewards
#
#     def set_model(
#         self,
#         model: FrozenParametricLifetimeModel,
#         ar: Optional[NDArray[np.float64]] = None,
#     ) -> None:
#
#         if self.model is None:
#             # self._nb_assets = get_nb_assets(model_args)
#             self.timeline = np.zeros((model.args_nb_assets, self.size))
#             # counting arrays to catch values crossing t0 and tf bounds
#             self.stop_counter = np.zeros((model.args_nb_assets, self.size), dtype=np.int64)
#             self.start_counter = np.zeros((model.args_nb_assets, self.size), dtype=np.int64)
#
#             self.hpp_timeline = np.zeros((model.args_nb_assets, self.size))
#             self.failure_times = np.zeros((model.args_nb_assets, self.size))
#             self.ages = np.zeros((model.args_nb_assets, self.size))
#             self.entries = np.zeros((model.args_nb_assets, self.size))
#             self.is_new_asset = np.zeros((model.args_nb_assets, self.size), dtype=np.bool_)
#             self.renewals_ids = np.zeros((model.args_nb_assets, self.size), dtype=np.int64)
#
#         self.model = model
#         self.ar = (
#             ar if ar is not None else (np.ones(self.model) * np.inf).reshape(-1, 1)
#         )
#
#     def step(self) -> dict[str, AnyNDArray]:
#
#         # reset those who are replaced
#         self.ages[self.is_new_asset] = 0.0  # asset is replaced (0 aged asset)
#         # self._assets_ids[self.is_new_asset] += 1 # asset is replaced (new asset id)
#         self.hpp_timeline[self.is_new_asset] = 0.0  # reset timeline
#         self.failure_times[self.is_new_asset] = 0.0
#         self.entries[self.is_new_asset] = 0.0
#         self.renewals_ids[self.is_new_asset] += 1
#         self.is_new_asset.fill(False)  # reset to False
#
#         # generate new values
#         self.hpp_timeline += self.exponential_dist.rvs(
#             size=self.size * self.model.args_nb_assets, seed=self.seed
#         ).reshape((self.model.args_nb_assets, self.size))
#
#         failure_times = self.model.ichf(self.hpp_timeline)
#         durations = failure_times - self.failure_times  # t_i+1 - t_i
#         self.failure_times = failure_times.copy()  # update t_i <- t_i+1
#         self.timeline += durations
#         self.ages += durations
#
#         # create array of events_indicators
#         events_indicators = np.ones_like(self.ages, np.bool_)
#
#         # ar update (before because it changes timeline, thus start and stop conditions)
#         self.timeline = np.where(
#             self.ages >= self.ar,
#             self.timeline - (self.ages - np.ones_like(self.timeline) * self.ar),
#             self.timeline,
#         )  # substract time after ar
#         self.ages = np.where(
#             self.ages >= self.ar, np.ones_like(self.ages) * self.ar, self.ages
#         )  # set ages to ar
#         self.is_new_asset[
#             np.logical_and(self.ages >= self.ar, ~self.just_crossed_tf)
#         ] = True
#         events_indicators[
#             np.logical_and(self.ages >= self.ar, ~self.just_crossed_tf)
#         ] = False
#
#         # update stop conditions
#         self.start_counter[self.timeline > self.t0] += 1
#         self.stop_counter[self.timeline > self.tf] += 1
#
#         # t0 entries update
#         self.entries = np.where(
#             self.just_crossed_t0, self.ages - (self.timeline - self.t0), self.entries
#         )
#
#         # tf update censoring update
#         self.ages = np.where(
#             self.just_crossed_tf, self.ages - (self.timeline - self.tf), self.ages
#         )
#         self.timeline[self.just_crossed_tf] = self.tf
#         events_indicators[self.just_crossed_tf] = False
#
#         # returned entries
#         entries = self.entries.copy()
#         # update for next iteration (keep previous ages)
#         self.entries = np.where(
#             events_indicators, self.ages, self.entries
#         )  # keep previous ages as entry for next iteration
#
#         # update seed to avoid having the same rvs result
#         if self.seed is not None:
#             self.seed += 1
#
#         rewards = self.compute_rewards(self.timeline, self.ages)
#
#         return self.construct_sample_data1d(
#             ages=self.ages,
#             renewals_ids=self.renewals_ids,
#             entries=entries,
#             events_indicators=events_indicators,
#             rewards=rewards,
#         )


def _nb_events(selection: CountDataFunctions) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    from relife.policy import BaseOneCycleAgeReplacementPolicy
    from relife.stochastic_process import RenewalProcess

    if not issubclass(selection.obj_type, RenewalProcess) or not issubclass(
        selection.obj_type, BaseOneCycleAgeReplacementPolicy
    ):
        raise ValueError(f"{selection.obj_type} object type does not have nb_events from sample")

    sort = np.argsort(selection.count_data.struct_array["timeline"])
    timeline = selection.count_data.struct_array["timeline"][sort]
    counts = np.ones_like(timeline)

    timeline = np.insert(timeline, 0, selection.t0)
    counts = np.insert(counts, 0, 0)
    counts[timeline == selection.tf] = 0
    return timeline, np.cumsum(counts)


def _total_rewards(selection: CountDataFunctions) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    from relife.policy import BaseOneCycleAgeReplacementPolicy
    from relife.stochastic_process import RenewalRewardProcess

    if not issubclass(selection.obj_type, RenewalRewardProcess) or not issubclass(
        selection.obj_type, BaseOneCycleAgeReplacementPolicy
    ):
        raise ValueError(f"{selection.obj_type} object type does not have nb_events from sample")

    sort = np.argsort(selection.count_data.struct_array["timeline"])
    timeline = selection.count_data.struct_array["timeline"][sort]
    reward = selection.count_data.struct_array["reward"][sort]

    timeline = np.insert(timeline, 0, selection.t0)
    reward = np.insert(reward, 0, 0)
    reward[timeline == selection.tf] = 0

    return timeline, np.cumsum(reward)


class CountDataFunctions:

    count_data: CountData

    def __init__(self, obj_type: type[RenewalProcess] | type[BaseOneCycleAgeReplacementPolicy], count_data: CountData):
        self.obj_type = obj_type
        self.count_data = count_data

    @property
    def t0(self) -> float:
        return self.count_data.t0

    @property
    def tf(self) -> float:
        return self.count_data.tf

    def nb_events(self, sample_id: int, asset_id: int = 0) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        selection = self.select(sample_id=sample_id, asset_id=asset_id)
        return _nb_events(selection)

    def mean_nb_events(self, asset_id: int = 0) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        selection = self.select(asset_id=asset_id)
        timeline, counts = _nb_events(selection)
        nb_sample = len(np.unique(selection.count_data.struct_array["sample_id"]))
        return timeline, counts / nb_sample

    def total_rewards(self, sample_id: int, asset_id: int = 0) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        selection = self.select(sample_id=sample_id, asset_id=asset_id)
        return _total_rewards(selection)

    def mean_total_rewards(self, asset_id: int = 0) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        selection = self.select(asset_id=asset_id)
        timeline, rewards = _nb_events(selection)
        nb_sample = len(np.unique(selection.count_data.struct_array["sample_id"]))
        return timeline, rewards / nb_sample

    def select(self, sample_id: Optional[ArrayLike] = None, asset_id: Optional[ArrayLike] = None) -> Self:
        mask = np.ones(len(self.count_data.struct_array), dtype=np.bool_)
        if sample_id is not None:
            mask = mask & np.isin(self.count_data.struct_array["sample_id"], sample_id)
        if asset_id is not None:
            mask = mask & np.isin(self.count_data.struct_array["asset_id"], asset_id)
        struct_subarray = self.count_data.struct_array[mask].copy()
        return CountDataFunctions(self.obj_type, CountData(self.t0, self.tf, struct_subarray))
