from __future__ import annotations

from abc import ABC
from collections.abc import Iterator
from typing import TYPE_CHECKING, Optional, TypeVarTuple

import numpy as np
from numpy.typing import DTypeLike
from numpy.lib import recfunctions as rfn
from numpy.typing import NDArray
from typing_extensions import override

from relife.lifetime_model import ParametricLifetimeModel
from relife.sample import sample_lifetime_data

if TYPE_CHECKING:
    from relife.economic import Discounting, Reward, ExponentialDiscounting


Args = TypeVarTuple("Args")


class CountDataIterator(Iterator[NDArray[DTypeLike]], ABC):
    timeline : Optional[NDArray[np.float64]]
    start_counter: Optional[NDArray[np.int64]]
    end_counter: Optional[NDArray[np.int64]]

    def __init__(
        self,
        size: int | tuple[int]| tuple[int, int],
        tf: float,  # calendar end time
        t0: float = 0.0,  # calendar beginning time
        seed: Optional[int] = None,
        maxsample : 1e5 = int,
    ):
        self.size = size
        self.tf = tf
        self.t0 = t0
        self.seed = seed
        self.nb_assets = 1
        self.timeline = np.zeros(size)
        self.stop_counter = np.zeros(size, dtype=np.int64)
        self.start_counter = np.zeros(size, dtype=np.int64)
        self.maxsample = maxsample
        self.n = 0
        self.nb_sampled = 0

    @property
    def stop(self) -> Optional[bool]:
        return np.all(self.stop_counter > 0)

    @property
    def just_crossed_t0(self) -> NDArray[np.bool_]:
        return self.start_counter == 1

    @property
    def just_crossed_tf(self) -> NDArray[np.bool_]:
        return self.stop_counter == 1

    @property
    def selection(self) -> NDArray[np.bool_]:
        return np.logical_and(self.start_counter >= 1, self.stop_counter <= 1)

    def base_structarray(self) -> NDArray[DTypeLike]:
        selection = self.selection.copy()
        asset_id, sample_id = np.where(selection)
        struct_array = np.zeros(
            sample_id.size,
            dtype=np.dtype([
                ("timeline", np.float64),
                ("sample_id", np.uint32),  #  unsigned 32bit integer
                ("asset_id", np.uint32),  #  unsigned 32bit integer
            ])
        )
        struct_array["sample_id"] = sample_id.astype(np.uint32)
        struct_array["asset_id"] = asset_id.astype(np.uint32)
        struct_array["timeline"] = self.timeline[selection]
        return struct_array


class RenewalProcessIterator(CountDataIterator):

    model: ParametricLifetimeModel[()]
    model1: Optional[ParametricLifetimeModel[()]]
    a0: Optional[NDArray[np.float64]]
    ar: Optional[NDArray[np.float64]]

    def __init__(
        self,
        size: int|tuple[int]|tuple[int,int],
        tf: float,  # calendar end time
        model : ParametricLifetimeModel[()],
        t0: float = 0.0,  # calendar start time
        model1 : Optional[ParametricLifetimeModel[()]] = None,
        maxsample : int = 1e5,
        seed: Optional[int] = None,
    ):
        super().__init__(size, tf, t0, maxsample=maxsample, seed=seed)
        self._model_queue = [model, model1]

    @property
    def model(self) -> ParametricLifetimeModel[()]:
        if len(self._model_queue) == 2:
            if self._model_queue[1] is not None:
                return self._model_queue.pop()
            self._model_queue.pop()
        return self._model_queue[0]

    def __next__(self) -> NDArray[DTypeLike]:
        if not self.stop:
            time, event, entry = sample_lifetime_data(self.model, size=self.size, seed=self.seed)
            if self.n == 0:
                self.timeline = np.zeros_like(time) # ensure broadcasting
                self.stop_counter = np.zeros_like(self.timeline, dtype=np.int64)
                self.start_counter = np.zeros_like(self.timeline, dtype=np.int64)
            else:
                entry = np.zeros_like(entry)

            # update timeline
            self.timeline += time

            # update start and stop counter
            self.start_counter[self.timeline > self.t0] += 1
            self.stop_counter[self.timeline > self.tf] += 1

            # tf right censorings
            time = np.where(
                self.just_crossed_tf, time - (self.timeline - self.tf), time
            )
            self.timeline[self.just_crossed_tf] = self.tf
            event[self.just_crossed_tf] = False

            # t0 left truncations, only applied on time not being truncated by model
            entry = np.where(
                self.just_crossed_t0, self.t0 + entry - (self.timeline - time), entry
            )

            # update seed to avoid having the same rvs result
            if self.seed is not None:
                self.seed += 1
            self.n += 1

            struct_arr = rfn.append_fields( # works on structured_array too
                self.base_structarray(),
                ("time", "event", "entry"),
                (time[self.selection], event[self.selection], entry[self.selection]),
                (np.float64, np.bool_, np.float64),
                usemask=False,
                asrecarray=False,
            )
            self.nb_sampled += len(struct_arr)
            if self.nb_sampled > self.maxsample:
                raise RuntimeError("Max number of sample reached")
            return struct_arr

class RenewalRewardProcessIterator(RenewalProcessIterator):
    reward: Reward
    discounting: Discounting

    def __init__(
        self,
        nb_sample: int,
        tf: float,  # calendar end time
        model : ParametricLifetimeModel[()],
        reward : Reward,
        t0: float = 0.0,  # calendar start time
        model1 : Optional[ParametricLifetimeModel[()]] = None,
        discounting_rate : Optional[float] = None,
        maxsample : int = 1e5,
        seed: Optional[int] = None,
    ):
        super().__init__(nb_sample, tf, model, t0=t0, model1=model1, maxsample=maxsample, seed=seed)
        self.reward = reward
        self.discounting = ExponentialDiscounting(discounting_rate)

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
#
#
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
