from abc import ABC
from collections.abc import Iterator
from typing import (
    Iterable,
    Optional,
    TypeVarTuple,
)

import numpy as np
from numpy.lib import recfunctions as rfn
from numpy.typing import NDArray
from typing_extensions import override

from relife.economic import ExponentialDiscounting, Reward
from relife.lifetime_model import (
    FrozenAgeReplacementModel,
    FrozenLeftTruncatedModel,
    FrozenLifetimeRegression,
    LifetimeDistribution,
)

from .renewal_process import RenewalProcess, RenewalRewardProcess

Args = TypeVarTuple("Args")


class CountDataIterator(Iterator[NDArray[np.void]], ABC):
    timeline: Optional[NDArray[np.float64]]
    start_counter: Optional[NDArray[np.uint32]]
    end_counter: Optional[NDArray[np.uint32]]

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
    def stop(self) -> bool:
        if self.stop_counter is not None:
            return np.all(self.stop_counter > 0)
        return False

    @property
    def just_crossed_t0(self) -> Optional[NDArray[np.bool_]]:
        if self.start_counter is not None:
            return self.start_counter == 1
        return None

    @property
    def just_crossed_tf(self) -> Optional[NDArray[np.bool_]]:
        if self.stop_counter is not None:
            return self.stop_counter == 1
        return None

    @property
    def selection(self) -> Optional[NDArray[np.bool_]]:
        if self.start_counter is not None and self.stop_counter is not None:
            return np.logical_and(self.start_counter >= 1, self.stop_counter <= 1)
        return None

    def base_structarray(self) -> Optional[NDArray[np.void]]:
        selection = self.selection
        if selection is None:
            return None
        asset_id, sample_id = np.where(np.atleast_2d(selection))
        struct_array = np.zeros(
            sample_id.size,
            dtype=np.dtype(
                [
                    ("sample_id", np.uint32),  #  unsigned 32bit integer
                    ("asset_id", np.uint32),  #  unsigned 32bit integer
                ]
            ),
        )
        struct_array["sample_id"] = sample_id.astype(np.uint32)
        struct_array["asset_id"] = asset_id.astype(np.uint32)
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
        # time is not residual age if return_entry is True (see LeftTruncatedModel)
        time, event, model_entry = self.model.rvs(size=self.size, return_event=True, return_entry=True, seed=self.seed)
        if self.cycle == 0:
            # initialize timeline, stop_counter and start_counter
            self.timeline = np.zeros_like(time, dtype=np.float64)  # ensure broadcasting
            self.stop_counter = np.zeros_like(self.timeline, dtype=np.uint32)
            self.start_counter = np.zeros_like(self.timeline, dtype=np.uint32)
        else:
            model_entry.fill(0.0)  # cancel any model entry after the first cycle

        # update timeline
        self.timeline += time

        # update start and stop counter
        self.start_counter[self.timeline > self.t0] += 1
        self.stop_counter[self.timeline > self.tf] += 1

        # tf right censorings
        # censored time : self.timeline - self.tf
        # observed time : time - censored time
        time = np.where(self.just_crossed_tf, time - (self.timeline - self.tf), time)
        self.timeline[self.just_crossed_tf] = self.tf
        event[self.just_crossed_tf] = False

        entry = np.zeros_like(model_entry)
        if self.cycle == 0:  # t0 added to model_entry
            entry = np.where(self.just_crossed_t0, self.t0 + model_entry, entry)
        else:
            # previous timeline step = self.timeline - time
            entry = np.where(self.just_crossed_t0, self.t0 - (self.timeline - time), entry)

        # update seed to avoid having the same rvs result
        if self.seed is not None:
            self.seed += 1
        # update cycle
        self.cycle += 1
        base_structarray = self.base_structarray()
        nb_renewal = np.full_like(self.timeline, self.cycle, dtype=np.uint32)

        struct_arr = rfn.append_fields(  #  works on structured_array too
            base_structarray,  # add to struct of timeline, sample_id, asset_id
            ("timeline", "time", "event", "entry", "nb_renewal"),
            (
                self.timeline[self.selection],
                time[self.selection],
                event[self.selection],
                entry[self.selection],
                nb_renewal[self.selection],
            ),
            (np.float64, np.float64, np.bool_, np.float64, np.uint32),
            usemask=False,
            asrecarray=False,
        )
        return struct_arr

    def __next__(self) -> NDArray[np.void]:
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
    def __next__(self) -> NDArray[np.void]:
        struct_array = super().__next__()
        # may be type hint error in rfn.append_fields overload
        return rfn.append_fields(
            struct_array,
            "reward",
            self.reward.sample(struct_array["time"]) * self.discounting.factor(struct_array["timeline"]),
            np.float64,
            usemask=False,
            asrecarray=False,
        )  # type: ignore


class RenewalProcessIterable(Iterable):

    def __init__(
        self,
        process: RenewalProcess[
            LifetimeDistribution | FrozenLifetimeRegression | FrozenAgeReplacementModel | FrozenLeftTruncatedModel
        ],
        size: int | tuple[int] | tuple[int, int],
        window: tuple[float, float],
        seed: Optional[int] = None,
    ):
        self.process = process
        self.size = size
        self.window = window
        self.seed = seed

    def __iter__(self) -> RenewalProcessIterator:
        if isinstance(self.process, RenewalProcess):
            return RenewalProcessIterator(self.process, self.size, self.window, self.seed)
        else:
            return RenewalRewardProcessIterator(self.process, self.size, self.window, self.seed)


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


#
# @sample_count_data.register
# def _(
#     obj: Union[
#         NonHomogeneousPoissonProcess,
#         NonHomogeneousPoissonAgeReplacementPolicy,
#     ],
#     size: int,
#     tf: float,
#     t0: float = 0.0,
#     maxsample: int = 1e5,
#     seed: Optional[int] = None,
# ):
#     keys = (
#         "timeline",
#         "ages",
#         "events_indicators",
#         "samples_ids",
#         "assets_ids",
#         "rewards",
#     )
#     iterator = NonHomogeneousPoissonProcessIterator(size, tf, t0=t0, seed=seed)
#     iterator.set_model(obj.model, ar=getattr(obj, "ar", None))
#     if isinstance(
#         obj,
#         NonHomogeneousPoissonAgeReplacementPolicy,
#     ):
#         iterator.rewards = age_replacement_rewards(obj.ar, obj.cr, obj.cp)
#         iterator.discounting = obj.discounting
#     stack = stack1d(iterator, keys, maxsample=maxsample)
#
#     return NHPPCountData(t0, tf, **stack)


# @failure_data_sample.register
# def _(
#     obj: Union[
#         NonHomogeneousPoissonProcess,
#         NonHomogeneousPoissonAgeReplacementPolicy,
#     ],
#     size: int,
#     tf: float,
#     t0: float = 0.0,
#     maxsample: int = 1e5,
#     seed: Optional[int] = None,
#     use: str = "model",
# ):
#     keys = (
#         "timeline",
#         "samples_ids",
#         "assets_ids",
#         "ages",
#         "entries",
#         "events_indicators",
#         "renewals_ids",
#     )
#
#     if use != "model":
#         raise ValueError("Invalid 'use' value. Only 'model' can be set")
#
#     iterator = NonHomogeneousPoissonProcessIterator(size, tf, t0=t0, seed=seed, keep_last=True)
#     iterator.set_model(obj.model, ar=obj.ar if hasattr(obj, "ar") else None)
#
#     stack = stack1d(iterator, keys, maxsample=maxsample)
#
#     str_samples_ids = np.char.add(
#         np.full_like(stack["samples_ids"], "S", dtype=np.str_),
#         stack["samples_ids"].astype(np.str_),
#     )
#     str_assets_ids = np.char.add(
#         np.full_like(stack["assets_ids"], "A", dtype=np.str_),
#         stack["assets_ids"].astype(np.str_),
#     )
#     str_renewals_ids = np.char.add(
#         np.full_like(stack["assets_ids"], "R", dtype=np.str_),
#         stack["renewals_ids"].astype(np.str_),
#     )
#     assets_ids = np.char.add(str_samples_ids, str_assets_ids)
#     assets_ids = np.char.add(assets_ids, str_renewals_ids)
#
#     sort_ind = np.lexsort((stack["timeline"], assets_ids))
#
#     entries = stack["entries"][sort_ind]
#     events_indicators = stack["events_indicators"][sort_ind]
#     ages = stack["ages"][sort_ind]
#     assets_ids = assets_ids[sort_ind]
#
#     # print("assets_ids", assets_ids)
#     # print("timeline", timeline)
#     # print("ages", ages)
#     # print("events_indicators", events_indicators)
#     # print("entries", entries)
#
#     first_ages_index = np.roll(assets_ids, 1) != assets_ids
#     last_ages_index = np.roll(first_ages_index, -1)
#
#     immediatly_replaced = np.logical_and(~events_indicators, first_ages_index)
#
#     # print("first_ages_index", first_ages_index)
#     # print("last_ages_index", last_ages_index)
#     # print("immediatly_replaced", immediatly_replaced)
#
#     # prefix = np.full_like(assets_ids[immediatly_replaced], "Z", dtype=np.str_)
#     # _assets_ids = np.char.add(prefix, assets_ids[immediatly_replaced])
#     _assets_ids = assets_ids[immediatly_replaced]
#     first_ages = entries[immediatly_replaced].copy()
#     last_ages = ages[immediatly_replaced].copy()
#
#     # print("assets_ids", _assets_ids)
#     # print("first_ages", first_ages)
#     # print("last_ages", last_ages)
#
#     events_assets_ids = assets_ids[events_indicators]
#     events_ages = ages[events_indicators]
#     other_assets_ids = np.unique(events_assets_ids)
#     _assets_ids = np.concatenate((_assets_ids, other_assets_ids))
#     first_ages = np.concatenate(
#         (first_ages, entries[first_ages_index & events_indicators])
#     )
#     last_ages = np.concatenate(
#         (last_ages, ages[last_ages_index & ~immediatly_replaced])
#     )
#
#     # print("events_assets_ids", events_assets_ids)
#     # print("events_ages", events_ages)
#     # print("assets_ids", _assets_ids)
#     # print("first_ages", first_ages)
#     # print("last_ages", last_ages)
#
#     # last sort (optional but convenient to control data)
#     sort_ind = np.argsort(events_assets_ids)
#     events_assets_ids = events_assets_ids[sort_ind]
#
#     sort_ind = np.argsort(_assets_ids)
#     _assets_ids = _assets_ids[sort_ind]
#     first_ages = first_ages[sort_ind]
#     last_ages = last_ages[sort_ind]
#
#     return events_assets_ids, events_ages, _assets_ids, first_ages, last_ages
