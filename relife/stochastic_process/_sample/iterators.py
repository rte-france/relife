from abc import ABC, abstractmethod
from typing import Iterator, Optional

import numpy as np
from numpy.lib import recfunctions as rfn
from numpy.typing import NDArray
from typing_extensions import override

from relife.economic import ExponentialDiscounting, Reward
from relife.lifetime_model import (
    Exponential,
)


class CountDataIterator(Iterator[NDArray[np.void]], ABC):

    def __init__(
        self,
        size: int,
        tf: float,
        t0: float = 0.0,
        nb_assets: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        if nb_assets is None:
            np_size = (1, size)
        else:
            np_size = (nb_assets, size)
        self.t0, self.tf = t0, tf
        self.seed = seed
        # broadcasting of the size must be done before instanciation
        self.timeline = np.zeros(np_size, dtype=np.float64)

        self._crossed_t0_counter = np.zeros(np_size, dtype=np.uint32)
        self._crossed_tf_counter = np.zeros(np_size, dtype=np.uint32)
        self._observed_event = np.zeros(np_size, dtype=np.bool_)
        self.replacement_cycle = 0

        self.mask = np.zeros(
            np_size,
            dtype=np.dtype(
                [
                    ("observed_event", np.bool_),
                    ("just_crossed_t0", np.bool_),
                    ("just_crossed_tf", np.bool_),
                    ("crossed_tf", np.bool_),
                ]
            ),
        )

    def update_mask(self) -> None:
        self._crossed_t0_counter[self.timeline > self.t0] += 1
        self._crossed_tf_counter[self.timeline > self.tf] += 1
        self.mask["just_crossed_t0"] = self._crossed_t0_counter == 1
        self.mask["just_crossed_tf"] = self._crossed_tf_counter == 1
        self.mask["crossed_tf"] = self._crossed_tf_counter >= 1
        self.mask["observed_event"] = np.atleast_2d(
            np.logical_and(self._crossed_t0_counter >= 1, self._crossed_tf_counter <= 1)
        )

    def get_base_structarray(self) -> NDArray[np.void]:
        asset_id, sample_id = np.where(self.mask["observed_event"])

        struct_array = np.zeros(
            sample_id.size,
            dtype=np.dtype(
                [
                    ("sample_id", np.uint32),  #  unsigned 32bit integer
                    ("asset_id", np.uint32),  #  unsigned 32bit integer
                    ("timeline", np.float64),  #  64bit float
                ]
            ),
        )
        struct_array["timeline"] = self.timeline[self.mask["observed_event"]]
        struct_array["sample_id"] = sample_id.astype(np.uint32)
        struct_array["asset_id"] = asset_id.astype(np.uint32)
        return struct_array

    @abstractmethod
    def step(self) -> NDArray[np.void]: ...

    def __next__(self) -> NDArray[np.void]:
        if not np.all(self.mask["crossed_tf"]):
            struct_arr = self.step()
            while struct_arr.size == 0:  # skip cycles while arrays are empty (if t0 != 0.)
                struct_arr = self.step()
                if np.all(self.mask["crossed_tf"]):
                    raise StopIteration
            return struct_arr
        raise StopIteration


class RenewalProcessIterator(CountDataIterator):

    def __init__(
        self,
        process,
        size: int,
        tf: float,
        t0: float = 0.0,
        nb_assets: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(size, tf, t0=t0, nb_assets=nb_assets, seed=seed)
        self.process = process

    @property
    def lifetime_model(self):
        if self.replacement_cycle == 0 and self.process.first_lifetime_model is not None:
            return self.process.first_lifetime_model
        return self.process.lifetime_model

    def step(self) -> NDArray[np.void]:
        # time is not residual age if return_entry is True (see LeftTruncatedModel)
        # time may be ar if model is AgeReplacementModel, then event is False
        # for first_cycle, model_entry may not be zero
        time, event, model_entry = self.lifetime_model.rvs(
            self.timeline.shape[1],
            nb_assets=self.timeline.shape[0],
            return_event=True,
            return_entry=True,
            seed=self.seed,
        )
        if self.replacement_cycle == 0 and np.any(model_entry != 0):
            # fixed age process
            self.timeline.fill(self.t0)
        residual_time = time - model_entry

        self.timeline += residual_time
        self.update_mask()

        # compute tf right censorings
        time = np.where(self.mask["just_crossed_tf"], time - (self.timeline - self.tf), time)
        self.timeline[self.mask["just_crossed_tf"]] = self.tf
        event[self.mask["just_crossed_tf"]] = False

        # compute t0 left truncations
        if self.replacement_cycle == 0 and np.any(model_entry != 0):
            entry = np.where(self.mask["just_crossed_t0"], model_entry, 0.0)
        else:
            entry = np.where(self.mask["just_crossed_t0"], time - (self.timeline - self.t0), 0.0)

        # update seed to avoid having the same lifetime rvs
        if self.seed is not None:
            self.seed += 1
        self.replacement_cycle += 1

        struct_arr = rfn.append_fields(  #  works on structured_array too
            self.get_base_structarray(),
            ("time", "event", "entry"),
            (
                time[self.mask["observed_event"]],
                event[self.mask["observed_event"]],
                entry[self.mask["observed_event"]],
            ),
            (np.float64, np.bool_, np.float64),
            usemask=False,
            asrecarray=False,
        )
        return struct_arr


class RenewalRewardProcessIterator(RenewalProcessIterator):
    reward: Reward
    discounting: ExponentialDiscounting

    def __init__(
        self,
        process,
        nb_samples: int,
        tf: float,
        t0: float = 0.0,
        nb_assets: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(process, nb_samples, tf, t0=t0, nb_assets=nb_assets, seed=seed)
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


class NonHomogeneousPoissonProcessIterator(CountDataIterator):

    def __init__(
        self,
        process,
        nb_samples: int,
        tf: float,
        t0: float = 0.0,
        nb_assets: Optional[int] = None,
        ar: Optional[float | NDArray[np.float64]] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(nb_samples, tf, t0=t0, nb_assets=nb_assets, seed=seed)
        self.process = process
        self.hpp_timeline = np.zeros_like(self.timeline, dtype=np.float64)
        self.failure_instant = np.zeros_like(self.timeline, dtype=np.float64)
        self.age = np.zeros_like(self.timeline, dtype=np.float64)
        self.entry = np.zeros_like(self.timeline, dtype=np.float64)
        self.is_new_asset = np.zeros_like(
            self.timeline, dtype=np.bool_
        )  # boolean flags indicating which assets are replaced
        self.renewals_ids = np.zeros_like(self.timeline, dtype=np.uint32)
        self.ar = (
            np.asarray(ar, dtype=np.float64).reshape(-1, 1)
            if ar is not None
            else np.full_like(self.timeline, np.inf, dtype=np.float64)
        )
        self.exponential_dist = Exponential(1.0)

    def step(self) -> NDArray[np.void]:

        # reset those who are replaced
        self.age[self.is_new_asset] = 0.0  # asset is replaced (0 aged asset)
        # self._assets_ids[self.is_new_asset] += 1 # asset is replaced (new asset id)
        self.hpp_timeline[self.is_new_asset] = 0.0
        self.failure_instant[self.is_new_asset] = 0.0
        self.entry[self.is_new_asset] = 0.0
        self.renewals_ids[self.is_new_asset] += 1
        self.is_new_asset.fill(False)  # reset to False

        # generate new values
        self.hpp_timeline += self.exponential_dist.rvs(
            self.timeline.shape[1], nb_assets=self.timeline.shape[0], seed=self.seed
        )
        failure_instant = self.process.unfrozen_model.lifetime_model.ichf(
            self.hpp_timeline, *getattr(self.process, "args", ())
        )
        duration = failure_instant - self.failure_instant  # t_i+1 - t_i
        self.failure_instant = failure_instant.copy()  # update t_i <- t_i+1
        self.timeline += duration
        self.age += duration

        # ar update (before because it changes timeline, thus start and stop conditions)
        self.timeline = np.where(
            self.age >= self.ar,
            self.timeline - (self.age - self.ar),
            self.timeline,
        )  # substract time after ar
        self.age = np.where(self.age >= self.ar, self.ar, self.age)
        self.is_new_asset[np.logical_and(self.age >= self.ar, ~self.mask["just_crossed_tf"])] = True
        event = np.where(np.logical_and(self.age >= self.ar, ~self.mask["just_crossed_tf"]), False, True)

        self.update_mask()

        # compute tf right censorings
        self.age = np.where(self.mask["just_crossed_tf"], self.age - (self.timeline - self.tf), self.age)
        self.timeline[self.mask["just_crossed_tf"]] = self.tf
        event[self.mask["just_crossed_tf"]] = False

        # compute t0 left truncations
        self.entry = np.where(self.mask["just_crossed_t0"], self.age - (self.timeline - self.t0), self.entry)

        entry = self.entry.copy()  # returned entry
        self.entry = np.where(event, self.age, self.entry)  # keep previous ages as entry for next iteration

        # update seed to avoid having the same rvs result
        if self.seed is not None:
            self.seed += 1

        struct_arr = rfn.append_fields(  #  works on structured_array too
            self.get_base_structarray(),
            ("age", "event", "entry"),
            (
                self.age[self.mask["observed_event"]],
                event[self.mask["observed_event"]],
                entry[self.mask["observed_event"]],
            ),
            (np.float64, np.bool_, np.float64),
            usemask=False,
            asrecarray=False,
        )
        return struct_arr
