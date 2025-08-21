from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Iterator, Optional

import numpy as np
from numpy.lib import recfunctions as rfn
from numpy.typing import NDArray
from typing_extensions import override

from relife.economic import ExponentialDiscounting, Reward
from relife.lifetime_model import (
    Exponential,
    FrozenAgeReplacementModel,
    FrozenLeftTruncatedModel,
    LeftTruncatedModel
)
from relife.lifetime_model.distribution import LifetimeDistribution
from relife.lifetime_model.regression import FrozenLifetimeRegression

if TYPE_CHECKING:
    from relife.stochastic_process import RenewalRewardProcess


class StochasticDataIterator(Iterator[NDArray[np.void]], ABC):

    def __init__(
        self,
        nb_samples: int,
        tf: float,
        t0: float = 0.0,
        nb_assets: Optional[int] = None,
    ):
        if nb_assets is None:
            np_size = (1, nb_samples)
        else:
            np_size = (nb_assets, nb_samples)
        self.t0, self.tf = t0, tf

        # broadcasting of the size must be done before instanciation
        # Timeline for sampling always starts at 0, independant of the observation window

        # Assets timeline restarts at 0 when a replacement is done.
        # Used to identify the current age of the asset
        self.asset_timeline = np.zeros(np_size, dtype=np.float64)

        # Full timeline never restarts
        # Used to identify the observation window
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
                    ("asset_id", np.uint32),  #  unsigned 32bit integer
                    ("sample_id", np.uint32),  #  unsigned 32bit integer
                    ("timeline", np.float64),  #  64bit float
                ]
            ),
        )
        
        struct_array["asset_id"] = asset_id.astype(np.uint32)
        struct_array["sample_id"] = sample_id.astype(np.uint32)
        struct_array["timeline"] = self.timeline[self.mask["observed_event"]]
        
        return struct_array

    @abstractmethod
    def sample_step(self):
        ...

    def apply_observation_window(self, time, event, entry) -> NDArray[np.void]:
        
        residual_time = time - entry

        self.timeline += residual_time
        self.update_mask()

        # compute tf right censorings
        time = np.where(self.mask["just_crossed_tf"], time - (self.timeline - self.tf), time)
        self.timeline[self.mask["just_crossed_tf"]] = self.tf
        event[self.mask["just_crossed_tf"]] = False

        # compute t0 left truncations
        entry = np.where(self.mask["just_crossed_t0"], time - (self.timeline - self.t0), entry)
        
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
    

    def step(self) -> NDArray[np.void]:
        time, event, entry = self.sample_step()
        struct_arr = self.apply_observation_window(time,event,entry)
        return struct_arr

    def __next__(self) -> NDArray[np.void]:
        if not np.all(self.mask["crossed_tf"]):
            struct_arr = self.step()
            while struct_arr.size == 0:  # skip cycles while arrays are empty (if t0 != 0.)
                struct_arr = self.step()
                if np.all(self.mask["crossed_tf"]):
                    raise StopIteration
            return struct_arr
        raise StopIteration


class RenewalProcessIterator(StochasticDataIterator):

    def __init__(
        self,
        process,
        nb_samples: int,
        tf: float,
        t0: float = 0.0,
    ):
        nb_assets = getattr(process.lifetime_model, "nb_assets", 1)
        super().__init__(nb_samples, tf, t0=t0, nb_assets=nb_assets)
        self.process = process

    @property
    def lifetime_model(self):
        if self.replacement_cycle == 0 and self.process.first_lifetime_model is not None:
            return self.process.first_lifetime_model
        return self.process.lifetime_model
    

    def sample_step(self):

        # Generate new values with ages = 0
        time, event, entry = self.lifetime_model.rvs(
            size=self.timeline.shape[1],
            return_event=True,
            return_entry=True,
        )

        # Reshape as timeline
        time = time.reshape(self.timeline.shape)
        event = event.reshape(self.timeline.shape)
        entry = entry.reshape(self.timeline.shape)

        # No need to update asset_timeline for a RenewalProcess

        return time, event, entry

    


class RenewalRewardProcessIterator(RenewalProcessIterator):
    reward: Reward
    discounting: ExponentialDiscounting

    def __init__(
        self,
        process: RenewalRewardProcess[
            LifetimeDistribution | FrozenLifetimeRegression | FrozenAgeReplacementModel | FrozenLeftTruncatedModel,
            Reward,
        ],
        nb_samples: int,
        tf: float,
        t0: float = 0.0,
    ):
        super().__init__(process, nb_samples, tf, t0=t0)
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


class NonHomogeneousPoissonProcessIterator(StochasticDataIterator):

    def __init__(
        self,
        process,
        nb_samples: int,
        tf: float,
        t0: float = 0.0,
    ):
        nb_assets = getattr(process.lifetime_model, "nb_assets", 1)
        super().__init__(nb_samples, tf, t0=t0, nb_assets=nb_assets)
        self.process = process

    
    def sample_step(self):

        # Broadcast frozen args to asset_timeline shape
        args = getattr(self.process.lifetime_model, "args", ())
        broadcasted_args = list(np.broadcast_to(arg,self.asset_timeline.shape).flatten() for arg in args)
        broadcasted_model = self.process.lifetime_model.unfreeze().freeze(*broadcasted_args) if len(broadcasted_args) > 0 else self.process.lifetime_model

        # Build LeftTruncatedModel
        ages = self.asset_timeline.flatten()
        truncated_model = LeftTruncatedModel(broadcasted_model).freeze(ages)

        time, event, entry = truncated_model.rvs(
            size=1,
            return_event=True,
            return_entry=True,
        )

        # Reshape as timeline
        time = time.reshape(self.timeline.shape)
        event = event.reshape(self.timeline.shape)
        entry = entry.reshape(self.timeline.shape)

        # TODO : delete when bug fixed for event in LeftTrunated(AgeReplacement(Distrib))
        # Only works when ar is the last args of the model
        event = (time != np.broadcast_to(args[0],self.asset_timeline.shape)) if len(broadcasted_args) > 0 else event

        # Update asset timeline
        self.asset_timeline += time - entry
        # If no events (replacement), restarts asset timeline for next step
        self.asset_timeline[~event] = 0

        return time, event, entry
