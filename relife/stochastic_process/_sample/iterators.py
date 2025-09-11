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
    LeftTruncatedModel,
)
from relife.lifetime_model.distribution import LifetimeDistribution
from relife.lifetime_model.regression import FrozenLifetimeRegression

if TYPE_CHECKING:
    from relife.stochastic_process import RenewalRewardProcess


def broadcast_lifetime_model(lifetime_model, nb_samples: int):
    # Broadcast process to duplicate each asset nb_samples time
    args = getattr(lifetime_model, "args", ())
    if len(args) == 0:
        return LeftTruncatedModel(lifetime_model).freeze(
            a0=np.zeros(nb_samples)
        )  # il faut changer ça, on dirait un hack
    broadcasted_args = list(
        np.repeat(arg, nb_samples, axis=0) for arg in args
    )  # Works for 1D and 2D arrays
    return lifetime_model.unfreeze().freeze(*broadcasted_args)


class StochasticDataIterator(Iterator[NDArray[np.void]], ABC):
    def __init__(
        self,
        process,
        nb_samples: int,
        tf: float,
        t0: float = 0.0,
        nb_assets: int = None,
        seed=None,
    ):
        self.nb_assets = nb_assets
        if nb_assets is None:
            self.nb_assets = getattr(process.lifetime_model, "nb_assets", 1)

        self.nb_samples = nb_samples

        self.lifetime_model = broadcast_lifetime_model(
            process.lifetime_model, self.nb_samples
        )

        np_size = (self.nb_assets * self.nb_samples,)

        self.t0, self.tf = t0, tf

        self.seed = np.random.default_rng(seed)

        # broadcasting of the size must be done before instanciation
        # Timeline for sampling always starts at 0, independant of the observation window

        # Assets timeline restarts at 0 when a replacement is done.
        # Used to identify the current age of the asset
        self.asset_ages = np.zeros(np_size, dtype=np.float64)

        # Full timeline never restarts
        # Used to identify the observation window
        self.timeline = np.zeros(np_size, dtype=np.float64)

        self._crossed_t0_counter = np.zeros(np_size, dtype=np.uint32)
        self._crossed_tf_counter = np.zeros(np_size, dtype=np.uint32)
        self.replacement_cycle = 0

        self.mask = np.zeros(
            np_size,
            dtype=np.dtype(
                [
                    ("observed_step", np.bool_),
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
        self.mask["observed_step"] = np.logical_and(
            self._crossed_t0_counter >= 1, self._crossed_tf_counter <= 1
        )

    def get_base_structarray(self) -> NDArray[np.void]:
        observed_index = np.where(self.mask["observed_step"])[0]

        asset_id = observed_index // self.nb_samples
        sample_id = observed_index % self.nb_samples

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
        struct_array["timeline"] = self.timeline[self.mask["observed_step"]]

        return struct_array

    @abstractmethod
    def sample_step(self): ...

    def apply_observation_window(self, time, event, entry) -> NDArray[np.void]:
        residual_time = time - entry

        self.timeline += residual_time
        self.update_mask()

        # compute tf right censorings
        time = np.where(
            self.mask["just_crossed_tf"], time - (self.timeline - self.tf), time
        )
        self.timeline[self.mask["just_crossed_tf"]] = self.tf
        event[self.mask["just_crossed_tf"]] = False

        # compute t0 left truncations
        entry = np.where(
            self.mask["just_crossed_t0"], time - (self.timeline - self.t0), entry
        )

        self.replacement_cycle += 1

        struct_arr = rfn.append_fields(  #  works on structured_array too
            self.get_base_structarray(),
            ("time", "event", "entry"),
            (
                time[self.mask["observed_step"]],
                event[self.mask["observed_step"]],
                entry[self.mask["observed_step"]],
            ),
            (np.float64, np.bool_, np.float64),
            usemask=False,
            asrecarray=False,
        )
        return struct_arr

    def step(self) -> NDArray[np.void]:
        time, event, entry = self.sample_step()
        struct_arr = self.apply_observation_window(time, event, entry)
        return struct_arr

    def __next__(self) -> NDArray[np.void]:
        if not np.all(self.mask["crossed_tf"]):
            struct_arr = self.step()
            while (
                struct_arr.size == 0
            ):  # skip cycles while arrays are empty (if t0 != 0.)
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
        if process.first_lifetime_model is not None:
            nb_assets = getattr(process.first_lifetime_model, "nb_assets", None)
            super().__init__(process, nb_samples, tf, t0, nb_assets)
            self.first_lifetime_model = broadcast_lifetime_model(
                process.first_lifetime_model, self.nb_samples
            )
            self.lifetime_model = broadcast_lifetime_model(
                process.lifetime_model, self.nb_samples * nb_assets
            )
        else:
            super().__init__(process, nb_samples, tf, t0)
            self.first_lifetime_model = self.lifetime_model

    def sample_step(self):
        if self.replacement_cycle == 0:
            lifetime_model = self.first_lifetime_model
        else:
            lifetime_model = self.lifetime_model

        # Generate new values with ages = 0
        time, event, entry = lifetime_model.rvs(
            size=1, return_event=True, return_entry=True, seed=self.seed
        )

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
            LifetimeDistribution
            | FrozenLifetimeRegression
            | FrozenAgeReplacementModel
            | FrozenLeftTruncatedModel,
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
            self.reward.sample(struct_array["time"])
            * self.discounting.factor(struct_array["timeline"]),
            np.float64,
            usemask=False,
            asrecarray=False,
        )  # type: ignore


class NonHomogeneousPoissonProcessIterator(StochasticDataIterator):
    def sample_step(self):
        args = getattr(self.lifetime_model, "args", ())
        ages = self.asset_ages.copy().reshape(-1, 1)
        truncated_lifetime_model = LeftTruncatedModel(
            self.lifetime_model.unfreeze()
        ).freeze(ages, *args)

        time, event, entry = truncated_lifetime_model.rvs(
            size=1, return_event=True, return_entry=True, seed=self.seed
        )

        time = time.reshape(self.timeline.shape)
        event = event.reshape(self.timeline.shape)
        entry = entry.reshape(self.timeline.shape)

        # Update asset ages
        self.asset_ages += time - entry
        # If no events (replacement), restarts asset timeline for next step
        self.asset_ages[~event] = 0

        return time, event, entry
