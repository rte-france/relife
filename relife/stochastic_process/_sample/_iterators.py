from abc import ABC, abstractmethod
from typing import Iterator, Optional, Tuple

import numpy as np
from numpy.lib import recfunctions as rfn
from numpy.typing import NDArray
from typing_extensions import override

from relife.economic import ExponentialDiscounting, Reward
from relife.lifetime_model import LeftTruncatedModel
from relife.utils import get_model_nb_assets, is_frozen


class _StochasticDataIterator(Iterator[NDArray[np.void]], ABC):
    """Abstract class for all stochastic processes iterator.
    Used to build the structarrays, get the shapes, iterate through steps and identify the observation window.
    Abstract method is sample_next_step, that is unique for each stochastic process.
    """

    def __init__(
        self,
        process,
        nb_samples: int,
        time_window: tuple[float, float],
        nb_assets: int = 1,
        seed=None,
    ):
        self.process = process
        self.nb_samples = nb_samples
        self.nb_assets = nb_assets
        self.t0, self.tf = time_window  # t0, tf are checked in iterable
        self.seed = np.random.default_rng(seed)
        sample_size = (self.nb_assets * self.nb_samples,)
        # Assets ages restart at 0 when replacements are done.
        # Used to identify the current ages of the assets in the process
        self.ages = np.zeros(sample_size, dtype=np.float64)
        # Full timeline never restarts, sums at each iteration
        # Used to identify current time for the observer
        self.timeline = np.zeros(sample_size, dtype=np.float64)
        self.replacement_cycle = 0

        self._crossed_t0_counter = np.zeros(sample_size, dtype=np.uint32)
        self._crossed_tf_counter = np.zeros(sample_size, dtype=np.uint32)

        self._time_window_mask = np.zeros(
            sample_size,
            dtype=np.dtype(
                [
                    ("observed_step", np.bool_),
                    ("just_crossed_t0", np.bool_),
                    ("just_crossed_tf", np.bool_),
                    ("crossed_tf", np.bool_),
                ]
            ),
        )

    @abstractmethod
    def sample_time_event_entry(self):
        """
        Routine that samples time, event, entry at each step
        IMPORTANT: time, event, entry must be 1D.
        """

    def __next__(self) -> NDArray[np.void]:
        """function to iterate"""
        if not np.all(self._time_window_mask["crossed_tf"]):
            time, event, entry = self.sample_time_event_entry()
            struct_arr = self._collect_time_window_observations(time, event, entry)
            while (
                struct_arr.size == 0
            ):  # skip cycles while arrays are empty (if t0 != 0.)
                time, event, entry = self.sample_time_event_entry()
                struct_arr = self._collect_time_window_observations(time, event, entry)
                if np.all(self._time_window_mask["crossed_tf"]):
                    raise StopIteration
            return struct_arr
        raise StopIteration

    def _collect_time_window_observations(self, time, event, entry) -> NDArray[np.void]:
        """Collect observed time, event, entry inside during the time window"""
        if time.ndim > 1:
            raise ValueError(
                f"sample_time_event_entry must return 1d entry. Got {time.ndim} dim for time"
            )
        if event.ndim > 1:
            raise ValueError(
                f"sample_time_event_entry must return 1d entry. Got {event.ndim} dim for event"
            )
        if entry.ndim > 1:
            raise ValueError(
                f"sample_time_event_entry must return 1d entry. Got {entry.ndim} dim for entry"
            )

        residual_time = time - entry

        # Timeline increases by residual time
        self.timeline += residual_time
        self._update_time_window_mask()

        # Replace times that exceeds tf with right censorings at tf
        time = np.where(
            self._time_window_mask["just_crossed_tf"],
            time - (self.timeline - self.tf),
            time,
        )
        self.timeline[self._time_window_mask["just_crossed_tf"]] = self.tf
        event[self._time_window_mask["just_crossed_tf"]] = False

        # Replace entries to take account of the left truncation at t0
        entry = np.where(
            self._time_window_mask["just_crossed_t0"],
            time - (self.timeline - self.t0),
            entry,
        )

        self.replacement_cycle += 1

        struct_arr = rfn.append_fields(  #  works on structured_array too
            self._init_structarray(),
            ("time", "event", "entry"),
            (
                time[self._time_window_mask["observed_step"]],
                event[self._time_window_mask["observed_step"]],
                entry[self._time_window_mask["observed_step"]],
            ),
            (np.float64, np.bool_, np.float64),
            usemask=False,
            asrecarray=False,
        )
        return struct_arr

    def _update_time_window_mask(self) -> None:
        """Update mask to keep track of the observed and non observed events"""
        self._crossed_t0_counter[self.timeline > self.t0] += 1
        self._crossed_tf_counter[self.timeline > self.tf] += 1
        self._time_window_mask["just_crossed_t0"] = self._crossed_t0_counter == 1
        self._time_window_mask["just_crossed_tf"] = self._crossed_tf_counter == 1
        self._time_window_mask["crossed_tf"] = self._crossed_tf_counter >= 1
        self._time_window_mask["observed_step"] = np.logical_and(
            self._crossed_t0_counter >= 1, self._crossed_tf_counter <= 1
        )

    def _init_structarray(self) -> NDArray[np.void]:
        """Construct the struct array to return"""
        observed_index = np.where(self._time_window_mask["observed_step"])[0]

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
        struct_array["timeline"] = self.timeline[
            self._time_window_mask["observed_step"]
        ]
        return struct_array


class _RenewalProcessIterator(_StochasticDataIterator):
    @property
    def lifetime_model(self):
        if (
            self.replacement_cycle == 0
            and self.process.first_lifetime_model is not None
        ):
            return self.process.first_lifetime_model
        return self.process.lifetime_model

    @property
    def _rvs_size(self) -> Tuple[int,int]:
        """
        Property to get the size that we should pass to rvs function depending on the args of the model used to sample.
        """
        model_nb_assets = get_model_nb_assets(self.lifetime_model)
        if model_nb_assets == 1:
            return (self.nb_assets * self.nb_samples,1)
        if model_nb_assets == self.nb_assets:
            return (model_nb_assets,self.nb_samples)
        raise ValueError

    def sample_time_event_entry(self):
        time, event, entry = self.lifetime_model.rvs(
            self._rvs_size,
            return_event=True,
            return_entry=True,
            seed=self.seed,
        )
        # flatten to return 1d
        time = time.flatten()
        event = event.flatten()
        entry = entry.flatten()

        return time, event, entry


class _RenewalRewardProcessIterator(_RenewalProcessIterator):
    reward: Reward
    discounting: ExponentialDiscounting

    def __init__(
        self,
        process,
        nb_samples: int,
        time_window: tuple[float, float],
        nb_assets: int = 1,
        seed: Optional[int] = None,
    ):
        super().__init__(
            process, nb_samples, time_window, nb_assets=nb_assets, seed=seed
        )
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


class _NonHomogeneousPoissonProcessIterator(_StochasticDataIterator):
    def __init__(
        self,
        process,
        nb_samples: int,
        time_window: tuple[float, float],
        nb_assets: int = 1,
        seed=None,
    ):
        super().__init__(
            process, nb_samples, time_window, nb_assets=nb_assets, seed=seed
        )
        # Here, all samples for each assets are considered individually because of the usage of LeftTruncatedModel
        # We need all samples and assets in 1D so we must broadcast the original lifetime model to repeat its args
        if is_frozen(self.process.lifetime_model):
            broadcasted_args = list(
                np.repeat(arg, self.nb_samples, axis=0)
                for arg in self.process.lifetime_model.args
            )
            self._expanded_lifetime_model = (
                self.process.lifetime_model.unfreeze().freeze(*broadcasted_args)
            )  # TODO: use a copy method of parametric models
        else:
            self._expanded_lifetime_model = self.process.lifetime_model

    @property
    def _truncated_lifetime_model(self):
        # Apply a Left truncation based on current ages on the model
        # self.ages is always 1d in LeftTruncatedModel
        ages = self.ages.copy()
        return LeftTruncatedModel(self._expanded_lifetime_model).freeze(ages)

    @property
    def _rvs_size(self) -> Tuple[int,int]:
        """
        Helper property, get the size that we should pass to rvs function depending on the args of the model used to sample.
        """
        model_nb_assets = get_model_nb_assets(self._truncated_lifetime_model)
        return (model_nb_assets,1)
        # if model_nb_assets == 1:
        #     return self.nb_samples
        # return 1

    def sample_time_event_entry(self):
        # Sample using truncated lifetime model truncation
        time, event, entry = self._truncated_lifetime_model.rvs(
            self._rvs_size,
            return_event=True,
            return_entry=True,
            seed=self.seed,
        )

        time = time.flatten()
        event = event.flatten()
        entry = entry.flatten()

        # Update asset ages
        self.ages += time - entry
        # If no events (replacement), restarts asset timeline for next step
        self.ages[~event] = 0

        return time, event, entry
