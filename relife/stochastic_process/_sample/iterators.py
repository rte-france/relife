from abc import ABC, abstractmethod
from typing import Iterator, Optional, Tuple

import numpy as np
from numpy.lib import recfunctions as rfn
from numpy.typing import NDArray
from typing_extensions import override

from relife.economic import ExponentialDiscounting, Reward
from relife.lifetime_model.conditional_model import LeftTruncatedModel
from relife.stochastic_process import NonHomogeneousPoissonProcess

from relife.utils import get_lifetime_model_nb_assets


class StochasticDataIterator(Iterator[NDArray[np.void]], ABC):
    """Abstract class for all stochastic processes iterator.
    Used to build the structarrays, get the shapes, iterate through steps and identify the observation window.
    Abstract method is sample_next_step, that is unique for each stochastic process.
    """

    def __init__(
        self,
        process,
        nb_assets: int,
        nb_samples: int,
        tf: float,
        t0: float = 0.0,
        seed=None,
    ):
        self.process = process

        self.nb_assets = nb_assets
        self.nb_samples = nb_samples

        sample_size = (self.nb_assets * self.nb_samples,)

        self.t0, self.tf = t0, tf

        self.seed = np.random.default_rng(seed)

        # Assets ages restart at 0 when replacements are done.
        # Used to identify the current ages of the assets in the process
        self.ages = np.zeros(sample_size, dtype=np.float64)

        # Full timeline never restarts, sums at each iteration
        # Used to identify current time for the observer
        self.timeline = np.zeros(sample_size, dtype=np.float64)

        self._crossed_t0_counter = np.zeros(sample_size, dtype=np.uint32)
        self._crossed_tf_counter = np.zeros(sample_size, dtype=np.uint32)
        self.replacement_cycle = 0

        self.mask = np.zeros(
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

    def update_mask(self) -> None:
        """Update mask to keep track of the observed and non observed events"""
        self._crossed_t0_counter[self.timeline > self.t0] += 1
        self._crossed_tf_counter[self.timeline > self.tf] += 1
        self.mask["just_crossed_t0"] = self._crossed_t0_counter == 1
        self.mask["just_crossed_tf"] = self._crossed_tf_counter == 1
        self.mask["crossed_tf"] = self._crossed_tf_counter >= 1
        self.mask["observed_step"] = np.logical_and(
            self._crossed_t0_counter >= 1, self._crossed_tf_counter <= 1
        )

    def get_base_structarray(self) -> NDArray[np.void]:
        """Construct the struct array to return"""
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

    def get_rvs_size(self, args_size: int) -> Tuple[int]:
        """Helper function, get the size that we should pass to rvs function depending on the args of the model used to sample.
        A lifetime_model random variable sampling called with size=n will sample n values for each asset in its args.
        """
        if args_size is None or args_size == 1:
            return self.nb_assets * self.nb_samples

        if args_size == self.nb_assets:
            return self.nb_samples

        if args_size == self.nb_assets * self.nb_samples:
            return 1

        # A commenter
        raise ValueError

    @abstractmethod
    def sample_step(self):
        """Abstract method to implement in child classes"""
        raise NotImplementedError

    def apply_observation_window(self, time, event, entry) -> NDArray[np.void]:
        """Apply the conditioning observations windows to the sampled values"""

        residual_time = time - entry

        # Timeline increases by residual time
        self.timeline += residual_time
        self.update_mask()

        # Replace times that exceeds tf with right censorings at tf
        time = np.where(
            self.mask["just_crossed_tf"], time - (self.timeline - self.tf), time
        )
        self.timeline[self.mask["just_crossed_tf"]] = self.tf
        event[self.mask["just_crossed_tf"]] = False

        # Replace entries to take account of the left truncation at t0
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
        """Sample one step, apply observation window and return the struct array"""
        time, event, entry = self.sample_step()
        struct_arr = self.apply_observation_window(time, event, entry)
        return struct_arr

    def __next__(self) -> NDArray[np.void]:
        """function to iterate"""
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
        seed: Optional[int] = None,
    ):
        lifetime_model_nb_assets = get_lifetime_model_nb_assets(process.lifetime_model)
        first_lifetime_model_nb_assets = get_lifetime_model_nb_assets(
            process.first_lifetime_model
        )

        if (
            lifetime_model_nb_assets != 1
            and first_lifetime_model_nb_assets != 1
            and lifetime_model_nb_assets != first_lifetime_model_nb_assets
        ):
            raise ValueError(
                f"Args of the first_lifetime_model and lifetime_model should have coherent shapes. Got {first_lifetime_model_nb_assets, lifetime_model_nb_assets}"
            )

        nb_assets = max(lifetime_model_nb_assets, first_lifetime_model_nb_assets)

        super().__init__(
            process=process,
            nb_assets=nb_assets,
            nb_samples=nb_samples,
            tf=tf,
            t0=t0,
            seed=seed,
        )

    @property
    def lifetime_model(self):
        if (
            self.replacement_cycle == 0
            and self.process.first_lifetime_model is not None
        ):
            return self.process.first_lifetime_model
        return self.process.lifetime_model

    def sample_step(self):
        args_size = get_lifetime_model_nb_assets(self.lifetime_model)

        # Generate new values with ages = 0
        time, event, entry = self.lifetime_model.rvs(
            self.get_rvs_size(args_size),
            return_event=True,
            return_entry=True,
            seed=self.seed,
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
        process,
        nb_samples: int,
        tf: float,
        t0: float = 0.0,
        seed: Optional[int] = None,
    ):
        super().__init__(process, nb_samples, tf, t0=t0, seed=seed)
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
    def __init__(
        self,
        process,
        nb_samples: int,
        tf: float,
        t0: float = 0.0,
        seed=None,
    ):
        # All samples for each assets is considered individually in an NHPP
        # We use a LeftTruncatedModel to sample according to current age
        # We need all samples and assets in 1D, so we must broadcast the original model to repeat its args

        nb_assets = get_lifetime_model_nb_assets(process.lifetime_model)

        broadcasted_model = process.lifetime_model
        args = getattr(process.lifetime_model, "args", None)
        if args:
            broadcasted_args = list(np.repeat(arg, nb_samples, axis=0) for arg in args)
            broadcasted_model = broadcasted_model.unfreeze().freeze(
                *broadcasted_args
            )  # TODO: use a copy method of parametric models

        super().__init__(
            process=NonHomogeneousPoissonProcess(broadcasted_model),
            nb_assets=nb_assets,
            nb_samples=nb_samples,
            tf=tf,
            t0=t0,
            seed=seed,
        )

    def sample_step(self):
        # Apply a Left truncation based on current ages on the model

        ages = self.ages.copy().reshape(-1, 1)
        truncated_lifetime_model = LeftTruncatedModel(
            self.process.lifetime_model
        ).freeze(ages)

        # Sample using this truncation

        args_size = get_lifetime_model_nb_assets(truncated_lifetime_model)

        time, event, entry = truncated_lifetime_model.rvs(
            self.get_rvs_size(args_size),
            return_event=True,
            return_entry=True,
            seed=self.seed,
        )

        time = time.reshape(self.timeline.shape)
        event = event.reshape(self.timeline.shape)
        entry = entry.reshape(self.timeline.shape)

        # Update asset ages
        self.ages += time - entry
        # If no events (replacement), restarts asset timeline for next step
        self.ages[~event] = 0

        return time, event, entry
