# pyright: basic

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from numpy.lib import recfunctions as rfn
from numpy.typing import NDArray
from typing_extensions import override

from relife.base import is_frozen
from relife.lifetime_model import LeftTruncatedModel
from relife.lifetime_model._base import ParametricLifetimeModel
from relife.lifetime_model._distribution import (
    EquilibriumDistribution,
)
from relife.typing._scalars import NumpyFloat

__all__ = [
    "StochasticDataIterator",
    "RenewalProcessIterator",
    "RenewalRewardProcessIterator",
    "NonHomogeneousPoissonProcessIterator",
]


def _expand_lifetime_model(
    lifetime_model: ParametricLifetimeModel, nb_samples: int
) -> ParametricLifetimeModel:
    """
    Expand a lifetime model by duplicating its arguments. A regression with n_assets assets will result in a new regression with n_assets * n_samples assets.
    """
    if isinstance(lifetime_model, EquilibriumDistribution):
        return EquilibriumDistribution(
            _expand_lifetime_model(lifetime_model.baseline, nb_samples)
        )

    expanded_lifetime_model = lifetime_model

    if is_frozen(lifetime_model):
        broadcasted_args = list(
            np.repeat(arg, nb_samples, axis=0) for arg in lifetime_model.args
        )
        expanded_lifetime_model = lifetime_model.unfreeze().freeze(
            *broadcasted_args
        )  # TODO: use a copy method of parametric models

    return expanded_lifetime_model


@dataclass
class SampleStep:
    residual_time: NDArray[np.float64]
    event: NDArray[np.bool_]
    entry: NDArray[np.float64]


class TimeWindowObserver:
    def __init__(self, sample_size: int, time_window: tuple[float, float]):

        self.t0, self.tf = time_window

        self._crossed_t0_counter: NDArray[np.int_] = np.zeros(
            sample_size, dtype=np.int64
        )
        self._crossed_tf_counter: NDArray[np.int_] = np.zeros(
            sample_size, dtype=np.int64
        )

    def update(self, timeline: NDArray[np.float64]):

        self._crossed_t0_counter[timeline > self.t0] += 1
        self._crossed_tf_counter[timeline > self.tf] += 1

    @property
    def just_crossed_t0(self):
        return self._crossed_t0_counter == 1

    @property
    def just_crossed_tf(self):
        return self._crossed_tf_counter == 1

    @property
    def observed_step(self):
        return np.logical_and(
            self._crossed_t0_counter >= 1, self._crossed_tf_counter <= 1
        )

    @property
    def all_finished(self):
        return np.all(self._crossed_tf_counter >= 1)

    def apply_observation_window(
        self, sample_step: SampleStep, timeline: NDArray[np.float64]
    ) -> tuple[SampleStep, NDArray[np.float64]]:
        entry = np.where(
            self.just_crossed_t0,
            sample_step.entry + self.t0 - (timeline - sample_step.residual_time),
            sample_step.entry,
        )

        residual_time = np.where(
            self.just_crossed_tf,
            self.tf - entry,
            sample_step.residual_time,
        )

        event = np.where(self.just_crossed_tf, False, sample_step.event)

        timeline[self.just_crossed_tf] = self.tf

        return SampleStep(residual_time, event, entry), timeline


class StructArrayBuilder:
    def __init__(self, nb_assets: int, nb_samples: int):
        self.nb_assets = nb_assets
        self.nb_samples = nb_samples

    def _init_structarray(self, observed_step: NDArray[np.bool_]) -> NDArray[np.void]:
        """Construct the struct array to return"""
        observed_index = np.where(observed_step)[0]

        asset_id = observed_index // self.nb_samples
        sample_id = observed_index % self.nb_samples

        struct_array = np.zeros(
            sample_id.size,
            dtype=np.dtype(
                [
                    ("asset_id", np.uint32),  #  unsigned 32bit integer
                    ("sample_id", np.uint32),  #  unsigned 32bit integer
                ]
            ),
        )

        struct_array["asset_id"] = asset_id.astype(np.uint32)
        struct_array["sample_id"] = sample_id.astype(np.uint32)
        return struct_array

    def build_structarray(
        self,
        observed_step: NDArray[np.bool_],
        timeline: NDArray[np.float64],
        sample_step: SampleStep,
    ) -> NDArray[np.void]:

        base_struct_array = self._init_structarray(observed_step)
        struct_arr = rfn.append_fields(  #  works on structured_array too
            base_struct_array,
            ("timeline", "time", "event", "entry"),
            (
                timeline[observed_step],
                sample_step.residual_time[observed_step]
                + sample_step.entry[observed_step],
                sample_step.event[observed_step],
                sample_step.entry[observed_step],
            ),
            (np.float64, np.float64, np.bool_, np.float64),
            usemask=False,
            asrecarray=False,
        )
        return struct_arr

    @staticmethod
    def add_field(
        struct_arr: NDArray[np.void], new_label: str, new_values: NDArray[np.float64]
    ):
        return rfn.append_fields(
            struct_arr,
            new_label,
            new_values,
            np.float64,
            usemask=False,
            asrecarray=False,
        )  # type: ignore


class StochasticDataIterator(Iterator[NDArray[np.void]], ABC):
    """Abstract class for all stochastic processes iterator.
    Used to build the structarrays, get the shapes, iterate through steps and identify the observation window.
    Abstract method is sample_next_step, that is unique for each stochastic process.
    """

    def __init__(
        self,
        process,
        nb_samples: int,
        time_window: tuple[float, float],
        a0: NumpyFloat | None = None,
        ar: NumpyFloat | None = None,
        nb_assets: int = 1,
        seed=None,
    ) -> None:
        self.process = process
        self.sample_size = nb_assets * nb_samples

        self._expanded_lifetime_model = _expand_lifetime_model(
            self.process.lifetime_model, nb_samples
        )

        self.time_window_observer = TimeWindowObserver(
            sample_size=self.sample_size, time_window=time_window
        )

        self.structarray_builder = StructArrayBuilder(
            nb_assets=nb_assets, nb_samples=nb_samples
        )

        if not a0:
            a0 = 0
        a0 = np.broadcast_to(a0, (nb_assets,)).astype(np.float64)
        self.ages = np.repeat(a0, nb_samples, axis=0)

        if ar :
            ar = np.broadcast_to(ar, (nb_assets,)).astype(np.float64)
            self.ar = np.repeat(ar, nb_samples)
        else:
            self.ar = None

        self.timeline = np.zeros(self.sample_size)

        self.replacement_cycle = 0
        self.seed = np.random.default_rng(seed)

    @property
    @abstractmethod
    def _expanded_dynamic_lifetime_model(self) -> ParametricLifetimeModel:
        """
        Use the lifetime model modified at each iteration according to each stochastic process specific properties
        """

    @abstractmethod
    def update_ages(
        self,
        time: NDArray[np.float64],
    ) -> None:
        """
        Update ages at each iteration according to each stochastic process specific properties
        """

    def sample_step(
        self,
    ) -> SampleStep:

        residual_time = self._expanded_dynamic_lifetime_model.rvs(
            (self.sample_size, 1),
            seed=self.seed,
        )
        residual_time = residual_time.flatten()

        event = np.ones_like(residual_time, dtype=np.bool_)
        entry = self.ages.copy()

        if self.ar is not None:
            preventive_replacements = self.ages + residual_time >= self.ar
            residual_time[preventive_replacements] = (
                self.ar[preventive_replacements] - self.ages[preventive_replacements]
            )
            event = ~preventive_replacements

        return SampleStep(residual_time, event, entry)

    def apply_observation_bias(
        self,
        sample_step: SampleStep,
    ) -> SampleStep:
        """Collect observed time, event, entry inside during the time window"""

        sample_step = self.sample_step()

        # Timeline increases by residual time
        self.timeline += sample_step.residual_time
        self.time_window_observer.update(self.timeline)

        # Apply observation window conditions
        sample_step, self.timeline = self.time_window_observer.apply_observation_window(
            sample_step, self.timeline
        )
        return sample_step

    def make_one_step(self):
        sample_step = self.sample_step()
        sample_step = self.apply_observation_bias(sample_step)

        self.update_ages(sample_step.residual_time)
        self.replacement_cycle += 1

        return self.structarray_builder.build_structarray(
            self.time_window_observer.observed_step, self.timeline, sample_step
        )

    def __next__(self) -> NDArray[np.void]:
        """function to iterate"""
        if not self.time_window_observer.all_finished:
            struct_arr = self.make_one_step()
            while (
                struct_arr.size == 0
            ):  # skip cycles while arrays are empty (if t0 != 0.)
                struct_arr = self.make_one_step()
                if self.time_window_observer.all_finished and struct_arr.size > 0:
                    return struct_arr
            return struct_arr
        raise StopIteration


class RenewalProcessIterator(StochasticDataIterator):
    def __init__(
        self,
        process,
        nb_samples: int,
        time_window: tuple[float, float],
        a0: NumpyFloat | None = None,
        ar: NumpyFloat | None = None,
        nb_assets: int = 1,
        seed=None,
    ) -> None:
        super().__init__(
            process=process,
            nb_samples=nb_samples,
            time_window=time_window,
            a0=a0,
            ar=ar,
            nb_assets=nb_assets,
            seed=seed,
        )
        first_lifetime_model = (
            _expand_lifetime_model(self.process.first_lifetime_model, nb_samples)
            if self.process.first_lifetime_model is not None
            else self._expanded_lifetime_model
        )
        self._expanded_first_lifetime_model = LeftTruncatedModel(
            first_lifetime_model
        ).freeze(self.ages.copy())

    @property
    def _expanded_dynamic_lifetime_model(self) -> ParametricLifetimeModel:
        return (
            self._expanded_first_lifetime_model
            if self.replacement_cycle == 0
            else self._expanded_lifetime_model
        )

    def update_ages(
        self,
        residual_time: NDArray[np.float64],
    ):
        """
        In a Renewal process, ages are reset to 0 after each iteration.
        """
        self.ages = np.zeros(self.sample_size, dtype=np.float64)


class RenewalRewardProcessIterator(RenewalProcessIterator):
    @override
    def __next__(self) -> NDArray[np.void]:
        struct_arr = super().__next__()
        return StructArrayBuilder.add_field(
            struct_arr,
            "reward",
            self.process.reward.sample(struct_arr["time"])
            * self.process.discounting.factor(struct_arr["timeline"]),
        )


class NonHomogeneousPoissonProcessIterator(StochasticDataIterator):
    @property
    def _expanded_dynamic_lifetime_model(self) -> ParametricLifetimeModel:
        # Apply a Left truncation based on current ages on the model
        # self.ages is always 1d in LeftTruncatedModel
        return LeftTruncatedModel(self._expanded_lifetime_model).freeze(
            self.ages.copy()
        )

    def update_ages(
        self,
        residual_time: NDArray[np.float64],
    ):
        """
        In a NHPP, ages are reset to 0 only when a replacement is made
        """
        # Update asset ages
        self.ages += residual_time

        if self.ar is not None:
            self.ages[self.ages >= self.ar] = 0


class VirtualAgeProcessIterator(StochasticDataIterator):
    def __init__(
        self,
        process,
        nb_samples: int,
        time_window: tuple[float, float],
        a0: NumpyFloat | None = None,
        ar: NumpyFloat | None = None,
        nb_assets: int = 1,
        seed=None,
    ) -> None:
        # TODO: peut-être bloquer le a0 ici
        super().__init__(
            process,
            nb_samples,
            time_window,
            ar=ar,
            a0=a0,
            nb_assets=nb_assets,
            seed=seed,
        )
        self.virtual_ages = self.ages.copy()

    @property
    def _expanded_dynamic_lifetime_model(self) -> ParametricLifetimeModel:
        # Apply a Left truncation based on current ages on the model
        # self.ages is always 1d in LeftTruncatedModel
        return LeftTruncatedModel(self._expanded_lifetime_model).freeze(
            self.virtual_ages.copy()
        )

    @override
    def __next__(self) -> NDArray[np.void]:
        struct_arr = super().__next__()
        return StructArrayBuilder.add_field(
            struct_arr,
            "virtual_age",
            self.virtual_ages[self.time_window_observer.observed_step],
        )


class Kijima1ProcessIterator(VirtualAgeProcessIterator):
    def update_ages(
        self,
        residual_time: NDArray[np.float64],
    ):
        """
        In a Kijima Process, the concept of age is virtual, and depends on the q parameter of the process
        """
        # Update asset ages
        self.virtual_ages = self.ages + self.process.q * residual_time
        self.ages += residual_time

        if self.ar is not None:
            self.ages[self.ages >= self.ar] = 0
            self.virtual_ages[self.ages >= self.ar] = 0


class Kijima2ProcessIterator(VirtualAgeProcessIterator):
    def update_ages(
        self,
        residual_time: NDArray[np.float64],
    ):
        """
        In a Kijima Process, the concept of age is virtual, and depends on the q parameter of the process
        """
        # Update asset ages
        self.virtual_ages = self.process.q * (self.ages + residual_time)
        self.ages += residual_time

        if self.ar is not None:
            self.ages[self.ages >= self.ar] = 0
            self.virtual_ages[self.ages >= self.ar] = 0
