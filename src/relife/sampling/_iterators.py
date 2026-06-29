from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from numpy.lib import recfunctions as rfn
from numpy.typing import NDArray
from optype.numpy import Array1D, ArrayND
from typing_extensions import override

from relife.lifetime_models._base import (
    LeftTruncatedModel,
    ParametricLifetimeModel,
)
from relife.typing import ST, NumpyST

__all__ = [
    "StochasticDataIterator",
    "RenewalProcessIterator",
    "RenewalRewardProcessIterator",
    "NonHomogeneousPoissonProcessIterator",
]


@dataclass
class SampleStep:
    residual_time: NDArray[np.float64]
    event: NDArray[np.bool_]
    entry: NDArray[np.float64]


class TimeWindowObserver:
    t0: float
    tf: float
    _crossed_t0_counter: ArrayND[np.int64]
    _crossed_tf_counter: ArrayND[np.int64]

    def __init__(self, sample_shape: tuple[int, ...], time_window: tuple[float, float]):
        self.t0, self.tf = time_window
        self._crossed_t0_counter = np.zeros(sample_shape, dtype=np.int64)
        self._crossed_tf_counter = np.zeros(sample_shape, dtype=np.int64)

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

        last_date = timeline - sample_step.residual_time
        installation_date = last_date - sample_step.entry

        entry = np.where(
            self.just_crossed_t0,
            self.t0 - installation_date,
            sample_step.entry,
        )
        residual_time = np.where(
            self.just_crossed_t0, timeline - self.t0, sample_step.residual_time
        )

        residual_time = np.where(
            self.just_crossed_tf, residual_time - (timeline - self.tf), residual_time
        )
        event = np.where(self.just_crossed_tf, False, sample_step.event)
        timeline[self.just_crossed_tf] = self.tf

        return SampleStep(residual_time, event, entry), timeline


class StructArrayBuilder:
    sample_id: ArrayND[np.int64]

    def __init__(self, sample_shape: tuple[int, ...]):
        self.sample_id = np.arange(np.prod(sample_shape)).reshape(sample_shape)

    def build_structarray(
        self,
        observed_step: NDArray[np.bool_],
        timeline: NDArray[np.float64],
        sample_step: SampleStep,
    ) -> NDArray[np.void]:

        struct_arr = np.zeros(
            observed_step.sum(),
            dtype=np.dtype(
                [
                    ("timeline", np.float64),
                    ("time", np.float64),
                    ("event", np.bool_),
                    ("entry", np.float64),
                    ("id", np.int64),
                ]
            ),
        )

        struct_arr["timeline"] = timeline[observed_step]
        struct_arr["time"] = (
            sample_step.residual_time[observed_step] + sample_step.entry[observed_step]
        )
        struct_arr["event"] = sample_step.event[observed_step]
        struct_arr["entry"] = sample_step.entry[observed_step]
        struct_arr["id"] = self.sample_id[observed_step]

        return struct_arr

    @staticmethod
    def add_field(
        struct_arr: NDArray[np.void], new_label: str, new_values: NDArray[np.float64]
    ):
        return rfn.append_fields(
            struct_arr,
            new_label,
            new_values,
            (np.dtype(np.float64),),
            usemask=False,
            asrecarray=False,
        )


class StochasticDataIterator(Iterator[NDArray[np.void]], ABC):
    """Abstract class for all stochastic processes iterator.
    Used to build the structarrays, get the shapes, iterate through steps and identify the observation window.
    Abstract method is sample_next_step, that is unique for each stochastic process.
    """  # noqa: E501

    nb_samples: int
    a0: ST | NumpyST | Array1D[NumpyST] | None
    ar: ST | NumpyST | Array1D[NumpyST] | None
    seed: (
        int
        | np.random.Generator
        | np.random.BitGenerator
        | np.random.RandomState
        | None
    )
    sample_shape: tuple[int, ...]
    timeline: ArrayND[np.float64]
    ages: ArrayND[np.float64]
    time_window_observer: TimeWindowObserver
    structarr_builder: StructArrayBuilder
    replacement_cycle: int

    def __init__(
        self,
        process,
        nb_samples: int,
        time_window: tuple[float, float],
        a0: ST | NumpyST | Array1D[NumpyST] | None = None,
        ar: ST | NumpyST | Array1D[NumpyST] | None = None,
        seed: int
        | np.random.Generator
        | np.random.BitGenerator
        | np.random.RandomState
        | None = None,
    ) -> None:
        self.process = process
        self.sample_shape = self._get_rvs_shape(nb_samples, a0, ar)
        self.ar = np.broadcast_to(np.asarray(ar), self.sample_shape)

        self.ages = (
            np.broadcast_to(a0, self.sample_shape)
            if a0 is not None
            else np.zeros(self.sample_shape)
        )
        self.timeline = np.zeros(self.sample_shape)

        self.time_window_observer = TimeWindowObserver(
            sample_shape=self.sample_shape, time_window=time_window
        )

        self.structarr_builder = StructArrayBuilder(self.sample_shape)

        self.replacement_cycle = 0

        self.seed = seed

    def _get_rvs_shape(
        self,
        nb_samples: int,
        a0: ST | NumpyST | Array1D[NumpyST] | None,
        ar: ST | NumpyST | Array1D[NumpyST] | None,
    ) -> tuple[int, ...]:
        a0_shape = np.array(a0).shape
        ar_shape = np.array(ar).shape
        broadcasted_shape = np.broadcast_shapes(
            a0_shape, ar_shape, self.process.lifetime_model.shape
        )
        return (nb_samples, *broadcasted_shape)

    @property
    @abstractmethod
    def _dynamic_lifetime_model(self) -> ParametricLifetimeModel:
        """
        Use the lifetime model modified at each iteration according to each stochastic process specific properties
        """  # noqa: E501

    @abstractmethod
    def update_ages(
        self,
        time: NDArray[np.float64],
    ) -> None:
        """
        Update ages at each iteration according to each stochastic process specific properties
        """  # noqa: E501

    def sample_step(
        self,
    ) -> SampleStep:

        residual_time = self._dynamic_lifetime_model.rvs(
            self.sample_shape,
            seed=self.seed,
        )

        residual_time = np.asarray(residual_time)

        event = np.ones_like(residual_time, dtype=np.bool_)
        entry = self.ages.copy()

        if self.ar is not None:
            preventive_replacements = (self.ages + residual_time) >= self.ar
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

        struct_arr = self.structarr_builder.build_structarray(
            self.time_window_observer.observed_step, self.timeline, sample_step
        )

        self.update_ages(sample_step.residual_time)
        self.replacement_cycle += 1

        return struct_arr

    @override
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
    @property
    def _dynamic_lifetime_model(self) -> ParametricLifetimeModel:
        return (
            LeftTruncatedModel(self.process.first_lifetime_model, self.ages)
            if self.replacement_cycle == 0
            else self.process.lifetime_model
        )

    def update_ages(
        self,
        residual_time: NDArray[np.float64],
    ):
        """
        In a Renewal process, ages are reset to 0 after each iteration.
        """
        self.ages = np.zeros(self.sample_shape, dtype=np.float64)


class RenewalRewardProcessIterator(RenewalProcessIterator):
    @override
    def make_one_step(self):
        sample_step = self.sample_step()
        sample_step = self.apply_observation_bias(sample_step)

        struct_arr = self.structarr_builder.build_structarray(
            self.time_window_observer.observed_step, self.timeline, sample_step
        )
        struct_arr = self.structarr_builder.add_field(
            struct_arr,
            "reward",
            self.process.reward.sample(struct_arr["time"])
            * self.process.discounting.factor(struct_arr["timeline"]),
        )

        self.update_ages(sample_step.residual_time)
        self.replacement_cycle += 1

        return struct_arr


class NonHomogeneousPoissonProcessIterator(StochasticDataIterator):
    @property
    def _dynamic_lifetime_model(self) -> ParametricLifetimeModel:
        # Apply a Left truncation based on current ages on the model
        # self.ages is always 1d in LeftTruncatedModel
        return LeftTruncatedModel(self.process.lifetime_model, self.ages)

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
        a0: ST | NumpyST | Array1D[NumpyST] | None = None,
        ar: ST | NumpyST | Array1D[NumpyST] | None = None,
        seed=None,
    ) -> None:
        super().__init__(
            process,
            nb_samples,
            time_window,
            ar=ar,
            a0=a0,
            seed=seed,
        )
        self.virtual_ages = self.ages.copy()

    @property
    def _dynamic_lifetime_model(self) -> ParametricLifetimeModel:
        # Apply a Left truncation based on current ages on the model
        # self.ages is always 1d in LeftTruncatedModel
        return LeftTruncatedModel(self.process.lifetime_model, self.virtual_ages)

    @override
    def make_one_step(self):
        sample_step = self.sample_step()
        sample_step = self.apply_observation_bias(sample_step)

        struct_arr = self.structarr_builder.build_structarray(
            self.time_window_observer.observed_step, self.timeline, sample_step
        )
        struct_arr = self.structarr_builder.add_field(
            struct_arr,
            "virtual_age",
            self.virtual_ages[self.time_window_observer.observed_step],
        )

        self.update_ages(sample_step.residual_time)
        self.replacement_cycle += 1

        return struct_arr


class Kijima1ProcessIterator(VirtualAgeProcessIterator):
    def update_ages(
        self,
        residual_time: NDArray[np.float64],
    ):
        """
        In a Kijima Process, the concept of age is virtual, and depends on the q parameter of the process
        """  # noqa: E501
        # Update asset ages
        self.virtual_ages += self.process.q * residual_time
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
        """  # noqa: E501
        # Update asset ages
        self.virtual_ages = self.process.q * (self.virtual_ages + residual_time)
        self.ages += residual_time

        if self.ar is not None:
            self.ages[self.ages >= self.ar] = 0
            self.virtual_ages[self.ages >= self.ar] = 0
