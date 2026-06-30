from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np
from numpy.lib import recfunctions as rfn
from optype.numpy import Array, Array1D, ArrayND, AtMost1D
from typing_extensions import override

from relife.lifetime_models import ParametricLifetimeModel
from relife.stochastic_processes import (
    Kijima1Process,
    Kijima2Process,
    NonHomogeneousPoissonProcess,
    RenewalProcess,
    RenewalRewardProcess,
)
from relife.typing import ST, NumpyST

PT = TypeVar(
    "PT",
    RenewalProcess,
    RenewalRewardProcess,
    Kijima1Process[()],
    Kijima2Process[()],
    NonHomogeneousPoissonProcess[()],
)


class StochasticDataIterable(Iterable[ArrayND[np.void]], Generic[PT], ABC):
    process: PT
    nb_samples: int
    time_window: tuple[float, float]
    a0: ST | NumpyST | Array1D[NumpyST] | None
    ar: ST | NumpyST | Array1D[NumpyST] | None
    seed: (
        int
        | np.random.Generator
        | np.random.BitGenerator
        | np.random.RandomState
        | None
    )

    def __init__(
        self,
        process: PT,
        nb_samples: int,
        time_window: tuple[float, float],
        a0: ST | NumpyST | Array1D[NumpyST] | None = None,
        ar: ST | NumpyST | Array1D[NumpyST] | None = None,
        seed: int
        | np.random.Generator
        | np.random.BitGenerator
        | np.random.RandomState
        | None = None,
    ):
        self.process = process

        t0, tf = time_window
        if t0 < 0 or tf < 0 or t0 > tf:
            raise ValueError(
                f"Incorrect time window. Got {time_window}. Values must be positive and first value can't lower than second value."  # noqa: E501
            )
        self.time_window = t0, tf

        self.a0 = a0
        self.ar = ar
        self.nb_samples = nb_samples
        self.seed = seed

    @property
    def t0(self) -> float:
        return self.time_window[0]

    @property
    def tf(self) -> float:
        return self.time_window[1]

    @override
    @abstractmethod
    def __iter__(self) -> StochasticDataIterator[PT]: ...


@dataclass
class SampleStep:
    residual_time: ArrayND[np.float64]
    event: ArrayND[np.bool_]
    entry: ArrayND[np.float64]


class TimeWindowObserver:
    t0: float
    tf: float
    _crossed_t0_counter: ArrayND[np.int64]
    _crossed_tf_counter: ArrayND[np.int64]

    def __init__(self, sample_shape: tuple[int, ...], time_window: tuple[float, float]):
        self.t0, self.tf = time_window
        self._crossed_t0_counter = np.zeros(sample_shape, dtype=np.int64)
        self._crossed_tf_counter = np.zeros(sample_shape, dtype=np.int64)

    def update(self, timeline: ArrayND[np.float64]):
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
        self, sample_step: SampleStep, timeline: ArrayND[np.float64]
    ) -> tuple[SampleStep, ArrayND[np.float64]]:

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
        observed_step: ArrayND[np.bool_],
        timeline: ArrayND[np.float64],
        sample_step: SampleStep,
    ) -> ArrayND[np.void]:

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
        struct_arr: ArrayND[np.void], new_label: str, new_values: ArrayND[np.float64]
    ) -> ArrayND[np.void]:
        return rfn.append_fields(
            struct_arr,
            new_label,
            new_values,
            (np.dtype(np.float64),),
            usemask=False,
            asrecarray=False,
        )


class StochasticDataIterator(Iterator[ArrayND[np.void]], Generic[PT], ABC):
    """Abstract class for all stochastic processes iterator.
    Used to build the structarrays, get the shapes, iterate through steps and identify the observation window.
    """  # noqa: E501

    process: PT
    nb_samples: int
    a0: ST | NumpyST | Array1D[NumpyST] | None
    ar: Array[AtMost1D, NumpyST] | None
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
        process: PT,
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
    def _dynamic_lifetime_model(self) -> ParametricLifetimeModel[()]:
        """
        Use the lifetime model modified at each iteration according to each stochastic process specific properties
        """  # noqa: E501

    @abstractmethod
    def update_ages(
        self,
        time: ArrayND[np.float64],
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
    def __next__(self) -> ArrayND[np.void]:
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
