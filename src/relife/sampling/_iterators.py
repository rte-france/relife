from abc import ABC
from typing import TypeVar, reveal_type

import numpy as np
from optype.numpy import Array1D, ArrayND
from typing_extensions import override

from relife.lifetime_models._base import (
    ParametricLifetimeModel,
)
from relife.stochastic_processes import (
    Kijima1Process,
    Kijima2Process,
    NonHomogeneousPoissonProcess,
    RenewalProcess,
)
from relife.typing import ST, NumpyST

from ._base import StochasticDataIterator


class RenewalProcessIterator(StochasticDataIterator[RenewalProcess]):
    ages: ArrayND[np.float64]

    @property
    @override
    def _dynamic_lifetime_model(self) -> ParametricLifetimeModel[()]:
        return (
            self.process.first_lifetime_model.apply_condition(a0=self.ages)
            if self.replacement_cycle == 0
            else self.process.lifetime_model
        )

    @override
    def update_ages(
        self,
        time: ArrayND[np.float64],
    ) -> None:
        """
        In a Renewal process, ages are reset to 0 after each iteration.
        """
        self.ages = np.zeros(self.sample_shape, dtype=np.float64)


# TODO : group RenewalProcessIterator and RenewalRewardProcessIterator ?
# here process has no reward
# make_one_step must treat both cases, with and without reward, in one class
class RenewalRewardProcessIterator(RenewalProcessIterator):
    @override
    def make_one_step(self):
        sample_step = self.sample_step()
        sample_step = self.apply_observation_bias(sample_step)

        struct_arr = self.structarr_builder.build_structarray(
            self.time_window_observer.observed_step, self.timeline, sample_step
        )
        reveal_type(self.process)
        struct_arr = self.structarr_builder.add_field(
            struct_arr,
            "reward",
            self.process.reward.sample(struct_arr["time"])
            * self.process.discounting.factor(struct_arr["timeline"]),
        )

        self.update_ages(sample_step.residual_time)
        self.replacement_cycle += 1

        return struct_arr


class NonHomogeneousPoissonProcessIterator(
    StochasticDataIterator[NonHomogeneousPoissonProcess[()]]
):
    ages: ArrayND[np.float64]

    @property
    @override
    def _dynamic_lifetime_model(self) -> ParametricLifetimeModel[()]:
        # Apply a Left truncation based on current ages on the model
        # self.ages is always 1d in LeftTruncatedModel
        return self.process.lifetime_model.apply_condition(a0=self.ages)

    @override
    def update_ages(
        self,
        time: ArrayND[np.float64],
    ):
        """
        In a NHPP, ages are reset to 0 only when a replacement is made
        """
        # Update asset ages
        self.ages += time

        if self.ar is not None:
            self.ages[self.ages >= self.ar] = 0


# narrowed typevar
KPT = TypeVar(
    "KPT",
    Kijima1Process[()],
    Kijima2Process[()],
)


class VirtualAgeProcessIterator(StochasticDataIterator[KPT], ABC):
    virtual_ages: ArrayND[np.float64]
    replacement_cycle: int

    def __init__(
        self,
        process: KPT,
        nb_samples: int,
        time_window: tuple[float, float],
        a0: ST | NumpyST | Array1D[NumpyST] | None = None,
        ar: ST | NumpyST | Array1D[NumpyST] | None = None,
        seed: (
            int
            | np.random.Generator
            | np.random.BitGenerator
            | np.random.RandomState
            | None
        ) = None,
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
    @override
    def _dynamic_lifetime_model(self) -> ParametricLifetimeModel[()]:
        # Apply a Left truncation based on current ages on the model
        # self.ages is always 1d in LeftTruncatedModel
        return self.process.lifetime_model.apply_condition(a0=self.virtual_ages)

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


class Kijima1ProcessIterator(VirtualAgeProcessIterator[Kijima1Process[()]]):
    ages: ArrayND[np.float64]
    virtual_ages: ArrayND[np.float64]

    @override
    def update_ages(
        self,
        time: ArrayND[np.float64],
    ):
        """
        In a Kijima Process, the concept of age is virtual, and depends on the q parameter of the process
        """  # noqa: E501
        # Update asset ages
        self.virtual_ages += self.process.q * time
        self.ages += time

        if self.ar is not None:
            self.ages[self.ages >= self.ar] = 0
            self.virtual_ages[self.ages >= self.ar] = 0


class Kijima2ProcessIterator(VirtualAgeProcessIterator[Kijima2Process[()]]):
    ages: ArrayND[np.float64]
    virtual_ages: ArrayND[np.float64]

    @override
    def update_ages(
        self,
        time: ArrayND[np.float64],
    ):
        """
        In a Kijima Process, the concept of age is virtual, and depends on the q parameter of the process
        """  # noqa: E501
        # Update asset ages
        self.virtual_ages = self.process.q * (self.virtual_ages + time)
        self.ages += time

        if self.ar is not None:
            self.ages[self.ages >= self.ar] = 0
            self.virtual_ages[self.ages >= self.ar] = 0
