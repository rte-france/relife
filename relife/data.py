"""Dataclasses for survival analysis and renewal processes outputs."""

# Copyright (c) 2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# SPDX-License-Identifier: Apache-2.0 (see LICENSE.txt)
# This file is part of ReLife, an open source Python library for asset
# management based on reliability theory and lifetime data analysis.

from dataclasses import dataclass, astuple
from typing import Tuple, NamedTuple
import numpy as np

from .utils import args_size, args_take


@dataclass
class LifetimeData:
    """Lifetime data.

    Dataclass of lifetime data required by the maximum likelihood estimation.
    """

    time: np.ndarray  #: Age of the assets.
    event: np.ndarray = None  #: Type of event, by default None.
    entry: np.ndarray = None  #: Age of assets at the beginning of the observation period (left truncation), by default None.
    args: Tuple[np.ndarray] = ()  #: Extra arguments required by the lifetime model.

    def __post_init__(self) -> None:
        self._parse_data()
        self._format_data()
        self.size = self.time.size

    def _parse_data(self) -> None:
        """Parse lifetime data and check values.

        Notes
        -----
        Default value for `event` is 1 (no censoring), default value for `entry` is
        0 (no truncation).
        """
        if self.event is None:
            self.event = np.ones_like(self.time, int)
        if self.entry is None:
            self.entry = np.zeros_like(self.time, float)
        if np.any(self.time <= 0):
            raise ValueError("time values must be strictly positive")
        if not np.all(np.isin(self.event, [0, 1, 2])):
            raise ValueError("event values must be in [0,1,2]")
        if np.any(self.entry < 0):
            raise ValueError("entry values must be positive")
        if np.any(self.time <= self.entry):
            raise ValueError("entry must be strictly lower than the time to event")
        s = args_size(*self.args)
        if s > 0 and s != np.size(self.time):
            raise ValueError(
                "dimension mismatch for optional args: expected {} got {}".format(
                    np.size(self.time), s
                )
            )

    class DataByEvent(NamedTuple):
        """Group data by type of event."""

        D: np.ndarray  #: observed event.
        D_RC: np.ndarray  #: union of observed events and right-censored data.
        LC: np.ndarray  #: left-censored data.
        LT: np.ndarray  #: left-truncated data.

    def _format_data(self) -> None:
        """Format data according to DataByEvent categories.

        Notes
        -----
        Used in negative log-likelihood calculation in parametric.py.
        """
        # Event Observed, Event Observed + Right Censoring, Left Censoring, Left Truncation
        D, D_RC, LC, LT = map(
            np.nonzero,
            [
                self.event == 1,
                (self.event == 1) + (self.event == 0),
                self.event == 2,
                self.entry > 0,
            ],
        )
        self._time = self.DataByEvent(
            *[self.time[ind].reshape(-1, 1) for ind in [D, D_RC, LC]],
            self.entry[LT].reshape(-1, 1),
        )
        self._args = self.DataByEvent(
            *[args_take(ind[0], *self.args) for ind in [D, D_RC, LC, LT]]
        )

    def __getitem__(self, key):
        return LifetimeData(
            self.time[key], self.event[key], self.entry[key], args_take(key, *self.args)
        )

    def astuple(self) -> Tuple[np.ndarray, ...]:
        """Converts the dataclass attributes as a tuple.

        Returns
        -------
        Tuple[ndarray]
            The attributes of the class as the tuple
            `(time, event, entry, *args)`.
        """
        return self.time, self.event, self.entry, *self.args


@dataclass
class CountData:
    """Counting process data."""

    T: float  #: Time at the end of the observation.
    n_indices: int  #: Number of initial assets.
    n_samples: int  #: Numbers of samples.
    indices: np.ndarray  #: Indices of assets.
    samples: np.ndarray  #: Indices of samples.
    times: np.ndarray  #: Times of observed events.

    @property
    def size(self) -> int:
        """The number of indices.

        Returns
        -------
        int
            Size of indices array.
        """
        return self.indices.size

    def number_of_events(self, sample: int) -> Tuple[np.ndarray, np.ndarray]:
        """Counts the number of events with respect to times.

        Parameters
        ----------
        sample : int
            Index of the sample.

        Returns
        -------
        Tuple[ndarray, ndarray]
            The ordered times and total number of events as a tuple `(times, counts)`.
        """
        ind = (self.samples == sample) & (self.times <= self.T)
        times = np.insert(np.sort(self.times[ind]), 0, 0)
        counts = np.arange(times.size)
        return times, counts

    def mean_number_of_events(self) -> Tuple[np.ndarray, np.ndarray]:
        """Mean number of events with respect to time.

        Returns
        -------
        Tuple[ndarray, ndarray]
            The ordered times and mean number of events as a tuple `(times, counts)`.
        """
        ind = self.times <= self.T
        times = np.insert(np.sort(self.times[ind]), 0, 0)
        counts = np.arange(times.size) / self.n_samples
        return times, counts

    def astuple(self) -> tuple:
        """Converts the dataclass attributes as a tuple."""
        return astuple(self)


@dataclass
class RenewalData(CountData):
    """Renewal process data.

    Notes
    -----
    Inherit from CountData and add a `durations` attribute.
    """

    durations: np.ndarray  #: Time between events.


@dataclass
class RenewalRewardData(RenewalData):
    """Renewal reward process data.

    Notes
    -----
    Inherit from RenewalData and add a `rewards` attribute.
    """

    rewards: np.ndarray  #: Reward associated at each event.

    def total_reward(self, sample: int) -> Tuple[np.ndarray, np.ndarray]:
        """Total reward with respect to time.

        Parameters
        ----------
        sample : int
            Index of the sample.

        Returns
        -------
        Tuple[ndarray, ndarray]
            The ordered times and total rewards as a tuple `(times, z)`.
        """
        ind = (self.samples == sample) & (self.times <= self.T)
        s = np.argsort(self.times[ind])
        times = np.insert(self.times[ind][s], 0, 0)
        z = np.insert(self.rewards[ind][s].cumsum(), 0, 0)
        return times, z

    def mean_total_reward(self) -> Tuple[np.ndarray, np.ndarray]:
        """Mean total reward with respect to time.

        Returns
        -------
        Tuple[ndarray, ndarray]
            The ordered times and mean total rewards as a tuple `(times, z)`.
        """
        ind = self.times <= self.T
        s = np.argsort(self.times[ind])
        times = np.insert(self.times[ind][s], 0, 0)
        z = np.insert(self.rewards[ind][s].cumsum(), 0, 0) / self.n_samples
        return times, z


@dataclass
class ReplacementPolicyData(RenewalRewardData):
    """Replacement policy data."""

    events: np.ndarray  #: Event types.
    args: np.ndarray  #: Extra arguments required by the lifetime model.
    a0: np.ndarray  #: Age of the assets at the first replacement.

    def to_lifetime_data(
        self, t0: float = 0, tf: float = None, sample: int = None
    ) -> LifetimeData:
        """Builds a lifetime data sample.

        Parameters
        ----------
        t0 : float, optional
            Start of the observation period, by default 0.
        tf : float, optional
            End of the observation period, by default the time at the end of the
            observation.
        sample : int, optional
            Index of the sample, by default all sample are mixed.

        Returns
        -------
        LifetimeData
            The lifetime data sample built from the observation period `[t0,tf]`
            of the renewal process.

        Raises
        ------
        ValueError
            if `t0` is greater than `tf`.
        """
        if tf is None or tf > self.T:
            tf = self.T
        if t0 >= tf:
            raise ValueError("`t0` must be strictly lesser than `tf`")

        # Filtering sample and sorting by times
        s = self.samples == sample if sample is not None else Ellipsis
        order = np.argsort(self.times[s])
        indices = self.indices[s][order]
        samples = self.samples[s][order]
        uindices = np.ravel_multi_index(
            (indices, samples), (self.n_indices, self.n_samples)
        )
        times = self.times[s][order]
        durations = self.durations[s][order] + self.a0[s][order]
        events = self.events[s][order]

        # Indices of interest
        ind0 = (times > t0) & (
            times <= tf
        )  # Indices of replacement occuring inside the obervation window
        ind1 = (
            times > tf
        )  # Indices of replacement occuring after the observation window which include right censoring

        # Replacements occuring inside the observation window
        time0 = durations[ind0]
        event0 = events[ind0]
        entry0 = np.zeros(time0.size)
        _, LT = np.unique(
            uindices[ind0], return_index=True
        )  # get the indices of the first replacements ocurring in the observation window
        b0 = (
            times[ind0][LT] - durations[ind0][LT]
        )  # time at birth for the firt replacements
        entry0[LT] = np.where(b0 >= t0, 0, t0 - b0)
        args0 = args_take(indices[ind0], *self.args)

        # Right censoring
        _, RC = np.unique(uindices[ind1], return_index=True)
        bf = (
            times[ind1][RC] - durations[ind1][RC]
        )  # time at birth for the right censored
        b1 = bf[
            bf < tf
        ]  # ensure that time of birth for the right censored is not equal to tf.
        time1 = tf - b1
        event1 = np.zeros(b1.size)
        entry1 = np.where(b1 >= t0, 0, t0 - b1)
        args1 = args_take(indices[ind1][RC][bf < tf], *self.args)

        # Concatenate
        time = np.concatenate((time0, time1))
        event = np.concatenate((event0, event1))
        entry = np.concatenate((entry0, entry1))
        args = tuple(
            np.concatenate((arg0, arg1), axis=0) for arg0, arg1 in zip(args0, args1)
        )
        return LifetimeData(time, event, entry, args)
