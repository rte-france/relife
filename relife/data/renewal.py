from dataclasses import dataclass, field
from itertools import filterfalse
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from relife.types import TupleArrays

from .counting import CountData, CountDataIterable


@dataclass
class RenewalData(CountData):
    lifetimes: NDArray[np.float64] = field(repr=False)
    events: NDArray[np.bool_] = field(
        repr=False
    )  # event indicators (right censored or not)

    model_args: TupleArrays = field(repr=False)
    with_model1: bool = field(repr=False)

    def iter(self, sample_id: Optional[int] = None):
        if sample_id is None:
            return CountDataIterable(self, ("event_times", "lifetimes", "events"))
        else:
            if sample_id not in self.samples_unique_ids:
                raise ValueError(f"{sample_id} is not part of samples index")
            return filterfalse(
                lambda x: x[0] != sample_id,
                CountDataIterable(self, ("event_times", "lifetimes", "events")),
            )

    def _get_args(
        self, index, previous_args: Optional[tuple[NDArray[np.float64], ...]] = None
    ) -> tuple[NDArray[np.float64], ...]:
        args = ()
        if bool(self.model_args):
            if self.nb_assets > 1:
                if previous_args:
                    args = tuple(
                        (
                            np.concatenate((p, np.take(a, index, axis=0)))
                            for p, a in zip(previous_args, self.model_args)
                        )
                    )
                else:
                    args = tuple((np.take(a, index, axis=0) for a in self.model_args))
            if self.nb_assets == 1:
                if previous_args:
                    args = tuple(
                        (
                            np.concatenate((p, np.tile(a, (len(index), 1))))
                            for p, a in zip(previous_args, self.model_args)
                        )
                    )
                else:
                    args = tuple((np.tile(a, (len(index), 1)) for a in self.model_args))
        return args

    def to_fit(
        self,
        t0: float = 0,
        tf: Optional[float] = None,
        sample: Optional[int] = None,
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.bool_],
        NDArray[np.float64],
        None,
        tuple[NDArray[np.float64], ...],
    ]:
        """
        Prepares and filters the dataset attributes for model fitting on specified
        time constraints and optional sample ID.

        Parameters
        ----------
        t0 : float, optional
            The lower bound of the time interval (inclusive). Default is 0.
        tf : float, optional
            The upper bound of the time interval (inclusive). If not provided, the
            maximum event time in the dataset is used.
        sample : int, optional
            An identifier for filtering a subset of the dataset. If provided, only
            data associated with this sample ID is processed. Default is None.

        Returns
        -------
        tuple of np.ndarray
            A tuple containing:
                - time: An array of observed lifetimes.
                - event: A boolean array indicating events (True for observed events,
                          False for censored).
                - entry: An array of entry times representing the left truncation values.
                - None: A placeholder returning no additional value for this component.
                - args: Additional arguments as a tuple of arrays prepared for
                        subsequent operations.
        """

        max_event_time = np.max(self.timeline)
        if tf > max_event_time:
            tf = max_event_time

        if t0 >= tf:
            raise ValueError("`t0` must be strictly lower than `tf`")

        time = np.array([], dtype=np.float64)
        event = np.array([], dtype=np.bool_)
        entry = np.array([], dtype=np.float64)

        s = self.samples_ids == sample if sample is not None else Ellipsis

        complete_left_truncated = (self.timeline[s] > t0) & (self.timeline[s] <= tf)

        _timeline = self.timeline[s][complete_left_truncated]
        _lifetimes = self.lifetimes[s][complete_left_truncated]
        _events = self.events[s][complete_left_truncated]
        _assets_index = self.assets_ids[s][complete_left_truncated]

        shift_left = _timeline - _lifetimes
        left_truncated = (t0 - shift_left) >= 0
        left_truncations = (t0 - shift_left)[left_truncated]

        time = np.concatenate(
            (time, _lifetimes[left_truncated], _lifetimes[~left_truncated])
        )
        event = np.concatenate(
            (
                event,
                np.ones_like(left_truncations, dtype=np.bool_),
                _events[~left_truncated],
            )
        )
        entry = np.concatenate(
            (entry, left_truncations, np.zeros_like(_lifetimes[~left_truncated]))
        )

        args = self._get_args(_assets_index[left_truncated])
        args = self._get_args(_assets_index[~left_truncated], previous_args=args)

        right_censored = self.timeline[s] > tf

        _timeline = self.timeline[s][right_censored]
        _lifetimes = self.lifetimes[s][right_censored]
        _events = self.events[s][right_censored]
        _assets_index = self.assets_ids[s][right_censored]

        shift_left = _timeline - _lifetimes
        right_censored = (tf - shift_left) >= 0
        right_censoring = (tf - shift_left)[right_censored]

        time = np.concatenate((time, right_censoring))
        event = np.concatenate((event, np.zeros_like(right_censoring, dtype=np.bool_)))
        entry = np.concatenate((entry, np.zeros_like(right_censoring)))

        args = self._get_args(_assets_index[right_censored], previous_args=args)

        return time, event, entry, None, args


@dataclass
class RenewalRewardData(RenewalData):
    total_rewards: NDArray[np.float64] = field(repr=False)

    def cum_total_rewards(
        self, sample: int
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        ind = self.samples_ids == sample
        s = np.argsort(self.timeline[ind])
        times = np.insert(self.timeline[ind][s], 0, 0)
        z = np.insert(self.total_rewards[ind][s].cumsum(), 0, 0)
        return times, z

    def mean_total_reward(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        s = np.argsort(self.timeline)
        times = np.insert(self.timeline[s], 0, 0)
        z = np.insert(self.total_rewards[s].cumsum(), 0, 0) / self.nb_samples
        return times, z
