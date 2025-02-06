from dataclasses import dataclass, field
from itertools import filterfalse
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from relife.types import ModelArgs

from .counting import CountData, CountDataIterable


@dataclass
class RenewalData(CountData):
    lifetimes: NDArray[np.float64] = field(repr=False)
    events: NDArray[np.bool_] = field(
        repr=False
    )  # event indicators (right censored or not)

    model_args: ModelArgs = field(repr=False)
    with_model1: bool = field(repr=False)

    def iter(self, sample: Optional[int] = None):
        if sample is None:
            return CountDataIterable(self, ("event_times", "lifetimes", "events"))
        else:
            if sample not in self.samples_unique_index:
                raise ValueError(f"{sample} is not part of samples index")
            return filterfalse(
                lambda x: x[0] != sample,
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
        consider only model_args (not model1_args)
        if t0 is lower than first event_times => raise  Error

        Parameters
        ----------
        t0 : start (time) of the observation period
        tf : end (time) of the observation period
        sample :

        Returns
        -------

        """

        max_event_time = np.max(self.event_times)
        if tf > max_event_time:
            tf = max_event_time

        if t0 >= tf:
            raise ValueError("`t0` must be strictly lower than `tf`")

        time = np.array([], dtype=np.float64)
        event = np.array([], dtype=np.bool_)
        entry = np.array([], dtype=np.float64)

        s = self.samples_index == sample if sample is not None else Ellipsis

        complete_left_truncated = (self.event_times[s] > t0) & (
            self.event_times[s] <= tf
        )

        _timeline = self.event_times[s][complete_left_truncated]
        _lifetimes = self.lifetimes[s][complete_left_truncated]
        _events = self.events[s][complete_left_truncated]
        _assets_index = self.assets_index[s][complete_left_truncated]

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

        right_censored = self.event_times[s] > tf

        _timeline = self.event_times[s][right_censored]
        _lifetimes = self.lifetimes[s][right_censored]
        _events = self.events[s][right_censored]
        _assets_index = self.assets_index[s][right_censored]

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
        ind = self.samples_index == sample
        s = np.argsort(self.event_times[ind])
        times = np.insert(self.event_times[ind][s], 0, 0)
        z = np.insert(self.total_rewards[ind][s].cumsum(), 0, 0)
        return times, z

    def mean_total_reward(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        s = np.argsort(self.event_times)
        times = np.insert(self.event_times[s], 0, 0)
        z = np.insert(self.total_rewards[s].cumsum(), 0, 0) / self.nb_samples
        return times, z
