"""
Below iterators are used in policies to :
1. sample and create CountData object that holds sampled data to compute empirical costs and plots
2. sample and return arrays that can be used to fit the model's policy

Those iterators needs a model and model_args. They are set manually so one can decompose each iteration
steps in case there is a model1

They hold t0 and tf. tf controls the stop iteration condition. If t0 is not zero, the iterator can return
a selection of empty arrays

They also memory the timeline array internally, it serves to compute the stop condition with t0 and tf but
it is not returned.

All returned arrays are 1d.

The iterotors allows to :
- isolate the iteration code in order to reuse it in several parts
- decompose each iteration steps with next command
"""

from typing import Optional
import numpy as np
from collections.abc import Iterator


from relife.models import Exponential
from relife.types import Arg


def get_nb_assets(args_tuple: tuple[Arg, ...]) -> int:
    def as_2d():
        for x in args_tuple:
            if not isinstance(x, np.ndarray):
                x = np.asarray(x)
            if len(x.shape) > 2:
                raise ValueError
            yield np.atleast_2d(x)

    return max(map(lambda x: x.shape[0], as_2d()), default=1)


class LifetimeIterator(Iterator):
    """
    returns time, event_indicators, entries in 2D  - shape : (nb_assets, nb_samples)
    censoring and truncations only based on t0 and tf values

    selection is done in iterable

    note that model_args is not constructed yet, in sample.py
    """

    def __init__(
        self,
        size: int,
        tf: float,
        t0: float = 0.0,
        *,
        seed: Optional[int] = None,
    ):

        self.size = size
        self.tf = tf
        self.t0 = t0
        self.seed = seed

        # hidden attributes, control set/get interface
        self._model = None
        self._model_args = None
        self._model_type = None
        self._nb_assets = None
        self._timeline = None
        self._start = None
        self._stop = None

    def set_model(self, model, model_args):
        if self._model is None:
            self._nb_assets = get_nb_assets(model_args)
            self._timeline = np.zeros((self._nb_assets, self.size))
            # counting arrays to catch values crossing t0 and tf bounds
            self._stop = np.zeros((self._nb_assets, self.size), dtype=np.int64)
            self._start = np.zeros((self._nb_assets, self.size), dtype=np.int64)
        else:
            nb_assets = get_nb_assets(model_args)
            if nb_assets != self._nb_assets:
                raise ValueError("Can't change nb assets")
            if type(model) != self._model_type:
                raise ValueError("Can't change model type")
        self._model = model
        self._model_type = type(model)
        self._model_args = model_args

    @property
    def timeline(self):
        return self._timeline

    @property
    def stop(self):
        """stop condition is based on a counter to keep track of last elements before tf (censoring)"""
        return np.all(self._stop > 0)

    @property
    def nb_samples(self):
        return self.size

    def _sample_routine(self):
        durations = self._model.rvs(
            *self._model_args,
            size=self.nb_samples,
            seed=self.seed,
        ).reshape((self._nb_assets, self.nb_samples))

        # update timeline
        self._timeline += durations

        # count values that have just crossed t0 or tf
        self._start[self._timeline > self.t0] += 1
        self._stop[self.timeline > self.tf] += 1

        left_truncated = self._start == 1
        right_censored = self._stop == 1

        # censor values
        event_indicators = np.ones_like(self.timeline, dtype=np.bool_)
        durations = np.where(
            right_censored, durations - (self._timeline - self.tf), durations
        )
        self._timeline = np.where(right_censored, self.tf, self.timeline)
        event_indicators[right_censored] = False

        # get left truncations
        entries = np.zeros_like(self.timeline)
        entries = np.where(
            left_truncated, self.t0 - (self.timeline - durations), entries
        )

        # update seed to avoid having the same rvs result
        if self.seed is not None:
            self.seed += 1

        return (
            durations,
            event_indicators,
            entries,
        )

    def __next__(self):
        if self._model is None:
            raise ValueError("Set model first")
        while not self.stop:  # recompute stop condition automatically
            return self._sample_routine()
        raise StopIteration


class NonHomogeneousPoissonIterator(Iterator):
    """
    returns a0, af, ages in 2D - shape : (nb_assets, nb_samples)

    selection is done in iterable
    """

    def __init__(
        self,
        size: int,
        tf: float,  # calendar beginning time
        t0: float = 0.0,  # calendar end time
        *,
        seed: Optional[int] = None,
    ):

        self.size = size
        self.tf = tf
        self.t0 = t0
        self.seed = seed

        self._model = None
        self._model_args = None
        self._nb_assets = None
        self._timeline = None
        self._hpp_timeline = None
        self._model_type = None
        self._stop = None

        self._exponential_dist = Exponential(1.0)

    @property
    def nb_samples(self):
        return self.size

    @property
    def timeline(self):
        return self._timeline

    @property
    def stop(self):
        return self._stop

    def set_model(self, model, model_args):
        self._nb_assets = get_nb_assets(model_args)
        self._timeline = np.zeros((self._nb_assets, self.size))
        self._hpp_timeline = np.zeros((self._nb_assets, self.size))

        self._stop = np.all(self._timeline >= self.tf)
        self._model = model
        self._model_type = type(model)
        self._model_args = model_args

    def _sample_routine(self):
        self._hpp_timeline += self._exponential_dist.rvs(
            size=self.nb_samples * self._nb_assets, seed=self.seed
        ).reshape((self._nb_assets, self.nb_samples))
        self._timeline = self._model.ichf(self._hpp_timeline, *self._model_args)
        ages = self._timeline.copy()

        # update seed to avoid having the same rvs result
        if self.seed is not None:
            self.seed += 1

        # update stop condition
        self._stop = np.all(self._timeline >= self.tf)

        return ages

    def __next__(self):
        if self._model is None:
            raise ValueError("Set model first")
        while not self.stop:
            return self._sample_routine()
        raise StopIteration
