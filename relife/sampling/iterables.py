from typing import Iterable, Optional, Callable

import numpy as np
from numpy.typing import NDArray

from relife.core import LifetimeModel, AgeReplacementModel, LeftTruncatedModel
from .iterators import LifetimeIterator, NonHomogeneousPoissonIterator
from relife.types import TupleArrays


class RenewalIterable(Iterable):
    """
    Iterable returning dict of 1D arrays selected from each iterations:
        - samples_ids
        - assets_ids
        - timeline
        - time
        - event
        - entry
        - rewards
    when RenewalIterable is used, one can pick only needed data

    reward_func and discount_func are optional but must take one array (durations) and return one array (costs)
    """

    def __init__(
        self,
        size: int,
        tf: float,
        model: LifetimeModel[*TupleArrays],
        reward_func: Optional[
            Callable[[NDArray[np.float64]], NDArray[np.float64]]
        ] = None,
        discount_factor: Optional[
            Callable[[NDArray[np.float64]], NDArray[np.float64]]
        ] = None,
        model_args: TupleArrays = (),
        model1: Optional[LifetimeModel[*TupleArrays]] = None,
        model1_args: TupleArrays = (),
        t0: float = 0.0,
        seed: Optional[int] = None,
    ):
        self.size = size
        self.tf = tf
        self.t0 = t0

        # control model1/model types and extract a0, ar and ar1 to compute truncations and censorings
        if isinstance(model, AgeReplacementModel):
            self.ar = model_args[0].copy()
            self.model_args = model_args[1:]
            if isinstance(model.baseline, LeftTruncatedModel):
                raise ValueError("Only LefTruncatedModel allowed for model1")
        else:
            self.ar = None
        self.model = model
        self.model_args = model_args

        if isinstance(model1, AgeReplacementModel):
            self.ar1 = model1_args[0].copy()
            if isinstance(model1.baseline, LeftTruncatedModel):
                self.a0 = model1_args[1].copy()
        elif isinstance(model1, LeftTruncatedModel):
            self.a0 = model1_args[0].copy()
            if isinstance(model1.baseline, AgeReplacementModel):
                self.ar1 = model1_args[1].copy()
        else:
            self.ar1 = None
            self.a0 = None
        self.model1 = model1
        self.model1_args = model1_args

        self.reward_func = reward_func
        self.discount_func = discount_factor
        self.seed = seed

        self.returned_dict = {
            "samples_ids": np.array([], dtype=np.int64),
            "assets_ids": np.array([], dtype=np.int64),
            "timeline": np.array([], dtype=np.float64),
            "time": np.array([], dtype=np.float64),
            "event": np.array([], dtype=np.float64),
            "entry": np.array([], dtype=np.float64),
            "rewards": np.array([], dtype=np.float64),
        }

    def update_returned_dict(self, timeline, time, event, entry, selection):
        # parameters are all 2D arrays (nb_assets, nb_assets)
        assets_ids, samples_ids = np.where(selection)

        self.returned_dict["assets_ids"] = assets_ids
        self.returned_dict["samples_ids"] = samples_ids
        self.returned_dict["timeline"] = timeline[selection]
        self.returned_dict["time"] = time[selection]

        rewards = np.zeros_like(time)
        if self.reward_func and self.discount_func:
            rewards = self.reward_func(time) * self.discount_func(time)
        if self.reward_func and not self.discount_func:
            rewards = self.reward_func(time)
        self.returned_dict["rewards"] = rewards[selection]

        self.returned_dict["entry"] = np.max(entry, self.a0)[selection]
        if self.ar is not None:
            self.returned_dict["event"] = np.where(time < self.ar, event, False)[
                selection
            ]
        else:
            self.returned_dict["event"] = entry[selection]
        return self.returned_dict

    def __iter__(self):
        # construct iterator
        iterator = LifetimeIterator(self.size, self.tf, self.t0, seed=self.seed)

        # first cycle : set model1 in iterator
        iterator.set_model(self.model1, self.model1_args)

        # yield first dict of results (return empty arrays if iterator stops at first iteration)
        try:
            time, event, entry = next(iterator)  # 2d returns (nb_assets, nb_samples)
            yield self.update_returned_dict(
                iterator.timeline,
                time,
                event,
                entry,
                self.t0 < iterator.timeline < self.tf,
            )
        except StopIteration:
            yield self.returned_dict

        # next cycles : set model in iterator
        iterator.set_model(self.model, self.model_args)

        # yield dict of results until iterator stops
        for time, event, entry in iterator:
            yield self.update_returned_dict(
                iterator.timeline,
                time,
                event,
                entry,
                self.t0 < iterator.timeline < self.tf,
            )


class NonHomogeneousPoissonIterable(Iterable):
    """
    Iterable returning dict of 1D arrays selected from each iterations
    """

    def __init__(
        self,
        size: int,
        tf: float,
        model: LifetimeModel[*TupleArrays],
        reward_func: Optional[
            Callable[[NDArray[np.float64]], NDArray[np.float64]]
        ] = None,
        discount_factor: Optional[
            Callable[[NDArray[np.float64]], NDArray[np.float64]]
        ] = None,
        model_args: TupleArrays = (),
        t0: float = 0.0,
        seed: Optional[int] = None,
    ):
        self.size = size
        self.tf = tf
        self.t0 = t0

        self.model = model
        self.model_args = model_args

        self.reward_func = reward_func
        self.discount_func = discount_factor
        self.seed = seed

        self.returned_dict = {
            "samples_ids": np.array([], dtype=np.int64),
            "assets_ids": np.array([], dtype=np.int64),
            "ages": np.array([], dtype=np.float64),
            "durations": np.array([], dtype=np.float64),
            "rewards": np.array([], dtype=np.float64),
            "is_a0": np.array([], dtype=np.bool_),  # booleans indicating if ages are a0
            "is_af": np.array([], dtype=np.bool_),  # booleans indicating if ages are af
        }

    def __iter__(self):
        # construct iterator
        iterator = NonHomogeneousPoissonIterator(
            self.size, self.tf, self.t0, seed=self.seed
        )

        # set model in iterator
        iterator.set_model(self.model, self.model_args)

        previous_ages = np.zeros_like(iterator.timeline)
        cpt_a0 = np.zeros_like(
            iterator.timeline, dtype=np.int64
        )  # count iterator to catch a0 (first iteration only)

        # yield dict of results until iterator stops
        for ages in iterator:
            selection = self.t0 < iterator.timeline < self.tf
            assets_ids, samples_ids = np.where(selection)
            self.returned_dict["assets_ids"] = assets_ids
            self.returned_dict["samples_ids"] = samples_ids
            self.returned_dict["ages"] = ages[selection]

            # update cpt_a0 and catch a0 ages
            cpt_a0[selection] += 1
            self.returned_dict["is_a0"] = np.where(cpt_a0 == 1, True, False)[selection]

            # check if iterator will stop at the next iteration:
            self.returned_dict["is_af"] = np.where(ages >= self.tf, True, False)[
                selection
            ]

            self.returned_dict["durations"] = (ages - previous_ages)[selection]
            previous_ages = ages.copy()

            rewards = np.zeros_like(ages)
            if self.reward_func and self.discount_func:
                rewards = self.reward_func(ages) * self.discount_func(ages)
            if self.reward_func and not self.discount_func:
                rewards = self.reward_func(ages)
            self.returned_dict["rewards"] = rewards[selection]

            yield self.returned_dict
