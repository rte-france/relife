from functools import singledispatch
from typing import Union, Optional

import numpy as np

from relife.data import RenewalData

from relife.process import RenewalRewardProcess, NonHomogeneousPoissonProcess
from relife.process.renewal import RenewalProcess
from relife.policies import (
    OneCycleRunToFailurePolicy,
    DefaultRunToFailurePolicy,
    OneCycleAgeReplacementPolicy,
    DefaultAgeReplacementPolicy,
    NonHomogeneousPoissonAgeReplacementPolicy,
)
from relife.rewards import (
    age_replacement_rewards,
    run_to_failure_rewards,
)
from .iterators import LifetimeIterator, NonHomogeneousPoissonIterator
from relife.process import NonHomogeneousPoissonProcessWithRewards
from relife.data.counting import NHPPCountData


# noinspection PyUnusedLocal
@singledispatch
def sample_count_data(
    obj,
    size: int,
    tf: float,
    t0: float = 0.0,
    maxsample: int = 1e5,
    seed: Optional[int] = None,
):
    raise ValueError(f"No sample for {type(obj)}")


@sample_count_data.register
def _(
    obj: Union[RenewalProcess, RenewalRewardProcess],
    size: int,
    tf: float,
    t0: float = 0.0,
    maxsample: int = 1e5,
    seed: Optional[int] = None,
):
    durations = np.array([], dtype=np.float64)
    timeline = np.array([], dtype=np.float64)
    samples_ids = np.array([], dtype=np.int64)
    assets_ids = np.array([], dtype=np.int64)

    iterator = LifetimeIterator(size, tf, t0, nb_assets=obj.nb_assets, seed=seed)

    rewards = None
    if isinstance(obj, RenewalRewardProcess):
        rewards = np.array([], dtype=np.float64)
        iterator.set_rewards(obj.rewards1)
        iterator.set_discounting(obj.discounting)

    # first cycle : set model1 in iterator
    if obj.model1 is not None:
        iterator.set_sampler(obj.model1, obj.model1_args)
        try:
            data = next(iterator)
            durations = np.concatenate((durations, data["durations"]))
            timeline = np.concatenate((timeline, data["timeline"]))
            samples_ids = np.concatenate((samples_ids, data["samples_ids"]))
            assets_ids = np.concatenate((assets_ids, data["assets_ids"]))
            if isinstance(obj, RenewalRewardProcess):
                rewards = np.concatenate((rewards, data["rewards"]))

        except StopIteration:
            return RenewalData(
                t0,
                tf,
                samples_ids,
                assets_ids,
                timeline,
                durations,
                rewards,
            )

    # next cycles : set model in iterator and change rewards
    iterator.set_sampler(obj.model, obj.model_args)
    if isinstance(obj, RenewalRewardProcess):
        iterator.set_rewards(obj.rewards)

    for data in iterator:
        if data["timeline"].size == 0:
            continue
        durations = np.concatenate((durations, data["durations"]))
        timeline = np.concatenate((timeline, data["timeline"]))
        samples_ids = np.concatenate((samples_ids, data["samples_ids"]))
        assets_ids = np.concatenate((assets_ids, data["assets_ids"]))
        if isinstance(obj, RenewalRewardProcess):
            rewards = np.concatenate((rewards, data["rewards"]))

        if len(samples_ids) > maxsample:
            raise ValueError(
                f"Max number of sample has been reach : {maxsample}. Modify maxsample or set different arguments"
            )

    if rewards is None:
        rewards = np.zeros_like(timeline)

    return RenewalData(
        t0,
        tf,
        samples_ids,
        assets_ids,
        timeline,
        durations,
        rewards,
    )


@sample_count_data.register
def _(
    obj: OneCycleRunToFailurePolicy,
    size: int,
    tf: float,
    t0: float = 0.0,
    maxsample: int = 1e5,
    seed: Optional[int] = None,
):

    iterator = LifetimeIterator(size, tf, t0, nb_assets=obj.nb_assets, seed=seed)
    iterator.set_rewards(run_to_failure_rewards(obj.cf))
    iterator.set_discounting(obj.discounting)
    iterator.set_sampler(obj.model, obj.model_args)

    data = next(iterator)

    durations = data["durations"]
    timeline = data["timeline"]
    rewards = data["rewards"]
    samples_ids = data["samples_ids"]
    assets_ids = data["assets_ids"]

    if len(samples_ids) > maxsample:
        raise ValueError(
            f"Max number of sample has been reach : {maxsample}. Modify maxsample or set different arguments"
        )

    return RenewalData(
        t0,
        tf,
        samples_ids,
        assets_ids,
        timeline,
        durations,
        rewards,
    )


@sample_count_data.register
def _(
    obj: OneCycleAgeReplacementPolicy,
    size: int,
    tf: float,
    t0: float = 0.0,
    maxsample: int = 1e5,
    seed: Optional[int] = None,
):

    iterator = LifetimeIterator(size, tf, t0, nb_assets=obj.nb_assets, seed=seed)
    iterator.set_rewards(age_replacement_rewards(obj.ar, obj.cf, obj.cp))
    iterator.set_discounting(obj.discounting)
    iterator.set_sampler(obj.model, obj.model_args)

    data = next(iterator)

    durations = data["durations"]
    timeline = data["timeline"]
    rewards = data["rewards"]
    samples_ids = data["samples_ids"]
    assets_ids = data["assets_ids"]

    if len(samples_ids) > maxsample:
        raise ValueError(
            f"Max number of sample has been reach : {maxsample}. Modify maxsample or set different arguments"
        )

    return RenewalData(
        t0,
        tf,
        samples_ids,
        assets_ids,
        timeline,
        durations,
        rewards,
    )


@sample_count_data.register
def _(
    obj: Union[DefaultRunToFailurePolicy, DefaultAgeReplacementPolicy],
    size: int,
    tf: float,
    t0: float = 0.0,
    maxsample: int = 1e5,
    seed: Optional[int] = None,
):
    return sample_count_data(obj.process, size, tf, t0, maxsample, seed)


@sample_count_data.register
def _(
    obj: Union[
        NonHomogeneousPoissonProcess,
        NonHomogeneousPoissonAgeReplacementPolicy,
        NonHomogeneousPoissonProcessWithRewards,
    ],
    size: int,
    tf: float,
    t0: float = 0.0,
    maxsample: int = 1e5,
    seed: Optional[int] = None,
):

    samples_ids = np.array([], dtype=np.int64)
    assets_ids = np.array([], dtype=np.int64)
    timeline = np.array([], dtype=np.float64)
    ages = np.array([], dtype=np.float64)
    events = np.array([], dtype=np.bool_)
    rewards = None
    if isinstance(
        obj,
        (
            NonHomogeneousPoissonAgeReplacementPolicy,
            NonHomogeneousPoissonProcessWithRewards,
        ),
    ):
        rewards = np.array([], dtype=np.float64)

    iterator = NonHomogeneousPoissonIterator(
        size, tf, t0=t0, nb_assets=obj.nb_assets, seed=seed
    )
    iterator.set_sampler(
        obj.model, obj.model_args, ar=obj.ar if hasattr(obj, "ar") else None
    )
    if isinstance(
        obj,
        (
            NonHomogeneousPoissonAgeReplacementPolicy,
            NonHomogeneousPoissonProcessWithRewards,
        ),
    ):
        iterator.set_rewards(obj.rewards)
        iterator.set_discounting(obj.discounting)

    for data in iterator:
        events = np.concatenate((events, data["is_repaired"]))
        ages = np.concatenate((ages, data["ages"]))
        timeline = np.concatenate((timeline, data["timeline"]))
        samples_ids = np.concatenate((samples_ids, data["samples_ids"]))
        assets_ids = np.concatenate((assets_ids, data["assets_ids"]))
        if isinstance(
            obj,
            (
                NonHomogeneousPoissonAgeReplacementPolicy,
                NonHomogeneousPoissonProcessWithRewards,
            ),
        ):
            rewards = np.concatenate((rewards, data["rewards"]))

        if len(samples_ids) > maxsample:
            raise ValueError(
                f"Max number of sample has been reach : {maxsample}. Modify maxsample or set different arguments"
            )

    if rewards is None:
        rewards = np.zeros_like(timeline)

    return NHPPCountData(
        t0, tf, samples_ids, assets_ids, timeline, ages, events, rewards
    )


def get_baseline_type(model):
    if hasattr(model, "baseline"):
        return get_baseline_type(model.baseline)
    return type(model)


def get_model_model1(model, model1, model_args, model1_args, use: str):
    if use == "both" and model1 is not None:
        if get_baseline_type(model) != get_baseline_type(model1):
            raise ValueError(
                "Can't collect lifetime data from model and model1 because they have not the same type. Set use to 'model' or 'model1'"
            )
    elif use == "model1":
        model = model1
        model_args = model1_args
    elif use == "model":
        pass
    else:
        raise ValueError(
            f"Invalid 'use' value. Got {use}. Expected : 'both', 'model', or 'model1'"
        )
    return model, model_args


# noinspection PyUnusedLocal
@singledispatch
def sample_failure_data(
    obj,
    size: int,
    tf: float,
    t0: float = 0.0,
    maxsample: int = 1e5,
    seed: Optional[int] = None,
    use: str = "model",
):
    """


    Parameters
    ----------
    obj : Any
    size : int
    tf : float
    t0 : float
    size : int
    maxsample : int
    seed : int
    use : str
    """
    ValueError(f"No sample for {type(obj)}")


@sample_failure_data.register
def _(
    obj: Union[RenewalProcess, RenewalRewardProcess],
    size: int,
    tf: float,
    t0: float = 0.0,
    maxsample: int = 1e5,
    seed: Optional[int] = None,
    use: str = "model",
):
    durations = np.array([], dtype=np.float64)
    event_indicators = np.array([], dtype=np.float64)
    entries = np.array([], dtype=np.float64)

    model, model_args = get_model_model1(
        obj.model, obj.model1, obj.model_args, obj.model1_args, use
    )

    iterator = LifetimeIterator(size, tf, t0, nb_assets=obj.nb_assets, seed=seed)
    iterator.set_sampler(obj.model, obj.model_args)

    stack_model_args = tuple(([] for _ in range(len(model_args))))
    for data in iterator:
        durations = np.concatenate((durations, data["durations"]))
        event_indicators = np.concatenate((event_indicators, data["event_indicators"]))
        entries = np.concatenate((entries, data["entries"]))

        for i, v in enumerate(stack_model_args):
            v.append(np.take(model_args[i], data["assets_ids"], axis=0))

        if len(durations) > maxsample:
            raise ValueError(
                f"Max number of sample has been reach : {maxsample}. Modify maxsample or set different arguments"
            )

    model_args = tuple((np.concatenate(x) for x in stack_model_args))

    return durations, event_indicators, entries, model_args


@sample_failure_data.register
def _(
    obj: Union[OneCycleRunToFailurePolicy, OneCycleAgeReplacementPolicy],
    size: int,
    tf: float,
    t0: float = 0.0,
    maxsample: int = 1e5,
    seed: Optional[int] = None,
    use: str = "model",
):
    durations = np.array([], dtype=np.float64)
    event_indicators = np.array([], dtype=np.float64)
    entries = np.array([], dtype=np.float64)

    if use in ("both", "model1"):
        raise ValueError(
            "Invalid 'use' argument for OneCycleRunToFailurePolicy. 'use' can only be 'model'"
        )

    iterator = LifetimeIterator(size, tf, t0, nb_assets=obj.nb_assets, seed=seed)
    iterator.set_sampler(obj.model, obj.model_args)

    model_args = ()
    for data in iterator:
        durations = np.concatenate((durations, data["durations"]))
        event_indicators = np.concatenate((event_indicators, data["event_indicators"]))
        entries = np.concatenate((entries, data["entries"]))

        model_args = tuple(
            (np.take(v, data["assets_ids"], axis=0) for v in obj.model_args)
        )

        if len(durations) > maxsample:
            raise ValueError(
                f"Max number of sample has been reach : {maxsample}. Modify maxsample or set different arguments"
            )

        # break loop after first iteration (one cycle only)
        break

    return durations, event_indicators, entries, model_args


@sample_failure_data.register
def _(
    obj: Union[DefaultRunToFailurePolicy, DefaultAgeReplacementPolicy],
    size: int,
    tf: float,
    t0: float = 0.0,
    maxsample: int = 1e5,
    seed: Optional[int] = None,
    use: str = "model",
):
    durations = np.array([], dtype=np.float64)
    event_indicators = np.array([], dtype=np.float64)
    entries = np.array([], dtype=np.float64)

    model, model_args = get_model_model1(
        obj.model, obj.model1, obj.model_args, obj.model1_args, use
    )

    iterator = LifetimeIterator(size, tf, t0, nb_assets=obj.nb_assets, seed=seed)
    iterator.set_sampler(model, model_args)

    stack_model_args = tuple(([] for _ in range(len(model_args))))
    for data in iterator:
        durations = np.concatenate((durations, data["durations"]))
        event_indicators = np.concatenate((event_indicators, data["event_indicators"]))
        entries = np.concatenate((entries, data["entries"]))

        for i, v in enumerate(stack_model_args):
            v.append(np.take(model_args[i], data["assets_ids"], axis=0))

        if len(durations) > maxsample:
            raise ValueError(
                f"Max number of sample has been reach : {maxsample}. Modify maxsample or set different arguments"
            )

    model_args = tuple((np.concatenate(x) for x in stack_model_args))

    return durations, event_indicators, entries, model_args


@sample_failure_data.register
def _(
    obj: Union[
        NonHomogeneousPoissonProcess,
        NonHomogeneousPoissonProcessWithRewards,
        NonHomogeneousPoissonAgeReplacementPolicy,
    ],
    size: int,
    tf: float,
    t0: float = 0.0,
    maxsample: int = 1e5,
    seed: Optional[int] = None,
    use: str = "model",
):

    if use != "model":
        raise ValueError("Invalid 'use' value. Only 'model' can be set")

    events_assets_ids = np.array([], dtype=np.int64)
    ages_at_events = np.array([], dtype=np.float64)

    iterator = NonHomogeneousPoissonIterator(
        size, tf, t0=t0, nb_assets=obj.nb_assets, seed=seed
    )
    iterator.set_sampler(
        obj.model, obj.model_args, ar=obj.ar if hasattr(obj, "ar") else None
    )

    for data in iterator:
        if data["timeline"].size == 0:
            continue

        is_repaired = data["is_repaired"]
        ages_at_events = np.concatenate((ages_at_events, data["ages"][is_repaired]))
        events_assets_ids = np.concatenate(
            (events_assets_ids, data["assets_ids"][is_repaired])
        )

        if len(events_assets_ids) > maxsample:
            raise ValueError(
                "Max number of sample has been reach : 1e5. Modify maxsample or set different arguments"
            )

    sort_ind = np.lexsort((ages_at_events, events_assets_ids))
    ages_at_events = ages_at_events[sort_ind]
    events_assets_ids = events_assets_ids[sort_ind]

    first_age_index = np.where(np.roll(events_assets_ids, 1) != events_assets_ids)[0]
    last_age_index = np.append(first_age_index[1:] - 1, len(events_assets_ids) - 1)

    assets_ids = events_assets_ids[first_age_index]
    if t0 == 0.0:
        start_ages = np.zeros_like(first_age_index, dtype=np.float64)
    else:
        start_ages = np.ones_like(first_age_index) * t0

    end_ages = ages_at_events[last_age_index]
    ages_at_events = np.delete(ages_at_events, last_age_index)
    events_assets_ids = np.delete(events_assets_ids, last_age_index)

    model_args = tuple((np.take(arg, assets_ids) for arg in obj.model_args))

    return (
        events_assets_ids,
        ages_at_events,
        assets_ids,
        start_ages,
        end_ages,
        model_args,
    )
