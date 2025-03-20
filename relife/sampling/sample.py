from functools import singledispatch
from typing import Optional, Union

import numpy as np

from relife.data import RenewalData
from relife.data.counting import NHPPCountData
from relife.policies import (
    DefaultAgeReplacementPolicy,
    DefaultRunToFailurePolicy,
    NonHomogeneousPoissonAgeReplacementPolicy,
    OneCycleAgeReplacementPolicy,
    OneCycleRunToFailurePolicy,
)
from relife.process import (
    NonHomogeneousPoissonProcess,
    NonHomogeneousPoissonProcessWithRewards,
    RenewalRewardProcess,
)
from relife.process.renewal import RenewalProcess
from relife.rewards import (
    age_replacement_rewards,
    run_to_failure_rewards,
)

from .iterators import LifetimeIterator, NonHomogeneousPoissonIterator


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

    timeline = np.array([], dtype=np.float64)
    events_assets_ids = np.array([], dtype=np.str_)
    ages_at_events = np.array([], dtype=np.float64)
    is_repaired = np.array([], dtype=np.bool_)
    is_new_asset = np.array([], dtype=np.bool_)
    entries = np.array([], dtype=np.float64)
    event_indicators = np.array([], dtype=np.bool_)
    new_start_ages = np.array([], dtype=np.float64)
    previous_end_ages = np.array([], dtype=np.float64)

    iterator = NonHomogeneousPoissonIterator(
        size, tf, t0=t0, nb_assets=obj.nb_assets, seed=seed, keep_last=True
    )
    iterator.set_sampler(
        obj.model, obj.model_args, ar=obj.ar if hasattr(obj, "ar") else None
    )

    for data in iterator:
        if data["timeline"].size == 0:
            continue

        timeline = np.concatenate((timeline, data["timeline"]))
        is_repaired = np.concatenate((is_repaired, data["is_repaired"])) # ~ right censoring indicators
        is_new_asset = np.concatenate((is_new_asset, data["is_new_asset"]))
        entries = np.concatenate((entries, data["entries"]))
        event_indicators = np.concatenate((event_indicators, data["event_indicators"]))
        new_start_ages = np.concatenate((new_start_ages, data["new_start_ages"]))
        previous_end_ages = np.concatenate((previous_end_ages, data["previous_end_ages"]))
        ages_at_events = np.concatenate((ages_at_events, data["ages"]))
        prefix = np.char.add(np.full_like(data["samples_ids"], "S", dtype=np.str_), data["samples_ids"].astype(np.str_))
        events_assets_ids = np.concatenate(
            (events_assets_ids, np.char.add(prefix, data["assets_ids"].astype(np.str_)))
        )


        if len(events_assets_ids) > maxsample:
            raise ValueError(
                "Max number of sample has been reach : 1e5. Modify maxsample or set different arguments"
            )

    sort_ind = np.lexsort((timeline, events_assets_ids))

    timeline = timeline[sort_ind]
    is_repaired = is_repaired[sort_ind]
    is_new_asset = is_new_asset[sort_ind]
    entries = entries[sort_ind]
    event_indicators = event_indicators[sort_ind]
    new_start_ages = new_start_ages[sort_ind]
    previous_end_ages = previous_end_ages[sort_ind]
    ages_at_events = ages_at_events[sort_ind]
    events_assets_ids = events_assets_ids[sort_ind]


    # if is_new_asset but entry is not 0, a new asset was created immediatly loosing information about previous asset
    immediatly_replaced = np.logical_and(is_new_asset, entries != 0)
    prefix = np.full_like(events_assets_ids[immediatly_replaced], "Z", dtype=np.str_)
    forgotten_assets = np.char.add(prefix, events_assets_ids[immediatly_replaced])
    print("forgotten_assets", forgotten_assets)
    print(entries[immediatly_replaced])
    print(previous_end_ages[immediatly_replaced])


    print("timeline", timeline)
    print("is_repaired", is_repaired)
    print("is_new_asset", is_new_asset)
    print("ages_at_events", ages_at_events)
    print("events_assets_ids", events_assets_ids)
    print("entries", entries)
    print("event_indicators", event_indicators)

    assets_ids = np.unique(events_assets_ids)
    print("assets_ids", assets_ids)
    print("new_start_ages", new_start_ages)
    print("previous_end_ages", previous_end_ages)



    # first_age_index = np.where(np.roll(events_assets_ids, 1) != events_assets_ids)[0]
    # last_age_index = np.append(first_age_index[1:] - 1, len(events_assets_ids) - 1)
    #
    # print(first_age_index)
    # print(last_age_index)
    #
    #
    #
    # t0_entry_index = first_age_index[:iterator.size * iterator._nb_assets]
    # assets_ids = events_assets_ids[t0_entry_index]
    # start_ages = ages_at_events[t0_entry_index] - t0
    #
    # is_new_asset = np.logical_and(~is_repaired, ages_at_events == 0.)
    # start_ages = np.concatenate((start_ages, np.zeros_like(is_repaired, dtype=np.float64)))
    #
    #
    # if t0 == 0.0:
    #     start_ages = np.zeros_like(first_age_index, dtype=np.float64)
    # else:
    #     start_ages = np.ones_like(first_age_index) * t0
    #
    # end_ages = ages_at_events[last_age_index]
    # ages_at_events = np.delete(ages_at_events, last_age_index)
    # events_assets_ids = np.delete(events_assets_ids, last_age_index)
    #
    # model_args = tuple((np.take(arg, assets_ids) for arg in obj.model_args))
    #
    # return (
    #     events_assets_ids,
    #     ages_at_events,
    #     assets_ids,
    #     start_ages,
    #     end_ages,
    #     model_args,
    # )
