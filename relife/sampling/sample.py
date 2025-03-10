from functools import singledispatch
from typing import Union, Optional

import numpy as np

from relife.data import RenewalData, NHPPData

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


@singledispatch
def sample_count_data(
    obj,
    size,
    tf,
    t0: float = 0.0,
    maxsample: int = 1e5,
    seed: Optional[int] = None,
):
    # '_param' just for IDE warning of unused param
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
    rewards = None
    if isinstance(obj, RenewalRewardProcess):
        rewards = np.array([], dtype=np.float64)

    iterator = LifetimeIterator(size, tf, t0, seed=seed)
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

    # next cycles : set model in iterator
    iterator.set_sampler(obj.model, obj.model_args)

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
                "Max number of sample has been reach : 1e5. Modify maxsample or set different arguments"
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
    obj: OneCycleRunToFailurePolicy,
    size: int,
    tf: float,
    t0: float = 0.0,
    maxsample: int = 1e5,
    seed: Optional[int] = None,
):

    iterator = LifetimeIterator(size, tf, t0, seed=seed)
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
            "Max number of sample has been reach : 1e5. Modify maxsample or set different arguments"
        )

    return RenewalRewardData(
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

    iterator = LifetimeIterator(size, tf, t0, seed=seed)
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
            "Max number of sample has been reach : 1e5. Modify maxsample or set different arguments"
        )

    return RenewalRewardData(
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
    obj: Union[NonHomogeneousPoissonProcess, NonHomogeneousPoissonAgeReplacementPolicy],
    size: int,
    tf: float,
    t0: float = 0.0,
    maxsample: int = 1e5,
    seed: Optional[int] = None,
):

    samples_ids = np.array([], dtype=np.int64)
    assets_ids = np.array([], dtype=np.int64)
    timeline = np.array([], dtype=np.float64)
    durations = np.array([], dtype=np.float64)

    iterator = NonHomogeneousPoissonIterator(size, tf, t0=t0, seed=seed)
    iterator.set_sampler(
        obj.model, obj.model_args, ar=obj.ar if hasattr(obj, "ar") else None
    )

    for data in iterator:
        if data["timeline"].size == 0:
            continue
        _samples_ids = data["samples_ids"]
        _assets_ids = data["assets_ids"]
        _timeline = data["timeline"]
        _durations = data["durations"]

        selection = ~np.isnan(_durations)
        durations = np.concatenate((durations, _durations[selection]))
        timeline = np.concatenate((timeline, _timeline[selection]))
        samples_ids = np.concatenate((samples_ids, _samples_ids[selection]))
        assets_ids = np.concatenate((assets_ids, _assets_ids[selection]))

        # select a0, af only for sample_failure_data
        # selection = ~np.isnan(_a0)
        # a0 = np.concatenate((a0, _a0[selection]))
        # samples_ids = np.concatenate((samples_ids, _samples_ids[selection]))
        # assets_ids = np.concatenate((samples_ids, _samples_ids[selection]))
        # TODO : update assets_ids if replacement in iterator

        if len(samples_ids) > maxsample:
            raise ValueError(
                "Max number of sample has been reach : 1e5. Modify maxsample or set different arguments"
            )

    return NHPPData(t0, tf, samples_ids, assets_ids, timeline, durations)


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


@singledispatch
def sample_failure_data(
    obj,
    _size,
    _tf,
    _t0,
    _seed,
    _use,
):
    # '_param' just for IDE warning of unused param
    ValueError(f"No sample for {type(obj)}")


@sample_failure_data.register
def _(
    obj: Union[RenewalProcess, RenewalRewardProcess],
    size: int,
    tf: float,
    t0: float = 0.0,
    seed: Optional[int] = None,
    use: str = "model",
):
    durations = np.array([], dtype=np.float64)
    event_indicators = np.array([], dtype=np.float64)
    entries = np.array([], dtype=np.float64)

    model, model_args = get_model_model1(
        obj.model, obj.model1, obj.model_args, obj.model1_args, use
    )

    iterator = LifetimeIterator(size, tf, t0, seed=seed)
    iterator.set_sampler(obj.model, obj.model_args)

    stack_model_args = tuple(([] for _ in range(len(model_args))))
    for data in iterator:
        durations = np.concatenate((durations, data["durations"]))
        event_indicators = np.concatenate((event_indicators, data["event_indicators"]))
        entries = np.concatenate((entries, data["entries"]))

        for i, v in enumerate(stack_model_args):
            v.append(np.take(model_args[i], data["assets_ids"], axis=0))

    model_args = tuple((np.concatenate(x) for x in stack_model_args))

    return durations, event_indicators, entries, model_args


@sample_failure_data.register
def _(
    obj: Union[OneCycleRunToFailurePolicy, OneCycleAgeReplacementPolicy],
    size: int,
    tf: float,
    t0: float = 0.0,
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

    iterator = LifetimeIterator(size, tf, t0, seed=seed)
    iterator.set_sampler(obj.model, obj.model_args)

    model_args = ()
    for data in iterator:
        durations = np.concatenate((durations, data["durations"]))
        event_indicators = np.concatenate((event_indicators, data["event_indicators"]))
        entries = np.concatenate((entries, data["entries"]))

        model_args = tuple(
            (np.take(v, data["assets_ids"], axis=0) for v in obj.model_args)
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
    seed: Optional[int] = None,
    use: str = "model",
):
    durations = np.array([], dtype=np.float64)
    event_indicators = np.array([], dtype=np.float64)
    entries = np.array([], dtype=np.float64)

    model, model_args = get_model_model1(
        obj.model, obj.model1, obj.model_args, obj.model1_args, use
    )

    iterator = LifetimeIterator(size, tf, t0, seed=seed)
    iterator.set_sampler(model, model_args)

    stack_model_args = tuple(([] for _ in range(len(model_args))))
    for data in iterator:
        durations = np.concatenate((durations, data["durations"]))
        event_indicators = np.concatenate((event_indicators, data["event_indicators"]))
        entries = np.concatenate((entries, data["entries"]))

        for i, v in enumerate(stack_model_args):
            v.append(np.take(model_args[i], data["assets_ids"], axis=0))

    model_args = tuple((np.concatenate(x) for x in stack_model_args))

    return durations, event_indicators, entries, model_args


# def sample_non_homogeneous_data(
#     obj: NonHomogeneousPoissonProcess,
#     size: int,
#     tf: float,
#     t0: float = 0.0,
#     seed: Optional[int] = None,
# ):
#     a0 = np.array([], dtype=np.float64)
#     af = np.array([], dtype=np.float64)
#     ages = np.array([], dtype=np.float64)
#     assets = np.array([], dtype=np.int64)
#
#     iterable = NonHomogeneousPoissonIterable(
#         size,
#         tf,
#         obj.model,
#         reward_func=None,
#         discount_factor=None,
#         seed=seed,
#     )
#
#     for data in iterable:
#         index = data["samples_ids"] + data["assets_ids"]
#
#         entries = data["entries"]
#         event_indicators = data["event_indicators"]
#
#         # collect a0
#         a0 = np.concatenate((a0, entries[entries != 0]))
#         assets = np.concatenate((assets, index[entries != 0]))
#
#         # collect af
#         af = np.concatenate((af, index[event_indicators != 0]))
#         assets = np.concatenate((assets, index[event_indicators != 0]))
#
#         # collect ages
#         timeline = data["timeline"]
#         ages = np.concatenate(
#             (ages, timeline[np.logical_and(event_indicators, entries == 0)])
#         )
#
#     return a0, af, assets, ages
