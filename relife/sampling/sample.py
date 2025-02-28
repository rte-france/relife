from functools import partial, singledispatch
from typing import Union, Optional

import numpy as np

from relife.data import RenewalData, RenewalRewardData

from relife.process import RenewalRewardProcess, NonHomogeneousPoissonProcess
from relife.process.renewal import RenewalProcess
from .iterables import RenewalIterable, NonHomogeneousPoissonIterable
from relife.core.discounting import exponential_discounting
from relife.policies import (
    OneCycleRunToFailurePolicy,
    DefaultRunToFailurePolicy,
    OneCycleAgeReplacementPolicy,
    DefaultAgeReplacementPolicy,
)
from relife.policies.age_replacement import age_replacement_cost
from relife.policies.run_to_failure import run_to_failure_cost


@singledispatch
def sample(obj, _size, _tf, _seed):
    # '_param' just for IDE warning of unused param
    raise ValueError(f"No sample for {type(obj)}")


@sample.register
def _(
    obj: RenewalProcess,
    size,
    tf,
    seed,
    maxsample: int = 1e5,
):
    lifetimes = np.array([], dtype=np.float64)
    timeline = np.array([], dtype=np.float64)
    samples_ids = np.array([], dtype=np.int64)
    assets_ids = np.array([], dtype=np.int64)

    iterable = RenewalIterable(
        size,
        tf,
        obj.model,
        model_args=obj.model_args,
        model1=obj.model1,
        model1_args=obj.model1_args,
        seed=seed,
    )

    for data in iterable:
        samples_ids = np.concatenate((timeline, data["samples_ids"]))
        assets_ids = np.concatenate((timeline, data["assets_ids "]))
        timeline = np.concatenate((timeline, data["timeline"]))
        lifetimes = np.concatenate((timeline, data["time"]))

        if len(samples_ids) > maxsample:
            raise ValueError(
                "Max number of sample has been reach : 1e5. Modify maxsample or set different arguments"
            )

    return RenewalData(
        samples_ids,
        assets_ids,
        timeline,
        lifetimes,
    )


@sample.register
def _(
    obj: RenewalRewardProcess,
    size,
    tf,
    seed,
    maxsample: int = 1e5,
):
    lifetimes = np.array([], dtype=np.float64)
    event_times = np.array([], dtype=np.float64)
    rewards = np.array([], dtype=np.float64)
    events = np.array([], dtype=np.bool_)
    samples_ids = np.array([], dtype=np.int64)
    assets_ids = np.array([], dtype=np.int64)

    iterable = RenewalIterable(
        size,
        tf,
        obj.model,
        obj.reward,
        partial(exponential_discounting.factor, rate=obj.discounting_rate),
        model_args=obj.model_args,
        model1=obj.model1,
        model1_args=obj.model1_args,
        seed=seed,
    )

    for data in iterable:
        lifetimes = np.concatenate((lifetimes, data["time"]))
        event_times = np.concatenate((event_times, data["timeline"]))
        rewards = np.concatenate((rewards, data["rewards"]))
        events = np.concatenate((events, data["event"]))
        samples_ids = np.concatenate((samples_ids, data["sample_ids"]))
        assets_ids = np.concatenate((assets_ids, data["assets_ids"]))

        if len(samples_ids) > maxsample:
            raise ValueError(
                "Max number of sample has been reach : 1e5. Modify maxsample or set different arguments"
            )

    return RenewalRewardData(
        samples_ids,
        assets_ids,
        event_times,
        lifetimes,
        rewards,
    )


@sample.register
def _(
    obj: OneCycleRunToFailurePolicy,
    size,
    tf,
    seed,
    maxsample: int = 1e5,
):
    lifetimes = np.array([], dtype=np.float64)
    event_times = np.array([], dtype=np.float64)
    rewards = np.array([], dtype=np.float64)
    samples_ids = np.array([], dtype=np.int64)
    assets_ids = np.array([], dtype=np.int64)

    iterable = RenewalIterable(
        size,
        tf,
        obj.model,
        partial(run_to_failure_cost, cf=obj.cf),
        partial(exponential_discounting.factor, rate=obj.discounting_rate),
        model_args=obj.model_args,
        seed=seed,
    )

    for data in iterable:
        lifetimes = np.concatenate((lifetimes, data["time"]))
        event_times = np.concatenate((event_times, data["timeline"]))
        rewards = np.concatenate((rewards, data["rewards"]))
        samples_ids = np.concatenate((samples_ids, data["sample_ids"]))
        assets_ids = np.concatenate((assets_ids, data["assets_ids"]))

        if len(samples_ids) > maxsample:
            raise ValueError(
                "Max number of sample has been reach : 1e5. Modify maxsample or set different arguments"
            )

        # break loop after first iteration (one cycle only)
        break

    return RenewalRewardData(
        samples_ids,
        assets_ids,
        event_times,
        lifetimes,
        rewards,
    )


@sample.register
def _(
    obj: DefaultRunToFailurePolicy,
    size,
    tf,
    seed,
    maxsample: int = 1e5,
):
    lifetimes = np.array([], dtype=np.float64)
    event_times = np.array([], dtype=np.float64)
    rewards = np.array([], dtype=np.float64)
    samples_ids = np.array([], dtype=np.int64)
    assets_ids = np.array([], dtype=np.int64)

    iterable = RenewalIterable(
        size,
        tf,
        obj.model,
        partial(run_to_failure_cost, cf=obj.cf),
        partial(exponential_discounting.factor, rate=obj.discounting_rate),
        model_args=obj.model_args,
        model1=obj.model1,
        model1_args=obj.model1_args,
        seed=seed,
    )

    for data in iterable:
        lifetimes = np.concatenate((lifetimes, data["time"]))
        event_times = np.concatenate((event_times, data["timeline"]))
        rewards = np.concatenate((rewards, data["rewards"]))
        samples_ids = np.concatenate((samples_ids, data["sample_ids"]))
        assets_ids = np.concatenate((assets_ids, data["assets_ids"]))

        if len(samples_ids) > maxsample:
            raise ValueError(
                "Max number of sample has been reach : 1e5. Modify maxsample or set different arguments"
            )

    return RenewalRewardData(
        samples_ids,
        assets_ids,
        event_times,
        lifetimes,
        rewards,
    )


@sample.register
def _(
    obj: NonHomogeneousPoissonProcess,
    size,
    tf,
    seed,
    maxsample: int = 1e5,
):
    ages = np.array([], dtype=np.float64)
    durations = np.array([], dtype=np.float64)
    rewards = np.array([], dtype=np.float64)
    samples_ids = np.array([], dtype=np.int64)
    assets_ids = np.array([], dtype=np.int64)

    iterable = NonHomogeneousPoissonIterable(
        size,
        tf,
        obj.model,
        reward_func=None,
        discount_factor=None,
        model_args=obj.model_args,
        seed=seed,
    )

    for data in iterable:
        ages = np.concatenate((ages, data["ages"]))
        durations = np.concatenate((durations, data["durations"]))
        rewards = np.concatenate((rewards, data["rewards"]))
        samples_ids = np.concatenate((samples_ids, data["sample_ids"]))
        assets_ids = np.concatenate((assets_ids, data["assets_ids"]))

        if len(samples_ids) > maxsample:
            raise ValueError(
                "Max number of sample has been reach : 1e5. Modify maxsample or set different arguments"
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
    elif use == "model":
        model1 = model
        model1_args = model_args
    elif use == "model1":
        model = model1
        model1_args = model1_args
    else:
        raise ValueError(
            f"Invalid 'use' value. Got {use}. Expected : 'both', 'model', or 'model1'"
        )
    return model, model1, model_args, model1_args


@singledispatch
def sample_lifetime_data(
    obj,
    _size,
    _tf,
    _t0,
    _seed,
    _use,
):
    # '_param' just for IDE warning of unused param
    ValueError(f"No sample for {type(obj)}")


@sample_lifetime_data.register
def _(
    obj: Union[RenewalProcess, RenewalRewardProcess],
    size: int,
    tf: float,
    t0: float = 0.0,
    seed: Optional[int] = None,
    use: str = "model",
):
    time = np.array([], dtype=np.float64)
    event = np.array([], dtype=np.float64)
    entry = np.array([], dtype=np.float64)

    model, model1, model_args, model1_args = get_model_model1(
        obj.model, obj.model1, obj.model_args, obj.model1_args, use
    )

    iterable = RenewalIterable(
        size,
        tf,
        model,
        t0=t0,
        model_args=model_args,
        model1=model1,
        model1_args=model1_args,
        seed=seed,
    )

    for data in iterable:
        time = np.concatenate((time, data["time"]))
        event = np.concatenate((event, data["event"]))
        entry = np.concatenate((entry, data["entry"]))

    return time, event, entry


@sample_lifetime_data.register
def _(
    obj: RenewalRewardProcess,
    size: int,
    tf: float,
    t0: float = 0.0,
    seed: Optional[int] = None,
    use: str = "model",
):
    time = np.array([], dtype=np.float64)
    event = np.array([], dtype=np.float64)
    entry = np.array([], dtype=np.float64)

    model, model1, model_args, model1_args = get_model_model1(
        obj.model, obj.model1, obj.model_args, obj.model1_args, use
    )

    iterable = RenewalIterable(
        size,
        tf,
        model,
        t0=t0,
        model_args=model_args,
        model1=model1,
        model1_args=model1_args,
        seed=seed,
    )

    for data in iterable:
        time = np.concatenate((time, data["time"]))
        event = np.concatenate((event, data["event"]))
        entry = np.concatenate((entry, data["entry"]))

    return time, event, entry


@sample_lifetime_data.register
def _(
    obj: OneCycleRunToFailurePolicy,
    size: int,
    tf: float,
    t0: float = 0.0,
    seed: Optional[int] = None,
    use: str = "model",
):
    time = np.array([], dtype=np.float64)
    event = np.array([], dtype=np.float64)
    entry = np.array([], dtype=np.float64)

    if use in ("both", "model1"):
        raise ValueError(
            "Invalid 'use' argument for OneCycleRunToFailurePolicy. 'use' can only be 'model'"
        )

    iterable = RenewalIterable(
        size,
        tf,
        obj.model,
        partial(run_to_failure_cost, cf=obj.cf),
        partial(exponential_discounting.factor, rate=obj.discounting_rate),
        t0=t0,
        model_args=obj.model_args,
        seed=seed,
    )

    for data in iterable:
        time = np.concatenate((time, data["time"]))
        event = np.concatenate((event, data["event"]))
        entry = np.concatenate((entry, data["entry"]))

        # break loop after first iteration (one cycle only)
        break

    return time, event, entry


@sample_lifetime_data.register
def _(
    obj: DefaultRunToFailurePolicy,
    size: int,
    tf: float,
    t0: float = 0.0,
    seed: Optional[int] = None,
    use: str = "model",
):
    time = np.array([], dtype=np.float64)
    event = np.array([], dtype=np.float64)
    entry = np.array([], dtype=np.float64)

    model, model1, model_args, model1_args = get_model_model1(
        obj.model, obj.model1, obj.model_args, obj.model1_args, use
    )

    iterable = RenewalIterable(
        size,
        tf,
        model,
        partial(run_to_failure_cost, cf=obj.cf),
        partial(exponential_discounting.factor, rate=obj.discounting_rate),
        t0=t0,
        model_args=model_args,
        model1=model1,
        model1_args=model1_args,
        seed=seed,
    )

    for data in iterable:
        time = np.concatenate((time, data["time"]))
        event = np.concatenate((event, data["event"]))
        entry = np.concatenate((entry, data["entry"]))

    return time, event, entry


@sample_lifetime_data.register
def _(
    obj: OneCycleAgeReplacementPolicy,
    size: int,
    tf: float,
    t0: float = 0.0,
    seed: Optional[int] = None,
    use: str = "model",
):
    time = np.array([], dtype=np.float64)
    event = np.array([], dtype=np.float64)
    entry = np.array([], dtype=np.float64)

    if use in ("both", "model1"):
        raise ValueError(
            "Invalid 'use' argument for OneCycleRunToFailurePolicy. 'use' can only be 'model'"
        )

    iterable = RenewalIterable(
        size,
        tf,
        obj.model,
        partial(age_replacement_cost, cp=obj.cp, cf=obj.cf),
        partial(exponential_discounting.factor, rate=obj.discounting_rate),
        t0=t0,
        model_args=obj.model_args,
        seed=seed,
    )

    for data in iterable:
        time = np.concatenate((time, data["time"]))
        event = np.concatenate((event, data["event"]))
        entry = np.concatenate((entry, data["entry"]))

        # break loop after first iteration (one cycle only)
        break

    return time, event, entry


@sample_lifetime_data.register
def _(
    obj: DefaultAgeReplacementPolicy,
    size: int,
    tf: float,
    t0: float = 0.0,
    seed: Optional[int] = None,
    use: str = "model",
):
    time = np.array([], dtype=np.float64)
    event = np.array([], dtype=np.float64)
    entry = np.array([], dtype=np.float64)

    model, model1, model_args, model1_args = get_model_model1(
        obj.model, obj.model1, obj.model_args, obj.model1_args, use
    )

    iterable = RenewalIterable(
        size,
        tf,
        model,
        partial(age_replacement_cost, cp=obj.cp, cf=obj.cf),
        partial(exponential_discounting.factor, rate=obj.discounting_rate),
        t0=t0,
        model_args=model_args,
        model1=model1,
        model1_args=model1_args,
        seed=seed,
    )

    for data in iterable:
        time = np.concatenate((time, data["time"]))
        event = np.concatenate((event, data["event"]))
        entry = np.concatenate((entry, data["entry"]))

    return time, event, entry


def sample_non_homogeneous_data(
    obj: NonHomogeneousPoissonProcess,
    size: int,
    tf: float,
    t0: float = 0.0,
    seed: Optional[int] = None,
):
    a0 = np.array([], dtype=np.float64)
    af = np.array([], dtype=np.float64)
    ages = np.array([], dtype=np.float64)
    assets = np.array([], dtype=np.int64)

    iterable = NonHomogeneousPoissonIterable(
        size,
        tf,
        obj.model,
        reward_func=None,
        discount_factor=None,
        model_args=obj.model_args,
        seed=seed,
    )

    for data in iterable:

        is_a0 = data["is_a0"]
        is_af = data["is_af"]
        not_af_a0 = np.logical_and(~is_a0, ~is_af)
        a0 = np.concatenate((a0, np.full((is_a0.sum(),), t0)))
        af = np.concatenate((af, np.full((is_af.sum(),), tf)))
        ages = np.concatenate((ages, data["ages"][not_af_a0]))
        assets = np.concatenate(
            (assets, (data["samples_ids"] + data["assets_ids"])[not_af_a0])
        )
