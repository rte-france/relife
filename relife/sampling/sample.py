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
from relife.costs import age_replacement_cost, run_to_failure_cost


@singledispatch
def sample_count_data(obj, size, tf, t0, maxsample, seed):
    # '_param' just for IDE warning of unused param
    raise ValueError(f"No sample for {type(obj)}")


@sample_count_data.register
def _(
    obj: RenewalProcess,
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

    iterable = RenewalIterable(
        size,
        tf,
        obj.model,
        t0=t0,
        model_args=obj.model_args,
        model1=obj.model1,
        model1_args=obj.model1_args,
        seed=seed,
        keep_last=False,
    )

    for data in iterable:
        samples_ids = np.concatenate((timeline, data["samples_ids"]))
        assets_ids = np.concatenate((timeline, data["assets_ids "]))
        timeline = np.concatenate((timeline, data["timeline"]))
        durations = np.concatenate((timeline, data["durations"]))

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
    )


@sample_count_data.register
def _(
    obj: RenewalRewardProcess,
    size: int,
    tf: float,
    t0: float = 0.0,
    maxsample: int = 1e5,
    seed: Optional[int] = None,
):
    durations = np.array([], dtype=np.float64)
    timeline = np.array([], dtype=np.float64)
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
        t0=t0,
        model_args=obj.model_args,
        model1=obj.model1,
        model1_args=obj.model1_args,
        seed=seed,
        keep_last=False,
    )

    for data in iterable:
        durations = np.concatenate((durations, data["durations"]))
        timeline = np.concatenate((timeline, data["timeline"]))
        rewards = np.concatenate((rewards, data["rewards"]))
        events = np.concatenate((events, data["event_indicators"]))
        samples_ids = np.concatenate((samples_ids, data["sample_ids"]))
        assets_ids = np.concatenate((assets_ids, data["assets_ids"]))

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
    obj: OneCycleRunToFailurePolicy,
    size: int,
    tf: float,
    t0: float = 0.0,
    maxsample: int = 1e5,
    seed: Optional[int] = None,
):
    durations = np.array([], dtype=np.float64)
    timeline = np.array([], dtype=np.float64)
    rewards = np.array([], dtype=np.float64)
    samples_ids = np.array([], dtype=np.int64)
    assets_ids = np.array([], dtype=np.int64)

    iterable = RenewalIterable(
        size,
        tf,
        obj.model,
        run_to_failure_cost(cf=obj.cf),
        partial(exponential_discounting.factor, rate=obj.discounting_rate),
        t0=t0,
        model_args=obj.model_args,
        seed=seed,
        keep_last=False,
    )

    for data in iterable:
        durations = np.concatenate((durations, data["durations"]))
        timeline = np.concatenate((timeline, data["timeline"]))
        rewards = np.concatenate((rewards, data["rewards"]))
        samples_ids = np.concatenate((samples_ids, data["samples_ids"]))
        assets_ids = np.concatenate((assets_ids, data["assets_ids"]))

        if len(samples_ids) > maxsample:
            raise ValueError(
                "Max number of sample has been reach : 1e5. Modify maxsample or set different arguments"
            )

        # break loop after first iteration (one cycle only)
        break

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
    obj: DefaultRunToFailurePolicy,
    size: int,
    tf: float,
    t0: float = 0.0,
    maxsample: int = 1e5,
    seed: Optional[int] = None,
):
    durations = np.array([], dtype=np.float64)
    timeline = np.array([], dtype=np.float64)
    rewards = np.array([], dtype=np.float64)
    samples_ids = np.array([], dtype=np.int64)
    assets_ids = np.array([], dtype=np.int64)

    iterable = RenewalIterable(
        size,
        tf,
        obj.model,
        run_to_failure_cost(cf=obj.cf),
        partial(exponential_discounting.factor, rate=obj.discounting_rate),
        t0=t0,
        model_args=obj.model_args,
        model1=obj.model1,
        model1_args=obj.model1_args,
        seed=seed,
        keep_last=False,
    )

    for data in iterable:
        durations = np.concatenate((durations, data["durations"]))
        timeline = np.concatenate((timeline, data["timeline"]))
        rewards = np.concatenate((rewards, data["rewards"]))
        samples_ids = np.concatenate((samples_ids, data["samples_ids"]))
        assets_ids = np.concatenate((assets_ids, data["assets_ids"]))

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
    obj: NonHomogeneousPoissonProcess,
    size: int,
    tf: float,
    t0: float = 0.0,
    maxsample: int = 1e5,
    seed: Optional[int] = None,
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
        samples_ids = np.concatenate((samples_ids, data["samples_ids"]))
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
    durations = np.array([], dtype=np.float64)
    event_indicators = np.array([], dtype=np.float64)
    entries = np.array([], dtype=np.float64)

    model, model_args = get_model_model1(
        obj.model, obj.model1, obj.model_args, obj.model1_args, use
    )

    iterable = RenewalIterable(
        size,
        tf,
        model,
        t0=t0,
        model_args=model_args,
        model1=model,
        model1_args=model_args,
        seed=seed,
    )

    stack_model_args = tuple(([] for _ in range(len(model_args))))
    for data in iterable:
        durations = np.concatenate((durations, data["durations"]))
        event_indicators = np.concatenate((event_indicators, data["event_indicators"]))
        entries = np.concatenate((entries, data["entries"]))

        for i, v in enumerate(stack_model_args):
            v.append(np.take(model_args[i], data["assets_ids"], axis=0))

    model_args = tuple((np.concatenate(x) for x in stack_model_args))

    return durations, event_indicators, entries, model_args


@sample_lifetime_data.register
def _(
    obj: RenewalRewardProcess,
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

    iterable = RenewalIterable(
        size,
        tf,
        model,
        t0=t0,
        model_args=model_args,
        model1=model,
        model1_args=model_args,
        seed=seed,
    )

    stack_model_args = tuple(([] for _ in range(len(model_args))))
    for data in iterable:
        durations = np.concatenate((durations, data["durations"]))
        event_indicators = np.concatenate((event_indicators, data["event_indicators"]))
        entries = np.concatenate((entries, data["entries"]))

        for i, v in enumerate(stack_model_args):
            v.append(np.take(model_args[i], data["assets_ids"], axis=0))

    model_args = tuple((np.concatenate(x) for x in stack_model_args))

    return durations, event_indicators, entries, model_args


@sample_lifetime_data.register
def _(
    obj: OneCycleRunToFailurePolicy,
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

    iterable = RenewalIterable(
        size,
        tf,
        obj.model,
        run_to_failure_cost(cf=obj.cf),
        partial(exponential_discounting.factor, rate=obj.discounting_rate),
        t0=t0,
        model_args=obj.model_args,
        seed=seed,
    )

    model_args = ()
    for data in iterable:
        durations = np.concatenate((durations, data["durations"]))
        event_indicators = np.concatenate((event_indicators, data["event_indicators"]))
        entries = np.concatenate((entries, data["entries"]))

        model_args = tuple(
            (np.take(v, data["assets_ids"], axis=0) for v in obj.model_args)
        )

        # break loop after first iteration (one cycle only)
        break

    return durations, event_indicators, entries, model_args


@sample_lifetime_data.register
def _(
    obj: DefaultRunToFailurePolicy,
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

    iterable = RenewalIterable(
        size,
        tf,
        model,
        run_to_failure_cost(cf=obj.cf),
        partial(exponential_discounting.factor, rate=obj.discounting_rate),
        t0=t0,
        model_args=model_args,
        model1=model,
        model1_args=model_args,
        seed=seed,
    )

    stack_model_args = tuple(([] for _ in range(len(model_args))))
    for data in iterable:
        durations = np.concatenate((durations, data["durations"]))
        event_indicators = np.concatenate((event_indicators, data["event_indicators"]))
        entries = np.concatenate((entries, data["entries"]))

        for i, v in enumerate(stack_model_args):
            v.append(np.take(model_args[i], data["assets_ids"], axis=0))

    model_args = tuple((np.concatenate(x) for x in stack_model_args))

    return durations, event_indicators, entries, model_args


@sample_lifetime_data.register
def _(
    obj: OneCycleAgeReplacementPolicy,
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

    iterable = RenewalIterable(
        size,
        tf,
        obj.model,
        age_replacement_cost(ar=obj.ar, cp=obj.cp, cf=obj.cf),
        partial(exponential_discounting.factor, rate=obj.discounting_rate),
        t0=t0,
        model_args=obj.model_args,
        seed=seed,
    )

    model_args = ()
    for data in iterable:
        durations = np.concatenate((durations, data["durations"]))
        event_indicators = np.concatenate((event_indicators, data["event_indicators"]))
        entries = np.concatenate((entries, data["entries"]))

        model_args = tuple(
            (np.take(v, data["assets_ids"], axis=0) for v in obj.model_args)
        )

        # break loop after first iteration (one cycle only)
        break

    return durations, event_indicators, entries, model_args


@sample_lifetime_data.register
def _(
    obj: DefaultAgeReplacementPolicy,
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

    iterable = RenewalIterable(
        size,
        tf,
        model,
        age_replacement_cost(ar=obj.ar, cp=obj.cp, cf=obj.cf),
        partial(exponential_discounting.factor, rate=obj.discounting_rate),
        t0=t0,
        model_args=model_args,
        model1=model,
        model1_args=model_args,
        seed=seed,
    )

    stack_model_args = tuple(([] for _ in range(len(model_args))))
    for data in iterable:
        durations = np.concatenate((durations, data["durations"]))
        event_indicators = np.concatenate((event_indicators, data["event_indicators"]))
        entries = np.concatenate((entries, data["entries"]))

        for i, v in enumerate(stack_model_args):
            v.append(np.take(model_args[i], data["assets_ids"], axis=0))

    model_args = tuple((np.concatenate(x) for x in stack_model_args))

    return durations, event_indicators, entries, model_args


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
