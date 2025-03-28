from functools import singledispatch
from itertools import islice
from typing import Iterator, Optional, Union

import numpy as np

from relife.data import RenewalData
from relife.data.counting import NHPPCountData
from relife.economics.rewards import (
    age_replacement_rewards,
    run_to_failure_rewards,
)
from relife.policies import (
    DefaultAgeReplacementPolicy,
    DefaultRunToFailurePolicy,
    NonHomogeneousPoissonAgeReplacementPolicy,
    OneCycleAgeReplacementPolicy,
    OneCycleRunToFailurePolicy,
)
from relife.processes import (
    NonHomogeneousPoissonProcess,
    RenewalRewardProcess,
)
from relife.processes.renewal import RenewalProcess
from .iterators import LifetimeIterator, NonHomogeneousPoissonIterator


def stack1d(
    iterator: Iterator,
    keys: tuple[str],
    maxsample: int = 1e5,
    stack: Optional[dict[str, np.ndarray]] = None,
):
    stack = {} if stack is None else stack
    for i, data in enumerate(iterator):
        if i == 0 and not bool(stack):
            stack.update(
                {k: np.array([], dtype=v.dtype) for k, v in data.items() if k in keys}
            )

        if len(stack[list(stack.keys())[0]]) > maxsample:
            raise ValueError(
                "Max number of sample has been reach : 1e5. Modify maxsample or set different arguments"
            )

        stack.update(
            {k: np.concatenate((stack[k], v)) for k, v in data.items() if k in keys}
        )

    return stack


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
    keys = ("durations", "timeline", "samples_ids", "assets_ids", "rewards")
    iterator = LifetimeIterator(size, tf, t0, nb_assets=obj.nb_assets, seed=seed)

    if isinstance(obj, RenewalRewardProcess):
        iterator.rewards = obj.rewards1
        iterator.discounting = obj.discounting

    stack = None
    if obj.distribution1 is not None:
        iterator.set_distribution(obj.distribution1, obj.model1_args)
        stack = stack1d(islice(iterator, 1), keys, maxsample=maxsample)

    iterator.set_distribution(obj.distribution, obj.model_args)
    if isinstance(obj, RenewalRewardProcess):
        iterator.rewards = obj.rewards

    stack = stack1d(iterator, keys, maxsample=maxsample, stack=stack)

    return RenewalData(
        t0,
        tf,
        **stack,
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
    keys = ("durations", "timeline", "samples_ids", "assets_ids", "rewards")
    iterator = LifetimeIterator(size, tf, t0, nb_assets=obj.nb_assets, seed=seed)
    iterator.rewards = run_to_failure_rewards(obj.cf)
    iterator.discounting = obj.discounting
    iterator.set_distribution(obj.model, obj.model_args)

    stack = stack1d(islice(iterator, 1), keys, maxsample=maxsample)

    return RenewalData(
        t0,
        tf,
        **stack,
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
    keys = ("durations", "timeline", "samples_ids", "assets_ids", "rewards")
    iterator = LifetimeIterator(size, tf, t0, nb_assets=obj.nb_assets, seed=seed)
    iterator.rewards = age_replacement_rewards(obj.ar, obj.cf, obj.cp)
    iterator.discounting = obj.discounting
    iterator.set_distribution(obj.model, obj.model_args)

    stack = stack1d(islice(iterator, 1), keys, maxsample=maxsample)

    return RenewalData(
        t0,
        tf,
        **stack,
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
    keys = (
        "timeline",
        "ages",
        "events_indicators",
        "samples_ids",
        "assets_ids",
        "rewards",
    )
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
        iterator.rewards = obj.rewards
        iterator.discounting = obj.discounting
    stack = stack1d(iterator, keys, maxsample=maxsample)

    return NHPPCountData(t0, tf, **stack)


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
def failure_data_sample(
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


@failure_data_sample.register
def _(
    obj: Union[RenewalProcess, RenewalRewardProcess],
    size: int,
    tf: float,
    t0: float = 0.0,
    maxsample: int = 1e5,
    seed: Optional[int] = None,
    use: str = "model",
):

    keys = ("durations", "events_indicators", "entries", "assets_ids")

    model, model_args = get_model_model1(
        obj.distribution, obj.distribution1, obj.model_args, obj.model1_args, use
    )

    iterator = LifetimeIterator(size, tf, t0, nb_assets=obj.nb_assets, seed=seed)
    iterator.set_distribution(obj.distribution, obj.model_args)

    stack = stack1d(iterator, keys, maxsample=maxsample)
    model_args = tuple((np.take(arg, stack["assets_ids"]) for arg in model_args))

    return stack["durations"], stack["events_indicators"], stack["entries"], model_args


@failure_data_sample.register
def _(
    obj: Union[OneCycleRunToFailurePolicy, OneCycleAgeReplacementPolicy],
    size: int,
    tf: float,
    t0: float = 0.0,
    maxsample: int = 1e5,
    seed: Optional[int] = None,
    use: str = "model",
):
    keys = ("durations", "events_indicators", "entries", "assets_ids")

    if use in ("both", "model1"):
        raise ValueError(
            "Invalid 'use' argument for OneCycleRunToFailurePolicy. 'use' can only be 'model'"
        )

    iterator = LifetimeIterator(size, tf, t0, nb_assets=obj.nb_assets, seed=seed)
    iterator.set_distribution(obj.model, obj.model_args)

    stack = stack1d(islice(iterator, 1), keys, maxsample=maxsample)
    model_args = tuple((np.take(arg, stack["assets_ids"]) for arg in obj.model_args))

    return stack["durations"], stack["events_indicators"], stack["entries"], model_args


@failure_data_sample.register
def _(
    obj: Union[DefaultRunToFailurePolicy, DefaultAgeReplacementPolicy],
    size: int,
    tf: float,
    t0: float = 0.0,
    maxsample: int = 1e5,
    seed: Optional[int] = None,
    use: str = "model",
):
    keys = ("durations", "events_indicators", "entries", "assets_ids")

    model, model_args = get_model_model1(
        obj.distribution, obj.distribution1, obj.model_args, obj.model1_args, use
    )

    iterator = LifetimeIterator(size, tf, t0, nb_assets=obj.nb_assets, seed=seed)
    iterator.set_distribution(model, model_args)

    stack = stack1d(iterator, keys, maxsample=maxsample)
    model_args = tuple((np.take(arg, stack["assets_ids"]) for arg in model_args))

    return stack["durations"], stack["events_indicators"], stack["entries"], model_args


@failure_data_sample.register
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
    keys = (
        "timeline",
        "samples_ids",
        "assets_ids",
        "ages",
        "entries",
        "events_indicators",
        "renewals_ids",
    )

    if use != "model":
        raise ValueError("Invalid 'use' value. Only 'model' can be set")

    iterator = NonHomogeneousPoissonIterator(
        size, tf, t0=t0, nb_assets=obj.nb_assets, seed=seed, keep_last=True
    )
    iterator.set_sampler(
        obj.model, obj.model_args, ar=obj.ar if hasattr(obj, "ar") else None
    )

    stack = stack1d(iterator, keys, maxsample=maxsample)

    str_samples_ids = np.char.add(
        np.full_like(stack["samples_ids"], "S", dtype=np.str_),
        stack["samples_ids"].astype(np.str_),
    )
    str_assets_ids = np.char.add(
        np.full_like(stack["assets_ids"], "A", dtype=np.str_),
        stack["assets_ids"].astype(np.str_),
    )
    str_renewals_ids = np.char.add(
        np.full_like(stack["assets_ids"], "R", dtype=np.str_),
        stack["renewals_ids"].astype(np.str_),
    )
    assets_ids = np.char.add(str_samples_ids, str_assets_ids)
    assets_ids = np.char.add(assets_ids, str_renewals_ids)

    sort_ind = np.lexsort((stack["timeline"], assets_ids))

    entries = stack["entries"][sort_ind]
    events_indicators = stack["events_indicators"][sort_ind]
    ages = stack["ages"][sort_ind]
    assets_ids = assets_ids[sort_ind]

    # print("assets_ids", assets_ids)
    # print("timeline", timeline)
    # print("ages", ages)
    # print("events_indicators", events_indicators)
    # print("entries", entries)

    first_ages_index = np.roll(assets_ids, 1) != assets_ids
    last_ages_index = np.roll(first_ages_index, -1)

    immediatly_replaced = np.logical_and(~events_indicators, first_ages_index)

    # print("first_ages_index", first_ages_index)
    # print("last_ages_index", last_ages_index)
    # print("immediatly_replaced", immediatly_replaced)

    # prefix = np.full_like(assets_ids[immediatly_replaced], "Z", dtype=np.str_)
    # _assets_ids = np.char.add(prefix, assets_ids[immediatly_replaced])
    _assets_ids = assets_ids[immediatly_replaced]
    first_ages = entries[immediatly_replaced].copy()
    last_ages = ages[immediatly_replaced].copy()

    # print("assets_ids", _assets_ids)
    # print("first_ages", first_ages)
    # print("last_ages", last_ages)

    events_assets_ids = assets_ids[events_indicators]
    events_ages = ages[events_indicators]
    other_assets_ids = np.unique(events_assets_ids)
    _assets_ids = np.concatenate((_assets_ids, other_assets_ids))
    first_ages = np.concatenate(
        (first_ages, entries[first_ages_index & events_indicators])
    )
    last_ages = np.concatenate(
        (last_ages, ages[last_ages_index & ~immediatly_replaced])
    )

    # print("events_assets_ids", events_assets_ids)
    # print("events_ages", events_ages)
    # print("assets_ids", _assets_ids)
    # print("first_ages", first_ages)
    # print("last_ages", last_ages)

    # last sort (optional but convenient to control data)
    sort_ind = np.argsort(events_assets_ids)
    events_assets_ids = events_assets_ids[sort_ind]

    sort_ind = np.argsort(_assets_ids)
    _assets_ids = _assets_ids[sort_ind]
    first_ages = first_ages[sort_ind]
    last_ages = last_ages[sort_ind]

    return events_assets_ids, events_ages, _assets_ids, first_ages, last_ages
