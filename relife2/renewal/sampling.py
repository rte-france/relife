from typing import Iterator, Optional, TypeVar

import numpy as np
from numpy.typing import NDArray

from relife2.fiability.addons import AgeReplacementModel
from relife2.model import LifetimeModel
from relife2.renewal.discounts import Discount
from relife2.renewal.rewards import Reward

M = TypeVar("M", tuple[NDArray[np.float64], ...], tuple[()])
M1 = TypeVar("M1", tuple[NDArray[np.float64], ...], tuple[()])
R = TypeVar("R", tuple[NDArray[np.float64], ...], tuple[()])
R1 = TypeVar("R1", tuple[NDArray[np.float64], ...], tuple[()])
D = TypeVar("D", tuple[NDArray[np.float64], ...], tuple[()])


def model_rvs(
    model: LifetimeModel[*M],
    size: int,
    args: M = (),
):
    return model.rvs(*args, size=size)


def rvs_size(
    nb_samples: int,
    nb_assets: int,
    model_args: M = (),
):
    if bool(model_args) and model_args[0].ndim == 2:
        size = nb_samples  # rvs size
    else:
        size = nb_samples * nb_assets
    return size


def compute_rewards(
    reward: Reward[*R],
    lifetimes: NDArray[np.float64],
    args: R = (),
):
    return reward(lifetimes, *args)


def compute_events(
    lifetimes: NDArray[np.float64],
    model: LifetimeModel[*M],
    model_args: M = (),
) -> NDArray[np.bool_]:
    """
    tag lifetimes as being right censored or not depending on model used
    """
    events = np.ones_like(lifetimes, dtype=np.bool_)
    ar = 0.0
    if isinstance(model, AgeReplacementModel):
        ar = model_args[0]
    events[lifetimes < ar] = False
    return events


def lifetimes_generator(
    model: LifetimeModel[*M],
    nb_samples: int,
    nb_assets: int,
    end_time: float,
    *,
    model_args: M = (),
    model1: Optional[LifetimeModel[*M1]] = None,
    model1_args: M1 = (),
) -> Iterator[
    tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]
]:

    event_times = np.zeros((nb_assets, nb_samples))
    order = np.zeros((nb_assets, nb_samples))
    still_valid = event_times < end_time

    def sample_routine(target_model, args):
        nonlocal event_times, order, still_valid  # modify these variables
        lifetimes = model_rvs(target_model, size, args=args).reshape(
            (nb_assets, nb_samples)
        )
        event_times += lifetimes
        order += 1
        events = compute_events(lifetimes, target_model, args)
        still_valid = event_times < end_time
        return lifetimes, event_times, events, still_valid

    if model1:
        size = rvs_size(nb_samples, nb_assets, model1_args)
        gen_data = sample_routine(model1, model1_args)
        if np.any(gen_data[-1]) > 0:
            yield gen_data
        else:
            return

    size = rvs_size(nb_samples, nb_assets, model_args)
    while True:
        gen_data = sample_routine(model, model_args)
        if np.any(gen_data[-1]) > 0:
            yield gen_data
        else:
            break
    return


def lifetimes_rewards_generator(
    model: LifetimeModel[*M],
    reward: Reward[*R],
    discounting: Discount[*D],
    nb_samples: int,
    nb_assets: int,
    end_time: float,
    *,
    model_args: M = (),
    reward_args: R = (),
    discount_args: D = (),
    model1: Optional[LifetimeModel[*M1]] = None,
    model1_args: M1 = (),
    reward1: Optional[Reward[*R1]] = None,
    reward1_args: R1 = (),
) -> Iterator[
    tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]
]:
    total_rewards = np.zeros((nb_assets, nb_samples))

    lifetimes_gen = lifetimes_generator(
        model,
        nb_samples,
        nb_assets,
        end_time,
        model_args=model_args,
        model1=model1,
        model1_args=model1_args,
    )

    def sample_routine(target_reward, args):
        nonlocal total_rewards  # modify these variables
        lifetimes, event_times, events, still_valid = next(lifetimes_gen)
        rewards = target_reward(lifetimes, *args)
        discountings = discounting.factor(event_times, *discount_args)
        total_rewards += rewards * discountings
        return lifetimes, event_times, total_rewards, events, still_valid

    if reward1 is None:
        reward1 = reward
        reward1_args = reward_args

    if model1:
        try:
            yield sample_routine(reward1, reward1_args)
        except StopIteration:
            return

    while True:
        try:
            yield sample_routine(reward, reward_args)
        except StopIteration:
            break
    return


class Array1DField:
    """
    data descriptor preventing field attribute to be set
    + avoid repeating code when using property decorator
    """

    def __init__(self, *, dtype):
        self._default = np.array([], dtype=dtype)

    def __set_name__(self, owner, name):
        self.public_name = name
        self.private_name = "_" + name

    def __get__(self, obj, type):
        if obj is None:
            return self._default
        return getattr(obj, self.private_name, self._default)

    def __set__(self, obj, value):
        raise AttributeError(f"{self.public_name} can't be set")


class CountData:
    """
    descriptor so that they can only be filled through populate method
    in order to control way CountData object are made
    only _hidden version can be set
    """

    samples: Array1DField = Array1DField(dtype=np.int64)
    assets: Array1DField = Array1DField(dtype=np.int64)
    order: Array1DField = Array1DField(dtype=np.int64)
    event_times: Array1DField = Array1DField(dtype=np.float64)

    def __init__(self, nb_samples: int, nb_assets: int):
        self.nb_samples = nb_samples
        self.nb_assets = nb_assets
        self._populate_call = 0
        self.fields = []

    def number_of_events(
        self, sample: int
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        ind = self.samples == sample
        times = np.insert(np.sort(self.event_times[ind]), 0, 0)
        counts = np.arange(times.size)
        return times, counts

    def mean_number_of_events(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        times = np.insert(np.sort(self.event_times), 0, 0)
        counts = np.arange(times.size) / self.nb_samples
        return times, counts

    def iter(self):
        return CountDataIterable(self)

    def populate(
        self,
        samples: NDArray[np.int64],
        assets: NDArray[np.int64],
        event_times: NDArray[np.float64],
        **kwdata: NDArray[np.float64 | np.bool_],
    ):
        if not set(self.fields) != set(kwdata.keys()):
            raise ValueError(f"expected only {set(self.fields)}")

        fields_values = (samples, assets, event_times, *kwdata.values())
        if not all(arr.ndim == 1 for arr in fields_values):
            raise ValueError("all arrays must be 1d")
        if not len(set(arr.shape[0] for arr in fields_values)) == 1:
            raise ValueError("all arrays must have the same shape")

        self._samples = np.concatenate((self.samples, samples))
        self._assets = np.concatenate((self.assets, assets))
        self._order = np.concatenate(
            (self.order, np.ones_like(samples) * self._populate_call)
        )
        self._event_times = np.concatenate((self.event_times, event_times))

        for k, v in kwdata.items():
            old_v = getattr(self, k)
            setattr(self, "_" + k, np.concatenate((old_v, v)))

        self._populate_call += 1


class CountDataIterable:
    def __init__(self, data: CountData):

        self.samples_index = np.unique(data.samples)
        self.assets_index = np.unique(data.assets)
        self.nb_samples = len(self.samples_index)
        self.nb_assets = len(self.assets_index)

        sorted_index = np.lexsort((data.order, data.assets, data.samples))

        self.sorted_fields = {
            field_name: getattr(data, field_name).copy()[sorted_index]
            for field_name in data.fields
        }

    def __len__(self) -> int:
        return self.nb_samples * self.nb_assets

    def __iter__(self) -> Iterator[tuple[int, int, dict[str, NDArray[np.float64]]]]:

        for sample in self.samples_index:
            sample_mask = self.sorted_fields["samples"] == sample
            for asset in self.assets_index:
                asset_mask = self.sorted_fields["assets"][sample_mask] == asset
                values_dict = {
                    k: v[sample_mask][asset_mask]
                    for k, v in self.sorted_fields.items()
                    if k not in ("samples", "assets", "events")
                }
                yield int(sample), int(asset), values_dict


class RenewalData(CountData):
    lifetimes: Array1DField = Array1DField(dtype=np.float64)
    events: Array1DField = Array1DField(
        dtype=np.bool_
    )  # event indicators (right censored or not)

    def __init__(self, nb_samples: int, nb_assets: int):
        super().__init__(nb_samples, nb_assets)
        self.fields.extend(["lifetimes", "events"])


class RenewalRewardData(RenewalData):
    total_rewards: Array1DField = Array1DField(dtype=np.float64)

    def __init__(self, nb_samples: int, nb_assets: int):
        super().__init__(nb_samples, nb_assets)
        self.fields.append("total_rewards")

    def cum_total_rewards(
        self, sample: int
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        ind = self.samples == sample
        s = np.argsort(self.event_times[ind])
        times = np.insert(self.event_times[ind][s], 0, 0)
        z = np.insert(self.total_rewards[ind][s].cumsum(), 0, 0)
        return times, z

    def mean_total_reward(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        s = np.argsort(self.event_times)
        times = np.insert(self.event_times[s], 0, 0)
        z = np.insert(self.total_rewards[s].cumsum(), 0, 0) / self.nb_samples
        return times, z
