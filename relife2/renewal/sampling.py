from dataclasses import field, fields, dataclass
from typing import Iterator, Optional, TypeVar, Generic

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


@dataclass
class CountData:
    samples: NDArray[np.int64] = field(repr=False)  # samples index
    assets: NDArray[np.int64] = field(repr=False)  # assets index
    order: NDArray[np.int64] = field(
        repr=False
    )  # order index (order in generation process)
    event_times: NDArray[np.float64] = field(repr=False)

    nb_samples: int = field(init=False)
    nb_assets: int = field(init=False)
    samples_index: NDArray[np.int64] = field(
        init=False, repr=False
    )  # unique samples index
    assets_index: NDArray[np.int64] = field(
        init=False, repr=False
    )  # unique assets index

    def __post_init__(self):
        fields_values = [
            getattr(self, _field.name) for _field in fields(self) if _field.init
        ]
        if not all(arr.ndim == 1 for arr in fields_values):
            raise ValueError("All array values must be 1d")
        if not len(set(arr.shape[0] for arr in fields_values)) == 1:
            raise ValueError("All array values must have the same shape")

        self.samples_index = np.unique(self.samples)
        self.assets_index = np.unique(self.assets)
        self.nb_samples = len(self.samples_index)
        self.nb_assets = len(self.assets_index)

    def __len__(self) -> int:
        return self.nb_samples * self.nb_assets

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


class CountDataIterable:
    def __init__(self, data: CountData):
        self.data = data

        sorted_index = np.lexsort(
            (self.data.order, self.data.assets, self.data.samples)
        )
        self.sorted_fields = {
            _field.name: getattr(self.data, _field.name).copy()[sorted_index]
            for _field in fields(self.data)
            if _field.init
        }

    def __len__(self) -> int:
        return self.data.nb_samples * self.data.nb_assets

    def __iter__(self) -> Iterator[tuple[int, int, dict[str, NDArray[np.float64]]]]:

        for sample in self.data.samples_index:
            sample_mask = self.sorted_fields["samples"] == sample
            for asset in self.data.assets_index:
                asset_mask = self.sorted_fields["assets"][sample_mask] == asset
                values_dict = {
                    k: v[sample_mask][asset_mask]
                    for k, v in self.sorted_fields.items()
                    if k not in ("samples", "assets", "events")
                }
                yield int(sample), int(asset), values_dict


@dataclass
class RenewalData(CountData, Generic[M, M1]):
    lifetimes: NDArray[np.float64] = field(repr=False)
    events: NDArray[np.bool_] = field(
        repr=False
    )  # event indicators (right censored or not)

    # TODO: remove model, model_args from there. Not needed
    # 1. if init_params uses *args (see model.py) then
    # 2. init_params in Regression does not rely on covar stored in Sample
    # 3. Sample does not need to store args
    # 4. to_lifetime_data only returns a object constructing on time, event, entry only
    def to_lifetime_data(
        self,
        model: LifetimeModel[*M],
        model_args: M = (),
        t0: float = 0,
        tf: Optional[float] = None,
        sample: Optional[int] = None,
        model1: Optional[LifetimeModel[*M1]] = None,
        model1_args: M1 = (),
    ):

        if t0 >= tf:
            raise ValueError("`t0` must be strictly lesser than `tf`")

        # Filtering sample and sorting by times
        s = self.samples == sample if sample is not None else Ellipsis
        order = np.argsort(self.event_times[s])
        indices = self.assets[s][order]
        samples = self.samples[s][order]
        uindices = np.ravel_multi_index(
            (indices, samples), (self.nb_assets, self.nb_samples)
        )
        event_times = self.event_times[s][order]
        lifetimes = self.lifetimes[s][order]
        events = self.events[s][order]
        order = self.order[s][order]

        # Indices of interest
        ind0 = (event_times > t0) & (
            event_times <= tf
        )  # Indices of replacement occuring inside the obervation window
        ind1 = (
            event_times > tf
        )  # Indices of replacement occuring after the observation window which include right censoring

        # Replacements occuring inside the observation window
        time0 = lifetimes[ind0]
        event0 = events[ind0]
        entry0 = np.zeros(time0.size)
        _, LT = np.unique(
            uindices[ind0], return_index=True
        )  # get the indices of the first replacements ocurring in the observation window
        b0 = (
            event_times[ind0][LT] - lifetimes[ind0][LT]
        )  # time at birth for the firt replacements
        entry0[LT] = np.where(b0 >= t0, 0, t0 - b0)

        args0 = args_take(indices[ind0], *self.args)

        # Right censoring
        _, RC = np.unique(uindices[ind1], return_index=True)
        bf = (
            times[ind1][RC] - durations[ind1][RC]
        )  # time at birth for the right censored
        b1 = bf[
            bf < tf
        ]  # ensure that time of birth for the right censored is not equal to tf.
        time1 = tf - b1
        event1 = np.zeros(b1.size)
        entry1 = np.where(b1 >= t0, 0, t0 - b1)
        args1 = args_take(indices[ind1][RC][bf < tf], *self.args)

        # Concatenate
        time = np.concatenate((time0, time1))
        event = np.concatenate((event0, event1))
        entry = np.concatenate((entry0, entry1))
        args = tuple(
            np.concatenate((arg0, arg1), axis=0) for arg0, arg1 in zip(args0, args1)
        )
        return LifetimeData(time, event, entry, args)


@dataclass
class RenewalRewardData(RenewalData[M, M1], Generic[M, M1]):
    total_rewards: NDArray[np.float64] = field(repr=False)

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
