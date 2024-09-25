from functools import wraps
from typing import Callable, Optional

import numpy as np
from numpy.typing import NDArray

from relife2.core import LifetimeModel


def lifetime_rvs(
    model: LifetimeModel,
    nb_samples: int,
    nb_assets: int,
    args: tuple[NDArray[np.float64], ...] = (),
):
    if bool(args) and args[0].ndim == 2:
        if nb_assets != args[0].shape[0]:
            raise ValueError
        rvs_size = nb_samples  # rvs size
    else:
        rvs_size = nb_samples * nb_assets

    yield model.rvs(*args, size=rvs_size)


def lifetimes_generator(
    model, model_args, nb_samples, nb_assets, initmodel=None, initmodel_args=()
):
    if initmodel is not None:
        yield from lifetime_rvs(initmodel, nb_samples, nb_assets, initmodel_args)
    while True:
        yield from lifetime_rvs(model, nb_samples, nb_assets, model_args)


def argscheck(method: Callable) -> Callable:
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        for key, value in kwargs.items():
            if key.endswith("_args"):
                for array in value:
                    if array.shape[0] != self.nb_assets:
                        raise ValueError(
                            f"Expected {self.nb_assets} nb assets but got {array.shape[0]} in {key}"
                        )
        return method(self, *args, **kwargs)

    return wrapper


class LifetimesSampler:

    def __init__(
        self,
        model: LifetimeModel,
        nb_assets: int = 1,
        initmodel: Optional[LifetimeModel] = None,
    ):
        # nb_assets is mandotory to control all model's information coherence
        # and allow model with ndim 2 args and initmodel without args
        # see how it is used in lifetime_rvs

        self.model = model
        self.initmodel = initmodel
        self.nb_assets = nb_assets

        self.values = None
        self.nb_samples = None
        self.samples_index = None
        self.assets_index = None
        self.flatten_samples_index = None

    @argscheck
    def sample(
        self,
        nb_samples: int,
        end_time: float,
        model_args: tuple[NDArray[np.float64], ...] = (),
        initmodel_args: tuple[NDArray[np.float64], ...] = (),
    ):

        self.nb_samples = nb_samples

        spent_time = np.zeros(self.nb_samples * self.nb_assets)
        all_samples_index, all_assets_index = np.unravel_index(
            np.arange(self.nb_samples * self.nb_assets),
            (self.nb_samples, self.nb_assets),
        )

        generator = lifetimes_generator(
            self.model,
            model_args,
            self.nb_samples,
            self.nb_assets,
            initmodel=self.initmodel,
            initmodel_args=initmodel_args,
        )

        values = np.array([], dtype=np.float64)
        samples_index = np.array([], dtype=np.int64)
        assets_index = np.array([], dtype=np.int64)

        still_valid = spent_time < end_time
        while still_valid.any():
            event_times = next(generator).reshape(-1)[still_valid]
            spent_time[still_valid] += event_times
            values = np.concatenate((values, event_times))
            samples_index = np.concatenate(
                (samples_index, all_samples_index[still_valid])
            )
            assets_index = np.concatenate((assets_index, all_assets_index[still_valid]))
            still_valid = spent_time < end_time

        sorted_index = np.lexsort((assets_index, samples_index))

        self.values = values[sorted_index]
        self.samples_index = samples_index[sorted_index]
        self.assets_index = assets_index[sorted_index]

        self.flatten_samples_index = (
            np.where(self.samples_index[:-1] != self.samples_index[1:])[0] + 1
        )

    def __len__(self):
        if self.nb_samples is None:
            return 0
        if self.nb_assets == 1:
            return self.nb_samples
        else:
            return self.nb_assets

    @property
    def mean_number_of_events(self):
        if self.nb_samples is None:
            raise ValueError

        return np.mean(
            np.diff(
                self.flatten_samples_index,
                prepend=0,
                append=len(self.values) - 1,
            )
        )

    def __get(self, index: int):
        if self.flatten_samples_index.size == 0:
            slice_index = slice(None, None)
        else:
            if index == len(self.flatten_samples_index):
                start = self.flatten_samples_index[-1]
            elif index > len(self.flatten_samples_index):
                raise IndexError
            else:
                start = self.flatten_samples_index[index]
            if start == self.flatten_samples_index[0]:
                slice_index = slice(None, start)
            elif start != self.flatten_samples_index[-1]:
                stop = self.flatten_samples_index[index + 1]
                slice_index = slice(start, stop)
            else:
                slice_index = slice(start, None)
        return self.values[slice_index], self.assets_index[slice_index]

    def __getitem__(self, index: int):
        try:
            values, assets_index = self.__get(index)
        except IndexError as err:
            raise IndexError(
                f"index {index} is out of bounds for {self.nb_samples} samples on {self.nb_assets} assets"
            ) from err
        if self.nb_assets == 1:
            nb_events = len(values)
        else:
            values = np.split(
                values, np.where(assets_index[:-1] != assets_index[1:])[0] + 1
            )
            nb_events = list(map(len, values))
        return {"values": values, "nb_events": nb_events}
