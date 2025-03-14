from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import filterfalse
from sys import exec_prefix
from typing import Optional, Union

import numpy as np
from numpy.typing import NDArray

from .counting import CountData
from relife.plots import PlotNHPPData, PlotConstructor
from ..types import Args


def nhpp_data_factory(
    event_asset_ids: Union[Sequence[str], NDArray[np.int64]],
    ages_at_event: NDArray[np.float64],
    asset_ids: Optional[Union[Sequence[str], NDArray[np.int64]]] = None,
    start_ages: Optional[NDArray[np.float64]] = None,
    end_ages: Optional[NDArray[np.float64]] = None,
    model_args: tuple[Args, ...] = (),
):
    # convert inputs to arrays
    event_asset_ids = np.asarray(event_asset_ids)
    ages_at_event = np.asarray(ages_at_event, dtype=np.float64)
    if asset_ids is not None:
        asset_ids = np.asarray(asset_ids)
    if start_ages is not None:
        start_ages = np.asarray(start_ages, dtype=np.float64)
    if end_ages is not None:
        end_ages = np.asarray(end_ages, dtype=np.float64)

    # control shapes
    if event_asset_ids.ndim != 1:
        raise ValueError("Invalid array shape for event_asset_ids. Expected 1d-array")
    if ages_at_event.ndim != 1:
        raise ValueError("Invalid array shape for ages_at_event. Expected 1d-array")
    if len(event_asset_ids) != len(ages_at_event):
        raise ValueError(
            "Shape of event_asset_ids and ages_at_event must be equal. Expected equal length 1d-arrays"
        )
    if asset_ids is not None:
        if asset_ids.ndim != 1:
            raise ValueError("Invalid array shape for asset_ids. Expected 1d-array")
        if start_ages is not None:
            if start_ages.ndim != 1:
                raise ValueError(
                    "Invalid array shape for start_ages. Expected 1d-array"
                )
            if len(start_ages) != len(asset_ids):
                raise ValueError(
                    "Shape of asset_ids and start_ages must be equal. Expected equal length 1d-arrays"
                )
        if end_ages is not None:
            if end_ages.ndim != 1:
                raise ValueError("Invalid array shape for end_ages. Expected 1d-array")
            if len(end_ages) != len(asset_ids):
                raise ValueError(
                    "Shape of asset_ids and end_ages must be equal. Expected equal length 1d-arrays"
                )
        if bool(model_args):
            for arg in model_args:
                arg = np.atleast_2d(np.asarray(arg, dtype=np.float64))
                if arg.ndim > 2:
                    raise ValueError(
                        "Invalid arg shape in model_args. Arrays must be 0, 1 or 2d"
                    )
                try:
                    arg.reshape((len(asset_ids), -1))
                except ValueError:
                    raise ValueError(
                        "Invalid arg shape in model_args. Arrays must coherent with the number of assets given by assets_ids"
                    )
    else:
        if start_ages is not None:
            raise ValueError(
                "If start_ages is given, corresponding asset ids must be given in assets_ids"
            )
        if end_ages is not None:
            raise ValueError(
                "If end_ages is given, corresponding asset ids must be given in assets_ids"
            )
        if bool(model_args):
            raise ValueError(
                "If model_args is given, corresponding asset ids must be given in assets_ids"
            )

    if event_asset_ids.dtype != np.int64:
        event_asset_ids = np.unique(event_asset_ids, return_inverse=True)[1]
    # convert assets_id to int id
    if asset_ids is not None:
        if asset_ids.dtype != np.int64:
            asset_ids = np.unique(asset_ids, return_inverse=True)[1]
        # control ids correspondance
        if not np.all(np.isin(event_asset_ids, asset_ids)):
            raise ValueError(
                "If asset_ids is filled, all values of event_asset_ids must exist in asset_ids"
            )

    # sort fields
    sort_ind = np.lexsort((ages_at_event, event_asset_ids))
    event_asset_ids = event_asset_ids[sort_ind]
    ages_at_event = ages_at_event[sort_ind]

    # number of age value per asset id
    nb_ages_per_asset = np.unique_counts(event_asset_ids).counts
    # index of the first ages and last ages in ages_at_event
    first_age_index = np.where(np.roll(event_asset_ids, 1) != event_asset_ids)[0]
    last_age_index = np.append(first_age_index[1:] - 1, len(event_asset_ids) - 1)

    if asset_ids is not None:

        # sort fields
        sort_ind = np.sort(asset_ids)
        start_ages = start_ages[sort_ind] if start_ages is not None else start_ages
        end_ages = end_ages[sort_ind] if end_ages is not None else end_ages
        model_args = tuple((arg[sort_ind] for arg in model_args))

        if start_ages is not None:
            if np.any(
                ages_at_event[first_age_index] <= start_ages[nb_ages_per_asset != 0]
            ):
                raise ValueError(
                    "Each start_ages value must be lower than all of its corresponding ages_at_event values"
                )
        if end_ages is not None:
            if np.any(
                ages_at_event[last_age_index] >= end_ages[nb_ages_per_asset != 0]
            ):
                raise ValueError(
                    "Each end_ages value must be greater than all of its corresponding ages_at_event values"
                )

    event = np.ones_like(ages_at_event, dtype=np.bool_)
    # insert_index = np.cumsum(nb_ages_per_asset)
    # insert_index = last_age_index + 1
    if end_ages is not None:
        time = np.insert(ages_at_event, last_age_index + 1, end_ages)
        event = np.insert(event, last_age_index + 1, False)
        _ids = np.insert(event_asset_ids, last_age_index + 1, asset_ids)
        if start_ages is not None:
            entry = np.insert(
                ages_at_event, np.insert((last_age_index + 1)[:-1], 0, 0), start_ages
            )
        else:
            entry = np.insert(ages_at_event, first_age_index, 0.0)
    else:
        time = ages_at_event.copy()
        _ids = event_asset_ids.copy()
        if start_ages is not None:
            entry = np.roll(ages_at_event, 1)
            entry[first_age_index] = start_ages
        else:
            entry = np.roll(ages_at_event, 1)
            entry[first_age_index] = 0.0
    model_args = tuple((np.take(arg, _ids) for arg in model_args))

    return time, event, entry, model_args


@dataclass
class NHPPData(CountData):
    durations: NDArray[np.float64] = field(repr=False)

    def number_of_repairs(self):
        # alias name
        return self.nb_events()

    def mean_number_of_repairs(self):
        return self.mean_nb_events()

    # def number_of_repairs(self):
    #     sort_all = np.argsort(self.timeline)
    #     timeline = np.insert(self.timeline[sort_all], 0, 0)
    #     nb_repairs = np.cumsum(np.insert(self.nb_repairs[sort_all], 0, 0))
    #     return timeline, nb_repairs
    #
    # def mean_number_of_repairs(self):
    #     sort_all = np.argsort(self.timeline)
    #     timeline = np.insert(self.timeline[sort_all], 0, 0)
    #     nb_repairs = np.cumsum(np.insert(self.nb_repairs[sort_all], 0, 0) / len(self))
    #     return timeline, nb_repairs

    @property
    def plot(self) -> PlotConstructor:
        return PlotNHPPData(self)

    # def to_fit(
    #     self, t0: float = 0.0, tf: Optional[float] = None, sample: Optional[int] = None
    # ):
    #
    #     # step 1 : select and control
    #     s = self.samples_ids == sample if sample is not None else Ellipsis
    #
    #     max_event_time = np.max(self.timeline)
    #     if tf is not None:
    #         if max_event_time <= tf:
    #             tf = None
    #             warnings.warn(
    #                 f"Selected window might be too large with tf {tf}. If you explicitly want to convert all date, leave tf as None (default Value)"
    #             )
    #     if t0 != 0:
    #         if max_event_time <= t0:
    #             raise ValueError(
    #                 f"Invalid t0 {t0} where the maximum event time value is {max_event_time}"
    #             )
    #
    #     _assets_index = self.assets_ids[s]
    #     _samples_index = self.samples_ids[s]
    #     _ages = self.timeline[s]
    #     _durations = self.durations[s]
    #     _assets = (
    #         _assets_index + _samples_index
    #     )  # all i.i.d evaluated as multiple assets
    #
    #     # step 2 : sort
    #     sort = np.lexsort((_ages, _assets))
    #     _assets = _assets[sort]
    #     _ages = _ages[sort]
    #     _durations = _durations[sort]
    #
    #     # step 3 : collect index
    #     if t0 == 0.0 and tf is None:
    #         if self.nb_assets == self.nb_samples == 1:
    #             a0_index = np.array([0])
    #             af_index = np.array([len(_assets) - 1])
    #         else:
    #             a0_index = np.where(np.roll(_assets, 1) != _assets)[0]
    #             af_index = np.append(a0_index[1:] - 1, len(_assets) - 1)
    #     else:
    #         if tf is None:
    #             tf = max_event_time
    #         a0_index = np.where((_ages >= t0) == False)[0] + 1
    #
    #     assert len(a0_index) == len(af_index)
    #
    #     # step 4 : convert and return
    #     a0 = _ages[a0_index]  # in order asset 0, 1, ... , n
    #     af = _ages[af_index]
    #     to_delete = np.concatenate((a0_index, af_index))
    #     _ages = np.delete(_ages, to_delete)
    #     _assets = np.delete(_assets, to_delete)
    #
    #     return a0, af, _ages, _assets


@dataclass
class NHPPPolicyData(CountData):
    durations: NDArray[np.float64] = field(repr=False)  # durations between repairs
    repairs: NDArray[np.int64] = field(repr=False)  # nb of repairs

    def to_fit(self, sample: Optional[int] = None):

        s = self.samples_ids == sample if sample is not None else Ellipsis

        _assets_index = self.assets_ids[s]
        _samples_index = self.samples_ids[s]
        _ages = self.timeline[s]
        _assets = _assets_index + _samples_index

        sort = np.lexsort((_ages, _assets))
        _assets = _assets[sort]
        _ages = _ages[sort]
        if self.nb_assets == self.nb_samples == 1:
            a0_index = np.array([0])
            af_index = np.array([len(_assets) - 1])
        else:
            a0_index = np.where(np.roll(_assets, 1) != _assets)[0]
            af_index = np.append(a0_index[1:] - 1, len(_assets) - 1)

        assert len(a0_index) == len(af_index)

        a0 = _ages[a0_index]  # in order asset 0, 1, ... , n
        af = _ages[af_index]
        to_delete = np.concatenate((a0_index, af_index))
        _ages = np.delete(_ages, to_delete)
        _assets = np.delete(_assets, to_delete)

        return a0, af, _ages, _assets
