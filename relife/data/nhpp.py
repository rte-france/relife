from dataclasses import dataclass, field
from itertools import filterfalse
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from .counting import CountData, CountDataIterable


def nhpp_lifetime_data_factory(
    a0: NDArray[np.float64],
    af: NDArray[np.float64],
    ages: NDArray[np.float64],
    assets: NDArray[np.int64],
):
    """
    Parameters
    ----------
        a0 : ndarray of floats, shape (m,)
            Beginnings of the observation windows for each asset (m assets)
        af : ndarray of floats, shape (m,)
            Ends of the observation windows for each asset (m assets)
        ages : ndarray of floats, shape (K,)
            Ages values per asset at each failure time
        assets : ndarray of ints, shape (K,)
            Index values identifying each asset. It must correspond to `ages`.

    Examples
    --------
    >>> a0 = np.array([1., 3., 6.])
    >>> af = np.array([5., 8., 15.])
    >>> ages = np.array([2., 4., 12., 13., 14.])
    >>> assets = np.array([0, 0, 2, 2, 2], dtype=np.int64)
    >>> time, event, entry, nhpp_lifetime_data_factory(a0, af, ages, assets)

    """
    # PREPROCESS AND SANITY CHECKS

    nb_assets = len(a0)
    if nb_assets != len(af):
        raise ValueError("a0 and af must have the same number of elements")

    if np.any(a0 > af):
        raise ValueError("All a0 values must be greater than af values")

    sort = np.lexsort((ages, assets))
    ages = ages[sort]
    assets = assets[sort]

    # uncorrect
    # unique_assets_index, inverse_index = np.unique(assets, return_inverse=True)
    # if len(unique_assets_index) != nb_assets:
    #     raise ValueError(
    #         "Assets must have the same number of unique elements than t0 and tf"
    #     )
    # if unique_assets_index != np.arange(0, len(nb_assets)):
    #     assets = np.sort(inverse_index)
    # --------------------------

    # [2, 0, 3] # change np.arange(0, nb_assets) -> np.unique(nb_assets) (not 0 from nb_assets)
    nb_ages_per_asset = np.sum(
        np.tile(assets, (nb_assets, 1)) == np.arange(0, nb_assets)[:, None], axis=1
    )

    # control that tf is coherent with ages (remove observations with no ages)
    index_last_age_per_asset = np.append(
        np.where(assets[:-1] != assets[1:])[0], len(assets) - 1
    )
    last_age_per_asset = ages[index_last_age_per_asset]
    if np.any(last_age_per_asset >= af[nb_ages_per_asset != 0]):
        raise ValueError("Every af per asset must be greater than every ages per asset")

    #  control that t0 is coherent with ages (remove observations with no ages)
    index_first_age_per_asset = np.insert(index_last_age_per_asset + 1, 0, 0)[:-1]
    first_age_per_asset = ages[index_first_age_per_asset]
    if np.any(first_age_per_asset <= a0[nb_ages_per_asset != 0]):
        raise ValueError("Every t0 per asset must be lower than every ages per asset")

    insert_index = np.cumsum(nb_ages_per_asset)
    event = np.ones_like(ages, dtype=np.bool_)
    time = np.insert(ages, insert_index, af)
    event = np.insert(event, insert_index, False)
    entry = np.roll(np.insert(ages, insert_index, np.roll(a0, -1)), 1)

    return time, event, entry


@dataclass
class NHPPData(CountData):
    durations: NDArray[np.float64] = field(repr=False)

    def iter(self, sample: Optional[int] = None):
        # note that event_times == ages
        if sample is None:
            return CountDataIterable(self, ("event_times", "durations"))
        else:
            if sample not in self.samples_unique_index:
                raise ValueError(f"{sample} is not part of samples index")
            return filterfalse(
                lambda x: x[0] != sample,
                CountDataIterable(self, ("event_times", "durations")),
            )

    # durations in post_init ?

    def number_of_repairs(self):
        pass

    def mean_number_of_repairs(self):
        pass

    def to_fit(self, t0: float, tf: float, sample: Optional[int] = None):
        max_event_time = np.max(self.event_times)
        if tf > max_event_time:
            tf = max_event_time

        if t0 >= tf:
            raise ValueError("`t0` must be strictly lower than `tf`")

        ages = np.array([], dtype=np.float64)
        assets = np.array([], dtype=np.float64)

        s = self.samples_index == sample if sample is not None else Ellipsis

        complete_left_truncated = (self.event_times[s] > t0) & (
            self.event_times[s] <= tf
        )

        _assets_index = self.assets_index[s][complete_left_truncated]
        _samples_index = self.samples_index[s][complete_left_truncated]
        _assets = _assets_index + _samples_index

        sort = np.argsort(_assets)
        _assets = _assets[sort]
        _ages = self.event_times[s][complete_left_truncated][sort]
        _durations = self.durations[s][complete_left_truncated][sort]

        shift_left = _ages - _durations
        left_truncated = (t0 - shift_left) >= 0
        # left_truncations = (t0 - shift_left)[left_truncated]

        a0 = _ages[left_truncated]
        ages = np.concatenate((ages, _ages[~left_truncated]))
        assets = np.concatenate((assets, _assets[~left_truncated]))

        right_censored = self.event_times[s] > tf

        _assets_index = self.assets_index[s][right_censored]
        _samples_index = self.samples_index[s][right_censored]
        _assets = _assets_index + _samples_index

        sort = np.argsort(_assets)
        _assets = _assets[sort]
        _ages = self.event_times[s][right_censored][sort]
        _durations = self.durations[s][right_censored][sort]

        shift_left = _ages - _durations
        target_right_censored = (tf - shift_left) >= 0
        # right_censoring = (tf - shift_left)[target_right_censored]

        af = np.ones(np.sum(target_right_censored)) * tf

        assert len(a0) == len(af)

        return a0, af, ages, assets
