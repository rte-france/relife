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

    nb_ages = np.sum(
        np.tile(assets, (nb_assets, 1)) == np.arange(0, nb_assets)[:, None], axis=1
    )

    #  control that t0 is coherent with ages (remove observations with no ages)
    index_first = np.where(np.roll(assets, 1) != assets)[0]
    index_last = np.append(index_first[1:] - 1, len(assets) - 1)

    if np.any(ages[index_first] <= a0[nb_ages != 0]):
        raise ValueError("Every t0 per asset must be lower than every ages per asset")
    if np.any(ages[index_last] >= af[nb_ages != 0]):
        raise ValueError("Every af per asset must be greater than every ages per asset")

    insert_index = np.cumsum(nb_ages)
    event = np.ones_like(ages, dtype=np.bool_)
    time = np.insert(ages, insert_index, af)
    event = np.insert(event, insert_index, False)
    entry = np.insert(ages, np.insert(insert_index[:-1], 0, 0), a0)

    return time, event, entry


@dataclass
class NHPPData(CountData):
    durations: NDArray[np.float64] = field(repr=False)  # durations between repairs
    repairs: NDArray[np.int64] = field(repr=False)  # nb of repairs

    def iter(self, sample_id: Optional[int] = None):
        # note that event_times == ages
        if sample_id is None:
            return CountDataIterable(self, ("timeline", "durations", "repairs"))
        else:
            if sample_id not in self.samples_unique_ids:
                raise ValueError(f"{sample_id} is not part of samples index")
            return filterfalse(
                lambda x: x[0] != sample_id,
                CountDataIterable(self, ("timeline", "durations", "repairs")),
            )

    # durations in post_init ?

    def number_of_repairs(self):
        pass

    def mean_number_of_repairs(self):
        pass

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
