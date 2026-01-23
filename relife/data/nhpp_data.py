# pyright: basic

from dataclasses import dataclass, field
from typing import Optional, Sequence, Union

import numpy as np
from numpy.typing import NDArray


@dataclass
class NHPPData:
    ages_at_events: NDArray[np.float64]
    events_assets_ids: Union[Sequence[str], NDArray[np.int64]]
    first_ages: Optional[NDArray[np.float64]] = field(repr=False, default=None)
    last_ages: Optional[NDArray[np.float64]] = field(repr=False, default=None)
    model_args: Optional[tuple[float | NDArray[np.float64], ...]] = field(repr=False, default=None)
    assets_ids: Optional[Union[Sequence[str], NDArray[np.int64]]] = field(repr=False, default=None)

    first_age_index: NDArray[np.int64] = field(repr=False, init=False)
    last_age_index: NDArray[np.int64] = field(repr=False, init=False)

    def __post_init__(self):
        # convert inputs to arrays
        self.events_assets_ids = np.unique(np.asarray(self.events_assets_ids), return_inverse=True)[1].astype(np.uint32)
        self.ages_at_events = np.asarray(self.ages_at_events, dtype=np.float64)
        if self.assets_ids is not None:
            self.assets_ids = np.unique(np.asarray(self.assets_ids), return_inverse=True)[1].astype(np.uint32)
        if self.first_ages is not None:
            self.first_ages = np.asarray(self.first_ages, dtype=np.float64)
        if self.last_ages is not None:
            self.last_ages = np.asarray(self.last_ages, dtype=np.float64)

        # control shapes
        if self.events_assets_ids.ndim != 1:
            raise ValueError("Invalid array shape for events_assets_ids. Expected 1d-array")
        if self.ages_at_events.ndim != 1:
            raise ValueError("Invalid array shape for ages. Expected 1d-array")
        if len(self.events_assets_ids) != len(self.ages_at_events):
            raise ValueError("Shape of events_assets_ids and ages must be equal. Expected equal length 1d-arrays")
        if self.assets_ids is not None:
            if self.assets_ids.ndim != 1:
                raise ValueError("Invalid array shape for assets_ids. Expected 1d-array")
            if self.first_ages is not None:
                if self.first_ages.ndim != 1:
                    raise ValueError("Invalid array shape for start_ages. Expected 1d-array")
                if len(self.first_ages) != len(self.assets_ids):
                    raise ValueError(
                        "Shape of assets_ids and start_ages must be equal. Expected equal length 1d-arrays"
                    )
            if self.last_ages is not None:
                if self.last_ages.ndim != 1:
                    raise ValueError("Invalid array shape for last_ages. Expected 1d-array")
                if len(self.last_ages) != len(self.assets_ids):
                    raise ValueError("Shape of assets_ids and last_ages must be equal. Expected equal length 1d-arrays")
            if bool(self.model_args):
                for arg in self.model_args:
                    arg = np.atleast_2d(np.asarray(arg, dtype=np.float64))
                    if arg.ndim > 2:
                        raise ValueError("Invalid arg shape in model_args. Arrays must be 0, 1 or 2d")
                    try:
                        arg.reshape((len(self.assets_ids), -1))
                    except ValueError:
                        raise ValueError(
                            "Invalid arg shape in model_args. Arrays must coherent with the number of assets given by assets_ids"
                        )
        else:
            if self.first_ages is not None:
                raise ValueError("If first_ages is given, corresponding asset ids must be given in assets_ids")
            if self.last_ages is not None:
                raise ValueError("If last_ages is given, corresponding asset ids must be given in assets_ids")
            if bool(self.model_args):
                raise ValueError("If model_args is given, corresponding asset ids must be given in assets_ids")

        # if self.events_assets_ids.dtype != np.int64:
        #     events_assets_ids = np.unique(self.events_assets_ids, return_inverse=True)[1]
        # # convert assets_id to int id
        # if self.assets_ids is not None:
        #     if self.assets_ids.dtype != np.int64:
        #         assets_ids = np.unique(self.assets_ids, return_inverse=True)[1]
        #     # control ids correspondance
        #     if not np.all(np.isin(self.events_assets_ids, self.assets_ids)):
        #         raise ValueError("If assets_ids is filled, all values of events_assets_ids must exist in assets_ids")

        # sort fields
        sort_ind = np.lexsort((self.ages_at_events, self.events_assets_ids))
        self.events_assets_ids = self.events_assets_ids[sort_ind]
        self.ages_at_events = self.ages_at_events[sort_ind]

        # number of age value per asset id
        nb_ages_per_asset = np.unique_counts(self.events_assets_ids).counts
        # index of the first ages and last ages in ages
        self.first_age_index = np.where(np.roll(self.events_assets_ids, 1) != self.events_assets_ids)[0]
        self.last_age_index = np.append(self.first_age_index[1:] - 1, len(self.events_assets_ids) - 1)

        if self.assets_ids is not None:

            # sort fields
            sort_ind = np.argsort(self.assets_ids)
            self.assets_ids = self.assets_ids[sort_ind]
            self.first_ages = self.first_ages[sort_ind] if self.first_ages is not None else self.first_ages
            self.last_ages = self.last_ages[sort_ind] if self.last_ages is not None else self.last_ages
            self.model_args = (
                tuple((arg[sort_ind] for arg in self.model_args)) if self.model_args is not None else self.model_args
            )

            if self.first_ages is not None:
                if np.any(self.ages_at_events[self.first_age_index] <= self.first_ages[nb_ages_per_asset != 0]):
                    raise ValueError("Each first_ages value must be lower than all of its corresponding ages values")
            if self.last_ages is not None:
                if np.any(self.ages_at_events[self.last_age_index] >= self.last_ages[nb_ages_per_asset != 0]):
                    raise ValueError("Each last_ages value must be greater than all of its corresponding ages values")

    def to_lifetime_data(self):
        event = np.ones_like(self.ages_at_events, dtype=np.bool_)
        # insert_index = np.cumsum(nb_ages_per_asset)
        # insert_index = last_age_index + 1
        if self.last_ages is not None:
            time = np.insert(self.ages_at_events, self.last_age_index + 1, self.last_ages)
            event = np.insert(event, self.last_age_index + 1, False)
            _ids = np.insert(self.events_assets_ids, self.last_age_index + 1, self.assets_ids)
            if self.first_ages is not None:
                entry = np.insert(
                    self.ages_at_events,
                    np.insert((self.last_age_index + 1)[:-1], 0, 0),
                    self.first_ages,
                )
            else:
                entry = np.insert(self.ages_at_events, self.first_age_index, 0.0)
        else:
            time = self.ages_at_events.copy()
            _ids = self.events_assets_ids.copy()
            if self.first_ages is not None:
                entry = np.roll(self.ages_at_events, 1)
                entry[self.first_age_index] = self.first_ages
            else:
                entry = np.roll(self.ages_at_events, 1)
                entry[self.first_age_index] = 0.0
        model_args = tuple((np.take(arg, _ids) for arg in self.model_args)) if self.model_args is not None else ()
        return time, event, entry, model_args
