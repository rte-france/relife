from typing import Optional, Self, TypedDict

import numpy as np
from numpy.typing import ArrayLike, DTypeLike, NDArray


class SampleData(TypedDict):
    t0: float
    tf: float
    data: NDArray[DTypeLike]


def check_nb_events_call(obj_type: type):
    from .renewal_process import RenewalProcess

    if obj_type is not RenewalProcess:
        raise ValueError(f"{obj_type} object compute nb_events from sample")


class SampleFunction:

    sample_data: SampleData

    def __init__(self, obj_type: type, sample_data: SampleData):
        self.obj_type = obj_type
        self.sample_data = sample_data

    def _check_sample_data(self):
        if self.sample_data is None:
            raise ValueError(f"{self.obj_type} object has no sample_data yet. Call sample_count_data first")

    def nb_events(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        self._check_sample_data()
        check_nb_events_call(self.obj_type)
        sort = np.argsort(self.sample_data["data"]["timeline"])
        timeline = self.sample_data["data"]["timeline"][sort]
        counts = np.ones_like(timeline)
        timeline = np.insert(timeline, 0, self.sample_data["t0"])
        counts = np.insert(counts, 0, 0)
        counts[timeline == self.sample_data["tf"]] = 0
        return timeline, np.cumsum(counts)

    def mean_nb_events(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        timeline, counts = self.nb_events()
        return timeline, counts / len(self.sample_data["data"])

    def select(self, sample_id: Optional[ArrayLike] = None, asset_id: Optional[ArrayLike] = None) -> Self:
        mask = np.ones(len(self.sample_data["data"]), dtype=np.bool_)
        if sample_id is not None:
            mask = mask & np.isin(self.sample_data["data"]["sample_id"], sample_id)
        if asset_id is not None:
            mask = mask & np.isin(self.sample_data["data"]["asset_id"], asset_id)
        substruct_array = self.sample_data["data"][mask].copy()
        return SampleFunction(self.obj_type, substruct_array)
