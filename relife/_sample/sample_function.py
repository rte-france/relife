from typing import TypeVarTuple, Optional

import numpy as np
from numpy.typing import NDArray, DTypeLike

from relife.data import LifetimeData, NHPPData
from relife.frozen_model import FrozenParametricLifetimeModel
from relife.lifetime_model import ParametricLifetimeModel
from relife.stochastic_process import RenewalProcess, RenewalRewardProcess

Args = TypeVarTuple("Args")




def sample_failure_data(
    model: ParametricLifetimeModel[()] | RenewalProcess | RenewalRewardProcess,
    size : int,
    window : tuple[float, float],
    nb_assets : Optional[int] = None,
    astuple : bool = False,
    maxsample: int = 1e5,
    seed: Optional[int] = None,
) -> LifetimeData | NHPPData | tuple[NDArray[DTypeLike]|tuple[NDArray[np.float64], ...], ...]:

    size = (nb_assets, size) if nb_assets is not None else size
    t0, tf = window
    if tf <= t0:
        raise ValueError

    match model:
        case ParametricLifetimeModel():
            time, event, entry = model.sample_time_event_entry(size=size, seed=seed)
            # time, event, entry shape : (m, size)
            entry = np.where(time > t0, np.full_like(time, t0), entry)
            time = np.where(time > tf, np.full_like(time, tf), time)
            event[time > tf] = False
            selection = t0 <= time <= tf
            asset_id, sample_id = np.where(selection)
            args =
            if isinstance(model, FrozenParametricLifetimeModel):
                args = model.args
                args = tuple((np.take(arg, asset_id) for arg in args))
            if astuple:
                return time[selection].copy(), event[selection].copy(), entry[selection].copy(), args
            return LifetimeData(time[selection].copy(), event=event[selection].copy(), entry=entry[selection].copy(), args=args)

        case RenewalProcess() as process:
            from relife.lifetime_model import LeftTruncatedModel
            from relife.stochastic_process.iterator import RenewalProcessIterator

            model1 = getattr(process, "model1", None)
            if model1 is not None and model1 != process.model:
                if isinstance(model1, LeftTruncatedModel) and model1.baseline == process.model:
                    pass
                else:
                    raise ValueError(
                        f"Calling sample_failure_data on {type(process)} having different model and model1 is ambiguous. Instantiate {type(process)} with only one model"
                    )

            iterator = RenewalProcessIterator(size, tf, process.model, t0, model1=process.model1, maxsample=maxsample, seed=seed)
            struct_array = np.concatenate((arr for arr in iterator))
            args = ()
            if hasattr(process.model, "args"):  #  may be FrozenParametricLifetimeModel
                args = tuple((np.take(arg, struct_array["asset_id"]) for arg in process.model.args))
            if astuple:
                return struct_array["time"].copy(), struct_array["event"].copy(), struct_array["entry"].copy(), args
            return LifetimeData(struct_array["time"].copy(), event=struct_array["event"].copy(), entry=struct_array["entry"].copy(), args=args)


# class CountDataSample:
#     def __init__(self, struct_array, t0, tf):
#         self.t0 = t0
#         self.tf = tf
#         self._struct_array = struct_array
#         self.nb_samples = len(np.unique(self._struct_array["sample_id"]))
#         self.nb_assets = len(np.unique(self._struct_array["asset_id"]))
#
#     def select(self, sample_id: Optional[ArrayLike] = None, asset_id: Optional[ArrayLike] = None) -> Self:
#         mask = np.ones(len(self._struct_array), dtype=np.bool_)
#         if sample_id is not None:
#             mask = mask & np.isin(self._struct_array['sample_id'], sample_id)
#         if asset_id is not None:
#             mask = mask & np.isin(self._struct_array['asset_id'], asset_id)
#         substruct_array = self._struct_array[mask].copy()
#         return CountDataSample(iter((substruct_array,)), self.t0, self.tf) # iterator that returns one struct_array
#
#     @property
#     def timeline(self) -> NDArray[np.float64]:
#         return self._struct_array["timeline"]
#
#     @property
#     def fields(self) -> tuple[str, ...]:
#         return self._struct_array.dtype.names
#
#     def __len__(self) -> int:
#         return len(self._struct_array)
#
#
# def sample_count_data(
#     obj: RenewalProcess|RenewalRewardProcess,
#     size: int,
#     tf: float,
#     t0: float = 0.0,
#     maxsample: int = 1e5,
#     seed: Optional[int] = None,
# ) -> CountDataSample:
#
#     from .iterator import RenewalProcessIterator
#
#     match obj:
#         case RenewalProcess():
#             iterator = RenewalProcessIterator(size, tf, obj.model, t0=t0, model1=obj.model1, maxsample=maxsample, seed=seed)
#             struct_array = np.concatenate(tuple((arr for arr in iterator)))
#             struct_array = np.sort(struct_array, order=("sample_id", "asset_id", "timeline"))
#             return RenewalProcessSample(struct_array, t0, tf)
#
#
#
# def nb_events(count_data : CountDataSample) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
#     sort = np.argsort(self.timeline)
#     timeline = self.timeline[sort]
#     counts = np.ones_like(timeline)
#     timeline = np.insert(timeline, 0, self.t0)
#     counts = np.insert(counts, 0, 0)
#     counts[timeline == self.tf] = 0
#     return timeline, np.cumsum(counts)
#
# def mean_nb_events(count_data : CountDataSample) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
#     timeline, counts = self.nb_events()
#     return timeline, counts / len(self)
#
#
# def total_rewards(count_data : CountDataSample) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
#     sort = np.argsort(self.timeline)
#     timeline = self.timeline[sort]
#     rewards = self.rewards[sort]
#     timeline = np.insert(timeline, 0, self.t0)
#     rewards = np.insert(rewards, 0, 0)
#     rewards[timeline == self.tf] = 0
#     return timeline, rewards.cumsum()
#
#
# def mean_total_rewards(count_data : CountDataSample) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
#     timeline, rewards = self.total_rewards()
#     return timeline, rewards / len(self)

