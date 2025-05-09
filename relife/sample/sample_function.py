from typing import TypeVarTuple, Optional, overload, Self

import numpy as np
from numpy.typing import ArrayLike
from numpy.typing import NDArray

from relife.lifetime_model import ParametricLifetimeModel
from relife.sample.counting_data import RenewalProcessSample
from relife.stochastic_process import RenewalProcess, RenewalRewardProcess

Args = TypeVarTuple("Args")


def sample_lifetime_data(
    model : ParametricLifetimeModel[*Args],
    *args : *Args,
    size: int | tuple[int] | tuple[int, int] = 1,
    seed:Optional[int]=None
) -> tuple[np.float64, np.float64, np.float64] | tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]] :
    """
    Note that in case of LefTruncatedModel, returned time is not residual. Use LeftTruncated.rvs instead
    Parameters
    ----------
    model
    args
    size
    seed

    Returns
    -------

    """
    from relife.lifetime_model import LeftTruncatedModel, AgeReplacementModel

    rs = np.random.RandomState(seed=seed)
    probability = rs.uniform(size=size)
    time = model.isf(probability, *args)
    event = np.ones_like(time, dtype=np.bool_)
    entry = np.zeros_like(time)

    match model:
        case LeftTruncatedModel():
            from relife.lifetime_model.conditional_model import reshape_ar_or_a0
            a0 = reshape_ar_or_a0("a0", args[0])
            time = time + a0 # change time to real time, not residual
            entry = a0
        case AgeReplacementModel():
            from relife.lifetime_model.conditional_model import reshape_ar_or_a0
            ar = reshape_ar_or_a0("ar", args[0])
            ar = np.broadcast_to(ar, time.shape).copy()
            time = np.minimum(model.baseline.rvs(*args, size=size, seed=seed), ar)
            event = event != ar

    return time, event, entry



class CountDataSample:
    def __init__(self, struct_array, t0, tf):
        self.t0 = t0
        self.tf = tf
        self._struct_array = struct_array
        self.nb_samples = len(np.unique(self._struct_array["sample_id"]))
        self.nb_assets = len(np.unique(self._struct_array["asset_id"]))

    def select(self, sample_id: Optional[ArrayLike] = None, asset_id: Optional[ArrayLike] = None) -> Self:
        mask = np.ones(len(self._struct_array), dtype=np.bool_)
        if sample_id is not None:
            mask = mask & np.isin(self._struct_array['sample_id'], sample_id)
        if asset_id is not None:
            mask = mask & np.isin(self._struct_array['asset_id'], asset_id)
        substruct_array = self._struct_array[mask].copy()
        return CountDataSample(iter((substruct_array,)), self.t0, self.tf) # iterator that returns one struct_array

    @property
    def timeline(self) -> NDArray[np.float64]:
        return self._struct_array["timeline"]

    @property
    def fields(self) -> tuple[str, ...]:
        return self._struct_array.dtype.names

    def __len__(self) -> int:
        return len(self._struct_array)


def sample_count_data(
    obj: RenewalProcess|RenewalRewardProcess,
    size: int,
    tf: float,
    t0: float = 0.0,
    maxsample: int = 1e5,
    seed: Optional[int] = None,
) -> CountDataSample:

    from .iterator import RenewalProcessIterator

    match obj:
        case RenewalProcess():
            iterator = RenewalProcessIterator(size, tf, obj.model, t0=t0, model1=obj.model1, maxsample=maxsample, seed=seed)
            struct_array = np.concatenate(tuple((arr for arr in iterator)))
            struct_array = np.sort(struct_array, order=("sample_id", "asset_id", "timeline"))
            return RenewalProcessSample(struct_array, t0, tf)


# @overload
# def sample_count_data(
#     obj: RenewalRewardProcess,
#     size: int,
#     tf: float,
#     t0: float = 0.0,
#     maxsample: int = 1e5,
#     seed: Optional[int] = None,
# ) -> RenewalRewardProcessSample: ...

# actual implementation

# TODO : merge this sample_failure_data with sample_lifetime_data above to 1 sample_failure_data
# apply tf and t0 truncation and censoring on time, event, entry returned.
# if NHPP ?




def sample_failure_data(
    obj: RenewalProcess | RenewalRewardProcess,
    size: int,
    tf: float,
    t0: float = 0.0,
    maxsample: int = 1e5,
    seed: Optional[int] = None,
) -> :
    from relife.lifetime_model import LeftTruncatedModel
    from relife.sample.iterator import RenewalProcessIterator

    model1 = getattr(obj, "model1", None)
    if model1 is not None and model1 != obj.model:
        if isinstance(model1, LeftTruncatedModel) and model1.baseline == obj.model:
            pass
        else:
            raise ValueError(
                f"Calling sample_failure_data on {type(obj)} having different model and model1 is ambiguous. Instantiate {type(obj)} with only one model"
            )

    iterator = RenewalProcessIterator(size, tf, obj.model, t0, model1=obj.model1, maxsample=maxsample, seed=seed)
    struct_array = np.concatenate((arr for arr in iterator))
    args = ()
    if hasattr(obj.model, "args"): # may be FrozenParametricLifetimeModel
        args = tuple((np.take(arg, struct_array["asset_id"]) for arg in obj.model.args))

    return struct_array["time"], struct_array["event"], struct_array["entry"], args