from typing import Optional, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from relife.lifetime_model import FrozenParametricLifetimeModel

if TYPE_CHECKING:
    from relife.sample import CountData
    from relife import ParametricModel
    from relife.policy import RenewalPolicy



class SampleMixin:
    def sample(
        self : ParametricModel | RenewalPolicy,
        size: int,
        tf: float,
        t0: float = 0.0,
        maxsample: int = 1e5,
        seed: Optional[int] = None,
    ) -> CountData:
        from ._dispatch_sample import sample_count_data
        return sample_count_data(self, size, tf, t0=t0, maxsample=maxsample, seed=seed)



class FailureDataSampleMixin:
    def failure_data_sample(
        self : ParametricModel | RenewalPolicy,
        size: int,
        tf: float,
        t0: float = 0.0,
        maxsample: int = 1e5,
        seed: Optional[int] = None,
    ) -> tuple[NDArray[np.float64], ...]:
        from ._dispatch_sample import failure_data_sample
        if getattr(self, "model1", None) is not None:
            raise ValueError(f"Calling failure_data_sample on {type(self)} having model and model1 is ambiguous. Instanciate {type(self)} with only one model")
        return failure_data_sample(self, size, tf, t0=t0, maxsample=maxsample, seed=seed)
