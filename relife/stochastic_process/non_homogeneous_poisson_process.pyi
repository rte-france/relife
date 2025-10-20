from typing import Any, Optional, Self, Sequence

import numpy as np
from numpy.typing import NDArray

from relife._typing import _Any_Number, _Any_Numpy_Number
from relife.base import ParametricModel
from relife.lifetime_model import FittableParametricLifetimeModel
from relife.likelihood import FittingResults

class NonHomogeneousPoissonProcess(ParametricModel):
    lifetime_model: FittableParametricLifetimeModel[*tuple[_Any_Number, ...]]
    fitting_results: Optional[FittingResults]
    def __init__(self, lifetime_model: FittableParametricLifetimeModel[*tuple[_Any_Number, ...]]) -> None: ...
    def intensity(self, time: _Any_Number, *args: _Any_Number) -> _Any_Numpy_Number: ...
    def cumulative_intensity(self, time: _Any_Number, *args: _Any_Number) -> _Any_Numpy_Number: ...
    def freeze(self, *args: _Any_Number): ...
    def sample(
        self,
        size: int,
        tf: float,
        *args: _Any_Number,
        t0: float = 0.0,
        nb_assets: Optional[int] = None,
        seed: Optional[int] = None,
    ): ...
    def generate_failure_data(
        self,
        size: int,
        tf: float,
        *args: _Any_Number,
        t0: float = 0.0,
        nb_assets: Optional[int] = None,
        seed: Optional[int] = None,
    ): ...
    def fit(
        self,
        ages_at_events: NDArray[np.float64],
        events_assets_ids: Sequence[str] | NDArray[np.int64],
        first_ages: Optional[NDArray[np.float64]] = None,
        last_ages: Optional[NDArray[np.float64]] = None,
        lifetime_model_args: Optional[tuple[_Any_Number]] = None,
        assets_ids: Optional[Sequence[str] | NDArray[np.int64]] = None,
        **options: Any,
    ) -> Self: ...
