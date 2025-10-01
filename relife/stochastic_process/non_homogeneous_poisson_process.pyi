from typing import Any, Optional, Self, Sequence

import numpy as np
from numpy.typing import NDArray

from relife._typing import _AdditionalArgs, _Array_Float, _IntOrFloat
from relife.lifetime_model import FittableParametricLifetimeModel
from relife.likelihood import FittingResults

from ._base import FrozenStochasticProcess, StochasticProcess

class NonHomogeneousPoissonProcess(StochasticProcess[*_AdditionalArgs]):
    lifetime_model: FittableParametricLifetimeModel[*_AdditionalArgs]
    fitting_results: Optional[FittingResults]
    def __init__(self, lifetime_model: FittableParametricLifetimeModel[*_AdditionalArgs]) -> None: ...
    def intensity(self, time: _IntOrFloat, *args: *_AdditionalArgs) -> _Array_Float: ...
    def cumulative_intensity(self, time: _IntOrFloat, *args: *_AdditionalArgs) -> _Array_Float: ...
    def freeze(self, *args: *_AdditionalArgs): ...
    def sample(
        self,
        size: int,
        tf: float,
        *args: *_AdditionalArgs,
        t0: float = 0.0,
        nb_assets: Optional[int] = None,
        seed: Optional[int] = None,
    ): ...
    def generate_failure_data(
        self,
        size: int,
        tf: float,
        *args: *_AdditionalArgs,
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
        model_args: Optional[tuple[*_AdditionalArgs]] = None,
        assets_ids: Optional[Sequence[str] | NDArray[np.int64]] = None,
        **options: Any,
    ) -> Self: ...

class FrozenNonHomogeneousPoissonProcess(FrozenStochasticProcess[*_AdditionalArgs]):
    unfrozen_model: NonHomogeneousPoissonProcess[*_AdditionalArgs]
    args: tuple[*_AdditionalArgs]

    def __init__(self, model: NonHomogeneousPoissonProcess[*_AdditionalArgs], *args: *_AdditionalArgs) -> None: ...
    def intensity(self, time: _IntOrFloat) -> _Array_Float: ...
    def cumulative_intensity(self, time: _IntOrFloat) -> _Array_Float: ...
    def sample(
        self, size: int, tf: float, t0: float = 0.0, nb_assets: Optional[int] = None, seed: Optional[int] = None
    ): ...
    def generate_failure_data(
        self, size: int, tf: float, t0: float = 0.0, nb_assets: Optional[int] = None, seed: Optional[int] = None
    ): ...
