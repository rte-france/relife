from typing import Any, Optional, Self, Sequence

import numpy as np
from numpy.typing import NDArray

from relife._typing import _X, _Y, _Xs
from relife.lifetime_model import FittableParametricLifetimeModel
from relife.likelihood import FittingResults

from .base import FrozenStochasticProcess, StochasticProcess

class NonHomogeneousPoissonProcess(StochasticProcess[*_Xs]):
    lifetime_model: FittableParametricLifetimeModel[*_Xs]
    fitting_results: Optional[FittingResults]
    def __init__(self, lifetime_model: FittableParametricLifetimeModel[*_Xs]) -> None: ...
    def intensity(self, time: _X, *args: *_Xs) -> _Y: ...
    def cumulative_intensity(self, time: _X, *args: *_Xs) -> _Y: ...
    def freeze(self, *args: *_Xs): ...
    def sample(
        self,
        size: int,
        tf: float,
        *args: *_Xs,
        t0: float = 0.0,
        nb_assets: Optional[int] = None,
        seed: Optional[int] = None,
    ): ...
    def generate_failure_data(
        self,
        size: int,
        tf: float,
        *args: *_Xs,
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
        model_args: Optional[tuple[*_Xs]] = None,
        assets_ids: Optional[Sequence[str] | NDArray[np.int64]] = None,
        **options: Any,
    ) -> Self: ...

class FrozenNonHomogeneousPoissonProcess(FrozenStochasticProcess[*_Xs]):
    unfrozen_model: NonHomogeneousPoissonProcess[*_Xs]
    args: tuple[*_Xs]

    def __init__(self, model: NonHomogeneousPoissonProcess[*_Xs], *args: *_Xs) -> None: ...
    def intensity(self, time: _X) -> _Y: ...
    def cumulative_intensity(self, time: _X) -> _Y: ...
    def sample(
        self, size: int, tf: float, t0: float = 0.0, nb_assets: Optional[int] = None, seed: Optional[int] = None
    ): ...
    def generate_failure_data(
        self, size: int, tf: float, t0: float = 0.0, nb_assets: Optional[int] = None, seed: Optional[int] = None
    ): ...
