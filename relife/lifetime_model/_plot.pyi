from typing import Generic, TypeVarTuple

import numpy as np
from _typeshed import Incomplete
from matplotlib.axes import Axes
from numpy.typing import NDArray as NDArray

from relife.lifetime_model import (
    NonParametricLifetimeModel as NonParametricLifetimeModel,
)
from relife.lifetime_model import ParametricLifetimeModel as ParametricLifetimeModel

ALPHA_CI: float

def plot_prob_function(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    se: NDArray[np.float64] | None = None,
    ci_bounds: tuple[float, float] | None = None,
    label: str | None = None,
    drawstyle: str = "default",
    **kwargs,
) -> Axes: ...

Args = TypeVarTuple("Args")

class PlotParametricLifetimeModel(Generic[*Args]):
    model: Incomplete
    def __init__(self, model: ParametricLifetimeModel[*Args]) -> None: ...
    def sf(self, *args: *Args, **kwargs) -> Axes: ...
    def cdf(self, *args: *Args, **kwargs) -> Axes: ...
    def chf(self, *args: *Args, **kwargs) -> Axes: ...
    def hf(self, *args: *Args, **kwargs) -> Axes: ...
    def pdf(self, *args: *Args, **kwargs) -> Axes: ...

class PlotNonParametricLifetimeModel:
    model: Incomplete
    def __init__(self, model: NonParametricLifetimeModel) -> None: ...
    def plot(
        self, fname: str, plot_se: bool = True, ci_bounds=(0.0, 1.0), drawstyle: str = "steps-post", **kwargs
    ) -> Axes: ...

class PlotECDF(PlotNonParametricLifetimeModel):
    def sf(self, plot_se: bool = True, **kwargs) -> Axes: ...
    def cdf(self, plot_se: bool = True, **kwargs) -> Axes: ...

class PlotKaplanMeier(PlotNonParametricLifetimeModel):
    def sf(self, plot_se: bool = True, **kwargs) -> Axes: ...

class PlotNelsonAalen(PlotNonParametricLifetimeModel):
    def chf(self, plot_se: bool = True, **kwargs) -> Axes: ...

class PlotTurnbull(PlotNonParametricLifetimeModel):
    def sf(self, **kwargs) -> Axes: ...
