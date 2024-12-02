from __future__ import annotations

import inspect
from typing import Optional, TYPE_CHECKING

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from numpy.typing import NDArray
from scipy.stats import stats

from relife2.utils.types import ModelArgs

if TYPE_CHECKING:  # avoid circular imports due to typing
    from relife2.regression import Regression
    from relife2.distribution import Distribution


def plot(
    x: np.ndarray,
    y: np.ndarray,
    se: np.ndarray = None,
    alpha_ci: Optional[float] = 0.05,
    bounds=(-np.inf, np.inf),
    **kwargs,
) -> Axes:
    r"""Plot a function with a confidence interval.

    Parameters
    ----------
    x : 1D array
        x-axis values.
    y : np.ndarray
        y-axis values.
    se : np.ndarray, optional
        The standard error, by default None.
    alpha_ci : float, optional
        :math:`\alpha`-value to define the :math:`100(1-\alpha)\%` confidence
        interval, by default 0.05 corresponding to the 95\% confidence interval.
        If set to None or if se is None, no confidence interval is plotted, by
        default 0.05.
    bounds : tuple, optional
        Bounds for clipping the value of the confidence interval, by default
        (-np.inf, np.inf).
    **kwargs : dict, optional
        Extra arguments to specify the plot properties (see
        matplotlib.pyplot.plot documentation).

    """
    ax = kwargs.pop("ax", plt.gca())
    drawstyle = kwargs.pop("drawstyle", "default")
    (lines,) = ax.plot(x, y, drawstyle=drawstyle, **kwargs)
    if alpha_ci is not None and se is not None:
        z = stats.norm.ppf(1 - alpha_ci / 2)
        yl = np.clip(y - z * se, bounds[0], bounds[1])
        yu = np.clip(y + z * se, bounds[0], bounds[1])
        step = drawstyle.split("-")[1] if "steps-" in drawstyle else None
        ax.fill_between(x, yl, yu, facecolor=lines.get_color(), step=step, alpha=0.25)
    ax.legend()
    return ax


def get_plots(obj):
    if isinstance(obj, Distribution):
        return DistributionPlots(obj)
    else:
        raise NotImplementedError


class PlotAccessor: ...


class DistributionPlots(PlotAccessor):
    """Make plots of Distribution"""

    def __init__(self, model: Distribution):
        self.model = model

    @staticmethod
    def bounds(fname: str):
        return (0, 1) if fname in ("sf", "cdf") else (0, np.inf)

    def sf(
        self,
        timeline: Optional[NDArray[np.float64]] = None,
        alpha_ci: float = 0.05,
        **kwargs,
    ):
        if timeline is None:
            timeline = np.linspace(0, self.model.isf(1e-3), 200)

        return plot(
            timeline,
            self.model.sf(timeline),
            alpha_ci=None,
            bounds=DistributionPlots.bounds(inspect.stack()[0][3]),
            label="sf",
            **kwargs,
        )

    def cdf(
        self,
        timeline: Optional[NDArray[np.float64]] = None,
        alpha_ci: float = 0.05,
        **kwargs,
    ):
        if timeline is None:
            timeline = np.linspace(0, self.model.isf(1e-3), 200)

        return plot(
            timeline,
            self.model.sf(timeline),
            alpha_ci=None,
            bounds=DistributionPlots.bounds(inspect.stack()[0][3]),
            label="cdf",
            **kwargs,
        )


class RegressionPlots(PlotAccessor):
    def __init__(self, model: Regression):
        self.model = model

    def sf(
        self,
        covar: NDArray[np.float64],
        args: ModelArgs = (),
        timeline: Optional[NDArray[np.float64]] = None,
        **kwargs,
    ):
        if timeline is None:
            timeline = np.linspace(0, self.model.isf(1e-3, covar, *args), 200)

        return plot(
            timeline,
            self.model.sf(timeline, covar, *args),
            alpha_ci=None,
            bounds=DistributionPlots.bounds(inspect.stack()[0][3]),
            label="cdf",
            **kwargs,
        )
