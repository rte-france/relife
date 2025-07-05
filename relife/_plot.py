from __future__ import annotations

from typing import TYPE_CHECKING, Generic, Optional, TypeVarTuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from matplotlib.axes import Axes
from numpy.ma.core import zeros_like
from numpy.typing import NDArray

if TYPE_CHECKING:
    from relife.lifetime_model import (
        NonParametricLifetimeModel,
        ParametricLifetimeModel,
    )

ALPHA_CI: float = 0.05


def plot_prob_function(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    se: Optional[NDArray[np.float64]] = None,
    ci_bounds: Optional[tuple[float, float]] = None,
    label: Optional[str] = None,
    drawstyle: str = "default",
    **kwargs,
) -> Axes:
    ax = kwargs.pop("ax", plt.gca())
    ax.plot(x, y, drawstyle=drawstyle, label=label, **kwargs)
    if se is not None and ci_bounds is not None:
        z = stats.norm.ppf(1 - ALPHA_CI / 2)
        yl = np.clip(y - z * se, ci_bounds[0], ci_bounds[1])
        yu = np.clip(y + z * se, ci_bounds[0], ci_bounds[1])
        step = drawstyle.split("-")[1] if "steps-" in drawstyle else None
        ax.fill_between(x, yl, yu, facecolors=[ax.lines[-1].get_color()], step=step, alpha=0.25, label="IC-95%")
    if label is not None:
        ax.legend()
    return ax


Args = TypeVarTuple("Args")


class PlotParametricLifetimeModel(Generic[*Args]):
    def __init__(self, model: ParametricLifetimeModel[*Args]):
        self.model = model

    def _plot(self, fname: str, *args: *Args, ci_bounds: Optional[tuple[float, float]] = None, **kwargs) -> Axes:
        end_time = kwargs.pop("end_time", None)
        if end_time is None:
            end_time = np.squeeze(self.model.isf(1e-3, *args))  # () or (m,)
        timeline = np.linspace(0, end_time, 200, dtype=np.float64)  # (200,) or (200, m)
        timeline = np.transpose(timeline)  # (200,) (m, 200)
        f = getattr(self.model, fname)
        jac_f = getattr(self.model, "jac_" + fname, None)
        y = f(timeline, *args)
        se = None
        if self.model.fitting_results is not None and jac_f is not None:
            se = zeros_like(timeline)
            se[..., 1:] = self.model.fitting_results.se_estimation_function(
                jac_f(timeline[..., 1:], *args, asarray=True)
            )
        label = kwargs.pop("label", f"{self.model.__class__.__name__}" + f".{fname}")
        ax = plot_prob_function(timeline, y, se=se, ci_bounds=ci_bounds, label=label, **kwargs)
        return ax

    def sf(self, *args: *Args, **kwargs) -> Axes:
        end_time = kwargs.get("end_time", None)
        ax = self._plot("sf", *args, ci_bounds=(0.0, 1.0), **kwargs)
        ax.set_ylabel("$S(t) = P(T > t)$")
        ax.set_xlabel("t")
        ax.set_title("Survival function")
        ax.set_ylim(bottom=0)
        ax.set_xlim(left=0, right=end_time)
        return ax

    def cdf(self, *args: *Args, **kwargs) -> Axes:
        end_time = kwargs.get("end_time", None)
        ax = self._plot("cdf", *args, ci_bounds=(0.0, 1.0), **kwargs)
        ax.set_ylabel("$F(t) = P(T \leq t)$")
        ax.set_xlabel("t")
        ax.set_title("Cumulative distribution function")
        ax.set_ylim(bottom=0)
        ax.set_xlim(left=0, right=end_time)
        return ax

    def chf(self, *args: *Args, **kwargs) -> Axes:
        end_time = kwargs.get("end_time", None)
        ax = self._plot("chf", *args, ci_bounds=(0.0, np.inf), **kwargs)
        ax.set_ylabel("$H(t)$")
        ax.set_xlabel("t")
        ax.set_title("Cumulative hazard function")
        ax.set_ylim(bottom=0)
        ax.set_xlim(left=0, right=end_time)
        return ax

    def hf(self, *args: *Args, **kwargs) -> Axes:
        end_time = kwargs.get("end_time", None)
        ax = self._plot("hf", *args, ci_bounds=(0.0, np.inf), **kwargs)
        ax.set_ylabel("$h(t)$")
        ax.set_xlabel("t")
        ax.set_title("Hazard function")
        ax.set_ylim(bottom=0)
        ax.set_xlim(left=0, right=end_time)
        return ax

    def pdf(self, *args: *Args, **kwargs) -> Axes:
        end_time = kwargs.get("end_time", None)
        ax = self._plot("pdf", *args, ci_bounds=(0.0, np.inf), **kwargs)
        ax.set_ylabel("$f(t)$")
        ax.set_xlabel("t")
        ax.set_title("Probability density function")
        ax.set_ylim(bottom=0)
        ax.set_xlim(left=0, right=end_time)
        return ax


class PlotNonParametricLifetimeModel:
    def __init__(self, model: NonParametricLifetimeModel):
        self.model = model

    def plot(self, fname: str, plot_se: bool = True, ci_bounds=(0.0, 1.0), drawstyle="steps-post", **kwargs) -> Axes:
        label = kwargs.pop("label", f"{self.model.__class__.__name__}" + f".{fname}")
        res = getattr(self.model, fname)(se=plot_se)
        se = None if not plot_se else res[-1]
        timeline, y = res[:2]
        ax = plot_prob_function(timeline, y, se=se, ci_bounds=ci_bounds, label=label, drawstyle=drawstyle, **kwargs)
        return ax


class PlotECDF(PlotNonParametricLifetimeModel):
    def sf(self, plot_se: bool = True, **kwargs) -> Axes:
        ax = self.plot("sf", plot_se=plot_se, **kwargs)
        ax.set_ylabel("Estimated survival function")
        ax.set_xlabel("Time")
        ax.set_ylim(bottom=0)
        ax.set_xlim(left=0)
        return ax

    def cdf(self, plot_se: bool = True, **kwargs) -> Axes:
        ax = self.plot("cdf", plot_se=plot_se, **kwargs)
        ax.set_ylabel("Estimated cumulative distribution function")
        ax.set_xlabel("Time")
        ax.set_ylim(bottom=0)
        ax.set_xlim(left=0)
        return ax


class PlotKaplanMeier(PlotNonParametricLifetimeModel):
    def sf(self, plot_se: bool = True, **kwargs) -> Axes:
        ax = self.plot("sf", plot_se=plot_se, **kwargs)
        ax.set_ylabel("Estimated survival function")
        ax.set_xlabel("Time")
        ax.set_ylim(bottom=0)
        ax.set_xlim(left=0)
        return ax


class PlotNelsonAalen(PlotNonParametricLifetimeModel):
    def chf(self, plot_se: bool = True, **kwargs) -> Axes:
        ax = self.plot("chf", plot_se=plot_se, ci_bounds=(0.0, np.inf), **kwargs)
        ax.set_ylabel("Estimated cumulative hazard function")
        ax.set_xlabel("Time")
        ax.set_ylim(bottom=0)
        ax.set_xlim(left=0)
        return ax


class PlotTurnbull(PlotNonParametricLifetimeModel):
    def sf(self, **kwargs) -> Axes:
        ax = self.plot("sf", plot_se=False, **kwargs)
        ax.set_ylabel("Estimated survival function")
        ax.set_xlabel("Time")
        ax.set_xlim(left=0)
        return ax


# def count_data_plot(
#     fname: str,
#     obj: CountData,
#     **kwargs,
# ):
#     label = kwargs.pop("label", fname)
#     if not hasattr(obj, fname):
#         raise ValueError(f"No plot for {fname}")
#     timeline, values = getattr(obj, fname)()
#     return plot(timeline, values, drawstyle="steps-post", label=label, **kwargs)
#
#
# def nhpp_count_data_plot(
#     fname: str,
#     obj: NHPPCountData,
#     **kwargs,
# ):
#     label = kwargs.pop("label", fname)
#     if not hasattr(obj, fname):
#         raise ValueError(f"No plot for {fname}")
#     timeline, values = getattr(obj, fname)()
#     if fname in ("total_rewards", "mean_total_rewards"):
#         ax = kwargs.pop("ax", plt.gca())
#         alpha = kwargs.pop("alpha", 0.2)
#         ax.plot(timeline, values, drawstyle="steps-post", label=label, **kwargs)
#         ax.fill_between(timeline, values, where=values >= 0, step="post", alpha=alpha, **kwargs)
#         if label is not None:
#             ax.legend()
#         return ax
#     return plot(timeline, values, drawstyle="steps-post", label=label, **kwargs)
#
#
# def renewal_data_plot(
#     fname: str,
#     obj: CountData,
#     **kwargs,
# ):
#     label = kwargs.pop("label", fname)
#     if not hasattr(obj, fname):
#         raise ValueError(f"No plot for {fname}")
#     timeline, values = getattr(obj, fname)()
#     if fname in ("total_rewards", "mean_total_rewards"):
#         ax = kwargs.pop("ax", plt.gca())
#         alpha = kwargs.pop("alpha", 0.2)
#         ax.plot(timeline, values, drawstyle="steps-post", label=label, **kwargs)
#         ax.fill_between(timeline, values, where=values >= 0, step="post", alpha=alpha, **kwargs)
#         if label is not None:
#             ax.legend()
#         return ax
#     else:
#         return count_data_plot(fname, obj, label=label, **kwargs)


# def nhpp_plot(
#     fname: str,
#     obj: NonHomogeneousPoissonProcess,
#     timeline: NDArray[np.float64] = None,
#     **kwargs,
# ):
#
#     label = kwargs.pop("label", f"{obj.__class__.__name__}" + f".{fname}")
#     if not hasattr(obj, fname):
#         raise ValueError(f"No plot for {fname}")
#     f = getattr(obj, fname)
#     y = f(timeline)
#     return plot(timeline, y, label=label, **kwargs)
