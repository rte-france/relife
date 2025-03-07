from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from numpy.typing import ArrayLike, NDArray


from relife.types import Arg

if TYPE_CHECKING:  # avoid circular imports due to typing
    from relife.core.model import LifetimeModel, NonParametricModel
    from relife.data import CountData, NHPPData


def plot(
    x: np.ndarray,
    y: np.ndarray,
    se: np.ndarray = None,
    alpha_ci: float = 0.05,
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


def param_probfunc_plot(
    fname: str,
    obj: LifetimeModel[*tuple[Arg, ...]],
    timeline: NDArray[np.float64] = None,
    model_args: tuple[Arg, ...] = (),
    asset: Optional[ArrayLike] = None,
    alpha_ci: float = 0.05,
    **kwargs,
) -> Axes:
    r"""Plot functions of the distribution core.

    Parameters
    ----------
    asset :
    obj :
    timeline : 1D array, optional
        Timeline of the plot (x-axis), by default guessed by the millile.
    model_args : Tuple[ndarray], optional
        Extra arguments required by the parametric lifetime core, by
        default ().
    alpha_ci : float, optional
        :math:`\alpha`-value to define the :math:`100(1-\alpha)\%`
        confidence interval, by default 0.05 corresponding to the 95\%
        confidence interval. If set to None or if the core has not been
        fitted, no confidence interval is plotted.
    fname : str, optional
        Name of the function to be plotted, by default 'sf'. Should be one
        of:

        - 'sf': survival function,
        - 'cdf': cumulative distribution function,
        - 'chf': cumulative hazard function,
        - 'hf': hazard function,
        - 'pdf': probability density function.

    **kwargs : dict, optional
        Extra arguments to specify the plot properties (see
        matplotlib.pyplot.plot documentation).

    Raises
    ------
    ValueError
        If `fname` value is not among 'sf', 'cdf', 'chf', 'hf' or 'pdf'.
    """
    label = kwargs.pop("label", f"{obj.__class__.__name__}" + f".{fname}")
    if timeline is None:
        timeline = np.linspace(0, obj.isf(1e-3), 200)
    f = getattr(obj, fname)
    jac_f = getattr(obj, "jac_" + fname)

    if asset is not None:
        model_args = tuple(
            (
                np.take(v, asset, axis=0)
                for v in np.atleast_2d(*model_args)
                if bool(model_args)
            )
        )
        label += f" (asset {asset})"

    y = f(timeline, *model_args)
    se = None
    if alpha_ci is not None and hasattr(obj, "fitting_results"):
        if obj.fitting_results is not None:
            if obj.fitting_results.se is not None:
                i0 = 0
                se = np.empty_like(timeline)
                if timeline[0] == 0:
                    i0 = 1
                    se[0] = 0
                se[i0:] = obj.fitting_results.standard_error(
                    jac_f(timeline[i0:].reshape(-1, 1), *model_args)
                )

    bounds = (0, 1) if fname in ["sf", "cdf"] else (0, np.inf)

    return plot(
        timeline, y, se=se, alpha_ci=alpha_ci, bounds=bounds, label=label, **kwargs
    )


def nonparam_probfunc_plot(
    fname: str,
    obj: NonParametricModel,
    timeline: NDArray[np.float64] = None,
    alpha_ci: float = 0.05,
    **kwargs,
):
    label = kwargs.pop("label", f"{obj.__class__.__name__}" + f".{fname}")
    if not hasattr(obj, fname):
        raise ValueError(f"No plot for {fname}")

    if timeline is None:
        timeline = obj.estimates.get(fname).timeline
        y = obj.estimates.get(fname).values
        se = obj.estimates.get(fname).se
    else:
        y, se = obj.estimates.get(fname).nearest_1dinterp(timeline)
    return plot(
        timeline,
        y,
        se,
        alpha_ci,
        bounds=(0.0, 1.0),
        label=label,
        drawstyle="steps-post",
        **kwargs,
    )


def nelsonaalen_plot(
    fname: str,
    obj: NonParametricModel,
    timeline: NDArray[np.float64] = None,
    alpha_ci: float = 0.05,
    **kwargs,
):
    label = kwargs.pop("label", f"{obj.__class__.__name__}" + f".{fname}")
    if not hasattr(obj, fname):
        raise ValueError(f"No plot for {fname}")

    if timeline is None:
        timeline = obj.estimates.get(fname).timeline
        y = obj.estimates.get(fname).values
        se = obj.estimates.get(fname).se
    else:
        y, se = obj.estimates.get(fname).nearest_1dinterp(timeline)
    return plot(
        timeline,
        y,
        se,
        alpha_ci,
        bounds=(0.0, np.inf),
        label=label,
        drawstyle="steps-post",
        **kwargs,
    )


def count_data_plot(
    fname: str,
    obj: CountData,
    **kwargs,
):
    label = kwargs.pop("label", fname)
    if not hasattr(obj, fname):
        raise ValueError(f"No plot for {fname}")
    timeline, values = getattr(obj, fname)()

    return plot(timeline, values, drawstyle="steps-post", label=label, **kwargs)


class BoundPlot:
    def __init__(self, obj, plot_func, fname: str):
        self.obj = obj
        self.plot_func = plot_func
        self.fname = fname

    def __call__(self, *args, **kwargs):
        return self.plot_func(self.fname, self.obj, *args, **kwargs)


class PlotDescriptor:
    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, obj, objtype=None):
        from relife.models import (
            ECDF,
            KaplanMeier,
            NelsonAalen,
        )  # avoid circular import
        from relife.models.regression import Regression
        from relife.models.distributions import Distribution
        from relife.data import CountData, NHPPData

        if isinstance(obj.obj, Distribution):
            return BoundPlot(obj.obj, param_probfunc_plot, self.name)
        if isinstance(obj.obj, Regression):
            return BoundPlot(obj.obj, param_probfunc_plot, self.name)
        if isinstance(obj.obj, ECDF | KaplanMeier):
            return BoundPlot(obj.obj, nonparam_probfunc_plot, self.name)
        if isinstance(obj.obj, NelsonAalen):
            return BoundPlot(obj.obj, nelsonaalen_plot, self.name)
        if isinstance(obj.obj, CountData):
            return BoundPlot(obj.obj, count_data_plot, self.name)
        if isinstance(obj.obj, NHPPData):
            return BoundPlot(obj.obj, count_data_plot, self.name)
        raise NotImplementedError("No plot")


class PlotSurvivalFunc:
    sf = PlotDescriptor()
    cdf = PlotDescriptor()
    chf = PlotDescriptor()
    hf = PlotDescriptor()
    pdf = PlotDescriptor()

    def __init__(self, obj: LifetimeModel[*tuple[Arg, ...]] | NonParametricModel):
        self.obj = obj


class PlotCountingData:
    number_of_events = PlotDescriptor()
    mean_number_of_events = PlotDescriptor()

    def __init__(self, obj: CountData):
        self.obj = obj


class PlotNHPPData:
    number_of_events = PlotDescriptor()
    mean_number_of_events = PlotDescriptor()
    number_of_repairs = PlotDescriptor()
    mean_number_of_repairs = PlotDescriptor()

    def __init__(self, obj: NHPPData):
        self.obj = obj
