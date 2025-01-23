from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
import scipy.stats as stats
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from numpy.typing import ArrayLike, NDArray
from relife.utils.types import ModelArgs

if TYPE_CHECKING:  # avoid circular imports due to typing
    from relife.fiability import LifetimeModel
    from relife.nonparametric import NonParametricLifetimeEstimator


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
    model: LifetimeModel[*ModelArgs],
    timeline: NDArray[np.float64] = None,
    model_args: ModelArgs = (),
    asset: Optional[ArrayLike] = None,
    alpha_ci: float = 0.05,
    **kwargs,
) -> Axes:
    r"""Plot functions of the distribution model.

    Parameters
    ----------
    asset :
    model :
    timeline : 1D array, optional
        Timeline of the plot (x-axis), by default guessed by the millile.
    model_args : Tuple[ndarray], optional
        Extra arguments required by the parametric lifetime model, by
        default ().
    alpha_ci : float, optional
        :math:`\alpha`-value to define the :math:`100(1-\alpha)\%`
        confidence interval, by default 0.05 corresponding to the 95\%
        confidence interval. If set to None or if the model has not been
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
    label = kwargs.pop("label", f"{model.__class__.__name__}" + f".{fname}")
    if timeline is None:
        timeline = np.linspace(0, model.isf(1e-3), 200)
    f = getattr(model, fname)
    jac_f = getattr(model, "jac_" + fname)

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
    if alpha_ci is not None and hasattr(model, "fitting_results"):
        if model.fitting_results is not None:
            if model.fitting_results.se is not None:
                i0 = 0
                se = np.empty_like(timeline)
                if timeline[0] == 0:
                    i0 = 1
                    se[0] = 0
                se[i0:] = model.fitting_results.standard_error(
                    jac_f(timeline[i0:].reshape(-1, 1), *model_args)
                )

    bounds = (0, 1) if fname in ["sf", "cdf"] else (0, np.inf)

    return plot(
        timeline, y, se=se, alpha_ci=alpha_ci, bounds=bounds, label=label, **kwargs
    )


def nonparam_probfunc_plot(
    fname: str,
    model: NonParametricLifetimeEstimator,
    timeline: NDArray[np.float64] = None,
    alpha_ci: float = 0.05,
    **kwargs,
):
    label = kwargs.pop("label", f"{model.__class__.__name__}" + f".{fname}")
    if not hasattr(model, fname):
        raise ValueError(f"No plot for {fname}")

    if timeline is None:
        timeline = model.estimates.get(fname).timeline
        y = model.estimates.get(fname).values
        se = model.estimates.get(fname).se
    else:
        y, se = model.estimates.get(fname).nearest_1dinterp(timeline)
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
    model: NonParametricLifetimeEstimator,
    timeline: NDArray[np.float64] = None,
    alpha_ci: float = 0.05,
    **kwargs,
):
    label = kwargs.pop("label", f"{model.__class__.__name__}" + f".{fname}")
    if not hasattr(model, fname):
        raise ValueError(f"No plot for {fname}")

    if timeline is None:
        timeline = model.estimates.get(fname).timeline
        y = model.estimates.get(fname).values
        se = model.estimates.get(fname).se
    else:
        y, se = model.estimates.get(fname).nearest_1dinterp(timeline)
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
        from relife.distribution import Distribution  # avoid circular import
        from relife.regression import Regression
        from relife.nonparametric import ECDF, KaplanMeier, NelsonAalen

        if isinstance(obj.model, Distribution):
            return BoundPlot(obj.model, param_probfunc_plot, self.name)
        if isinstance(obj.model, Regression):
            return BoundPlot(obj.model, param_probfunc_plot, self.name)
        if isinstance(obj.model, ECDF | KaplanMeier):
            return BoundPlot(obj.model, nonparam_probfunc_plot, self.name)
        if isinstance(obj.model, NelsonAalen):
            return BoundPlot(obj.model, nelsonaalen_plot, self.name)
        raise NotImplementedError("No plot")


class PlotSurvivalFunc:
    sf = PlotDescriptor()
    cdf = PlotDescriptor()
    chf = PlotDescriptor()
    hf = PlotDescriptor()
    pdf = PlotDescriptor()

    def __init__(
        self, model: LifetimeModel[*ModelArgs] | NonParametricLifetimeEstimator
    ):
        self.model = model
