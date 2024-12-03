from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from numpy.typing import ArrayLike
from numpy.typing import NDArray
from scipy.stats import stats

from relife2.utils.types import ModelArgs

if TYPE_CHECKING:  # avoid circular imports due to typing
    from relife2.fiability import LifetimeModel


def args_take(indices: ArrayLike, *args) -> tuple[np.ndarray, ...]:
    """Take elements in each array of args on axis=-2.

    Parameters
    ----------
    indices : ndarray
        The indices of the values to extract for each array.
    *args : float or 2D array
        Sequence of arrays.

    Returns
    -------
    Tuple[ndarray, ...]
        The tuple of arrays where values at indices are extracted.
    """
    return tuple(
        (np.take(arg, indices, axis=0) if np.ndim(arg) == 2 else arg) for arg in args
    )


def prob_func_plot(
    fname: str,
    model: LifetimeModel[*ModelArgs],
    timeline: NDArray[np.float64] = None,
    args: ModelArgs = (),
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
    args : Tuple[ndarray], optional
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
    flist = ["sf", "cdf", "chf", "hf", "pdf"]
    if fname not in flist:
        raise ValueError(
            "Function name '{}' is not supported for plotting, `fname` must be in {}".format(
                fname, flist
            )
        )

    label = kwargs.pop("label", f"{model.__class__.__name__}" + f".{fname}")
    if timeline is None:
        timeline = np.linspace(0, model.isf(np.array(1e-3)), 200)
    f = getattr(model, fname)
    jac_f = getattr(model, "jac_" + fname)

    if asset is not None:
        print(timeline)
        timeline, *args = args_take(asset, timeline, *args)
        label += f" (asset {asset})"
        print(timeline, args)

    y = f(timeline, *args)
    if alpha_ci is not None and hasattr(model, "result"):
        i0 = 0
        se = np.empty_like(timeline, float)
        if timeline[0] == 0:
            i0 = 1
            se[0] = 0
        se[i0:] = model.result.standard_error(
            jac_f(model.result.opt.x, timeline[i0:].reshape(-1, 1), *args)
        )
    else:
        se = None

    bounds = (0, 1) if fname in ["sf", "cdf"] else (0, np.inf)

    ax = kwargs.pop("ax", plt.gca())
    drawstyle = kwargs.pop("drawstyle", "default")
    (lines,) = ax.plot(timeline, y, drawstyle=drawstyle, label=label, **kwargs)
    if alpha_ci is not None and se is not None:
        z = stats.norm.ppf(1 - alpha_ci / 2)
        yl = np.clip(y - z * se, bounds[0], bounds[1])
        yu = np.clip(y + z * se, bounds[0], bounds[1])
        step = drawstyle.split("-")[1] if "steps-" in drawstyle else None
        ax.fill_between(
            timeline, yl, yu, facecolor=lines.get_color(), step=step, alpha=0.25
        )
    ax.legend()
    return ax


class BoundPlot:
    def __init__(self, obj, plot_func, fname: str):
        self.obj = obj
        self.plot_func = plot_func
        self.fname = fname

    def __call__(self, *args, **kwargs):
        return self.plot_func(self.fname, self.obj, *args, **kwargs)


class PlotDescriptor:
    def __set_name__(self, owner, name):
        self.fname = name

    def __get__(self, obj, objtype=None):
        from relife2.distribution import Distribution  # avoid circular import
        from relife2.regression import Regression

        if isinstance(obj.model, Distribution):
            return BoundPlot(obj.model, prob_func_plot, self.fname)
        if isinstance(obj.model, Regression):
            pass
        raise NotImplementedError("No plot")


class PlotAccessor:
    sf = PlotDescriptor()
    cdf = PlotDescriptor()
    chf = PlotDescriptor()
    hf = PlotDescriptor()
    pdf = PlotDescriptor()

    def __init__(self, model: LifetimeModel[*ModelArgs]):
        self.model = model
