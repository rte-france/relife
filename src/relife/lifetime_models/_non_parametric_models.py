"""Nonparametric lifetime models."""

from typing import Any, Generic, Literal, NamedTuple, Self, TypeVar, final

import numpy as np
from matplotlib.axes import Axes
from optype.numpy import Array1D

from relife.lifetime_models._base import plot_probability_function

__all__ = [
    "NonParametricLifetimeModel",
    "ECDF",
    "KaplanMeier",
    "NelsonAalen",
]


class NonParametricEstimation(NamedTuple):
    timeline: Array1D[np.float64]
    values: Array1D[np.float64]
    se: Array1D[np.float64]


KT = TypeVar("KT", bound=str)


class NonParametricLifetimeModel(Generic[KT]):
    _ci_bounds: tuple[float, float] = (0.0, 1.0)
    _estimations: dict[KT, NonParametricEstimation | None]

    def __init__(self) -> None:
        self._estimations = {}

    def plot(
        self,
        fname: KT,
        ax: Axes | None = None,
        **kwargs: Any,
    ) -> Axes | None:
        ci = kwargs.pop("ci", True)
        drawstyle = kwargs.get("drawstyle", "steps-post")
        estimations = self._estimations.get(fname, None)
        if estimations is not None:
            time, y, se = estimations
            if ci:
                return plot_probability_function(
                    time,
                    y,
                    se=se,
                    ci_bounds=self._ci_bounds,
                    ax=ax,
                    drawstyle=drawstyle,
                    **kwargs,
                )
            return plot_probability_function(
                time,
                y,
                ax=ax,
                drawstyle=drawstyle,
                **kwargs,
            )


@final
class ECDF(NonParametricLifetimeModel[Literal["sf", "cdf"]]):
    """
    Empirical Cumulative Distribution Function.
    """

    def fit(self, time: Array1D[np.float64]) -> Self:
        """
        Compute the non-parametric estimations with respect to lifetime data.

        Parameters
        ----------
        time : 1darray
            Observed lifetime values.
        """
        timeline, counts = np.unique(time, return_counts=True)
        timeline = np.insert(timeline, 0, 0)
        cdf = np.insert(np.cumsum(counts), 0, 0) / np.sum(counts)
        se = np.sqrt((1 - cdf) / len(time))
        self._estimations["sf"] = NonParametricEstimation(timeline, 1 - cdf, se)
        self._estimations["cdf"] = NonParametricEstimation(timeline, cdf, se)
        return self

    def sf(self) -> NonParametricEstimation | None:
        """
        The estimation of the survival function.

        Returns
        -------
        out : tuple of 3 1darray
            A tuple containing the timeline, the estimated values and the
            estimated standard errors.
        """
        return self._estimations.get("sf", None)

    def cdf(self) -> NonParametricEstimation | None:
        """
        The estimation of the cumulative distribution function.

        Returns
        -------
        out : tuple of 3 1darray
            A tuple containing the timeline, the estimated values and the
            estimated standard errors.
        """
        return self._estimations.get("cdf", None)


@final
class KaplanMeier(NonParametricLifetimeModel[Literal["sf"]]):
    r"""Kaplan-Meier estimator.

    Compute the non-parametric Kaplan-Meier estimator (also known as the product
    limit estimator) of the survival function from lifetime data.

    Notes
    -----
    For a given time instant :math:`t` and :math:`n` total observations, this
    estimator is defined as:

    .. math::

        \hat{S}(t) = \prod_{i: t_i \leq t} \left( 1 - \frac{d_i}{n_i}\right)

    where :math:`d_i` is the number of failures until :math:`t_i` and
    :math:`n_i` is the number of assets at risk just prior to :math:`t_i`.

    The variance estimation is obtained by:

    .. math::

        \widehat{Var}[\hat{S}(t)] = \hat{S}(t)^2 \sum_{i: t_i \leq t}
        \frac{d_i}{n_i(n_i - d_i)}

    which is often referred to as Greenwood's formula.

    References
    ----------
    .. [1] Lawless, J. F. (2011). Statistical models and methods for lifetime
        data. John Wiley & Sons.

    .. [2] Kaplan, E. L., & Meier, P. (1958). Nonparametric estimation from
        incomplete observations. Journal of the American statistical
        association, 53(282), 457-481.

    """

    def fit(
        self,
        time: Array1D[np.float64],
        event: Array1D[np.bool_] | None = None,
        entry: Array1D[np.float64] | None = None,
    ) -> Self:
        """
        Compute the non-parametric estimations with respect to lifetime data.

        Parameters
        ----------
        time : ndarray
            Observed lifetime values.
        event : ndarray of boolean values (1d), default is None
            Boolean indicators tagging lifetime values as right censored or complete.
        entry : ndarray of float (1d), default is None
            Left truncations applied to lifetime values.
        """

        if event is None:
            event = np.ones_like(time).astype(bool)

        if entry is None:
            entry = np.zeros_like(time)

        timeline = np.unique(time)

        n = (
            (timeline <= time.reshape(-1, 1)) * (timeline >= entry.reshape(-1, 1))
        ).sum(axis=0)

        d = ((time.reshape(-1, 1) == timeline) * event.reshape(-1, 1)).sum(axis=0)
        sf = (1 - d / n).cumprod()

        with np.errstate(divide="ignore"):
            var = (sf**2) * (d / (n * (n - d))).cumsum()
            var = np.where(n > d, var, 0)

        self._estimations["sf"] = NonParametricEstimation(
            np.insert(timeline, 0, 0),
            np.insert(sf, 0, 1),
            np.insert(np.sqrt(var), 0, 0),
        )
        return self

    def sf(self) -> NonParametricEstimation | None:
        """
        The estimation of the survival function.

        Returns
        -------
        out : tuple of 3 1darray
            A tuple containing the timeline, the estimated values and the
            estimated standard errors.
        """
        return self._estimations.get("sf", None)


@final
class NelsonAalen(NonParametricLifetimeModel[Literal["chf"]]):
    r"""Nelson-Aalen estimator.

    Compute the non-parametric Nelson-Aalen estimator of the cumulative hazard
    function from lifetime data.

    Notes
    -----
    For a given time instant :math:`t` and :math:`n` total observations, this
    estimator is defined as:

    .. math::

        \hat{H}(t) = \sum_{i: t_i \leq t} \frac{d_i}{n_i}

    where :math:`d_i` is the number of failures until :math:`t_i` and
    :math:`n_i` is the number of assets at risk just prior to :math:`t_i`.

    The variance estimation is obtained by:

    .. math::

        \widehat{Var}[\hat{H}(t)] = \sum_{i: t_i \leq t} \frac{d_i}{n_i^2}

    Note that the alternative survivor function estimate:

    .. math::

        \tilde{S}(t) = \exp{(-\hat{H}(t))}

    is sometimes suggested for the continuous-time case.

    References
    ----------
    .. [1] Lawless, J. F. (2011). Statistical models and methods for lifetime
        data. John Wiley & Sons.
    """

    _ci_bounds: tuple[float, float] = (0.0, np.inf)

    def fit(
        self,
        time: Array1D[np.float64],
        event: Array1D[np.bool_] | None = None,
        entry: Array1D[np.float64] | None = None,
    ) -> Self:
        """
        Compute the non-parametric estimations with respect to lifetime data.

        Parameters
        ----------
        time : ndarray
            Observed lifetime values.
        event : ndarray of boolean values (1d), default is None
            Boolean indicators tagging lifetime values as right censored or complete.
        entry : ndarray of float (1d), default is None
            Left truncations applied to lifetime values.
        """

        if event is None:
            event = np.ones_like(time).astype(bool)

        if entry is None:
            entry = np.zeros_like(time)

        timeline = np.unique(time)

        n = (
            (timeline <= time.reshape(-1, 1)) * (timeline >= entry.reshape(-1, 1))
        ).sum(axis=0)

        d = ((time.reshape(-1, 1) == timeline) * event.reshape(-1, 1)).sum(axis=0)

        chf = (d / n).cumsum()

        with np.errstate(divide="ignore"):
            var = (d / n**2).cumsum()

        self._estimations["chf"] = NonParametricEstimation(
            np.insert(timeline, 0, 0),
            np.insert(chf, 0, 0),
            np.insert(np.sqrt(var), 0, 0),
        )
        return self

    def chf(self) -> NonParametricEstimation | None:
        """
        The estimation of the cumulative hazard function.

        Returns
        -------
        out : tuple of 3 1darray
            A tuple containing the timeline, the estimated values and the
            estimated standard errors.
        """
        return self._estimations.get("chf", None)
