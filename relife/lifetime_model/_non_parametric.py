"""Nonparametric lifetime models."""

from typing import Literal, Self, final, overload

import numpy as np
from numpy.typing import NDArray

from ._plot import (
    PlotECDF,
    PlotKaplanMeier,
    PlotNelsonAalen,
)

__all__: list[str] = ["ECDF", "KaplanMeier", "NelsonAalen"]


@final
class ECDF:
    """
    Empirical Cumulative Distribution Function.
    """

    _sf: NDArray[np.void] | None
    _cdf: NDArray[np.void] | None

    def __init__(self) -> None:
        self._sf = None
        self._cdf = None

    def fit(self, time: NDArray[np.float64]) -> Self:
        """
        Compute the non-parametric estimations with respect to lifetime data.

        Parameters
        ----------
        time : ndarray
            Observed lifetime values.
        """
        timeline, counts = np.unique(time, return_counts=True)
        timeline = np.insert(timeline, 0, 0)

        dtype = np.dtype([("timeline", np.float64), ("estimation", np.float64), ("se", np.float64)])
        self._sf = np.empty((timeline.size,), dtype=dtype)
        self._cdf = np.empty((timeline.size,), dtype=dtype)

        self._sf["timeline"] = timeline
        self._cdf["timeline"] = timeline
        cdf = np.insert(np.cumsum(counts), 0, 0) / np.sum(counts)
        self._cdf["estimation"] = cdf
        self._sf["estimation"] = 1 - cdf
        se = np.sqrt((1 - cdf) / len(time))
        self._sf["se"] = se
        self._cdf["se"] = se
        return self

    @overload
    def sf(self, se: Literal[False]) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None: ...
    @overload
    def sf(self, se: Literal[True]) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]] | None: ...
    @overload
    def sf(
        self, se: bool
    ) -> (
        tuple[NDArray[np.float64], NDArray[np.float64]]
        | tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
        | None
    ): ...

    def sf(
        self, se: bool = True
    ) -> (
        tuple[NDArray[np.float64], NDArray[np.float64]]
        | tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
        | None
    ):
        """
        The survival functions estimated values

        Parameters
        ----------
        se : bool, default is False
            If true, the estimated standard errors are returned too.

        Returns
        -------
        tuple of 2 or 3 ndarrays
            A tuple containing the timeline, the estimated values and optionally the estimated standard errors (if se is set to true)
        """
        if self._sf is None:
            return None
        if se:
            return self._sf["timeline"], self._sf["estimation"], self._sf["se"]
        return self._sf["timeline"], self._sf["estimation"]

    @overload
    def cdf(self, se: Literal[False]) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None: ...
    @overload
    def cdf(self, se: Literal[True]) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]] | None: ...
    @overload
    def cdf(
        self, se: bool
    ) -> (
        tuple[NDArray[np.float64], NDArray[np.float64]]
        | tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
        | None
    ): ...

    def cdf(
        self, se: bool = True
    ) -> (
        tuple[NDArray[np.float64], NDArray[np.float64]]
        | tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
        | None
    ):
        """
        The cumulative distribution function estimated values

        Parameters
        ----------
        se : bool, default is False
            If true, the estimated standard errors are returned too.

        Returns
        -------
        tuple of 2 or 3 ndarrays
            A tuple containing the timeline, the estimated values and optionally the estimated standard errors (if se is set to true)
        """
        if self._cdf is None:
            return None
        if se:
            return self._cdf["timeline"], self._cdf["estimation"], self._cdf["se"]
        return self._cdf["timeline"], self._cdf["estimation"]

    @property
    def plot(self):
        return PlotECDF(self)


@final
class KaplanMeier:
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

    _sf: NDArray[np.void] | None

    def __init__(self) -> None:
        self._sf = None

    def fit(
        self,
        time: NDArray[np.float64],
        event: NDArray[np.bool_] | None = None,
        entry: NDArray[np.float64] | None = None,
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

        n = ((timeline <= time.reshape(-1, 1)) * (timeline >= entry.reshape(-1, 1))).sum(axis=0)

        d = ((time.reshape(-1, 1) == timeline) * event.reshape(-1, 1)).sum(axis=0)
        sf = (1 - d / n).cumprod()

        with np.errstate(divide="ignore"):
            var = (sf**2) * (d / (n * (n - d))).cumsum()
            var = np.where(n > d, var, 0)

        dtype = np.dtype([("timeline", np.float64), ("estimation", np.float64), ("se", np.float64)])
        self._sf = np.empty((timeline.size + 1,), dtype=dtype)
        self._sf["timeline"] = np.insert(timeline, 0, 0)
        self._sf["estimation"] = np.insert(sf, 0, 1)
        self._sf["se"] = np.insert(np.sqrt(var), 0, 0)
        return self

    @overload
    def sf(self, se: Literal[False]) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None: ...
    @overload
    def sf(self, se: Literal[True]) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]] | None: ...
    @overload
    def sf(
        self, se: bool
    ) -> (
        tuple[NDArray[np.float64], NDArray[np.float64]]
        | tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
        | None
    ): ...

    def sf(
        self, se: bool = True
    ) -> (
        tuple[NDArray[np.float64], NDArray[np.float64]]
        | tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
        | None
    ):
        """
        The survival function estimation

        Parameters
        ----------
        se : bool, default is False
            If true, the estimated standard errors are returned too.

        Returns
        -------
        tuple of 2 or 3 ndarrays
            A tuple containing the timeline, the estimated values and optionally the estimated standard errors (if se is set to true)
        """
        if self._sf is None:
            return None
        if se:
            return self._sf["timeline"], self._sf["estimation"], self._sf["se"]
        return self._sf["timeline"], self._sf["estimation"]

    @property
    def plot(self):
        return PlotKaplanMeier(self)


@final
class NelsonAalen:
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

    _chf: NDArray[np.void] | None

    def __init__(self) -> None:
        self._chf = None

    def fit(
        self,
        time: NDArray[np.float64],
        event: NDArray[np.bool_] | None = None,
        entry: NDArray[np.float64] | None = None,
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

        n = ((timeline <= time.reshape(-1, 1)) * (timeline >= entry.reshape(-1, 1))).sum(axis=0)

        d = ((time.reshape(-1, 1) == timeline) * event.reshape(-1, 1)).sum(axis=0)

        chf = (d / n).cumsum()

        with np.errstate(divide="ignore"):
            var = (d / n**2).cumsum()

        dtype = np.dtype([("timeline", np.float64), ("estimation", np.float64), ("se", np.float64)])
        self._chf = np.empty((timeline.size + 1,), dtype=dtype)
        self._chf["timeline"] = np.insert(timeline, 0, 0)
        self._chf["estimation"] = np.insert(chf, 0, 0)
        self._chf["se"] = np.insert(np.sqrt(var), 0, 0)
        return self

    @overload
    def chf(self, se: Literal[False]) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None: ...
    @overload
    def chf(self, se: Literal[True]) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]] | None: ...
    @overload
    def chf(
        self, se: bool
    ) -> (
        tuple[NDArray[np.float64], NDArray[np.float64]]
        | tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
        | None
    ): ...

    def chf(
        self, se: bool = True
    ) -> (
        tuple[NDArray[np.float64], NDArray[np.float64]]
        | tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
        | None
    ):
        """
        The cumulative hazard function estimation

        Parameters
        ----------
        se : bool, default is False
            If true, the estimated standard errors are returned too.

        Returns
        -------
        tuple of 2 or 3 ndarrays
            A tuple containing the timeline, the estimated values and optionally the estimated standard errors (if se is set to true)
        """
        if self._chf is None:
            return None
        if se:
            return self._chf["timeline"], self._chf["estimation"], self._chf["se"]
        return self._chf["timeline"], self._chf["estimation"]

    @property
    def plot(self):
        return PlotNelsonAalen(self)
