from typing import Literal, Optional, Self, overload

import numpy as np
from numpy.typing import NDArray

from relife._plot import PlotECDF, PlotKaplanMeier, PlotNelsonAalen, PlotTurnbull
from relife.data import LifetimeData

from ._base import NonParametricLifetimeModel


class ECDF(NonParametricLifetimeModel):
    """
    Empirical Cumulative Distribution Function.
    """

    def __init__(self):
        self._sf: Optional[NDArray[np.void]] = None
        self._cdf: Optional[NDArray[np.void]] = None

    def fit(
        self,
        time: NDArray[np.float64],
        /,
        event: Optional[NDArray[np.float64]] = None,
        entry: Optional[NDArray[np.float64]] = None,
        departure: Optional[NDArray[np.float64]] = None,
    ) -> Self:
        """
        Compute the non-parametric estimations with respect to lifetime data.

        Parameters
        ----------
        time : ndarray (1d or 2d)
            Observed lifetime values.
        event : ndarray of boolean values (1d), default is None
            Boolean indicators tagging lifetime values as right censored or complete.
        entry : ndarray of float (1d), default is None
            Left truncations applied to lifetime values.
        departure : ndarray of float (1d), default is None
            Right truncations applied to lifetime values.
        """
        lifetime_data = LifetimeData(
            time,
            event=event,
            entry=entry,
            departure=departure,
        )
        timeline, counts = np.unique(lifetime_data.complete_or_right_censored.lifetime_values, return_counts=True)
        timeline = np.insert(timeline, 0, 0)

        dtype = np.dtype([("timeline", np.float64), ("estimation", np.float64), ("se", np.float64)])
        self._sf = np.empty((timeline.size,), dtype=dtype)
        self._cdf = np.empty((timeline.size,), dtype=dtype)

        self._sf["timeline"] = timeline
        self._cdf["timeline"] = timeline
        cdf = np.insert(np.cumsum(counts), 0, 0) / np.sum(counts)
        self._cdf["estimation"] = cdf
        self._sf["estimation"] = 1 - cdf
        se = np.sqrt(*(1 - cdf) / len(lifetime_data.complete_or_right_censored.lifetime_values))
        self._sf["se"] = se
        self._cdf["se"] = se
        return self

    @overload
    def sf(self, se: Literal[False] = False) -> Optional[tuple[NDArray[np.float64], NDArray[np.float64]]]: ...

    @overload
    def sf(
        self, se: Literal[True] = True
    ) -> Optional[tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]]: ...

    def sf(
        self, se: bool = False
    ) -> (
        Optional[tuple[NDArray[np.float64], NDArray[np.float64]]]
        | Optional[tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]]
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
    def cdf(self, se: Literal[False] = False) -> Optional[tuple[NDArray[np.float64], NDArray[np.float64]]]: ...

    @overload
    def cdf(
        self, se: Literal[True] = True
    ) -> Optional[tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]]: ...

    def cdf(
        self, se: bool = False
    ) -> (
        Optional[tuple[NDArray[np.float64], NDArray[np.float64]]]
        | Optional[tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]]
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
    def plot(self) -> PlotECDF:
        return PlotECDF(self)


class KaplanMeier(NonParametricLifetimeModel):
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

    def __init__(self):
        self._sf: Optional[np.void] = None

    def fit(
        self,
        time: NDArray[np.float64],
        event: Optional[NDArray[np.float64]] = None,
        entry: Optional[NDArray[np.float64]] = None,
        departure: Optional[NDArray[np.float64]] = None,
    ) -> Self:
        """
        Compute the non-parametric estimations with respect to lifetime data.

        Parameters
        ----------
        time : ndarray (1d or 2d)
            Observed lifetime values.
        event : ndarray of boolean values (1d), default is None
            Boolean indicators tagging lifetime values as right censored or complete.
        entry : ndarray of float (1d), default is None
            Left truncations applied to lifetime values.
        departure : ndarray of float (1d), default is None
            Right truncations applied to lifetime values.
        inplace : boolean, default is True
            If true, estimations are stored in the object
        """

        lifetime_data = LifetimeData(
            time,
            event=event,
            entry=entry,
            departure=departure,
        )

        if lifetime_data.left_censoring is not None:
            raise ValueError("KaplanMeier does not take left censored lifetimes")
        timeline, unique_indices, counts = np.unique(
            lifetime_data.complete_or_right_censored.lifetime_values,
            return_inverse=True,
            return_counts=True,
        )
        death_set = np.zeros_like(timeline, int)  # death at each timeline step
        complete_observation_indic = np.zeros_like(
            lifetime_data.complete_or_right_censored.lifetime_values
        )  # just creating an array to fill it next line
        complete_observation_indic[lifetime_data.complete.lifetime_index] = 1
        np.add.at(death_set, unique_indices, complete_observation_indic)
        x_in = np.histogram(
            np.concatenate(
                (
                    lifetime_data.left_truncation.lifetime_values.flatten(),
                    np.array(
                        [
                            0
                            for _ in range(
                                len(lifetime_data.complete_or_right_censored.lifetime_values)
                                - len(lifetime_data.left_truncation.lifetime_values)
                            )
                        ]
                    ),  # TODO : remplacer ça par self.entry en définissant self.entry plus haut?
                )
            ),
            np.insert(timeline, 0, 0),
        )[0]
        x_out = np.insert(counts[:-1], 0, 0)
        at_risk_assets = np.cumsum(x_in - x_out)
        s = np.cumprod(1 - death_set / at_risk_assets)
        sf = np.insert(s, 0, 1)

        with np.errstate(divide="ignore"):
            var = s**2 * np.cumsum(
                np.where(
                    at_risk_assets > death_set,
                    death_set / (at_risk_assets * (at_risk_assets - death_set)),
                    0,
                )
            )
        se = np.sqrt(np.insert(var, 0, 0))
        timeline = np.insert(timeline, 0, 0)

        dtype = np.dtype([("timeline", np.float64), ("estimation", np.float64), ("se", np.float64)])
        self._sf = np.empty((timeline.size,), dtype=dtype)
        self._sf["timeline"] = timeline
        self._sf["estimation"] = sf
        self._sf["se"] = se
        return self

    @overload
    def sf(self, se: Literal[False] = False) -> Optional[tuple[NDArray[np.float64], NDArray[np.float64]]]: ...

    @overload
    def sf(
        self, se: Literal[True] = True
    ) -> Optional[tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]]: ...

    def sf(
        self, se: bool = False
    ) -> (
        Optional[tuple[NDArray[np.float64], NDArray[np.float64]]]
        | Optional[tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]]
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
    def plot(self) -> PlotKaplanMeier:
        return PlotKaplanMeier(self)


class NelsonAalen(NonParametricLifetimeModel):
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

    def __init__(self):
        self._chf: Optional[NDArray[np.void]] = None

    def fit(
        self,
        time: NDArray[np.float64],
        event: Optional[NDArray[np.float64]] = None,
        entry: Optional[NDArray[np.float64]] = None,
        departure: Optional[NDArray[np.float64]] = None,
    ) -> Self:
        """
        Compute the non-parametric estimations with respect to lifetime data.

        Parameters
        ----------
        time : ndarray (1d or 2d)
            Observed lifetime values.
        event : ndarray of boolean values (1d), default is None
            Boolean indicators tagging lifetime values as right censored or complete.
        entry : ndarray of float (1d), default is None
            Left truncations applied to lifetime values.
        departure : ndarray of float (1d), default is None
            Right truncations applied to lifetime values.
        """

        lifetime_data = LifetimeData(
            time,
            event=event,
            entry=entry,
            departure=departure,
        )

        if lifetime_data.left_censoring is not None:
            raise ValueError("NelsonAalen does not accept left censored or interval censored lifetimes")

        timeline, unique_indices, counts = np.unique(
            lifetime_data.complete_or_right_censored.lifetime_values,
            return_inverse=True,
            return_counts=True,
        )
        death_set = np.zeros_like(timeline, dtype=np.int64)  # death at each timeline step

        complete_observation_indic = np.zeros_like(
            lifetime_data.complete_or_right_censored.lifetime_values
        )  # just creating an array to fill it next line
        complete_observation_indic[lifetime_data.complete.lifetime_index] = 1

        np.add.at(death_set, unique_indices, complete_observation_indic)
        x_in = np.histogram(
            np.concatenate(
                (
                    lifetime_data.left_truncation.lifetime_values.flatten(),
                    np.array(
                        [
                            0
                            for _ in range(
                                len(lifetime_data.complete_or_right_censored.lifetime_values)
                                - len(lifetime_data.left_truncation.lifetime_values)
                            )
                        ]
                    ),  # TODO : remplacer ça par self.entry en définissant self.entry plus haut?
                )
            ),
            np.insert(timeline, 0, 0),
        )[0]
        x_out = np.insert(counts[:-1], 0, 0)
        at_risk_assets = np.cumsum(x_in - x_out)
        s = np.cumsum(death_set / at_risk_assets)
        var = np.cumsum(death_set / np.where(at_risk_assets == 0, 1, at_risk_assets**2))
        chf = np.insert(s, 0, 0)
        se = np.sqrt(np.insert(var, 0, 0))
        timeline = np.insert(timeline, 0, 0)

        dtype = np.dtype([("timeline", np.float64), ("estimation", np.float64), ("se", np.float64)])
        self._chf = np.empty((timeline.size,), dtype=dtype)
        self._chf["timeline"] = timeline
        self._chf["estimation"] = chf
        self._chf["se"] = se
        return self

    @overload
    def chf(self, se: Literal[False] = False) -> Optional[tuple[NDArray[np.float64], NDArray[np.float64]]]: ...

    @overload
    def chf(
        self, se: Literal[True] = True
    ) -> Optional[tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]]: ...

    def chf(
        self, se: bool = False
    ) -> (
        Optional[tuple[NDArray[np.float64], NDArray[np.float64]]]
        | Optional[tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]]
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
    def plot(self) -> PlotNelsonAalen:
        return PlotNelsonAalen(self)


class Turnbull(NonParametricLifetimeModel):
    """Turnbull estimator."""

    def __init__(
        self,
        tol: Optional[float] = 1e-4,
        lowmem: Optional[bool] = False,
    ):
        super().__init__()
        self.tol = tol
        self.lowmem = lowmem
        self._sf: Optional[NDArray[np.void]] = None

    def fit(
        self,
        time: NDArray[np.float64],
        event: Optional[NDArray[np.float64]] = None,
        entry: Optional[NDArray[np.float64]] = None,
        departure: Optional[NDArray[np.float64]] = None,
        inplace: bool = False,
    ) -> Self:
        """
        Compute the non-parametric estimations with respect to lifetime data.

        Parameters
        ----------
        time : ndarray (1d or 2d)
            Observed lifetime values.
        event : ndarray of boolean values (1d), default is None
            Boolean indicators tagging lifetime values as right censored or complete.
        entry : ndarray of float (1d), default is None
            Left truncations applied to lifetime values.
        departure : ndarray of float (1d), default is None
            Right truncations applied to lifetime values.
        inplace : boolean, default is True
            If true, estimations are stored in the object
        """

        lifetime_data = LifetimeData(
            time,
            event=event,
            entry=entry,
            departure=departure,
        )

        timeline_temp = np.unique(np.insert(lifetime_data.interval_censoring.lifetime_values.flatten(), 0, 0))
        timeline_len = len(timeline_temp)
        if not self.lowmem:
            event_occurence = (
                np.greater_equal.outer(
                    timeline_temp[:-1],
                    lifetime_data.interval_censoring.lifetime_values[
                        :, 0
                    ],  # or self.observed_lifetimes.interval_censored.values.T[0][i]
                )
                * np.less_equal.outer(
                    timeline_temp[1:],
                    lifetime_data.interval_censoring.lifetime_values[:, 1],
                )
            ).T

            s = self._estimate_with_high_memory(
                lifetime_data,
                timeline_len,
                event_occurence,
                timeline_temp,
            )

        else:
            len_censored_data = len(lifetime_data.interval_censoring.lifetime_values)
            event_occurence = []
            for i in range(len_censored_data):
                event_occurence.append(
                    np.where(
                        (lifetime_data.interval_censoring.lifetime_values[:, 0][i] <= timeline_temp[:-1])
                        & (timeline_temp[1:] <= lifetime_data.interval_censoring.lifetime_values[:, 1][i])
                    )[0][[0, -1]]
                )
            event_occurence = np.array(event_occurence)
            s = self._estimate_with_low_memory(
                lifetime_data,
                timeline_temp,
                timeline_len,
                event_occurence,
                len_censored_data,
            )

        ind_del = np.where(timeline_temp == np.inf)
        sf = np.delete(s, ind_del)
        timeline = np.delete(timeline_temp, ind_del)
        dtype = np.dtype([("timeline", np.float64), ("estimation", np.float64)])
        self._sf = np.empty((timeline.size,), dtype=dtype)
        self._sf["timeline"] = timeline
        self._sf["estimation"] = sf
        return self

    def sf(self) -> Optional[tuple[NDArray[np.float64], NDArray[np.float64]]]:
        """
        The survival function estimation

        Returns
        -------
        tuple of 2 ndarrays
            A tuple containing the timeline and the estimated values
        """
        if self._sf is None:
            return None
        return self._sf["timeline"], self._sf["estimation"]

    def _estimate_with_low_memory(
        self,
        lifetime_data: LifetimeData,
        timeline_temp,
        timeline_len,
        event_occurence,
        len_censored_data,
    ):

        d_tilde = np.histogram(
            np.searchsorted(timeline_temp, lifetime_data.complete.lifetime_values),
            bins=range(timeline_len + 1),
        )[0][1:]
        s = np.linspace(1, 0, timeline_len)
        res = 1
        count = 1
        while res > self.tol:
            p = -np.diff(s)  # écart entre les points de S (survival function) => proba of an event occuring at
            if np.sum(p == 0) > 0:
                p = np.where(
                    p == 0,
                    1e-5 / np.sum(p == 0),
                    p - 1e-5 / ((timeline_len - 1) - np.sum(p == 0)),
                )  # remplace 0 par 1e-5 (et enlève cette quantité des autres proba pr sommer à 1)

            x = [p[event_occurence[i, 0] : (event_occurence[i, 1] + 1)] for i in range(event_occurence.shape[0])]
            d = np.repeat(0, timeline_len - 1)
            for i in range(len_censored_data):
                d = np.add(
                    d,
                    np.append(
                        np.insert(x[i] / x[i].sum(), 0, np.repeat(0, event_occurence[i][0])),
                        np.repeat(0, timeline_len - event_occurence[i][1] - 2),
                    ),
                )
            d += d_tilde
            y = np.cumsum(d[::-1])[::-1]
            _unsorted_entry = lifetime_data.left_truncation.lifetime_values.flatten()

            y -= len(_unsorted_entry) - np.searchsorted(np.sort(_unsorted_entry), timeline_temp[1:])
            s_updated = np.array(np.cumprod(1 - d / y))
            s_updated = np.insert(s_updated, 0, 1)
            res = max(abs(s - s_updated))
            s = s_updated
            count += 1
        return s

    def _estimate_with_high_memory(
        self,
        lifetime_data: LifetimeData,
        timeline_len,
        event_occurence,
        timeline_temp,
    ):

        d_tilde = np.histogram(
            np.searchsorted(timeline_temp, lifetime_data.complete.lifetime_values),
            bins=range(timeline_len + 1),
        )[0][1:]
        s = np.linspace(1, 0, timeline_len)
        res = 1
        count = 1

        while res > self.tol:
            p = -np.diff(s)  # écart entre les points de S (survival function) => proba of an event occuring at
            if np.sum(p == 0) > 0:
                p = np.where(
                    p == 0,
                    1e-5 / np.sum(p == 0),
                    p - 1e-5 / ((timeline_len - 1) - np.sum(p == 0)),
                )  # remplace 0 par 1e-5 (et enlève cette quantité des autres proba pr sommer à 1)

            if np.any(event_occurence):
                event_occurence_proba = event_occurence * p.T
                d = (event_occurence_proba.T / event_occurence_proba.sum(axis=1)).T.sum(axis=0)
                d += d_tilde
            else:
                d = d_tilde
            y = np.cumsum(d[::-1])[::-1]
            _unsorted_entry = lifetime_data.left_truncation.lifetime_values.flatten()
            y -= len(_unsorted_entry) - np.searchsorted(np.sort(_unsorted_entry), timeline_temp[1:])
            s_updated = np.array(np.cumprod(1 - d / y))
            s_updated = np.insert(s_updated, 0, 1)
            res = max(abs(s - s_updated))
            s = s_updated
            count += 1
        return s

    @property
    def plot(self) -> PlotTurnbull:
        return PlotTurnbull(self)
