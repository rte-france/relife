from typing import Optional, Self, Union

import numpy as np
from numpy.typing import NDArray

from relife.core.decorators import require_attributes
from relife.core.model import Estimates, NonParametricModel
from relife.data.lifetime import LifetimeData, lifetime_data_factory
from relife.process.nhpp import nhpp_data_factory


class ECDF(NonParametricModel):
    """
    Empirical Cumulative Distribution Function.
    """

    _sf: Union[Estimates, None]
    _cdf: Union[Estimates, None]

    def __init__(self):
        super().__init__()
        self._sf = None
        self._cdf = None

    def fit(
        self,
        time: NDArray[np.float64],
        event: Optional[NDArray[np.float64]] = None,
        entry: Optional[NDArray[np.float64]] = None,
        departure: Optional[NDArray[np.float64]] = None,
    ) -> Self:
        """
        Computes the EDCF estimators (``sf`` and ``cdf``) from given lifetime data.

        Parameters
        ----------
        time : np.ndarray
            Observed lifetime values. Dimension is ``1d`` as left censoring is not allowed here.
            Shape must be ``(n_samples,)``.
        event : np.ndarray, default is None
            Booleans that indicated if lifetime values are right censored. Shape must be ``(n_samples,)``.
            By default, all lifetimes are assumed to be complete.
        entry : np.ndarray, default is None
            Left truncations values. Shape is always ``(n_samples,)``.
        departure : np.ndarray, default is None
            Right truncations values. Shape is always ``(n_samples,)``.

        Returns
        -------
        Self
            The instance with computed estimators.
        """

        lifetime_data = lifetime_data_factory(
            time,
            event,
            entry,
            departure,
        )

        timeline, counts = np.unique(lifetime_data.rc.values, return_counts=True)
        timeline = np.insert(timeline, 0, 0)
        cdf = np.insert(np.cumsum(counts), 0, 0) / np.sum(counts)
        sf = 1 - cdf
        se = np.sqrt(cdf * (1 - cdf) / len(lifetime_data.rc.values))

        self._sf = Estimates(timeline, sf, se)
        self._cdf = Estimates(timeline, cdf, se)
        return self

    @require_attributes("_sf")
    def sf(self, time: float | NDArray[np.float64]) -> NDArray[np.float64]:
        return self._sf.nearest_1dinterp(time)[0]

    @require_attributes("_cdf")
    def cdf(self, time: float | NDArray[np.float64]) -> NDArray[np.float64]:
        return self._cdf.nearest_1dinterp(time)[0]


class KaplanMeier(NonParametricModel):
    r"""Kaplan-Meier estimator (also known as the product limit estimator).

    For a given time instant :math:`t` and :math:`n` total observations, this
    estimator is defined as [1]_ [2]_:

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

    _sf: Union[Estimates, None]

    def __init__(self):
        super().__init__()
        self._sf = None

    def fit(
        self,
        time: NDArray[np.float64],
        event: Optional[NDArray[np.float64]] = None,
        entry: Optional[NDArray[np.float64]] = None,
        departure: Optional[NDArray[np.float64]] = None,
    ) -> Union[Self, None]:
        """
        Computes the Kaplan-Meier estimator (``sf``) from given lifetime data.

        Parameters
        ----------
        time : np.ndarray
            Observed lifetime values. Dimension is ``1d`` as left censoring is not allowed here.
            Shape must be ``(n_samples,)``.
        event : np.ndarray, default is None
            Booleans that indicated if lifetime values are right censored. Shape must be ``(n_samples,)``.
            By default, all lifetimes are assumed to be complete.
        entry : np.ndarray, default is None
            Left truncations values. Shape is always ``(n_samples,)``.
        departure : np.ndarray, default is None
            Right truncations values. Shape is always ``(n_samples,)``.

        Returns
        -------
        Self
            The instance with computed estimators.

        Raises
        ------
        ValueError
            If the input lifetime data contains left-censored.
        """

        lifetime_data = lifetime_data_factory(
            time,
            event,
            entry,
            departure,
        )

        if len(lifetime_data.left_censoring) > 0:
            raise ValueError("KaplanMeier does not take left censored lifetimes")
        timeline, unique_indices, counts = np.unique(
            lifetime_data.rc.values, return_inverse=True, return_counts=True
        )
        death_set = np.zeros_like(timeline, int)  # death at each timeline step
        complete_observation_indic = np.zeros_like(
            lifetime_data.rc.values
        )  # just creating an array to fill it next line
        complete_observation_indic[lifetime_data.complete.index] = 1
        np.add.at(death_set, unique_indices, complete_observation_indic)
        x_in = np.histogram(
            np.concatenate(
                (
                    lifetime_data.left_truncation.values.flatten(),
                    np.array(
                        [
                            0
                            for _ in range(
                                len(lifetime_data.rc.values)
                                - len(lifetime_data.left_truncation.values)
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

        self._sf = Estimates(timeline, sf, se)
        return self

    @require_attributes("_sf")
    def sf(self, time: float | NDArray[np.float64]) -> NDArray[np.float64]:
        return self._sf.nearest_1dinterp(time)[0]


class NelsonAalen(NonParametricModel):
    r"""Nelson-Aalen estimator.

    For a given time instant :math:`t` and :math:`n` total observations, this
    estimator is defined as [1]_:

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

    _chf: Union[Estimates, None]

    def __init__(self):
        super().__init__()
        self._chf = None

    def fit(
        self,
        time: NDArray[np.float64],
        event: Optional[NDArray[np.float64]] = None,
        entry: Optional[NDArray[np.float64]] = None,
        departure: Optional[NDArray[np.float64]] = None,
    ) -> Self:
        """
        Computes the Nelson-Aalen estimator (``chf``) from given lifetime data.

        Parameters
        ----------
        time : np.ndarray
            Observed lifetime values. Dimension is ``1d`` as left censoring is not allowed here.
            Shape must be ``(n_samples,)``.
        event : np.ndarray, default is None
            Booleans that indicated if lifetime values are right censored. Shape must be ``(n_samples,)``.
            By default, all lifetimes are assumed to be complete.
        entry : np.ndarray, default is None
            Left truncations values. Shape is always ``(n_samples,)``.
        departure : np.ndarray, default is None
            Right truncations values. Shape is always ``(n_samples,)``.

        Returns
        -------
        Self
            The instance with computed estimators.

        Raises
        ------
        ValueError
            If the input lifetime data contains left-censored or interval-censored
            observations.
        """

        lifetime_data = lifetime_data_factory(
            time,
            event,
            entry,
            departure,
        )

        if len(lifetime_data.left_censoring) > 0:
            raise ValueError(
                "NelsonAalen does not accept left censored or interval censored lifetimes"
            )

        timeline, unique_indices, counts = np.unique(
            lifetime_data.rc.values, return_inverse=True, return_counts=True
        )
        death_set = np.zeros_like(
            timeline, dtype=np.int64
        )  # death at each timeline step

        complete_observation_indic = np.zeros_like(
            lifetime_data.rc.values
        )  # just creating an array to fill it next line
        complete_observation_indic[lifetime_data.complete.index] = 1

        np.add.at(death_set, unique_indices, complete_observation_indic)
        x_in = np.histogram(
            np.concatenate(
                (
                    lifetime_data.left_truncation.values.flatten(),
                    np.array(
                        [
                            0
                            for _ in range(
                                len(lifetime_data.rc.values)
                                - len(lifetime_data.left_truncation.values)
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

        self._chf = Estimates(timeline, chf, se)
        return self

    @require_attributes("_chf")
    def chf(self, time: float | NDArray[np.float64]) -> NDArray[np.float64]:
        return self._chf.nearest_1dinterp(time)[0]


class Turnbull(NonParametricModel):
    """Turnbull estimator"""

    _sf: Union[Estimates, None]

    def __init__(
        self,
        tol: Optional[float] = 1e-4,
        lowmem: Optional[bool] = False,
    ):
        super().__init__()
        self._sf = None
        self.tol = tol
        self.lowmem = lowmem

    def fit(
        self,
        time: float | NDArray[np.float64],
        event: Optional[NDArray[np.float64]] = None,
        entry: Optional[NDArray[np.float64]] = None,
        departure: Optional[NDArray[np.float64]] = None,
    ) -> Self:
        """
        Computes the Turnbull estimator (``sf``) from given lifetime data.

        Parameters
        ----------
        time : np.ndarray
            Observed lifetime values. Dimension is ``1d`` as left censoring is not allowed here.
            Shape must be ``(n_samples,)``.
        event : np.ndarray, default is None
            Booleans that indicated if lifetime values are right censored. Shape must be ``(n_samples,)``.
            By default, all lifetimes are assumed to be complete.
        entry : np.ndarray, default is None
            Left truncations values. Shape is always ``(n_samples,)``.
        departure : np.ndarray, default is None
            Right truncations values. Shape is always ``(n_samples,)``.

        Returns
        -------
        Self
            The instance with computed estimators.
        """

        lifetime_data = lifetime_data_factory(
            time,
            event,
            entry,
            departure,
        )

        timeline_temp = np.unique(
            np.insert(lifetime_data.interval_censoring.values.flatten(), 0, 0)
        )
        timeline_len = len(timeline_temp)
        if not self.lowmem:
            event_occurence = (
                np.greater_equal.outer(
                    timeline_temp[:-1],
                    lifetime_data.interval_censoring.values[
                        :, 0
                    ],  # or self.observed_lifetimes.interval_censored.values.T[0][i]
                )
                * np.less_equal.outer(
                    timeline_temp[1:],
                    lifetime_data.interval_censoring.values[:, 1],
                )
            ).T

            s = self._estimate_with_high_memory(
                lifetime_data,
                timeline_len,
                event_occurence,
                timeline_temp,
            )

        else:
            len_censored_data = len(lifetime_data.interval_censoring.values)
            event_occurence = []
            for i in range(len_censored_data):
                event_occurence.append(
                    np.where(
                        (
                            lifetime_data.interval_censoring.values[:, 0][i]
                            <= timeline_temp[:-1]
                        )
                        & (
                            timeline_temp[1:]
                            <= lifetime_data.interval_censoring.values[:, 1][i]
                        )
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

        self._sf = Estimates(timeline, sf)
        return self

    def _estimate_with_low_memory(
        self,
        lifetime_data: LifetimeData,
        timeline_temp,
        timeline_len,
        event_occurence,
        len_censored_data,
    ):

        d_tilde = np.histogram(
            np.searchsorted(timeline_temp, lifetime_data.complete.values),
            bins=range(timeline_len + 1),
        )[0][1:]
        s = np.linspace(1, 0, timeline_len)
        res = 1
        count = 1
        while res > self.tol:
            p = -np.diff(
                s
            )  # écart entre les points de S (survival function) => proba of an event occuring at
            if np.sum(p == 0) > 0:
                p = np.where(
                    p == 0,
                    1e-5 / np.sum(p == 0),
                    p - 1e-5 / ((timeline_len - 1) - np.sum(p == 0)),
                )  # remplace 0 par 1e-5 (et enlève cette quantité des autres proba pr sommer à 1)

            x = [
                p[event_occurence[i, 0] : (event_occurence[i, 1] + 1)]
                for i in range(event_occurence.shape[0])
            ]
            d = np.repeat(0, timeline_len - 1)
            for i in range(len_censored_data):
                d = np.add(
                    d,
                    np.append(
                        np.insert(
                            x[i] / x[i].sum(), 0, np.repeat(0, event_occurence[i][0])
                        ),
                        np.repeat(0, timeline_len - event_occurence[i][1] - 2),
                    ),
                )
            d += d_tilde
            y = np.cumsum(d[::-1])[::-1]
            _unsorted_entry = lifetime_data.left_truncation.values.flatten()

            y -= len(_unsorted_entry) - np.searchsorted(
                np.sort(_unsorted_entry), timeline_temp[1:]
            )
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
            np.searchsorted(timeline_temp, lifetime_data.complete.values),
            bins=range(timeline_len + 1),
        )[0][1:]
        s = np.linspace(1, 0, timeline_len)
        res = 1
        count = 1

        while res > self.tol:
            p = -np.diff(
                s
            )  # écart entre les points de S (survival function) => proba of an event occuring at
            if np.sum(p == 0) > 0:
                p = np.where(
                    p == 0,
                    1e-5 / np.sum(p == 0),
                    p - 1e-5 / ((timeline_len - 1) - np.sum(p == 0)),
                )  # remplace 0 par 1e-5 (et enlève cette quantité des autres proba pr sommer à 1)

            if np.any(event_occurence):
                event_occurence_proba = event_occurence * p.T
                d = (event_occurence_proba.T / event_occurence_proba.sum(axis=1)).T.sum(
                    axis=0
                )
                d += d_tilde
            else:
                d = d_tilde
            y = np.cumsum(d[::-1])[::-1]
            _unsorted_entry = lifetime_data.left_truncation.values.flatten()
            y -= len(_unsorted_entry) - np.searchsorted(
                np.sort(_unsorted_entry), timeline_temp[1:]
            )
            s_updated = np.array(np.cumprod(1 - d / y))
            s_updated = np.insert(s_updated, 0, 1)
            res = max(abs(s - s_updated))
            s = s_updated
            count += 1
        return s

    @require_attributes("_sf")
    def sf(self, time: float | NDArray[np.float64]) -> NDArray[np.float64]:
        return self._sf.nearest_1dinterp(time)[0]


class NHPPNonParametric(NonParametricModel):

    _chf: Union[Estimates, None]

    def __init__(self):
        super().__init__()
        self._chf = None

    def fit(
        self,
        t0: NDArray[np.float64],
        tf: NDArray[np.float64],
        ages: NDArray[np.float64],
        assets: NDArray[np.int64],
    ) -> Self:

        self.estimates["chf"] = (
            NelsonAalen().fit(nhpp_data_factory(t0, tf, ages, assets)).estimates["chf"]
        )
        return self

    @require_attributes("_chf")
    def chf(self, time):
        return self._chf.nearest_1dinterp(time)[0]


TIME_BASE_DOCSTRING = """
The estimation of {name}.

Parameters
----------
time : float or np.ndarray
    Elapsed time value(s) at which to report the estimation value(s).
    If ndarray, allowed shapes are ``()``, ``(n_values,)``.

Returns
-------
np.float64 or np.ndarray
    Estimation values at each given time(s).
"""


ECDF.sf.__doc__ = TIME_BASE_DOCSTRING.format(name="the survival function")
ECDF.cdf.__doc__ = TIME_BASE_DOCSTRING.format(
    name="the cumulative distribution function"
)
KaplanMeier.sf.__doc__ = TIME_BASE_DOCSTRING.format(name="the survival function")
NelsonAalen.chf.__doc__ = TIME_BASE_DOCSTRING.format(
    name="the cumulative hazard function"
)
Turnbull.sf.__doc__ = TIME_BASE_DOCSTRING.format(name="the survival function")
NHPPNonParametric.chf.__doc__ = TIME_BASE_DOCSTRING.format(
    name="the cumulative hazard function"
)
