from dataclasses import dataclass
from functools import wraps
from typing import Optional, Protocol, Union, Self

import numpy as np
from numpy.typing import NDArray

from relife.nhpp import nhpp_data_factory
from relife.utils.data import LifetimeData, lifetime_data_factory
from relife.utils.plots import PlotSurvivalFunc


@dataclass
class Estimates:
    """
    Stores the estimates for a non-parametric lifetime model.

    Parameters
    ----------
    timeline : np.ndarray of shape (n, )
        The timeline of the estimates.
    values : np.ndarray of shape (n, )
        The estimated values.
    se : np.ndarray of shape (n, ), optional
        The standard errors of the estimates. If not provided, defaults to an array of zeros.

    Raises
    ------
    ValueError
        If the shapes of `timeline`, `values`, and `se` are not compatible.
    """

    timeline: NDArray[np.float64]
    values: NDArray[np.float64]
    se: Optional[NDArray[np.float64]] = None

    def __post_init__(self):
        if self.se is None:
            self.se = np.zeros_like(
                self.values
            )  # garder None/Nan efaire le changement de valeur au niveau du plot

        if self.timeline.shape != self.values.shape != self.se:
            raise ValueError("Incompatible timeline, values and se in Estimates")

    def nearest_1dinterp(
        self, x: float | NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Returns x nearest interpolation based on timeline and values data points
        timeline has to be monotonically increasing

        Args:
            x (NDArray[np.float64]): 1d x coordinates to interpolate

        Returns:
            NDArray[np.float64]: interpolation values of x
        """
        spacing = np.diff(self.timeline) / 2
        xp = np.hstack([spacing, spacing[-1]]) + self.timeline
        values_p = np.concatenate([self.values, self.values[-1, None]])
        se_p = np.concatenate([self.se, self.se[-1, None]])
        return (
            values_p[np.searchsorted(xp, np.asarray(x))],
            se_p[np.searchsorted(xp, np.asarray(x))],
        )


def estimated(method):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        if None in self.estimates.values():
            return ValueError(
                f"{self.__class__.__name__} instance has not been fitted yet, call fit first"
            )
        return method(self, *args, **kwargs)

    return wrapper


class NonParametricEstimator(Protocol):
    """
    Non-parametric lifetime estimator.

    Attributes
    ----------
    estimates : Estimations
        The estimations produced when fitting the estimator.
    """

    estimates: dict[str, Optional[Estimates]]

    @property
    @estimated
    def plot(self):
        return PlotSurvivalFunc(self)


class ECDF(NonParametricEstimator):
    """
    Empirical Cumulative Distribution Function.
    """

    def __init__(self):
        super().__init__()
        self.estimates = {"sf": None, "cdf": None}

    def fit(
        self,
        time: NDArray[np.float64],
        event: Optional[NDArray[np.float64]] = None,
        entry: Optional[NDArray[np.float64]] = None,
        departure: Optional[NDArray[np.float64]] = None,
        inplace: bool = False,
    ) -> Union[Self, None]:
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

        if inplace:
            self.estimates["sf"] = Estimates(timeline, sf, se)
            self.estimates["cdf"] = Estimates(timeline, cdf, se)
        else:
            return_obj = ECDF()
            return_obj.estimates["sf"] = Estimates(timeline, sf, se)
            return_obj.estimates["cdf"] = Estimates(timeline, cdf, se)
            return return_obj

    @estimated
    def sf(self, time: float | NDArray[np.float64]) -> NDArray[np.float64]:
        """Survival function.

        Parameters
        ----------
        time : np.ndarray of shape (n, )
            The times at which to estimate the survival function.

        Returns
        -------
        np.ndarray of shape (n, )
            The estimated survival probabilities at each time.
        """
        return self.estimates["sf"].nearest_1dinterp(time)[0]

    @estimated
    def cdf(self, time: float | NDArray[np.float64]) -> NDArray[np.float64]:
        """Cumulative hazard function.

        Parameters
        ----------
        time : np.ndarray of shape (n, )
            The times at which to estimate the cumulative hazard function.

        Returns
        -------
        np.ndarray of shape (n, )
            The estimated cumulative hazard values at each time.
        """
        return self.estimates["cdf"].nearest_1dinterp(time)[0]


class KaplanMeier(NonParametricEstimator):
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
        super().__init__()
        self.estimates = {"sf": None}

    def fit(
        self,
        time: NDArray[np.float64],
        event: Optional[NDArray[np.float64]] = None,
        entry: Optional[NDArray[np.float64]] = None,
        departure: Optional[NDArray[np.float64]] = None,
        inplace: bool = False,
    ) -> Union[Self, None]:
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

        if inplace:
            self.estimates["sf"] = Estimates(timeline, sf, se)
        else:
            return_obj = KaplanMeier()
            return_obj.estimates["sf"] = Estimates(timeline, sf, se)
            return return_obj

    @estimated
    def sf(self, time: float | NDArray[np.float64]) -> NDArray[np.float64]:
        """Survival function.

        Parameters
        ----------
        time : np.ndarray of shape (n, )
            The times at which to estimate the survival function.

        Returns
        -------
        np.ndarray of shape (n, )
            The estimated survival probabilities at each time.
        """
        return self.estimates["sf"].nearest_1dinterp(time)[0]


class NelsonAalen(NonParametricEstimator):
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
        super().__init__()
        self.estimates = {"chf": None}

    def fit(
        self,
        time: NDArray[np.float64],
        event: Optional[NDArray[np.float64]] = None,
        entry: Optional[NDArray[np.float64]] = None,
        departure: Optional[NDArray[np.float64]] = None,
        inplace: bool = False,
    ) -> Union[Self, None]:
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

        if inplace:
            self.estimates["chf"] = Estimates(timeline, chf, se)
        else:
            return_obj = NelsonAalen()
            return_obj.estimates["chf"] = Estimates(timeline, chf, se)
            return return_obj

    @estimated
    def chf(self, time: float | NDArray[np.float64]) -> NDArray[np.float64]:
        """Cumulative hazard function.

        Parameters
        ----------
        time : np.ndarray of shape (n, )
            The times at which to estimate the cumulative hazard function.

        Returns
        -------
        np.ndarray of shape (n, )
            The estimated cumulative hazard values at each time.
        """
        return self.estimates["chf"].nearest_1dinterp(time)[0]


class Turnbull(NonParametricEstimator):
    """Turnbull estimator"""

    def __init__(
        self,
        tol: Optional[float] = 1e-4,
        lowmem: Optional[bool] = False,
    ):
        super().__init__()
        self.estimates = {"sf": None}
        self.tol = tol
        self.lowmem = lowmem

    def fit(
        self,
        time: float | NDArray[np.float64],
        event: Optional[NDArray[np.float64]] = None,
        entry: Optional[NDArray[np.float64]] = None,
        departure: Optional[NDArray[np.float64]] = None,
        inplace: bool = False,
    ) -> Union[Self, None]:
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

        if inplace:
            self.estimates["sf"] = Estimates(timeline, sf)
        else:
            return_obj = Turnbull(self.tol, self.lowmem)
            return_obj.estimates["sf"] = Estimates(timeline, sf)
            return return_obj

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

    @estimated
    def sf(self, time: float | NDArray[np.float64]) -> NDArray[np.float64]:
        """Survival function.

        Parameters
        ----------
        time : np.ndarray of shape (n, )
            The times at which to estimate the survival function.

        Returns
        -------
        np.ndarray of shape (n, )
            The estimated survival probabilities at each time.
        """
        return self.estimates["sf"].nearest_1dinterp(time)[0]


class NonParametricNHPP:

    def __init__(self):
        self.nelson_aalen = NelsonAalen()

    def fit(
        self,
        ages: NDArray[np.float64],
        assets: NDArray[np.float64],
    ) -> None:
        self.nelson_aalen.fit(nhpp_data_factory(ages, assets))

    def chf(self, time):
        """Cumulative hazard function.

        Parameters
        ----------
        time : np.ndarray of shape (n, )
            The times at which to estimate the cumulative hazard function.

        Returns
        -------
        np.ndarray of shape (n, )
            The estimated cumulative hazard values at each time.
        """
        return self.nelson_aalen.chf(time)
