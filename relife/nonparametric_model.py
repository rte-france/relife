from typing import NamedTuple, Optional

import numpy as np
from numpy.typing import NDArray

from relife._plots import PlotSurvivalFunc
from relife.likelihood.lifetime_data import LifetimeData, lifetime_data_factory


class Estimation(NamedTuple):
    timeline: NDArray[np.float64]
    values: NDArray[np.float64]
    se: Optional[NDArray[np.float64]] = None


class Estimator(dict):
    def __init__(
        self, mapping: Optional[dict[str, Estimation]] = None, /, **kwargs: Estimation
    ):
        if mapping is None:
            mapping = {}
        mapping.update(kwargs)
        super().__init__(mapping)

    def __getattr__(self, item):
        class_name = type(self).__name__
        if item in self.__dict__:
            return self.__dict__[item]
        if item in super().__getattribute__("keys")():
            return super().__getitem__(item)
        raise AttributeError(f"{class_name} has no attribute named {item}")

    def __setitem__(self, key, val):
        raise AttributeError("Can't set item")

    def update(self, *args, **kwargs):
        raise AttributeError("Can't update items")

    @property
    def plot(self):
        return PlotSurvivalFunc(self)


def ecdf(
    time: NDArray[np.float64],
    event: Optional[NDArray[np.float64]] = None,
    entry: Optional[NDArray[np.float64]] = None,
    departure: Optional[NDArray[np.float64]] = None,
) -> Estimator:
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

    return Estimator(sf=Estimation(timeline, sf, se), cdf=Estimation(timeline, cdf, se))


def kaplan_meier(
    time: NDArray[np.float64],
    event: Optional[NDArray[np.float64]] = None,
    entry: Optional[NDArray[np.float64]] = None,
    departure: Optional[NDArray[np.float64]] = None,
) -> Estimator:
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

    return Estimator(sf=Estimation(timeline, sf, se))


def nelson_aalen(
    time: NDArray[np.float64],
    event: Optional[NDArray[np.float64]] = None,
    entry: Optional[NDArray[np.float64]] = None,
    departure: Optional[NDArray[np.float64]] = None,
) -> Estimator:
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
    death_set = np.zeros_like(timeline, dtype=np.int64)  # death at each timeline step

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

    return Estimator(chf=Estimation(timeline, chf, se))


def turnbull(
    time: float | NDArray[np.float64],
    event: Optional[NDArray[np.float64]] = None,
    entry: Optional[NDArray[np.float64]] = None,
    departure: Optional[NDArray[np.float64]] = None,
    tol: Optional[float] = 1e-4,
    lowmem: Optional[bool] = False,
) -> Estimator:
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
    lowmem
    tol
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
    if not lowmem:
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

        s = _turnbull_estimation_with_high_memory(
            lifetime_data,
            timeline_len,
            event_occurence,
            timeline_temp,
            tol=tol,
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
        s = _turnbull_estimation_with_low_memory(
            lifetime_data,
            timeline_temp,
            timeline_len,
            event_occurence,
            len_censored_data,
            tol=tol,
        )

    ind_del = np.where(timeline_temp == np.inf)
    sf = np.delete(s, ind_del)
    timeline = np.delete(timeline_temp, ind_del)

    return Estimator(sf=Estimation(timeline, sf))


def _turnbull_estimation_with_low_memory(
    lifetime_data: LifetimeData,
    timeline_temp,
    timeline_len,
    event_occurence,
    len_censored_data,
    tol: float = 1e-4,
) -> NDArray[np.float64]:

    d_tilde = np.histogram(
        np.searchsorted(timeline_temp, lifetime_data.complete.values),
        bins=range(timeline_len + 1),
    )[0][1:]
    s = np.linspace(1, 0, timeline_len)
    res = 1
    count = 1
    while res > tol:
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


def _turnbull_estimation_with_high_memory(
    lifetime_data: LifetimeData,
    timeline_len,
    event_occurence,
    timeline_temp,
    tol: float = 1e-4,
) -> NDArray[np.float64]:

    d_tilde = np.histogram(
        np.searchsorted(timeline_temp, lifetime_data.complete.values),
        bins=range(timeline_len + 1),
    )[0][1:]
    s = np.linspace(1, 0, timeline_len)
    res = 1
    count = 1

    while res > tol:
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
