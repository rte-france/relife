from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from relife2.data import dataclass, lifetime_factory_template
from relife2.io import array_factory, preprocess_lifetime_data


def nearest_1dinterp(
    x: NDArray[np.float64], xp: NDArray[np.float64], yp: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Returns x nearest interpolation based on xp and yp data points
    xp has to be monotonically increasing

    Args:
        x (NDArray[np.float64]): 1d x coordinates to interpolate
        xp (NDArray[np.float64]): 1d known x coordinates
        yp (NDArray[np.float64]): 1d known y coordinates

    Returns:
        NDArray[np.float64]: interpolation values of x
    """
    spacing = np.diff(xp) / 2
    xp = xp + np.hstack([spacing, spacing[-1]])
    yp = np.concatenate([yp, yp[-1, None]])
    return yp[np.searchsorted(xp, x)]


@dataclass
class Estimates:
    """
    BLABLABLABLA
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


class NonParametricLifetimeEstimators(ABC):
    """_summary_"""

    def __init__(
        self,
    ):
        self.estimations = {}

    @abstractmethod
    def fit(
        self,
        time: NDArray[np.float64],
        event: Optional[NDArray[np.float64]] = None,
        entry: Optional[NDArray[np.float64]] = None,
        departure: Optional[NDArray[np.float64]] = None,
    ) -> Estimates:
        """_summary_

        Returns:
            Tuple[Estimates]: description
        """


class ECDF(NonParametricLifetimeEstimators):
    """_summary_"""

    def fit(
        self,
        time: NDArray[np.float64],
        event: Optional[NDArray[np.float64]] = None,
        entry: Optional[NDArray[np.float64]] = None,
        departure: Optional[NDArray[np.float64]] = None,
    ) -> None:
        """_summary_

        Returns:
            FloatArray: _description_"""

        time, event, entry, departure, _ = preprocess_lifetime_data(
            time, event, entry, departure
        )
        observed_lifetimes, truncations = lifetime_factory_template(
            time,
            event,
            entry,
            departure,
        )

        timeline, counts = np.unique(observed_lifetimes.rlc.values, return_counts=True)
        timeline = np.insert(timeline, 0, 0)
        cdf = np.insert(np.cumsum(counts), 0, 0) / np.sum(counts)
        sf = 1 - cdf
        se = np.sqrt(cdf * (1 - cdf) / len(observed_lifetimes.rlc.values))

        self.estimations["sf"] = Estimates(timeline, sf, se)
        self.estimations["cdf"] = Estimates(timeline, cdf, se)

    def sf(self, t: NDArray[np.float64]) -> NDArray[np.float64]:
        if "sf" not in self.estimations.keys():
            raise KeyError("sf values not yet estimated. First run ECDF.estimate(...)")
        t = array_factory(t)
        return nearest_1dinterp(
            t, self.estimations["sf"].timeline, self.estimations["sf"].values
        )

    def cdf(self, t: NDArray[np.float64]) -> NDArray[np.float64]:
        if "cdf" not in self.estimations.keys():
            raise KeyError("cdf values not yet estimated. First run ECDF.estimate(...)")
        t = array_factory(t)
        return nearest_1dinterp(
            t, self.estimations["cdf"].timeline, self.estimations["cdf"].values
        )


class KaplanMeier(NonParametricLifetimeEstimators):
    """_summary_"""

    def fit(
        self,
        time: NDArray[np.float64],
        event: Optional[NDArray[np.float64]] = None,
        entry: Optional[NDArray[np.float64]] = None,
        departure: Optional[NDArray[np.float64]] = None,
    ) -> None:
        """_summary_

        Returns:
            FloatArray: _description_"""

        time, event, entry, departure, _ = preprocess_lifetime_data(
            time, event, entry, departure
        )
        observed_lifetimes, truncations = lifetime_factory_template(
            time,
            event,
            entry,
            departure,
        )

        if len(observed_lifetimes.left_censored) > 0:
            raise ValueError("KaplanMeier does not take left censored lifetimes")
        timeline, timeline_indexes, counts = np.unique(
            observed_lifetimes.rlc.values, return_inverse=True, return_counts=True
        )
        death_set = np.zeros_like(timeline, int)  # death at each timeline step
        complete_observation_indic = np.zeros_like(
            observed_lifetimes.rlc.values
        )  # just creating an array to fill it next line
        complete_observation_indic[observed_lifetimes.complete.ids] = 1
        np.add.at(death_set, timeline_indexes, complete_observation_indic.flatten())
        x_in = np.histogram(
            np.concatenate(
                (
                    truncations.left.values.flatten(),
                    np.array(
                        [
                            0
                            for _ in range(
                                len(observed_lifetimes.rlc.values)
                                - len(truncations.left.values)
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
        self.estimations["sf"] = Estimates(timeline, sf, se)

    def sf(self, t: NDArray[np.float64]) -> NDArray[np.float64]:
        if "sf" not in self.estimations.keys():
            raise KeyError(
                "sf values not yet estimated. First run KaplanMeier.estimate(...)"
            )
        t = array_factory(t)
        return nearest_1dinterp(
            t, self.estimations["sf"].timeline, self.estimations["sf"].values
        )


class NelsonAalen(NonParametricLifetimeEstimators):
    """_summary_"""

    def fit(
        self,
        time: NDArray[np.float64],
        event: Optional[NDArray[np.float64]] = None,
        entry: Optional[NDArray[np.float64]] = None,
        departure: Optional[NDArray[np.float64]] = None,
    ) -> None:
        """_summary_

        Returns:
            FloatArray: _description_"""

        time, event, entry, departure, _ = preprocess_lifetime_data(
            time, event, entry, departure
        )
        observed_lifetimes, truncations = lifetime_factory_template(
            time,
            event,
            entry,
            departure,
        )
        if len(observed_lifetimes.left_censored) > 0:
            raise ValueError("NelsonAalen does not take left censored lifetimes")
        timeline, timeline_indexes, counts = np.unique(
            observed_lifetimes.rlc.values, return_inverse=True, return_counts=True
        )
        death_set = np.zeros_like(timeline, int)  # death at each timeline step
        complete_observation_indic = np.zeros_like(
            observed_lifetimes.rlc.values
        )  # just creating an array to fill it next line
        complete_observation_indic[observed_lifetimes.complete.ids] = 1
        np.add.at(death_set, timeline_indexes, complete_observation_indic.flatten())
        x_in = np.histogram(
            np.concatenate(
                (
                    truncations.left.values.flatten(),
                    np.array(
                        [
                            0
                            for _ in range(
                                len(observed_lifetimes.rlc.values)
                                - len(truncations.left.values)
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
        self.estimations["chf"] = Estimates(timeline, chf, se)

    def chf(self, t: NDArray[np.float64]) -> NDArray[np.float64]:
        if "chf" not in self.estimations.keys():
            raise KeyError(
                "chf values not yet estimated. First run NelsonAalen.estimate(...)"
            )
        t = array_factory(t)
        return nearest_1dinterp(
            t, self.estimations["chf"].timeline, self.estimations["chf"].values
        )


class Turnbull(NonParametricLifetimeEstimators):
    """
    BLABLABLA
    """

    def __init__(
        self,
        tol: Optional[float] = 1e-4,
        lowmem: Optional[bool] = False,
    ):
        super().__init__()
        self.tol = tol
        self.lowmem = lowmem

    def fit(
        self,
        time: NDArray[np.float64],
        event: Optional[NDArray[np.float64]] = None,
        entry: Optional[NDArray[np.float64]] = None,
        departure: Optional[NDArray[np.float64]] = None,
    ) -> None:
        """_summary_

        Returns:
            FloatArray: _description_"""

        time, event, entry, departure, _ = preprocess_lifetime_data(
            time, event, entry, departure
        )
        observed_lifetimes, truncations = lifetime_factory_template(
            time,
            event,
            entry,
            departure,
        )
        timeline_temp = np.unique(
            np.insert(observed_lifetimes.interval_censored.values.flatten(), 0, 0)
        )
        timeline_len = len(timeline_temp)
        if not self.lowmem:
            event_occurence = (
                np.greater_equal.outer(
                    timeline_temp[:-1],
                    observed_lifetimes.interval_censored.values[
                        :, 0
                    ],  # or self.observed_lifetimes.interval_censored.values.T[0][i]
                )
                * np.less_equal.outer(
                    timeline_temp[1:],
                    observed_lifetimes.interval_censored.values[:, 1],
                )
            ).T

            s = self._estimate_with_high_memory(
                observed_lifetimes,
                truncations,
                timeline_len,
                event_occurence,
                timeline_temp,
            )

        else:
            len_censored_data = len(observed_lifetimes.interval_censored.values)
            event_occurence = []
            for i in range(len_censored_data):
                event_occurence.append(
                    np.where(
                        (
                            observed_lifetimes.interval_censored.values[:, 0][i]
                            <= timeline_temp[:-1]
                        )
                        & (
                            timeline_temp[1:]
                            <= observed_lifetimes.interval_censored.values[:, 1][i]
                        )
                        == True
                    )[0][[0, -1]]
                )
            event_occurence = np.array(event_occurence)
            s = self._estimate_with_low_memory(
                observed_lifetimes,
                truncations,
                timeline_temp,
                timeline_len,
                event_occurence,
                len_censored_data,
            )

        ind_del = np.where(timeline_temp == np.inf)
        sf = np.delete(s, ind_del)
        timeline = np.delete(timeline_temp, ind_del)
        self.estimations["sf"] = Estimates(timeline, sf)

    def _estimate_with_low_memory(
        self,
        observed_lifetimes,
        truncations,
        timeline_temp,
        timeline_len,
        event_occurence,
        len_censored_data,
    ):

        d_tilde = np.histogram(
            np.searchsorted(timeline_temp, observed_lifetimes.complete.values),
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
                d = d + (
                    np.append(
                        np.insert(
                            x[i] / x[i].sum(), 0, np.repeat(0, event_occurence[i][0])
                        ),
                        np.repeat(0, timeline_len - event_occurence[i][1] - 2),
                    )
                )
            d = d + d_tilde
            y = np.cumsum(d[::-1])[::-1]
            _unsorted_entry = truncations.left.values.flatten()

            y -= len(_unsorted_entry) - np.searchsorted(
                np.sort(_unsorted_entry), timeline_temp[1:], side="left"
            )
            s_updated = np.array(np.cumprod(1 - d / y))
            s_updated = np.insert(s_updated, 0, 1)
            res = max(abs(s - s_updated))
            s = s_updated
            count += 1
        return s

    def _estimate_with_high_memory(
        self,
        observed_lifetimes,
        truncations,
        timeline_len,
        event_occurence,
        timeline_temp,
    ):

        d_tilde = np.histogram(
            np.searchsorted(timeline_temp, observed_lifetimes.complete.values),
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
            _unsorted_entry = truncations.left.values.flatten()
            y -= len(_unsorted_entry) - np.searchsorted(
                np.sort(_unsorted_entry), timeline_temp[1:], side="left"
            )
            s_updated = np.array(np.cumprod(1 - d / y))
            s_updated = np.insert(s_updated, 0, 1)
            res = max(abs(s - s_updated))
            s = s_updated
            count += 1
        return s

    def sf(self, t: NDArray[np.float64]) -> NDArray[np.float64]:
        if "sf" not in self.estimations.keys():
            raise KeyError(
                "sf values not yet estimated. First run Turnbull.estimate(...)"
            )
        t = array_factory(t)
        return nearest_1dinterp(
            t, self.estimations["sf"].timeline, self.estimations["sf"].values
        )
