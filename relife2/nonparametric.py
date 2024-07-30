"""
This module defines fundamental types of nonparametric functions used in survival analysis

Copyright (c) 2022, RTE (https://www.rte-france.com)
See AUTHORS.txt
SPDX-License-Identifier: Apache-2.0 (see LICENSE.txt)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike

from relife2.data import lifetime_factory_template
from relife2.utils.types import FloatArray


@dataclass
class Estimates:
    timeline: FloatArray
    values: FloatArray
    se: Optional[FloatArray] = None

    def __post_init__(self):
        if self.se is None:
            self.se = np.zeros_like(
                self.values
            )  # garder None/Nan efaire le changement de valeur au niveau du plot

        if self.timeline.shape != self.values.shape != self.se:
            raise ValueError("Incompatible timeline, values and se in Estimates")


class NonParametricEstimator(ABC):
    """_summary_"""

    def __init__(
        self,
        time: ArrayLike,
        entry: Optional[ArrayLike] = None,
        departure: Optional[ArrayLike] = None,
        lc_indicators: Optional[ArrayLike] = None,
        rc_indicators: Optional[ArrayLike] = None,
    ):

        self.observed_lifetimes, self.truncations = lifetime_factory_template(
            time, entry, departure, lc_indicators, rc_indicators
        )

        self._estimate()

    @property
    @abstractmethod
    def sf(self) -> Estimates:
        """"""

    @abstractmethod
    def _estimate(self) -> None:
        """_summary_

        Returns:
            Tuple[Estimates]: description
        """


class ECDF(NonParametricEstimator):
    """_summary_"""

    def _estimate(self) -> None:
        """_summary_

        Returns:
            FloatArray: _description_"""
        timeline, counts = np.unique(
            self.observed_lifetimes.rlc.values, return_counts=True
        )
        timeline = np.insert(timeline, 0, 0)
        cdf = np.insert(np.cumsum(counts), 0, 0) / np.sum(counts)
        sf = 1 - cdf
        se = np.sqrt(cdf * (1 - cdf) / len(self.observed_lifetimes.rlc.values))

        self._sf = Estimates(timeline, sf, se)
        self._cdf = Estimates(timeline, cdf, se)

    @property
    def sf(self) -> Estimates:
        return self._sf

    @property
    def cdf(self) -> Estimates:
        return self._cdf

    @property
    def timeline(self) -> FloatArray:
        """_summary_

        Returns:
            FloatArray: _description_"""
        return np.insert(np.unique(self.observed_lifetimes.rlc.values), 0, 0)


class KaplanMeier(NonParametricEstimator):
    """_summary_"""

    def _estimate(self) -> None:
        """_summary_

        Returns:
            FloatArray: _description_"""

        if len(self.observed_lifetimes.left_censored) > 0:
            raise ValueError("KaplanMeier does not take left censored lifetimes")
        timeline, timeline_indexes, counts = np.unique(
            self.observed_lifetimes.rlc.values, return_inverse=True, return_counts=True
        )
        death_set = np.zeros_like(timeline, int)  # death at each timeline step
        complete_observation_indic = np.zeros_like(
            self.observed_lifetimes.rlc.values
        )  # just creating an array to fill it next line
        complete_observation_indic[self.observed_lifetimes.complete.index] = 1
        np.add.at(death_set, timeline_indexes, complete_observation_indic.flatten())
        x_in = np.histogram(
            np.concatenate(
                (
                    self.truncations.left.values.flatten(),
                    np.array(
                        [
                            0
                            for _ in range(
                                len(self.observed_lifetimes.rlc.values)
                                - len(self.truncations.left.values)
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

    @property
    def sf(self) -> Estimates:
        return self._sf


class NelsonAalen(NonParametricEstimator):
    """_summary_"""

    def _estimate(self) -> Tuple[FloatArray]:
        """_summary_

        Returns:
            FloatArray: _description_"""
        if len(self.observed_lifetimes.left_censored) > 0:
            raise ValueError("NelsonAalen does not take left censored lifetimes")
        timeline, timeline_indexes, counts = np.unique(
            self.observed_lifetimes.rlc.values, return_inverse=True, return_counts=True
        )
        death_set = np.zeros_like(timeline, int)  # death at each timeline step
        complete_observation_indic = np.zeros_like(
            self.observed_lifetimes.rlc.values
        )  # just creating an array to fill it next line
        complete_observation_indic[self.observed_lifetimes.complete.index] = 1
        np.add.at(death_set, timeline_indexes, complete_observation_indic.flatten())
        x_in = np.histogram(
            np.concatenate(
                (
                    self.truncations.left.values.flatten(),
                    np.array(
                        [
                            0
                            for _ in range(
                                len(self.observed_lifetimes.rlc.values)
                                - len(self.truncations.left.values)
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

    @property
    def sf(self) -> Estimates:
        # TODO : verify following
        return np.exp(-self._chf)

    @property
    def chf(self) -> Estimates:
        return self._chf


class Turnbull(NonParametricEstimator):
    """
    BLABLABLA
    """

    def __init__(
        self,
        time: ArrayLike,
        entry: Optional[ArrayLike] = None,
        departure: Optional[ArrayLike] = None,
        lc_indicators: Optional[ArrayLike] = None,
        rc_indicators: Optional[ArrayLike] = None,
        tol: Optional[float] = 1e-4,
        lowmem: Optional[bool] = False,
    ):

        self.lowmem = lowmem
        self.tol = tol

        super().__init__(time, entry, departure, lc_indicators, rc_indicators)

        # TODO : qst => entry is not optional???

    def _estimate_with_low_memory(
        self, timeline_temp, timeline_len, event_occurence, len_censored_data
    ):

        d_tilde = np.histogram(
            np.searchsorted(timeline_temp, self.observed_lifetimes.complete.values),
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
            _unsorted_entry = self.truncations.left.values.flatten()

            y -= len(_unsorted_entry) - np.searchsorted(
                np.sort(_unsorted_entry), timeline_temp[1:], side="left"
            )
            s_updated = np.array(np.cumprod(1 - d / y))
            s_updated = np.insert(s_updated, 0, 1)
            res = max(abs(s - s_updated))
            s = s_updated
            count += 1
        return s

    def _estimate_with_high_memory(self, timeline_len, event_occurence, timeline_temp):

        d_tilde = np.histogram(
            np.searchsorted(timeline_temp, self.observed_lifetimes.complete.values),
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
            _unsorted_entry = self.truncations.left.values.flatten()
            y -= len(_unsorted_entry) - np.searchsorted(
                np.sort(_unsorted_entry), timeline_temp[1:], side="left"
            )
            s_updated = np.array(np.cumprod(1 - d / y))
            s_updated = np.insert(s_updated, 0, 1)
            res = max(abs(s - s_updated))
            s = s_updated
            count += 1
        return s

    def _estimate(self) -> Tuple[FloatArray]:
        """_summary_

        Returns:
            FloatArray: _description_"""

        timeline_temp = np.unique(
            np.insert(self.observed_lifetimes.interval_censored.values.flatten(), 0, 0)
        )
        timeline_len = len(timeline_temp)
        if not self.lowmem:
            event_occurence = (
                np.greater_equal.outer(
                    timeline_temp[:-1],
                    self.observed_lifetimes.interval_censored.values[
                        :, 0
                    ],  # or self.observed_lifetimes.interval_censored.values.T[0][i]
                )
                * np.less_equal.outer(
                    timeline_temp[1:],
                    self.observed_lifetimes.interval_censored.values[:, 1],
                )
            ).T

            s = self._estimate_with_high_memory(
                timeline_len, event_occurence, timeline_temp
            )

        else:
            len_censored_data = len(self.observed_lifetimes.interval_censored.values)
            event_occurence = []
            for i in range(len_censored_data):
                event_occurence.append(
                    np.where(
                        (
                            self.observed_lifetimes.interval_censored.values[:, 0][i]
                            <= timeline_temp[:-1]
                        )
                        & (
                            timeline_temp[1:]
                            <= self.observed_lifetimes.interval_censored.values[:, 1][i]
                        )
                        == True
                    )[0][[0, -1]]
                )
            event_occurence = np.array(event_occurence)
            s = self._estimate_with_low_memory(
                timeline_temp, timeline_len, event_occurence, len_censored_data
            )

        ind_del = np.where(timeline_temp == np.inf)
        sf = np.delete(s, ind_del)
        timeline = np.delete(timeline_temp, ind_del)
        self._sf = Estimates(timeline, sf)

    @property
    def sf(self) -> Estimates:
        return self._sf
