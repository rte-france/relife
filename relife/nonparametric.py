"""Non-parametric estimator for survival analysis."""

# Copyright (c) 2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# SPDX-License-Identifier: Apache-2.0 (see LICENSE.txt)
# This file is part of ReLife, an open source Python library for asset
# management based on reliability theory and lifetime data analysis.

from __future__ import annotations
from typing import Tuple
import numpy as np

from .data import LifetimeData
from .utils import plot


def _estimate(data: LifetimeData) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Counting failures and assets at risks

    Counts the number of failures and assets at risk sorted along the timeline
    of unique times, needed to compute the Kaplan-Meier and Nelson-Aalen
    estimators.

    Parameters
    ----------
    data : LifetimeData
        The lifetime data.

    Returns
    -------
    Tuple[1D array, 1D array, 1D array]
        timeline : sorted unique time-to-event.
        d : number of failures.
        n : number of assets at risk

    Raises
    ------
    ValueError
        If `data.event` type is not 0 or 1.

    Notes
    -----
    For :math:`N` the unique times in timeline:

    .. math::

        \begin{align*}
            d_j &= \sum_{i=1}^N \delta_i \mathbbm{1}_{\{t_j = t_i\}} \\ n_j &=
            \sum_{i=1}^N \mathbbm{1}_{\{t_j \leq t_i\}}
        \end{align*}

    """
    if not np.all(np.isin(data.event, [0, 1])):
        raise ValueError("event values must be in [0,1]")
    if len(data.time.shape) != 1: 
        raise NotImplementedError("did not yet adapt _estimate to handle 2d time data for KM and Nelson-A") # [ TODO ] implement it
    timeline, inv, counts = np.unique(
        data.time, return_inverse=True, return_counts=True
    )
    d = np.zeros_like(timeline, int)
    np.add.at(d, inv, data.event) # death at each step
    x_in = np.histogram(data.entry, np.insert(timeline, 0, 0))[0]
    x_out = np.insert(counts[:-1], 0, 0)
    n = np.cumsum(x_in - x_out) # number of assets at risk at each step
    return timeline, d, n

def _turnbull_estimate(data: LifetimeData, tol=1e-4, lowmem=False):
    """Computation of the Turnbull estimator on interval censored data.

                Parameters
                ----------
                data : data set of generalized interval censored measurements (can  be left, right, interval, exact)
                tol : stopping criteria for convergence. Algorithm stops when the infinite norm of the difference
                between the current and updated estimates for the survival function is less than tol.

                Returns
                -------
                Estimates of the survival function
    """

    # PRECEDENT approach : np.sort redondant et complexité  O(3nlog(n)) + O(n) (appel 2 fois np.unique et insert )
    # tau = np.unique(np.insert(np.sort(np.unique(data[:, 0:2].flatten())), 0, 0))
    
    tau = np.unique(data.time.flatten()) # flattens les intervalles et les tri et insère 0 au début
    if tau[0] != 0 :
        tau = np.insert(tau, 0, 0)
    k = len(tau)
    data_censored = LifetimeData(data.time[data.event == 0], entry = data.entry[data.event == 0 ])

    if not lowmem:

        alpha = (np.greater_equal.outer(tau[:-1], data_censored.xl ) *  # replaced np.logical_not(np.less.outer) by np.greater_equal.outer
            np.less_equal.outer(tau[1:], data_censored.xr)).T # replaced np.logical_not(np.greater.outer) by np.less_equal.outer
        # points of the timeline (tau) that are included in the interval (for each interval)
            # so for each interval, we have a vector of len(tau) of F and T, T if the point is included in the interval
    else:
        alpha_bis = []
        for i in range(data_censored.size): 
            alpha_bis.append(
                np.where((data_censored.xl[i] <= tau[:-1]) & (tau[1:] <= data_censored.xr[i]) == True)[0][[0, -1]])
        alpha_bis = np.array(alpha_bis)

    exact_survival_times = data.time[data.event == 1][:, 0] 
    d_tilde = np.histogram(np.searchsorted(tau, exact_survival_times), bins=range(k + 1))[0][1:] # TODO
    print(np.histogram(np.searchsorted(tau, exact_survival_times), bins=range(k + 1)))
    S = np.linspace(1, 0, k)
    res = 1
    count = 1

    while res > tol:
        p = -np.diff(S) # écart entre les points de S (survival function) => proba of an event occuring at 
        if np.sum(p == 0) > 0 : 
            p = np.where(p == 0, 1e-5 / np.sum(p == 0), p - 1e-5 / ((k - 1) - np.sum(p == 0))) # remplace 0 par 1e-5 (et enlève cette quantité des autres proba pr sommer à 1)
        
        if not lowmem:
            if np.any(alpha):
                alpha_p = alpha * p.T
                d = (alpha_p.T / alpha_p.sum(axis=1)).T.sum(axis=0)
                d += d_tilde
            else:
                d = d_tilde
        else:
            x = [p[alpha_bis[i, 0]:(alpha_bis[i, 1] + 1)] for i in range(alpha_bis.shape[0])]
            d = np.repeat(0, k - 1)
            for i in range(data_censored.time.shape[0]):
                d = d + (np.append(np.insert(x[i] / x[i].sum(), 0, np.repeat(0, alpha_bis[i][0])),
                                   np.repeat(0, k - alpha_bis[i][1] - 2)))
            d = d + d_tilde

        y = np.cumsum(d[::-1])[::-1]
        print(y)
        y -= len(data.entry) - np.searchsorted(np.sort(data.entry), tau[1:], side='left')  
        print(y)
        S_updated = np.array(np.cumprod(1 - d / y))
        S_updated = np.insert(S_updated, 0, 1)
        res = max(abs(S - S_updated))
        S = S_updated
        count += 1

    ind_del = np.where(tau == np.inf)
    timeline = np.delete(tau, ind_del)
    s = np.delete(S, ind_del)
    
    return timeline, s

class ECDF:
    """Empirical Cumulative Distribution Function."""

    def fit(self, time: np.ndarray) -> ECDF:
        """Fit the empirical cumuative distribution function.

        Parameters
        ----------
        time : 1D array
            Failure times.

        Returns
        -------
        ECDF
            Return the fitted empirical estimate as the current object.

        """
        if len(time.shape) != 1: 
            raise ValueError("ECDF handles only 1D format for time") 
        data = LifetimeData(time)
        self.n_samples = data.size
        x1, n1 = np.unique(data.time, return_counts=True)
        self.timeline = np.insert(x1, 0, 0)
        self.cdf = np.insert(np.cumsum(n1), 0, 0) / np.sum(n1)
        self.sf = 1 - self.cdf
        self.se = np.sqrt(self.cdf * (1 - self.cdf) / self.n_samples)
        return self

    def plot(
        self, alpha_ci: float = 0.05, fname: str = "cdf", **kwargs: np.ndarray
    ) -> None:
        r"""Plot the empirical cumulative distribution function.

        Parameters
        ----------
        alpha_ci : float, optional
            :math:`\alpha`-value to define the :math:`100(1-\alpha)\%`
            confidence interval, by default 0.05 corresponding to the 95\%
            confidence interval. If set to None or if the model has not been
            fitted, no confidence interval is plotted.

        fname : str, optional
            Name of the function to be plotted, by default 'cdf'. Should be one
            of:

                - 'sf' : survival function,
                - 'cdf': cumulative distribution function.

        **kwargs :
            Extra arguments to specify the plot properties (see
            matplotlib.pyplot.plot documentation).

        Raises
        ------
        ValueError
            If `fname` value is not 'sf' or 'cdf'.

        """
        flist = ["sf", "cdf"]
        if fname in flist:
            y = getattr(self, fname)
        else:
            raise ValueError(
                "Function name '{}' is not supported for plotting, 'fname' must be in {}".format(
                    fname, flist
                )
            )
        label = "ECDF"
        plot(
            self.timeline,
            y,
            self.se,
            alpha_ci,
            bounds=(0, 1),
            label=label,
            drawstyle="steps-post",
            **kwargs,
        )


class KaplanMeier:
    r"""Kaplan-Meier Estimator.

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
        self, time: np.ndarray, event: np.ndarray = None, entry: np.ndarray = None
    ) -> KaplanMeier:
        """Fit the Kaplan-Meier estimator to a time, event and entry arrays.

        Parameters
        ----------
        time : 1D array
            Array of time-to-event or durations.
        event : 1D array, optional
            Array of event types coded as follows:

            - 0 if observation ends before the event has occurred (right censoring)
            - 1 if the event has occured
            - 2 if observation starts after the event has occurred (left censoring)

            by default the event has occured for each asset.
        entry : 1D array, optional
            Array of delayed entry times (left truncation),
            by default None.

        Returns
        -------
        KaplanMeier
            Return the fitted Kaplan-Meier estimator as the current object.

        """
        data = LifetimeData(time, event, entry)
        timeline, d, n = _estimate(data)
        # [1] eq (3.2.2)
        s = np.cumprod(1 - d / n)
        with np.errstate(divide="ignore"):
            # [1] eq (3.2.3)
            var = s**2 * np.cumsum(np.where(n > d, d / (n * (n - d)), 0))
        self.timeline = np.insert(timeline, 0, 0)
        self.sf = np.insert(s, 0, 1)
        self.se = np.sqrt(np.insert(var, 0, 0))
        return self

    def plot(self, alpha_ci: float = 0.05, **kwargs) -> None:
        r"""Plot the Kaplan-Meier estimator of the survival function.

        Parameters
        ----------
        alpha_ci : float, optional
            :math:`\alpha`-value to define the :math:`100(1-\alpha)\%`
            confidence interval, by default 0.05 corresponding to the 95\%
            confidence interval. If set to None or if the model has not been
            fitted, no confidence interval is plotted.

        **kwargs :
            Extra arguments to specify the plot properties (see
            matplotlib.pyplot.plot documentation).
        """
        label = kwargs.pop("label", "Kaplan-Meier")
        plot(
            self.timeline,
            self.sf,
            self.se,
            alpha_ci,
            bounds=(0, 1),
            label=label,
            drawstyle="steps-post",
            **kwargs,
        )


class NelsonAalen:
    r"""Nelson-Aalen Estimator.

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

    def fit(
        self, time: np.ndarray, event: np.ndarray = None, entry: np.ndarray = None
    ) -> NelsonAalen:
        """Fit the Nelson-Aalen estimator to lifetime data.

        Parameters
        ----------
        time : 1D array
            Array of time-to-event or durations.
        event : 1D array, optional
            Array of event types coded as follows:

            - 0 if observation ends before the event has occurred (right censoring)
            - 1 if the event has occured
            - 2 if observation starts after the event has occurred (left censoring)

            by default the event has occured for each asset.
        entry : 1D array, optional
            Array of delayed entry times (left truncation),
            by default None.

        Returns
        -------
        NelsonAalen
            The fitted Nelson-Aalen estimator as the current object.

        """
        data = LifetimeData(time, event, entry)
        timeline, d, n = _estimate(data)
        # [1] eq (3.2.13)
        s = np.cumsum(d / n)
        # [1] eq (3.2.15)
        var = np.cumsum(d / np.where(n == 0, 1, n**2))
        self.timeline = np.insert(timeline, 0, 0)
        self.chf = np.insert(s, 0, 0)
        self.se = np.sqrt(np.insert(var, 0, 0))
        return self

    def plot(self, alpha_ci: float = 0.05, **kwargs: np.ndarray) -> None:
        r"""Plot the Nelson-Aalen estimator of the cumulative hazard function.

        Parameters
        ----------
        alpha_ci : float, optional
            :math:`\alpha`-value to define the :math:`100(1-\alpha)\%`
            confidence interval, by default 0.05 corresponding to the 95\%
            confidence interval. If set to None or if the model has not been
            fitted, no confidence interval is plotted.

        **kwargs :
            Extra arguments to specify the plot properties (see
            matplotlib.pyplot.plot documentation).

        """
        label = kwargs.pop("label", "Nelson-Aalen")
        plot(
            self.timeline,
            self.chf,
            self.se,
            alpha_ci,
            bounds=(0, np.inf),
            label=label,
            drawstyle="steps-post",
            **kwargs,
        )

class Turnbull:
    r"""[Aya : TODO Complete code and doc] Turnbull Estimator.
    
    Compute the non-parametric Turnbull estimator of the survival function 
    from interval censored lifetime data.  
    """

    def fit(
            self, time: np.ndarray, entry: np.ndarray = None, tol: float = 1e-4,
              lowmem: bool = False,
    ) -> Turnbull:
        """[Aya : TODO / Complete code and doc] Fit the Turnbull estimator to interval censored lifetime data.

        Parameters
        ----------

        """
        
        data = LifetimeData(time, entry = entry) 
        if data._time.IC.size + data._time.IC.size == 0 :
            # self = KaplanMeier().fit(data.time) 
            print("implement kaplan_meier estimator in case no LC and IC -> what data structure to use ? 2d not taken into account in KM estimete : should it take it, using the 1D rep of 2D ? ") # TODO 

        timeline, s = _turnbull_estimate(data, tol, lowmem)
        self.timeline = timeline
        self.sf = s
        return self
    
    def plot(self, alpha_ci: float = 0.05, **kwargs: np.ndarray) -> None:
        r"""[Aya : TODO / Complete code and doc] Plot the Turnbull estimator of the survival function.

        Parameters
        ----------
        alpha_ci : float, optional
            :math:`\alpha`-value to define the :math:`100(1-\alpha)\%`
            confidence interval, by default 0.05 corresponding to the 95\%
            confidence interval. If set to None or if the model has not been
            fitted, no confidence interval is plotted.

        **kwargs :
            Extra arguments to specify the plot properties (see
            matplotlib.pyplot.plot documentation).

        """
        label = kwargs.pop("label", "Turnbull")
        plot(
            self.timeline,
            self.sf,
            None,
            alpha_ci,
            bounds=(0, np.inf),
            label=label,
            drawstyle="steps-post",
            **kwargs,
        )
