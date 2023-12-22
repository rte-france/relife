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
        raise NotImplementedError("did not yet adapt _estimate to handle 2d time data for KM and Nelson-A") # [ TODO ] 
    timeline, inv, counts = np.unique(
        data.time, return_inverse=True, return_counts=True
    )
    d = np.zeros_like(timeline, int)
    np.add.at(d, inv, data.event)
    x_in = np.histogram(data.entry, np.insert(timeline, 0, 0))[0]
    x_out = np.insert(counts[:-1], 0, 0)
    n = np.cumsum(x_in - x_out)
    return timeline, d, n


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
            self, time: np.ndarray, event: np.ndarray = None, entry: np.ndarray = None
    ) -> Turnbull:
        """[Aya : TODO / Complete code and doc] Fit the Turnbull estimator to interval censored lifetime data.

        Parameters
        ----------

        """
        _data = LifetimeData(time, event, entry) # on aura self.xl, self.xr, et self.entry aussi
        data = np.column_stack((_data.time, _data.entry))
        # TODO : now that i've defined data, continue (frm censorship and all) using self.sf and self.timeline
        # self.timeline, self.sf = Turnbull(data).values().T
        return self

def turnbull(data, tol=1e-4, lowmem=False):
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

    data = data[['L', 'U', 'T']].values
    censorship = (data[:, 0] < data[:, 1])
    tau = np.unique(np.insert(np.sort(np.unique(data[:, 0:2].flatten())), 0, 0))
    # tau = np.unique(np.append(np.insert(np.sort(np.unique(data[:, 0:2].flatten())), 0, 0),np.inf))
    k = len(tau)

    # Let alpha_{i,j} be 1 if the interval (tau_{j−1}, tau_{j}] is contained in the interval (Li, Ui]
    # and 0 otherwise. alpha_{i,j} indicates whether the event which occurs in the
    # interval (L_i, U_i] could have occurred at tau_j.

    data_censored = data[censorship == True]

    if not lowmem:
        alpha = (np.logical_not(np.less.outer(tau[:-1], data_censored[:, 0])) * np.logical_not(
            np.greater.outer(tau[1:], data_censored[:, 1]))).T
        print(sys.getsizeof(alpha))

    else:
        ###
        alpha_bis = []
        for i in range(data_censored.shape[0]):
            alpha_bis.append(
                np.where((data_censored[i, 0] <= tau[:-1]) & (tau[1:] <= data_censored[i, 1]) == True)[0][[0, -1]])
        alpha_bis = np.array(alpha_bis)
        print(sys.getsizeof(alpha_bis))
        ###

    # count number of exact survival times falling in the intervall (tau_{j-1} ; tau_j]
    exact_survival_times = data[censorship == False][:, 0]
    d_tilde = np.histogram(np.searchsorted(tau, exact_survival_times), bins=range(len(tau) + 1))[0][1:]

    # Survival function initialization
    S = np.linspace(1, 0, k)

    res = 1
    count = 1

    while res > tol:
        # Step 1 : estimation of probability of an event occuring at times tau_j:
        p = -np.diff(S)
        if np.sum(p == 0) > 0:  # prevents alpha * p.T from having a row of 0's which prevents d from updating
            p = np.where(p == 0, 1e-5 / np.sum(p == 0), p - 1e-5 / ((k - 1) - np.sum(p == 0)))

        # Step 2 + 2.5 : estimation of number of events which occured at time tau_j +
        # add number of exact survival times falling in (tau_{j-1} ; tau_j] to d

        if not lowmem:
            if np.any(alpha):
                alpha_p = alpha * p.T
                d = (alpha_p.T / alpha_p.sum(axis=1)).T.sum(axis=0)
                d += d_tilde
            else:
                d = d_tilde
        else:
            #####
            x = [p[alpha_bis[i, 0]:(alpha_bis[i, 1] + 1)] for i in range(alpha_bis.shape[0])]
            d = np.repeat(0, len(tau) - 1)

            for i in range(data_censored.shape[0]):
                d = d + (np.append(np.insert(x[i] / x[i].sum(), 0, np.repeat(0, alpha_bis[i][0])),
                                   np.repeat(0, len(tau) - alpha_bis[i][1] - 2)))
            d = d + d_tilde
            #####

        # Step 3 : estimation of number of individuals at risk at tau_j:
        y = np.cumsum(d[::-1])[::-1]
        y -= len(data[:, 2]) - np.searchsorted(np.sort(data[:, 2]), tau[1:], side='left')  # add truncation effect

        # Step 4 : Compute the Product-Limit estimator using the data found in Steps 2 and 3.
        S_updated = np.array(np.cumprod(1 - d / y))
        S_updated = np.insert(S_updated, 0, 1)
        res = max(abs(S - S_updated))
        S = S_updated
        count += 1
    ind_del = np.where(tau == np.inf)
    tau = np.delete(tau, ind_del)
    S = np.delete(S, ind_del)
    return pd.DataFrame({'tau': tau, 'S': S})
