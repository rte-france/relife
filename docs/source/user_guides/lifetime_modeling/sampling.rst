Sampling
=========

ReLife can generate synthetic data from any fitted or hand-specified model. This is useful to
check that an estimator recovers the parameters it is supposed to recover, to build test data
when real observations are scarce, or to run Monte-Carlo experiments on a maintenance policy.

Three levels of sampling are available:

- ``rvs`` on a lifetime model, which draws independent lifetimes;
- ``sample_lifetimes_from_renewal_process``, which draws *observations* as they would be
  collected in the field, that is with censoring and truncation;
- ``sample_process``, which draws whole trajectories of a stochastic process. Only renewal
  processes are covered here; see :doc:`../counting_processes/sampling` and
  :doc:`../deterioration_processes/sampling` for the other processes.

Everything below is deterministic given a ``seed``.

Sampling from a lifetime model
--------------------------------

``rvs`` draws lifetimes from the model. Its first argument is the shape of the output array.

>>> import numpy as np
>>> from relife.lifetime_models import Weibull
>>> weibull = Weibull(7, 0.05)
>>> np.round(weibull.rvs((3, 2), seed=3), 4)
array([[22.7413, 21.0705],
       [16.126 , 18.3196],
       [22.6144, 19.4971]])

Conditioning the drawn lifetimes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``apply_condition`` returns a new model conditioned by an initial age ``a0`` (the asset is
already ``a0`` years old and has survived so far, see
:doc:`censoring_and_truncation`) or by an age of replacement ``ar``
(the asset is preventively replaced at ``ar`` if it has not failed before).

``a0`` and ``ar`` must be broadcastable with the ``rvs`` size. Here two initial ages are given
as a column so that they broadcast against the four columns of the requested ``(2, 4)`` output:

>>> conditioned = weibull.apply_condition(a0=np.array([[5], [10]]))
>>> np.round(conditioned.rvs((2, 4), seed=3), 4)
array([[17.7413, 16.0706, 11.1266, 13.3199],
       [12.625 ,  9.523 ,  9.1718, 11.8237]])

The first row holds residual lifetimes of assets aged 5, the second row of assets aged 10.
Conditioning on ``ar`` instead truncates the draws from above:

>>> np.round(weibull.apply_condition(ar=np.array([10, 20])).rvs((4, 2), seed=3), 4)
array([[10.    , 20.    ],
       [10.    , 18.3196],
       [10.    , 19.4971],
       [10.    , 20.    ]])

Every value of the first column is exactly 10, because the mean of this Weibull is far above 10
and the replacement is almost always reached before failure:

>>> round(float(weibull.mean()), 4)
18.7088

Sampling lifetime data from a renewal process
-----------------------------------------------

``rvs`` returns clean, complete lifetimes. Real datasets are not like that: assets are observed
through a finite time window, so some are already in service when observation starts
(left truncation) and some are still alive when it ends (right censoring).
``sample_lifetimes_from_renewal_process`` reproduces this.

>>> from relife.sampling import sample_lifetimes_from_renewal_process
>>> lifetime_sample = sample_lifetimes_from_renewal_process(
...     weibull, 40, time_window=(20, 100), ar=22, seed=3
... )
>>> len(lifetime_sample.time)
217

40 independent renewal trajectories observed between ``t = 20`` and ``t = 100`` yield 217
lifetime observations. The result is a named tuple whose fields are exactly the arguments
``fit`` expects:

>>> lifetime_sample._fields
('time', 'event', 'entry')

40 observations are left-truncated, one per trajectory: the cycle that was already running when
observation started at ``t = 20``.

>>> int((lifetime_sample.entry > 0).sum())
40

157 of the 217 observations are actual failures. The remaining 60 are censored, either because
the trajectory was still running at ``t = 100`` or because the asset reached the age of
replacement ``ar = 22``.

>>> int(lifetime_sample.event.sum())
157

Because the fields match ``fit``, the sample can be unpacked directly to refit the model it came
from and check that the parameters are recovered:

>>> np.round(weibull.get_params(), 4)
array([7.  , 0.05])
>>> np.round(Weibull().fit(*lifetime_sample).get_params(), 4)
array([7.6131, 0.0493])

Sampling with covariates
~~~~~~~~~~~~~~~~~~~~~~~~~~

For a regression, ``covar`` must be given. It is currently not possible to pass a different
covariate value per observation in a single call, so the sample is built by looping over the
covariate values and concatenating the results.

>>> from relife.lifetime_models import ParametricProportionalHazard
>>> regression = ParametricProportionalHazard(weibull, coefficients=(2, 1.5))
>>> np.round(regression.get_params(), 4)
array([2.  , 1.5 , 7.  , 0.05])

The coefficients come first, then the parameters of the baseline Weibull. Positive coefficients
raise the hazard, so covariates shorten the expected lifetime:

>>> round(float(regression.mean(0.21, 0.18)), 4)
16.9526

Below, four combinations of the two covariates are used, laid out as a factorial design so that
both coefficients stay identifiable, with 200 trajectories each:

>>> covar_values = [(0.1, 0.1), (0.1, 0.6), (0.6, 0.1), (0.6, 0.6)]
>>> time, event, entry, z1, z2 = [], [], [], [], []
>>> for seed, (v1, v2) in enumerate(covar_values):
...     sample = sample_lifetimes_from_renewal_process(
...         regression, 200, time_window=(20, 100), covar=(v1, v2), seed=seed
...     )
...     time.append(sample.time)
...     event.append(sample.event)
...     entry.append(sample.entry)
...     z1.append(np.full(len(sample.time), v1))
...     z2.append(np.full(len(sample.time), v2))
>>> time, event, entry = np.concat(time), np.concat(event), np.concat(entry)
>>> z1, z2 = np.concat(z1), np.concat(z2)
>>> len(time)
4814

Fitting a fresh regression on the concatenated data recovers the true parameters
``(2, 1.5, 7, 0.05)``:

>>> fitted = ParametricProportionalHazard(Weibull()).fit(
...     time, covar=(z1, z2), event=event, entry=entry
... )
>>> np.round(fitted.get_params(), 4)
array([1.964 , 1.4236, 6.7452, 0.0488])

Sampling a stochastic process
-------------------------------

The functions above collapse trajectories into a flat lifetime dataset. To keep the trajectories
themselves, use ``sample_process``. It returns a named tuple with a shared ``timeline`` and one
row per realization:

- ``timeline``, the sorted union of the event dates of every realization;
- ``events``, a boolean array of shape ``(nb_samples, len(timeline))``, ``True`` where a
  realization has an event;
- ``preventive_renewals``, ``True`` where that event is a preventive replacement;
- ``rewards``, only filled for reward processes.

Renewal process
~~~~~~~~~~~~~~~~~

Below, 100 realizations of a renewal process are drawn over the time window ``(0, 75)``. Taking
the cumulative sum of ``events`` along the timeline gives the counting process of each
realization, whose average estimates the renewal function:

.. plot::
    :context: close-figs

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from relife.lifetime_models import Weibull
    >>> from relife.sampling import sample_process
    >>> from relife.stochastic_processes import RenewalProcess
    >>> weibull = Weibull(7, 0.05)
    >>> renewal_process = RenewalProcess(weibull)
    >>> sample = sample_process(renewal_process, 100, (0, 75), seed=10)
    >>> sample.timeline.shape
    (349,)
    >>> sample.events.shape
    (100, 349)
    >>> fig, ax = plt.subplots(figsize=(8, 6))
    >>> cumulative_events = sample.events.cumsum(axis=1)
    >>> for sample_id in range(10):
    ...     _ = ax.plot(sample.timeline, cumulative_events[sample_id], alpha=0.3, color="gray")
    >>> mean = cumulative_events.mean(axis=0)
    >>> std = cumulative_events.std(axis=0)
    >>> _ = ax.plot(sample.timeline, mean, color="red")
    >>> _ = ax.fill_between(sample.timeline, mean - std, mean + std, color="red", alpha=0.2)
    >>> _ = ax.set_xlabel("Time")
    >>> _ = ax.set_ylabel("Cumulative number of renewals")
    >>> plt.show()

The grey staircases are ten individual realizations, the red line their mean over the 100
realizations and the shaded band one standard deviation. The mean stays flat until the first
failures occur and then grows linearly with slope ``1 / weibull.mean()``, as
:doc:`renewal_theory` predicts.
