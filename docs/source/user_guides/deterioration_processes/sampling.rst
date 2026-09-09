Sampling
=========

``sample_process`` draws whole trajectories of a Kijima process. It returns the named tuple
described in :doc:`../lifetime_modeling/sampling`: a shared ``timeline`` and one row per
realization.

Kijima processes sit between the two extremes of a renewal process
(:doc:`../lifetime_modeling/sampling`) and a non-homogeneous Poisson process
(:doc:`../counting_processes/sampling`): a repair removes part of the accumulated damage,
controlled by the rejuvenation parameter ``q``. Kijima 1 applies ``q`` to the last increment
of virtual age only:

.. plot::
    :context: close-figs

    >>> import matplotlib.pyplot as plt
    >>> from relife.lifetime_models import Weibull
    >>> from relife.sampling import sample_process
    >>> from relife.stochastic_processes import Kijima1Process
    >>> weibull = Weibull(7, 0.05)
    >>> kijima_1 = Kijima1Process(weibull, q=0.7)
    >>> sample = sample_process(kijima_1, 100, (0, 100), ar=25, seed=10)
    >>> int(sample.events.sum())
    653
    >>> fig, ax = plt.subplots(figsize=(8, 6))
    >>> cumulative_events = sample.events.cumsum(axis=1)
    >>> for sample_id in range(10):
    ...     _ = ax.plot(sample.timeline, cumulative_events[sample_id], alpha=0.3, color="gray")
    >>> mean = cumulative_events.mean(axis=0)
    >>> std = cumulative_events.std(axis=0)
    >>> _ = ax.plot(sample.timeline, mean, color="red")
    >>> _ = ax.fill_between(sample.timeline, mean - std, mean + std, color="red", alpha=0.2)
    >>> _ = ax.set_xlabel("Time")
    >>> _ = ax.set_ylabel("Cumulative number of failures")
    >>> plt.show()

Kijima 2 applies ``q`` to the whole virtual age, so the asset is rejuvenated more aggressively
and fewer events accumulate over the same window:

.. plot::
    :context: close-figs

    >>> from relife.stochastic_processes import Kijima2Process
    >>> kijima_2 = Kijima2Process(weibull, q=0.7)
    >>> sample = sample_process(kijima_2, 100, (0, 100), ar=25, seed=10)
    >>> int(sample.events.sum())
    612
    >>> fig, ax = plt.subplots(figsize=(8, 6))
    >>> cumulative_events = sample.events.cumsum(axis=1)
    >>> for sample_id in range(10):
    ...     _ = ax.plot(sample.timeline, cumulative_events[sample_id], alpha=0.3, color="gray")
    >>> mean = cumulative_events.mean(axis=0)
    >>> std = cumulative_events.std(axis=0)
    >>> _ = ax.plot(sample.timeline, mean, color="red")
    >>> _ = ax.fill_between(sample.timeline, mean - std, mean + std, color="red", alpha=0.2)
    >>> _ = ax.set_xlabel("Time")
    >>> _ = ax.set_ylabel("Cumulative number of failures")
    >>> plt.show()
