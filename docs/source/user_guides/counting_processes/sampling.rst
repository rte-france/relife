Sampling
=========

``sample_process`` draws whole trajectories of a non-homogeneous Poisson process. It returns
the named tuple described in :doc:`../lifetime_modeling/sampling`: a shared ``timeline`` and
one row per realization.

The call takes an age of replacement through ``ar``. A repair is minimal here: it does
not reset the asset, so failures pile up much faster and the timeline is far denser than for a
renewal process. Only the preventive replacements at ``ar = 30`` bring the asset back to a new
state, which shows up as the plateaus at ``t = 30``, ``60`` and ``90``:

.. plot::
    :context: close-figs

    >>> import matplotlib.pyplot as plt
    >>> from relife.lifetime_models import Weibull
    >>> from relife.sampling import sample_process
    >>> from relife.stochastic_processes import NonHomogeneousPoissonProcess
    >>> weibull = Weibull(7, 0.05)
    >>> nhpp = NonHomogeneousPoissonProcess(weibull)
    >>> sample = sample_process(nhpp, 100, (0, 100), ar=30, seed=10)
    >>> int(sample.events.sum())
    5182
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
