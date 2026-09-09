Preventive age replacement policy
====================================

Instead of always waiting for failure, an asset can be replaced preventively once it
reaches a fixed age ``ar``, at a (usually lower) cost ``cp``; it is still replaced
at cost ``cf`` if it fails before reaching ``ar`` [1]_. Choosing ``ar``
trades off the risk of an expensive failure against replacing an asset that still had useful
life left.

**Assumptions**

* the asset ages, i.e. its hazard rate :math:`h` is increasing;
* it is replaced either preventively at age ``ar`` at cost ``cp``, or on failure before that
  age at cost ``cf`` > ``cp``;
* it is replaced by an identical asset over successive cycles (no obsolescence);

**Objective**: determine the replacement age ``ar`` minimizing the asymptotic cost per unit
of time.

>>> from relife.datasets import load_circuit_breaker
>>> from relife.lifetime_models import Weibull
>>> from relife.policies import AgeReplacementPolicy
>>> dataset = load_circuit_breaker()
>>> weibull = Weibull().fit(dataset["time"], event=dataset["event"], entry=dataset["entry"])
>>> policy = AgeReplacementPolicy(weibull)


The expected annual cost depends on the chosen replacement age. Costs and ages are vectorized
over assets, so a 1d ``ar`` evaluates several candidate ages in a single call, and three of
them are already enough to show there is a sweet spot:

>>> import numpy as np
>>> np.round(policy.asymptotic_expected_equivalent_annual_cost(ar=np.array([25., 40., 60.]), cf=1500., cp=400.), 2)
array([16.59, 12.08, 12.54])

Sweeping the whole range traces out a curve with a clear minimum:

.. plot::
    :context: close-figs

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from relife.datasets import load_circuit_breaker
    >>> from relife.lifetime_models import Weibull
    >>> from relife.policies import AgeReplacementPolicy
    >>> dataset = load_circuit_breaker()
    >>> weibull = Weibull().fit(dataset["time"], event=dataset["event"], entry=dataset["entry"])
    >>> policy = AgeReplacementPolicy(weibull)
    >>> ar_range = np.arange(10., 90., 5.)
    >>> costs = policy.asymptotic_expected_equivalent_annual_cost(ar=ar_range, cf=1500., cp=400.)
    >>> ar_star = policy.compute_optimal_ar(cf=1500., cp=400.)
    >>> _ = plt.plot(ar_range, costs, marker="o", label="expected annual cost")
    >>> _ = plt.axvline(float(ar_star), color="grey", linestyle="--", label="optimal ar")
    >>> _ = plt.xlabel("replacement age ar")
    >>> _ = plt.ylabel("expected annual cost")
    >>> _ = plt.legend()
    >>> plt.show()

Replacing too early (small ``ar``) wastes useful life and pays the preventive cost too
often; replacing too late drifts back towards the run-to-failure cost as more failures slip
through before the preventive replacement age is reached.

Rather than scanning candidate ages by hand, ``compute_optimal_ar`` locates the minimum of
this curve [2]_:

>>> ar_star = policy.compute_optimal_ar(cf=1500., cp=400.)
>>> round(float(ar_star), 2)
47.44
>>> round(float(policy.asymptotic_expected_equivalent_annual_cost(ar=ar_star, cf=1500., cp=400.)), 2)
11.69

Replacing preventively at age 47.4 costs about 11.69 per unit of time in the long run,
against 20.47 for :doc:`run_to_failure` on the same fitted model and failure cost.

The same shape shows up regardless of the underlying distribution; only how deep and how
sharp the minimum is changes:

.. figure:: /_static/figures/cost_ratio_optimal_age.png
    :alt: Ratio of the expected annual cost with preventive replacement over the run-to-failure cost, for a Weibull and a Gompertz model, each showing a minimum below 1 with the failure probability at the optimal age annotated.
    :width: 100%

    Expected annual cost of an age-replacement policy, divided by the run-to-failure cost, as
    a function of ``ar``. Both curves dip below 1 (preventive replacement wins) before rising
    back towards it. The annotated ``F(ar)`` is the probability of failing *before* reaching
    the optimal age: the fraction of assets that will still cost ``cf`` even under the
    optimal policy, since not every asset can be caught in time.

Projecting the replacements
------------------------------

A cost figure says nothing about the workload that produces it. On top of the four cost
methods shared by every policy (see :doc:`../going_further/from_process_to_policy`),
``AgeReplacementPolicy`` projects the
replacements themselves, year by year:

``annual_number_of_replacements(nb_years, *, ar, a0=None)``
    the expected number of replacements during each year, all causes together.

``annual_number_of_failures(nb_years, *, ar, a0=None)``
    the part of them caused by a failure. Subtract one from the other and you have the
    preventive workload.

Both return a timeline of whole years and one value per year, so they read directly as a
maintenance schedule.

>>> timeline, n_replacements = policy.annual_number_of_replacements(60, ar=47.44, a0=np.array([0., 10., 20., 30.]))
>>> timeline, n_failures = policy.annual_number_of_failures(60, ar=47.44, a0=np.array([0., 10., 20., 30.]))

These two methods are the year-by-year form of the counters ``expected_number_of_events`` and
``expected_number_of_preventive_renewals`` described in
:doc:`../going_further/reward_framework`, which report
the same information cumulated since time 0 over an arbitrary timeline rather than aggregated
per year.

Stopping after one cycle
--------------------------

Like :doc:`run_to_failure`, this policy has a one-cycle counterpart for decisions that only
concern the asset currently in service. ``OneCycleAgeReplacementPolicy`` charges ``cp`` if
the asset reaches ``ar`` and ``cf`` if it fails before, then stops; there is no renewal, so
the expected cost of the cycle is simply the two costs weighted by their probabilities:

>>> from relife.policies import OneCycleAgeReplacementPolicy
>>> one_cycle = OneCycleAgeReplacementPolicy(weibull)
>>> round(float(one_cycle.asymptotic_expected_net_present_value(ar=ar_star, cf=1500., cp=400.)), 2)
539.15

Because the criterion being minimized is not the same, the optimal age isn't either:

>>> one_cycle_ar_star = one_cycle.compute_optimal_ar(cf=1500., cp=400.)
>>> round(float(one_cycle_ar_star), 2)
43.46
>>> round(float(one_cycle.asymptotic_expected_equivalent_annual_cost(ar=one_cycle_ar_star, cf=1500., cp=400.)), 2)
12.77

Both optima sit in the same region: the lifetime model, not the horizon, does most of the
work. But the one-cycle optimum lands slightly earlier here, and its annualized cost (12.77)
is not directly comparable to the renewal policy's 11.69: the two numbers annualize over
different horizons (see :doc:`../going_further/from_process_to_policy`). Use the one-cycle
policies to rank options
for a single asset, and the renewal policies to plan a fleet over the long run.

.. [1] Mazzuchi, T. A., Van Noortwijk, J. M., & Kallen, M. J. (2007). Maintenance
    optimization. Encyclopedia of Statistics in Quality and Reliability, 1000-1008.
.. [2] Coolen-Schrijner, P., & Coolen, F. P. A. (2006). On optimality criteria for age
    replacement. Proceedings of the Institution of Mechanical Engineers, Part O: Journal
    of Risk and Reliability, 220(1), 21-29.
