The reward and discounting framework
=======================================

:doc:`../lifetime_modeling/run_to_failure` and
:doc:`../lifetime_modeling/preventive_age_replacement` both reduce to the same question:
attach a cost to each renewal of the
:doc:`renewal process <../lifetime_modeling/renewal_theory>`, and work out
what that comes to per unit of time. 

``RenewalRewardProcess`` is where that happens; the
policies are wrappers over it. It takes a lifetime model, and two further ingredients
supplied as arguments to its methods: a **reward** (how much does *this* renewal cost) and a
**discounting rate** (how much is a cost incurred later worth today).

>>> from relife.datasets import load_circuit_breaker
>>> from relife.lifetime_models import Weibull
>>> from relife.stochastic_processes import RenewalRewardProcess
>>> dataset = load_circuit_breaker()
>>> weibull = Weibull().fit(dataset["time"], event=dataset["event"], entry=dataset["entry"])
>>> process = RenewalRewardProcess(weibull)

Rewards
--------

A reward answers "what does this renewal cost, given how long the asset lasted". It isn't a
separate object to build: you describe it with the cost keyword arguments accepted by every
method of the process.

Passing ``cf`` alone means every renewal is a failure, costing ``cf`` whatever its duration:

>>> round(float(process.asymptotic_expected_equivalent_annual_worth(cf=1500.)), 2)
20.47

Passing ``cf``, ``cp`` and ``ar`` together instead makes the cost depend on how long the
asset actually lasted: ``cf`` if it failed before reaching the replacement age ``ar``, ``cp``
if it reached that age and was replaced preventively.

>>> round(float(process.asymptotic_expected_equivalent_annual_worth(cf=1500., cp=400., ar=47.44)), 2)
11.69

The switch between the two costs can be seen by pushing ``ar`` far beyond the range of
plausible lifetimes, so that virtually no asset ever reaches the replacement age and ``cp``
is never paid:

>>> round(float(process.asymptotic_expected_equivalent_annual_worth(cf=1500., cp=400., ar=200.)), 6)
20.474811

The age-replacement reward has degenerated into the run-to-failure one. ``cp`` and ``ar``
describe a single decision and only make sense together, so setting one without the other is
impossible.

First-cycle costs
~~~~~~~~~~~~~~~~~~~

In a *delayed* process (one whose first asset does not start new, either because
``first_lifetime_model`` differs from ``lifetime_model`` or because an initial age ``a0`` is
given), the first renewal can be given its own costs through ``cf1`` and ``cp1``. Starting
from assets already 20 time units old, and making that first replacement twice as expensive
as the following ones:

>>> import numpy as np
>>> timeline, z = process.expected_total_reward(100., 5, cf=1500., a0=20.)
>>> np.round(z, 2)
array([   0.  ,  150.95,  662.81, 1328.36, 1788.86])
>>> timeline, z1 = process.expected_total_reward(100., 5, cf=1500., cf1=3000., a0=20.)
>>> np.round(z1, 2)
array([   0.  ,  301.4 , 1315.48, 2578.88, 3268.35])

These two arguments only affect the first cycle, so they leave the long-run rate untouched
and matter on a finite horizon only. Note that they are honoured for a delayed process only:
on a non-delayed one there is nothing to distinguish the first renewal from the others, and
``cf1`` is ignored.

Discounting
------------

A cost incurred later is worth less than the same cost incurred now. ReLife applies
exponential discounting at a rate given as ``discounting_rate`` to any method. Two factors
follow from that rate: one shrinks a single cost paid at a given date down to what it is
worth today, the other turns a cumulated cost into the constant yearly payment worth the
same. Writing the rate :math:`\delta`:

.. math::

    D(x) = e^{-\delta x}
    \qquad\qquad
    AF(t) = \int_0^t D(x)\,\mathrm{d}x = \frac{1 - e^{-\delta t}}{\delta}

:math:`D` is the **discounting factor**: a cost :math:`c` paid at date :math:`x` is worth
:math:`c\,D(x)` today. :math:`AF` is the **annuity factor**, the present value of paying 1
per unit of time from 0 to :math:`t`; dividing a cumulated cost by it gives the constant
yearly payment worth the same. Both at a 5 % rate:

>>> rate = 0.05
>>> times = np.array([0., 10., 50.])
>>> np.exp(-rate * times)
array([1.        , 0.60653066, 0.082085  ])
>>> (1 - np.exp(-rate * times)) / rate
array([ 0.        ,  7.86938681, 18.35830003])

A cost paid in 50 time units counts for about 8 % of its face value. With ``rate=0.``, the
default used everywhere above, both factors degenerate into no discounting at all
(:math:`D = 1` and :math:`AF(t) = t`) and every cost counts in full whenever it occurs.

Without discounting, the expected *total* cost of an indefinitely renewed asset diverges, 
since the renewals never stop accumulating, which is why the policies report an annualized 
rate instead.

>>> float(process.asymptotic_expected_total_reward(cf=1500.))
inf
>>> round(float(process.asymptotic_expected_total_reward(cf=1500., discounting_rate=0.05)), 2)
72.65

Discounting also shifts the decision itself, not just its accounting. Costs deferred into the
future being cheaper, waiting longer becomes more attractive, and the optimal replacement age
moves later:

>>> from relife.policies import AgeReplacementPolicy
>>> ar_star = AgeReplacementPolicy(weibull).compute_optimal_ar(cf=1500., cp=400., discounting_rate=0.05)
>>> round(float(ar_star), 2)
61.14

61.1 against the 47.4 obtained undiscounted in
:doc:`../lifetime_modeling/preventive_age_replacement`.

Two ways of expressing total costs
----------------------------------

The **total reward** is what has been spent since time 0, discounted back to today: a
cumulative curve, growing with the horizon. It is obtained by solving the renewal equation 
of :doc:`../lifetime_modeling/renewal_theory`, except that a renewal now contributes its discounted cost instead 
of a unit count. With a final time and a number of points, it returns the timeline 
together with the values:

>>> timeline, z = process.expected_total_reward(500., 51, cf=1500.)
>>> np.round(z[::10], 1)
array([   0. , 1435.2, 3418.6, 5459.5, 7508.7, 9547.7])

The **equivalent annual worth** is that same total re-expressed as a constant yearly payment
covering it, the total divided by the annuity factor seen above. This is the figure to
compare policies with, since unlike the total it does not grow with the horizon:

>>> timeline, q = process.expected_equivalent_annual_worth(500., 51, cf=1500.)
>>> np.round(q[::10], 2)
array([ 0.  , 14.35, 17.09, 18.2 , 18.77, 19.1 ])

Both come with an ``asymptotic_`` variant that reports the limit: what renewing forever 
costs, and what it costs per unit of time. The exact expressions are in the API documentation 
of 
:py:meth:`~relife.stochastic_processes.RenewalRewardProcess.asymptotic_expected_total_reward`.

Counting number of events
-------------------------

A cost says how much, not how many. ``RenewalRewardProcess`` also inherits two counters from
``RenewalProcess``, which split the renewals according to *why* the asset was replaced:

``expected_number_of_events()``
    the expected number of **failures** up to each point of the timeline. Preventive
    replacements are not events and are not counted.

``expected_number_of_preventive_renewals()``
    the expected number of **preventive replacements** up to each point of the timeline.
    ``ar`` is required here: without a replacement age there is nothing to count.

Both solve a renewal equation of the same shape as the renewal function's, and by
construction the two add back up to it, since every replacement is one or the other:

>>> timeline, m = process.renewal_function(200., 201, ar=47.44)
>>> timeline, m_failures = process.expected_number_of_events(200., 201, ar=47.44)
>>> timeline, m_preventive = process.expected_number_of_preventive_renewals(200., 201, ar=47.44)
>>> np.round(m_failures[::50], 3)
array([0.   , 0.127, 0.255, 0.384, 0.515])
>>> np.round(m_preventive[::50], 3)
array([0.   , 0.873, 1.748, 2.624, 3.504])
>>> bool(np.allclose(m, m_failures + m_preventive))
True

Over 200 time units the fleet goes through about four replacements per asset, of which barely
half a failure. Undiscounted, the expected total reward *is* those two counters weighted by 
their costs, nothing more:

>>> timeline, z = process.expected_total_reward(200., 201, cf=1500., cp=400., ar=47.44)
>>> np.round(z[::50], 2)
array([   0.  ,  539.97, 1081.47, 1625.66, 2173.95])
>>> bool(np.allclose(z, 1500. * m_failures + 400. * m_preventive))
True

Summary of the methods
----------------------

The six methods all share the same arguments: the costs ``cf``, ``cp`` and ``ar``, their 
first-cycle variants ``cf1`` and ``cp1``, the initial age ``a0``, and ``discounting_rate``.

The four **finite-horizon** ones take a final time ``tf`` and a number of points
``nb_steps``, and return a ``(timeline, values)`` tuple, the timeline being ``nb_steps``
points evenly spread from 0 to ``tf``. The two **asymptotic** ones take no timeline at all
and return a single value.

===============================================  ==========  =================================================
method                                           horizon     what it gives
===============================================  ==========  =================================================
``expected_total_reward``                        finite      cost accumulated since time 0
``expected_equivalent_annual_worth``             finite      that cost as a constant yearly payment
``expected_number_of_events``                    finite      failures accumulated since time 0
``expected_number_of_preventive_renewals``       finite      preventive replacements accumulated since time 0
``asymptotic_expected_total_reward``             infinite    cost of renewing forever
``asymptotic_expected_equivalent_annual_worth``  infinite    long-run annual cost of renewing forever
===============================================  ==========  =================================================
