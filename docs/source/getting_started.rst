Getting Started
===============

ReLife's workflow has common steps for most cases:

1. You **collect data** on a fleet of assets.
2. You **fit a statistical model** to that data.
3. You **model a maintenance policy** and optimize it against costs.
4. You **compute the consequences** of that policy (for instance the expected number of
   replacements per year).

These steps can change since the workflow depends on the **event of interest**, and on the 
modeling approach of that event:

- **lifetime models**, for a single terminal event: the asset fails once and is replaced. What
  is modeled is a duration.
- **non-homogeneous Poisson processes**, for repairable assets: the same asset fails several
  times and each repair is *minimal*, meaning it restores service without rejuvenating the
  asset. What is modeled is the rate of recurrent failures.
- **gamma processes** (not implemented yet), for gradual deterioration measured on the asset 
  (corrosion, loss of thickness): the event of interest is a threshold crossing rather than a failure.

That choice propagates through the whole workflow: it decides what the field data has to
record, which model is fitted to it, and which policies that model can then be plugged into.

.. figure:: /_static/workflow.png
    :align: center

    ReLife's workflow example (lifetime modeling): from data collection to the consequences of a 
    maintenance policy

This page walks the four steps for the **lifetime model** approach, on a built-in dataset: one
asset, one failure, one replacement. It is the most common case, not the only one. See the
:doc:`user_guides/index` for the concepts behind each step.

1. Data collection
------------------

ReLife ships example datasets so you can get started without your own data (see
:doc:`user_guides/lifetime_modeling/datasets`). Here we use the *power transformer*
dataset. If you're not familiar with the asset, see `the Wikipedia page
<https://en.wikipedia.org/wiki/Transformer>`_.

>>> from relife.datasets import load_power_transformer
>>> dataset = load_power_transformer()
>>> dataset.dtype.names
('time', 'event', 'entry')

``dataset`` is a `structured array
<https://numpy.org/doc/stable/user/basics.rec.html>`_ with three fields:

- ``time``: the observed lifetime values.
- ``event``: whether the failure actually happened inside the observation window
  (``False`` means the lifetime is *right-censored*, i.e. the asset was still alive at the 
  end of the observation window).
- ``entry``: the age of the asset at the beginning of the observation window (*left-truncation*).

>>> dataset["time"][:3]
array([34.3, 45.1, 53.2])
>>> dataset["event"][:3]
array([ True,  True,  True])
>>> dataset["entry"][:3]
array([34., 44., 52.])

Censoring and truncation are not details to be cleaned away. They carry information, and
most ReLife estimator accounts for them. See
:doc:`user_guides/lifetime_modeling/censoring_and_truncation`.

.. note::

    **Data is always the starting point, but not always the same data.** The three fields above
    are what a *lifetime* model needs: one duration per asset, plus what is known about how the
    observation window cut it. A non-homogeneous Poisson process is fitted on something else,
    the *sequence* of failure and repair dates observed on each asset, so a single asset
    contributes several records instead of one. A gamma process would need deterioration
    measurements: a measured level and the date it was measured, repeated over the asset's
    life. Which event you intend to model is therefore a decision to take before collecting,
    since it dictates what the records have to contain.

2. Lifetime model estimation
----------------------------

It's a good idea to start with a **non-parametric** model, which assumes no particular shape.
The Kaplan-Meier estimator is a step function that drops at each observed failure and stays
flat in between, so it follows the data without imposing any parameters on it:

>>> from relife.lifetime_models import KaplanMeier
>>> km = KaplanMeier(
...     dataset["time"], event=dataset["event"], entry=dataset["entry"]
... )

.. plot::
    :context: close-figs

    >>> import matplotlib.pyplot as plt
    >>> from relife.datasets import load_power_transformer
    >>> from relife.lifetime_models import KaplanMeier
    >>> dataset = load_power_transformer()
    >>> km = KaplanMeier(
    ...     dataset["time"], event=dataset["event"], entry=dataset["entry"]
    ... )
    >>> _ = km.plot("sf", label="Kaplan-Meier")
    >>> _ = plt.xlabel("Time")
    >>> _ = plt.ylabel("Estimated survival function")
    >>> _ = plt.legend()
    >>> plt.show()

Then fit a **parametric** model (here a Weibull distribution). *Parametric* means the whole
model is carried by a fixed, small set of numbers: fitting is estimating those numbers from the
data, and ``get_params`` reads them back. The two fitted values are the Weibull
``shape`` and ``rate``. A shape above 1 means the hazard rate increases with age: the assets ages with 
time, which is what makes preventive replacement worth considering at all. A shape equal to 1 means the
hazard rate is constant with time, the Weibull distribution reduces to an Exponential distribution, and
a shape below 1 means there is significant "infant mortality", or defective items failing early and the 
hazard rate decreasing over time as the defective items are out of the population.

>>> from relife.lifetime_models import Weibull
>>> weibull = Weibull().fit(
...     dataset["time"], event=dataset["event"], entry=dataset["entry"]
... )
>>> weibull.get_params()
array([3.46597396, 0.0122785 ])

Overlay of the two survival functions shows agreement between them and is a common sanity check: it means 
the parametric shape you chose is compatible with what the data shows on its own.

.. plot::
    :context: close-figs

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from relife.datasets import load_power_transformer
    >>> from relife.lifetime_models import KaplanMeier, Weibull
    >>> dataset = load_power_transformer()
    >>> km = KaplanMeier(
    ...     dataset["time"], event=dataset["event"], entry=dataset["entry"]
    ... )
    >>> weibull = Weibull().fit(
    ...     dataset["time"], event=dataset["event"], entry=dataset["entry"]
    ... )
    >>> timeline = np.arange(0, 145)
    >>> _ = km.plot("sf", label="Kaplan-Meier")
    >>> _ = weibull.plot("sf", timeline, label="Weibull")
    >>> _ = plt.xlabel("Time")
    >>> _ = plt.ylabel("Estimated survival function")
    >>> _ = plt.legend()
    >>> plt.show()

.. note::

    **Every object ReLife fits is a model, and they all share one interface.** They
    are built either unparametrized or with known parameter values, ``fit`` estimates the
    parameters from data by maximum likelihood, and ``get_params`` / ``set_params`` read and
    write them as a flat vector. That interface is not specific to distributions: a regression
    (:doc:`user_guides/lifetime_modeling/regressions`) is the same object with one
    extra coefficient per covariate, and a
    :py:class:`~relife.stochastic_processes.NonHomogeneousPoissonProcess` is parametrized by the
    lifetime model that defines its intensity, so it is fitted from repair histories in the same
    way. Non-parametric estimators such as ``KaplanMeier`` are the exception: they hold no
    parameters at all, which is precisely what makes them a good first look at the data.

3. Maintenance policy optimization
----------------------------------

A lifetime model alone doesn't say *when* to replace an asset. For that you wrap it in a
policy. A policy is a model as well, and it has parameters of two kinds: **a survival law**,
passed once when the policy is built, and **the costs** attached to each kind of event, passed
to the methods that need them. Here the two costs that drive the trade-off are:

- ``cp``, the cost of a **preventive** replacement, that you can choose;
- ``cf``, the cost of an **unexpected failure**, which is (almost always, some specific cases exists) 
  higher because it includes the undesirable consequences of the failure itself.

Costs have no imposed unit: euros, dollars, millions of either, or anything else works, as
long as ``cp`` and ``cf`` use the *same* unit. Whatever you put in is what comes back out, so
the annual costs below are in that same unit too. Nothing constrains what you put *into* a
cost either: the direct expense of the replacement, but also the socio-economic consequences
of the failure, such as unsupplied energy or the shadow price of carbon, which is often what
makes a renewal investment defensible.

The **time** unit is freer than the method names suggest. ReLife follows an annual accounting
convention in its naming (``asymptotic_expected_equivalent_annual_cost``,
``annual_number_of_replacements``, and a ``discounting_rate`` read as a yearly rate), because
that is the horizon asset managers budget on. But no method converts anything: the time unit
is whichever unit the lifetimes you fitted are expressed in. Fit on operating hours or on
switching cycles and "annual" reads as "per hour" or "per cycle", with the discounting rate
expressed per that same unit. Only consistency between lifetimes, horizons and rate matters.

Here ``cp`` = 3 and ``cf`` = 11, with a 4 % discounting rate. Replacing every asset
preventively at the optimal age costs less per year than waiting for failures:

>>> from relife.policies import AgeReplacementPolicy, RunToFailurePolicy
>>> policy = AgeReplacementPolicy(weibull)
>>> ar_star = policy.compute_optimal_ar(cf=11., cp=3., discounting_rate=0.04)
>>> round(ar_star, 4)
np.float64(59.1975)
>>> round(policy.asymptotic_expected_equivalent_annual_cost(
...     ar=ar_star, cf=11., cp=3., discounting_rate=0.04
... ), 6)
np.float64(0.035023)

>>> round(RunToFailurePolicy(weibull).asymptotic_expected_equivalent_annual_cost(
...     cf=11., discounting_rate=0.04
... ), 6)
np.float64(0.039082)

Replacing at age ``ar_star`` (about 59 years) is roughly 10 % cheaper per year than running
the assets to failure. :doc:`user_guides/lifetime_modeling/preventive_age_replacement`
explains where these numbers come from.

.. note::

    **Run-to-failure and age replacement are two ends of a spectrum, not the whole catalogue.**
    Both assume the asset is *replaced* at the terminal event, which is what a lifetime model
    describes. When the asset is *repaired* instead and goes on failing, the decision becomes
    when to stop repairing and replace: that is
    :py:class:`~relife.policies.NonHomogeneousPoissonAgeReplacementPolicy`, built on a
    non-homogeneous Poisson process rather than on a lifetime model. Each policy also comes in
    a *renewal* variant, for planning a fleet over the long run, and a *one-cycle* variant, for
    a decision about the asset currently in service. See
    :doc:`user_guides/background/from_process_to_policy`.

4. Projection of consequences
-----------------------------

Finally, project what the policy implies for a real fleet. We take 1000 assets whose current
ages are drawn from a binomial distribution, and ask for the expected number of replacements
over the next 170 years. ReLife answers by solving **the renewal equation** (see
:doc:`user_guides/lifetime_modeling/renewal_theory`).

>>> import numpy as np
>>> rng = np.random.default_rng(42)
>>> a0 = rng.binomial(60, 0.5, 1000).astype(float)  # current age of each asset
>>> timeline, replacements = policy.annual_number_of_replacements(170, ar=ar_star, a0=a0)
>>> _, failures = policy.annual_number_of_failures(170, ar=ar_star, a0=a0)
>>> timeline.shape, replacements.shape
((170,), (170, 1000))

You get one value per year and per asset. Summing over the fleet gives the workload to plan
for, and how much of it is unplanned failure rather than scheduled replacement:

>>> round(float(replacements.sum()), 1)
3113.5
>>> round(float(failures.sum()), 1)
889.8

ReLife has no built-in plot for this, but the arrays go straight into `Matplotlib
<https://matplotlib.org/>`_:

.. plot::
    :context: close-figs

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from relife.datasets import load_power_transformer
    >>> from relife.lifetime_models import Weibull
    >>> from relife.policies import AgeReplacementPolicy
    >>> dataset = load_power_transformer()
    >>> weibull = Weibull().fit(
    ...     dataset["time"], event=dataset["event"], entry=dataset["entry"]
    ... )
    >>> policy = AgeReplacementPolicy(weibull)
    >>> ar_star = policy.compute_optimal_ar(cf=11., cp=3., discounting_rate=0.04)
    >>> rng = np.random.default_rng(42)
    >>> a0 = rng.binomial(60, 0.5, 1000).astype(float)
    >>> timeline, replacements = policy.annual_number_of_replacements(
    ...     170, ar=ar_star, a0=a0
    ... )
    >>> _, failures = policy.annual_number_of_failures(170, ar=ar_star, a0=a0)
    >>> fig, ax = plt.subplots(figsize=(10, 4))
    >>> _ = ax.bar(
    ...     timeline + 2025, replacements.sum(axis=1), align="edge", width=1.,
    ...     label="all replacements", color="C1", edgecolor="black", linewidth=0.3,
    ... )
    >>> _ = ax.bar(
    ...     timeline + 2025, failures.sum(axis=1), align="edge", width=1.,
    ...     label="failure replacements", color="C0", edgecolor="black", linewidth=0.3,
    ... )
    >>> _ = ax.set_xlabel("Year")
    >>> _ = ax.set_ylabel("Number of annual replacements")
    >>> _ = ax.set_xlim(left=2025, right=2025 + 170)
    >>> _ = ax.set_ylim(bottom=0)
    >>> _ = ax.legend(loc="upper right")
    >>> plt.show()

The replacement waves you see are the fleet's current age distribution propagating forward:
assets installed around the same time come due around the same time, and the peaks damp out
over successive cycles.

.. note::

    **What gets projected follows from the model that was fitted.** With a lifetime model, the
    renewal equation counts *replacements*, over a horizon expressed in the time unit of the
    fitted data (170 years here, because the transformer lifetimes are in years). With a
    non-homogeneous Poisson process, the same step projects the expected number of *repairs*,
    and a replacement count only appears once a replacement decision is put on top of it.
    Either way what comes out is a plain NumPy array, one value per time step and per asset, so
    turning it into a budget, a spare-parts stock or a maintenance workload is left to you.

Next steps
----------

- :doc:`user_guides/index` for the concepts (censoring and truncation, lifetime models,
  maintenance policies) behind the calls above.
- :doc:`api/index` for the full reference.
