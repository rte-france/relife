Censoring and truncation
=========================

Asset lifetime data is rarely a clean list of "this asset failed after :math:`t` units of
time". Assets are still working when the study ends, monitoring starts after some assets
have already been in service for a while, and installation dates are sometimes only
approximately known. Any of these turns a simple "failed at time :math:`t`" record into
something less direct, and ignoring that distinction leads to systematically wrong
estimates. ReLife handles it through two arguments: ``event`` and ``entry``.

The observation scheme
-------------------------

For each asset, four dates matter: when the *observation window* starts (:math:`t_{start}`)
and ends (:math:`t_{end}`), and when the asset itself was installed (:math:`t_{init}`) and
failed (:math:`t_{fail}`, if that was actually observed). The figure below lines up six
assets on the same calendar to show every combination that can occur:

.. figure:: /_static/figures/observation_scheme.png
    :alt: Observation scheme for six assets, showing complete observations, right-censoring, left-censoring and left-truncation.
    :width: 100%

    Observation scheme for six assets. A filled circle is an observed failure; an open circle means the asset was still working
    when the observation window ended at :math:`t_{end}` (**right-censoring**); a
    right-pointing triangle before :math:`t_{start}` means the asset was already installed
    before observation began (**left-truncation** if it fails after :math:`t_{start}`, or
    **left-censoring**, dashed, if a failure before :math:`t_{start}` is only known to have
    happened, without knowing exactly when).

ReLife's ``event``/``entry`` pair covers the two cases that dominate industrial asset data
(right-censoring and left-truncation), which is why they are the two arguments widely used in ReLife's functions. 
Note that left-censoring (asset 6 above) is rarer in practice and isn't part of ReLife's data model.

Right censoring
----------------

An observation is **right-censored** when the actual failure time is unknown: the asset
was still working the last time it was observed, so all we know is that its lifetime is
*at least* ``time``. This is flagged with ``event=False``.

The ``load_power_transformer`` dataset (see :doc:`datasets`) has both kinds of rows:

>>> from relife.datasets import load_power_transformer
>>> dataset = load_power_transformer()
>>> dataset[dataset["event"] & (dataset["entry"] == 0)][:2]  # observed failures
array([(12.4,  True, 0.), (12.6,  True, 0.)],
      dtype=[('time', '<f8'), ('event', '?'), ('entry', '<f8')])
>>> dataset[~dataset["event"] & (dataset["entry"] == 0)][:2]  # still in service at "time"
array([(21.6, False, 0.), ( 4.2, False, 0.)],
      dtype=[('time', '<f8'), ('event', '?'), ('entry', '<f8')])

Out of the 1650 transformers in the dataset, only a small fraction were actually observed
failing:

>>> dataset["event"].sum(), len(dataset)
(np.int64(318), 1650)

Left truncation
-----------------

An observation is **left-truncated** when the asset had already survived up to some age
before it was observed. For instance, monitoring only started once a transformer
had already been in service for a while. Assets of the same population that failed *before*
observation are invisible to the dataset, which biases naive estimates towards
longer lifetimes if left unaccounted for. This is captured with the ``entry`` field: the age
at which the asset entered observation (``entry=0`` if it was observed from the start of its
life).

>>> dataset[dataset["event"] & (dataset["entry"] > 0)][:2]
array([(34.3,  True, 34.), (45.1,  True, 44.)],
      dtype=[('time', '<f8'), ('event', '?'), ('entry', '<f8')])

Here, both transformers were already 34 and 44 (respectively) units old when they were
first observed, and were then failing shortly after, at 34.3 and 45.1. More than 70% of
the fleet in this dataset was already in service before observation:

>>> (dataset["entry"] > 0).sum(), len(dataset)
(np.int64(1158), 1650)

Why getting this right matters
---------------------------------

Mishandling either one doesn't just add noise: it biases the result in a predictable
direction, and the effect can be large. On a fleet of high-voltage assets, estimating 
a mean lifetime under three assumptions gives:

.. figure:: /_static/figures/censoring_bias.png
    :alt: Survival curves and mean lifetimes under three assumptions about the same industrial dataset: 16, 38, and 46 years.
    :width: 100%

    Same dataset, three assumptions. Counting only
    observed failures and ignoring the still-working assets (**mortality bias**) gives a mean
    lifetime of 16 years. Accounting for right-censoring but ignoring left-truncation
    (**survivor bias**) gives 46 years. Accounting for both gives 38 years: the correct
    figure lies in between, not at either extreme.

Ignoring censoring entirely (mortality bias) systematically *underestimates* how long assets
last, since the (longer-lived) censored units get discarded or miscounted as short-lived
ones. Ignoring left-truncation (survivor bias) does the opposite: it *overestimates*
lifetimes, because units that failed before observation are invisible to the
sample, leaving only the units hardy enough to have survived that long. This is exactly why
every ReLife lifetime model accounts for both directly in its likelihood (see
:doc:`distributions`), rather than requiring you to drop or approximate
either kind of observation.

Fitting with censoring and truncation
---------------------------------------

Both are passed to ``fit`` together: ReLife builds the correct likelihood contribution for
each row depending on its ``event``/``entry`` combination:

>>> from relife.lifetime_models import Weibull
>>> weibull = Weibull().fit(dataset["time"], event=dataset["event"], entry=dataset["entry"])
>>> weibull.get_params()
array([3.46597396, 0.0122785 ])

Dropping ``entry`` (i.e. ignoring the left truncation) changes the fitted parameters, since
the model is then implicitly assuming every asset was observed from birth:

>>> weibull_no_truncation = Weibull().fit(dataset["time"], event=dataset["event"])
>>> weibull_no_truncation.get_params()
array([4.11911713, 0.0122451 ])
