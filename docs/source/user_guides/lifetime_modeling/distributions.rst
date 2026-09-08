Parametric distributions
=============================================

A parametric lifetime distribution assumes a fixed functional shape (Weibull, Gamma,
Gompertz, ...) for the survival function, and estimates its parameters from data. Fitting
one is a single call, accounting for censoring and truncation as described in
:doc:`censoring_and_truncation`:

>>> from relife.datasets import load_power_transformer
>>> from relife.lifetime_models import Weibull
>>> dataset = load_power_transformer()
>>> weibull = Weibull().fit(dataset["time"], event=dataset["event"], entry=dataset["entry"])
>>> weibull.get_params()
array([3.46597396, 0.0122785 ])

The two fitted values are the Weibull ``shape`` and ``rate``. A shape above 1 means the
hazard rate increases with age: the assets wear out.

Comparing candidate shapes
--------------------------------------------

Nothing in the data tells you which shape to assume, so fit several and look at them. The
Kaplan-Meier estimator (see :doc:`non_parametric_models`) is the natural reference here: it
makes no shape assumption, so a parametric curve that strays far from it is a poor
description of the data.

.. plot::
    :context: close-figs

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from relife.datasets import load_power_transformer
    >>> from relife.lifetime_models import KaplanMeier, Weibull, Gamma, Gompertz, LogLogistic
    >>> dataset = load_power_transformer()
    >>> km = KaplanMeier(dataset["time"], event=dataset["event"], entry=dataset["entry"])
    >>> _ = km.plot("sf", ci=False, color="black", linestyle="--", label="Kaplan-Meier")
    >>> timeline = np.arange(0, 145)
    >>> for distribution in (Weibull(), Gamma(), Gompertz(), LogLogistic()):
    ...     fitted = distribution.fit(
    ...         dataset["time"], event=dataset["event"], entry=dataset["entry"]
    ...     )
    ...     _ = fitted.plot("sf", timeline, ci=False, label=type(fitted).__name__)
    >>> _ = plt.xlabel("Time")
    >>> _ = plt.ylabel("Estimated survival function")
    >>> _ = plt.legend()
    >>> plt.show()

The confidence bands are hidden here with ``ci=False``, since four of them overlapping would
make the figure unreadable.

All four shapes agree while the fleet is young and only separate past the median, where the
data thins out: Gompertz and Weibull bend down faster, Gamma and LogLogistic keep a heavier
tail. Gompertz stays closest to the Kaplan-Meier step function over the observed range. That
choice matters well beyond the plot, because a maintenance policy weighs the cost of a
failure against the cost of a preventive replacement using exactly this model.

Reading curves only gets you so far, though. Two shapes can look equally plausible, and a
distribution with more parameters can always be made to hug the data more closely. The
comparison has to be made on the likelihood and the number of parameters.

Maximum likelihood
------------------------------------------

Fitting maximizes the likelihood of the observed data under the chosen shape. Each row
contributes a factor that depends on what was actually observed. Consider :math:`\theta` for
the parameters, :math:`f` for the probability density, :math:`S` for the survival function,
:math:`H = -\log S` for the cumulative hazard, :math:`\mathcal{D}` for the set of observed
failures and :math:`\mathcal{C}` for the right-censored ones, each entering the study at age
:math:`a_i`. A failure is informative about the density at :math:`t_i`; a right-censored unit 
only tells us it was still alive at :math:`t_i`, hence :math:`S(t_i;\theta)`; and left truncation
conditions each factor on having survived up to the entry age :math:`a_i`, which is what the
denominators do. ``fit`` minimizes the negative log-likelihood, which is the form actually
implemented:

.. math::

    -\log L(\theta) = - \sum_{i \in \mathcal{D}} \log f(t_i; \theta)
                      + \sum_{i \in \mathcal{C}} H(t_i; \theta)
                      - \sum_i H(a_i; \theta)

Dropping the ``event`` and ``entry`` arguments is not a neutral simplification, it 
changes which of these terms are used, and therefore the estimate.

The minimization is delegated to SciPy, and the value reached at the optimum is kept:

>>> round(weibull.fitting_results.neg_log_likelihood, 2)
np.float64(1698.24)

Information criteria
------------------------------------------

The log-likelihood alone can't arbitrate between shapes with different numbers of
parameters, since adding parameters can only improve the fit. Information criteria penalize
that: with :math:`k` parameters, :math:`n` observations and :math:`\hat{\ell}` the maximized
log-likelihood,

.. math::

    \textrm{AIC} = 2k - 2\hat{\ell} \qquad
    \textrm{AICc} = \textrm{AIC} + \frac{2k(k+1)}{n - k - 1} \qquad
    \textrm{BIC} = k \log n - 2\hat{\ell}

AICc is the small-sample correction to AIC, and BIC penalizes parameters more heavily as the
sample grows. All three are computed on ``fitting_results``, and in every case the lower the
better:

>>> print(weibull.fitting_results)
fitted params : [3.46597, 0.0122785]
AIC           : 3400.49
AICc          : 3400.49
BIC           : 3411.3

>>> from relife.lifetime_models import Gamma, Gompertz
>>> gamma = Gamma().fit(dataset["time"], event=dataset["event"], entry=dataset["entry"])
>>> print(gamma.fitting_results)
fitted params : [5.35711, 0.0662282]
AIC           : 3442.37
AICc          : 3442.37
BIC           : 3453.18

>>> gompertz = Gompertz().fit(dataset["time"], event=dataset["event"], entry=dataset["entry"])
>>> print(gompertz.fitting_results)
fitted params : [0.00865741, 0.0606263]
AIC           : 3374.22
AICc          : 3374.23
BIC           : 3385.04

All three shapes have two parameters here, so the penalty term is identical and the ranking
is the ranking of the likelihoods: Gompertz first, then Weibull, then Gamma. This confirms
what the plot suggested, and it is the usual outcome, the criteria settle comparisons the eye
can't, they rarely overturn a gap that is already visible.

Two cautions. These numbers are only comparable across models fitted on **the same data**
with the same likelihood, so refitting on a filtered dataset invalidates the comparison. And
a best-of-four is still only the best of what you tried: check the winner against
Kaplan-Meier before trusting its fit.

Covariates are the next step: see :doc:`regressions` for models where lifetime depends on the
conditions each asset operates under.
