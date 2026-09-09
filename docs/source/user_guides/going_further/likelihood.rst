The likelihood
===============

Every parametric object in ReLife is fitted by maximum likelihood. ``fit`` builds the
negative log-likelihood of the observed data, hands it to SciPy for minimization, and keeps
the value reached at the optimum in ``fitting_results.neg_log_likelihood``.

What changes from one modeling approach to the next is not the principle but the terms. Each
observation contributes a factor that depends on **what was actually observed**, and, for a
stochastic process, on how the observation window cut its trajectory. This page collects the
forms used across the library, since the same likelihood machinery serves lifetime models
(:doc:`../lifetime_modeling/index`), their regressions, and counting processes
(:doc:`../counting_processes/index`).

Throughout, :math:`\theta` denotes the parameters, :math:`f` the probability density,
:math:`S` the survival function, :math:`F = 1 - S` the cumulative distribution function and
:math:`H = -\log S` the cumulative hazard.

Contribution of a single observation
--------------------------------------

There are only four kinds of contribution, and every likelihood below is assembled from
them:

.. list-table::
    :header-rows: 1
    :widths: 45 55

    * - What is known about the asset
      - Contribution to :math:`-\log L(\theta)`
    * - it failed at :math:`t` (complete observation)
      - :math:`-\log f(t;\theta)`
    * - it was still alive at :math:`t` (right censoring)
      - :math:`H(t;\theta)`
    * - it failed somewhere in :math:`[a, b]` (interval censoring)
      - :math:`-\log \left[ F(b;\theta) - F(a;\theta) \right]`
    * - it was already :math:`a` units old when observation began (left truncation)
      - :math:`-H(a;\theta)`

The first three are mutually exclusive: a row is either complete, right-censored or
interval-censored. The fourth one adds to whichever of them applies, on every row that
entered observation after birth. Right censoring is the interval case with
:math:`b = +\infty` and left censoring the interval case with :math:`a = 0`, which is exactly
how they are encoded in ReLife (see :doc:`interval_censoring`).

.. note::

    The interval term is computed as :math:`-\log \left[ 10^{-10} + F(b) - F(a) \right]`. The
    constant only keeps the logarithm finite while the optimizer explores parameter values
    for which the interval carries almost no probability mass; it is negligible against any
    interval the data actually supports.

Lifetime models
-----------------

For a lifetime distribution, with :math:`\mathcal{D}` the set of observed failures,
:math:`\mathcal{C}` the right-censored ones, :math:`\mathcal{I}` the interval-censored ones
and :math:`a_i` the entry age of asset :math:`i`, ``fit`` minimizes:

.. math::

    -\log L(\theta) = - \sum_{i \in \mathcal{D}} \log f(t_i; \theta)
                      + \sum_{i \in \mathcal{C}} H(t_i; \theta)
                      - \sum_{i \in \mathcal{I}} \log \left[ F(b_i; \theta) - F(a_i; \theta) \right]
                      - \sum_i H(a_i; \theta)

A failure is informative about the density at :math:`t_i`; a right-censored unit only tells
us it was still alive at :math:`t_i`, hence :math:`S(t_i;\theta)`; an interval tells us the
failure fell inside it, hence the probability mass it contains; and left truncation
conditions each factor on having survived up to the entry age :math:`a_i`, which is what the
last sum does.

Which sets are non-empty is decided entirely by how the data is passed to ``fit``. A
one-dimensional ``time`` with an ``event`` flag splits the rows between :math:`\mathcal{D}`
and :math:`\mathcal{C}`; a two-column ``time`` of interval bounds fills :math:`\mathcal{I}`,
with ``0.`` and ``np.inf`` as the bounds that turn an interval into a left- or right-censored
observation; and ``entry`` populates the truncation sum. Dropping ``event`` or ``entry`` is
therefore not a neutral simplification: it changes which terms are used, and so the estimate.
:doc:`../lifetime_modeling/censoring_and_truncation` covers the one-dimensional form and
:doc:`interval_censoring` the two-column one.

Regressions
-------------

Adding covariates changes nothing structural. The covariate effect is substituted into
:math:`f`, :math:`H` and :math:`F`, and the coefficients :math:`\beta` join the baseline
parameters :math:`\theta` in the same minimization:

.. math::

    -\log L(\beta, \theta) = - \sum_{i \in \mathcal{D}} \log f(t_i, x_i; \beta, \theta)
                             + \sum_{i \in \mathcal{C}} H(t_i, x_i; \beta, \theta)
                             - \sum_{i \in \mathcal{I}} \log \left[ F(b_i, x_i; \beta, \theta) - F(a_i, x_i; \beta, \theta) \right]
                             - \sum_i H(a_i, x_i; \beta, \theta)

The important point is that :math:`\beta` and :math:`\theta` are estimated **jointly**, in a
single optimization: the covariate effect you read off depends on the baseline shape you
assumed. See :doc:`../lifetime_modeling/regressions`.

The Cox partial likelihood
----------------------------

The semi-parametric Cox model leaves the baseline hazard unspecified, so it cannot use the
likelihood above. It maximizes a **partial** likelihood instead, built from the conditional
probability that, among the units at risk, the one that actually failed is the one that did.
The unknown baseline appears in every term of that ratio and cancels.

With :math:`t_j` the ordered distinct failure times, :math:`\mathcal{R}_j` the risk set just
prior to :math:`t_j`, :math:`\mathcal{D}_j` the units failing at :math:`t_j` and
:math:`d_j = |\mathcal{D}_j|`, ReLife uses Breslow's handling of ties:

.. math::

    L(\beta) = \prod_j \frac{\exp \left( \beta \cdot \sum_{i \in \mathcal{D}_j} x_i \right)}
                            {\left( \sum_{i \in \mathcal{R}_j} e^{\beta \cdot x_i} \right)^{d_j}}

and switches to Efron's correction, which is more accurate, when failure times are heavily
tied (more than three failures sharing a date):

.. math::

    L(\beta) = \prod_j \frac{\exp \left( \beta \cdot \sum_{i \in \mathcal{D}_j} x_i \right)}
                            {\prod_{\alpha=0}^{d_j - 1} \left( \sum_{i \in \mathcal{R}_j} e^{\beta \cdot x_i}
                             - \frac{\alpha}{d_j} \sum_{i \in \mathcal{D}_j} e^{\beta \cdot x_i} \right)}

Censoring and truncation enter through the risk set only: unit :math:`i` belongs to
:math:`\mathcal{R}_j` when :math:`a_i < t_j \leq t_i`, that is when it had already entered
observation and had not yet left it. Interval-censored observations have no place in this
construction, since the partial likelihood is built on the *order* of the failure dates.

Non-homogeneous Poisson process
---------------------------------

A counting process is observed as a sequence of event ages per asset rather than as one
duration. For an asset observed from age :math:`t_{first}` to age :math:`t_{last}`, with
failures at :math:`t_1 < \dots < t_n`, the log-likelihood of a non-homogeneous Poisson process
of intensity :math:`\lambda` and cumulative intensity :math:`\Lambda` is:

.. math::

    \log L(\theta) = \sum_{k=1}^{n} \log \lambda(t_k; \theta)
                     - \Lambda(t_{last}; \theta) + \Lambda(t_{first}; \theta)

summed over assets. In ReLife the intensity *is* the hazard function of the lifetime model
that parametrizes the process, :math:`\lambda = h` and :math:`\Lambda = H`, which is why no
separate estimator is needed: each inter-event gap is handed to the lifetime likelihood above
as a complete observation at :math:`t_k` left-truncated at :math:`t_{k-1}`, and the tail of
the window as an observation right-censored at :math:`t_{last}` and left-truncated at
:math:`t_n`. Summing those contributions telescopes into the expression above.

What comes out of the optimization
------------------------------------

Whichever form was minimized, ``fitting_results`` carries the value at the optimum
(``neg_log_likelihood``), the number of observations and parameters it was computed on, and
the information criteria derived from them (see
:doc:`../lifetime_modeling/distributions`), which are only comparable between models fitted
on the same data with the same likelihood.
