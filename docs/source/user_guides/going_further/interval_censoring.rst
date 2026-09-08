Interval censoring
===================

:doc:`../lifetime_modeling/censoring_and_truncation` covers the two observation schemes that
dominate industrial asset data, right censoring and left truncation. Sometimes the failure
date is not known at all, only the window it falls in: the asset was found broken during a
periodic inspection, so all that is known is that it failed between the previous inspection
and this one. For these observations, ``time`` is passed as a
**two-column array** of interval bounds instead of a one-dimensional array of durations. Each
row is then read as :math:`[a_i, b_i]`, the failure being somewhere in between:

.. list-table::
    :header-rows: 1
    :widths: 30 70

    * - Row
      - Meaning
    * - ``[t, t]``
      - failure observed at ``t`` (complete observation)
    * - ``[a, b]``
      - failure between ``a`` and ``b`` (interval censoring)
    * - ``[0., b]``
      - failure before ``b``, exact date unknown (left censoring)
    * - ``[a, np.inf]``
      - still working at ``a`` (right censoring)

The two degenerate bounds, ``0.`` and ``np.inf``, are what make left and right censoring
special cases of the same interval form: an unknown lower bound is the start of the asset's
life, an unknown upper bound is "not yet". A left-censored asset, found already broken without
a recorded date, is a row ``[0., b]``; a right-censored one, still working when last seen, is
a row ``[a, np.inf]``. ``event`` is redundant with this encoding and must be left out, while
``entry`` keeps its meaning.

Concretely, the two bounds are assembled into an ``(n, 2)`` array, one row per asset:

.. code-block:: python

    import numpy as np

    # an exact failure, then a unit found broken between two inspections,
    # then a unit found broken with no date at all, then a unit still working
    time = np.array([
        [15.0, 15.0],
        [20.0, 25.0],
        [0.0, 8.0],
        [30.0, np.inf],
    ])

Each row then contributes to the likelihood according to which of the two bounds is degenerate:
a complete observation contributes a density, an interval the probability mass it contains,
and the two censored forms reduce to :math:`F(b)` and :math:`S(a)` respectively, since
:math:`F(0) = 0` and :math:`F(\infty) = 1`. :doc:`likelihood` writes out the four terms.

.. warning::

    The interval form is part of the data model but is not usable for fitting in the current
    release: ``fit`` rejects a two-column ``time`` before reaching the likelihood. Use the
    one-dimensional ``time`` with ``event`` and ``entry``, as in
    :doc:`../lifetime_modeling/censoring_and_truncation`.
