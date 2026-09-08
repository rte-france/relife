About NumPy
===========

ReLife is built on `NumPy <https://numpy.org/>`_, a fundamental Python library for numerical
computing.

Representing data
-----------------

There are 3 standard representations of data in ReLife :

- If you want to pass a scalar value, use the ``float`` built-in type (``np.float64`` is accepted but not required).
- If you want to pass a vector of :math:`\mathbb{R}^n`, e.g. :math:`n` values for one asset, use a ``np.ndarray`` of shape ``(n,)``.
- If you want to pass a matrix of :math:`\mathbb{R}^{m\times n}`, i.e. :math:`n` values for :math:`m` assets, use a ``np.ndarray`` of shape ``(m, n)``.

The last one is the convention to keep in mind: in a 2d array, **rows are assets and columns
are values**.

Broadcasting with NumPy
-----------------------

With broadcasting, the output has the same shape as the (broadcast) input.

>>> from relife.lifetime_models import Weibull
>>> weibull = Weibull(3.47, 0.012)
>>> round(weibull.sf(40.), 6)
np.float64(0.924663)

A scalar in, a scalar out. To compute :math:`P(T > 40)`, but also :math:`P(T > 50)` and
:math:`P(T > 60)`, we can benefit from `broadcasting
<https://numpy.org/doc/stable/user/basics.broadcasting.html>`_ and compute three survival
function evaluations in parallel.

>>> import numpy as np
>>> weibull.sf(np.array([40., 50., 60.])) # 1d array of shape (3,)
array([0.92466275, 0.84375201, 0.72625935])

This logic is extended **to any dimensions**. For instance, it is sometimes useful to pass
several values per asset.

>>> weibull.sf(np.array([[40., 50., 60.], [42., 55., 68.]])) # 2d array of shape (2, 3)
array([[0.92466275, 0.84375201, 0.72625935],
       [0.91139796, 0.78939177, 0.61029328]])

Each row encodes a vector of values for each asset.

Broadcasting is what makes those two dimensions cheap to obtain: you rarely have to build the
``(m, n)`` array yourself. When a model takes several arguments, giving one of them an asset
axis is enough for the result to acquire it. Here, two assets with different covariate values
are evaluated on the same three times, and the ``(2, 1)`` covariate array broadcasts against
the ``(3,)`` time array into a ``(2, 3)`` result:

>>> from relife.lifetime_models import ParametricProportionalHazard
>>> regression = ParametricProportionalHazard(weibull, coefficients=(0.1,))
>>> time = np.array([40., 50., 60.]) # shape (3,)
>>> covar = np.array([[1.], [2.]]) # shape (2, 1) : one covariate value per asset
>>> regression.sf(time, covar)
array([[0.917077  , 0.82880958, 0.70223526],
       [0.90876582, 0.81260329, 0.6766079 ]])

.. note::

    Two dimensions is a modeling convention, not a hard limit: nothing stops NumPy from
    broadcasting further. But past two dimensions the ``(assets, values)`` reading of the
    result is lost, and so is the meaning ReLife attaches to it.

Shape-constrained arguments
---------------------------

In maintenance policies, arguments such as the costs (``cf``, ``cp``, ``cr``), describe 
**the fleet itself**: there is exactly one value per asset, so these are **at most 1d**. 
A scalar means the same value for the whole fleet, and an array of shape ``(n,)`` means 
one value per asset.

>>> from relife.policies import OneCycleRunToFailurePolicy
>>> policy = OneCycleRunToFailurePolicy(weibull)
>>> round(policy.asymptotic_expected_equivalent_annual_cost(cf=1.), 6) # one cost for the fleet
np.float64(0.015357)
>>> policy.asymptotic_expected_equivalent_annual_cost(
...     cf=np.array([1., 2., 3.])
... ).round(6) # one cost per asset, shape (3,)
array([0.015357, 0.030713, 0.04607 ])

A 2d cost array is rejected: the asset axis is the only one such an argument can have, so a
second axis has no meaning to give it.
