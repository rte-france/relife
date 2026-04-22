About NumPy
===========

ReLife is built using `NumPy <https://numpy.org/>`_, a fundamental Python library for numerical computing.
While you don't need to be a NumPy expert, understanding its basics will help since ReLife often requires data input of ``np.ndarray`` type.

There are 3 standard representations of data in ReLife :

- If you want to pass a scalar value, use ``float`` built-in type (``np.float64`` is accepted but not required).
- If you want to pass a vector of :math:`\mathbb{R}^n`, e.g. :math:`n` values for one asset, use a ``np.ndarray`` of shape ``(n,)``.
- If you want to pass a matrix of :math:`\mathbb{R}^{m\times n}`, i.e. :math:`n` values for :math:`m` assets, use a ``np.ndarray`` of shape ``(m, n)``

**Broadcasting examples**

.. code-block:: python

    >>> from relife.lifetime_model import Weibull
    >>> weibull = Weibull(3.47, 0.012)
    >>> weibull.sf(40.)
    np.float64(0.9246627462729304)

The output has the same number of dimension than the input.
To compute :math:`P(T > 40)`, but also :math:`P(T > 50)` and :math:`P(T > 60)`, we can benefit from `broadcasting <https://numpy.org/doc/stable/user/basics.broadcasting.html>`_ and compute three survival function evaluations in parallel.

.. code-block:: python

    >>> import numpy as np
    >>> weibull.sf(np.array([40., 50., 60.])) # 1d array of shape (3,)
    array([0.92466275, 0.84375201, 0.72625935])

This logic is extended **until two dimensions**. For instance, it is sometimes usefull to pass several values per assets.

.. code-block:: python

    >>> weibull.sf(np.array([[40., 50., 60.], [42., 55., 68.]])) # 2d array of shape (2, 3)
    array([[0.92466275, 0.84375201, 0.72625935],
           [0.91139796, 0.78939177, 0.61029328]])

Each row encodes a vector of values for each asset.