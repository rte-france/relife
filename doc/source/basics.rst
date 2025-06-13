Basics
======

.. contents::
    :local:

The section will help you understand basic commands and math concepts to begin with ReLife.


ReLife and Numpy
----------------

ReLife is built using `NumPy <https://numpy.org/>`_, a fundamental Python library for numerical computing.
While you don't need to be a NumPy expert, understanding its basics will help since:

1. Most ReLife examples use the standard NumPy import: ``import numpy as np``
2. ReLife often requires data input of type ``np.ndarray``

There are 3 standard representation of data in ReLife :

- If you want to pass a scalar value, then use a ``float``
- If you want to pass a vector of :math:`\mathbb{R}^n`, eg :math:`n` values for one asset, then use a ``np.ndarray`` of shape ``(n,)``
- If you want to pass a matrix of :math:`\mathbb{R}^{m\times n}`, eg :math:`n` values for :math:`m` assets, then use a ``np.ndarray`` of shape ``(n,)``

**Broadcasting examples**

Here we create a very simple lifetime model (a lifetime distribution) called Weibull. To demonstrate input/output logic, we will begin with a ``float`` input. Here
we want to comput :math:`P(T > 40)`. The output has the same number of dimension than the input. It is a float-like object called ``np.float64`` that is compatible
with the NumPy interfaces. For instance, this object can answer to ``.ndim```and ``.shape``. Here it would be ``0`` and ``()``.

.. code-block:: python

    >>> from relife.lifetime_model import Weibull
    >>> weibull = Weibull(3.47, 0.012)
    >>> weibull.sf(40.)
    np.float64(0.9246627462729304)


Now, imagine that you want to compute not only :math:`P(T > 40)`, but also :math:`P(T > 50)` and :math:`P(T > 60)`. Because ReLife is built on NumPy, it benefits from
a concept called `broadcasting <https://numpy.org/doc/stable/user/basics.broadcasting.html>`_. It provides a way to vectorize operation so that the three survival function
values are computed in parallel. To do that, you need to pass a ``np.ndarray`` that encapsulate all your input values.

.. code-block:: python

    >>> import numpy as np
    >>> weibull.sf(np.array([40., 50., 60.])) # 1d array of shape (3,)
    array([0.92466275, 0.84375201, 0.72625935])

Note that the input is a ``np.ndarray`` of 1 dimension with a shape of ``(3,)``. The output is consistent to the input and has the same shape. This logic is extended **until
two dimensions**. With ReLife, asset managers may be interested to compute values on a fleet of assets. In this scenario, it is sometimes usefull to pass several values per
assets.

.. code-block:: python

    >>> weibull.sf(np.array([[40., 50., 60.], [42., 55., 68.]])) # 2d array of shape (2, 3)
    array([[0.92466275, 0.84375201, 0.72625935],
           [0.91139796, 0.78939177, 0.61029328]])

Note that the input is a ``np.ndarray`` of 2 dimensions with a shape of ``(2, 3)``. Each row is like a vector of values for each assets. Here the number of assets is 2.
The output shape is consistent to the input.

Variables dimensions
--------------------

