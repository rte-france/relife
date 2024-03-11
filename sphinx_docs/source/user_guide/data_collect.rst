Lifetime data format
====================

.. role:: python(code)
   :language: python

.. image:: ../img/main_process_bis_0.png
    :scale: 100 %
    :align: center

In ReLife, collected data must be loaded in `np.array <https://numpy.org/doc/stable/reference/generated/numpy.array.html>`_. ReLife can carry either 1d-array or 2d-array observed lifetimes. This section describes in-depth how to handle these two
formats. ReLife's lifetime arguments are :


+-------------------------------+-------------+--------------------------+-----------------------------------------------------------------------+
|          **Variable**         | **Default** |         **Value**        |                            **Description**                            |
+-------------------------------+-------------+--------------------------+-----------------------------------------------------------------------+
|     ``observed_lifetimes``    |    unset    |       ``np.float``       | shape : (n,) or (n, 2)                                                |
|                               |             |                          |                                                                       |
|                               |             |                          | Observed lifetime values, either 1d or 2d                             |
+-------------------------------+-------------+--------------------------+-----------------------------------------------------------------------+
|    ``complete_indicators``    |   ``None``  | ``np.bool_`` or ``None`` | shape : (n,)                                                          |
|                               |             |                          |                                                                       |
|                               |             |                          | Boolean indicators used to tag observations are complete or not       |
|                               |             |                          |                                                                       |
|                               |             |                          | Only used when ``observed_lifetimes`` shape is (n,)                   |
+-------------------------------+-------------+--------------------------+-----------------------------------------------------------------------+
|  ``left_censored_indicators`` |   ``None``  | ``np.bool_`` or ``None`` | shape : (n,)                                                          |
|                               |             |                          |                                                                       |
|                               |             |                          | Boolean indicators used to tag observations are left censored or not  |
|                               |             |                          |                                                                       |
|                               |             |                          | Only used when ``observed_lifetimes`` shape is (n,)                   |
+-------------------------------+-------------+--------------------------+-----------------------------------------------------------------------+
| ``right_censored_indicators`` |   ``None``  | ``np.bool_`` or ``None`` | shape : (n,)                                                          |
|                               |             |                          |                                                                       |
|                               |             |                          | Boolean indicators used to tag observations are right censored or not |
|                               |             |                          |                                                                       |
|                               |             |                          | Only used when ``observed_lifetimes`` shape is (n,)                   |
+-------------------------------+-------------+--------------------------+-----------------------------------------------------------------------+
|           ``entry``           |   ``None``  |       ``np.float``       | shape : (n,)                                                          |
|                               |             |                          |                                                                       |
|                               |             |                          | Left truncations values (value is 0 for untruncated lifetimes)        |
+-------------------------------+-------------+--------------------------+-----------------------------------------------------------------------+
|         ``departure``         |   ``None``  |       ``np.float``       | shape : (n,)                                                          |
|                               |             |                          |                                                                       |
|                               |             |                          | Right truncations values (value is 0 for untruncated lifetimes)       |
+-------------------------------+-------------+--------------------------+-----------------------------------------------------------------------+


All arrays have the **same length** but not the same shape. ``observed_lifetimes`` can be either 1d or 2d array.

.. note::
    One can't specify indicators arrays tagging lifetimes as right censored and left censored at the same time. For interval censored lifetimes,
    prefer using 2d lifetimes as below. 
    

Lifetimes as 1d array
---------------------

.. code-block:: python
    
    import numpy as np

    observed_lifetimes = np.array([10, 11, 9, 10, 12, 13, 11])
    event = np.array([1, 0, 1, 0, 0, 0, 1])
    entry = np.array([0, 0, 3, 5, 3, 1, 9])

    complete_indicators = event == 1
    right_censored_indicators = event == 0


When dealing with 1d lifetimes, if no indicators are written, all observed lifetimes are considered complete. Here we would have the following
information : 

+-----------+--------------------------+-------------------------+
| **Index** |      **Information**     |        **Values**       |
+-----------+--------------------------+-------------------------+
|     0     | complete, no truncations | **lifetime** : 10       |
|           |                          |                         |
|           |                          | **left truncation** : 0 |
+-----------+--------------------------+-------------------------+
|     1     | complete, no truncations | **lifetime** : 10       |
|           |                          |                         |
|           |                          | **left truncation** : 0 |
+-----------+--------------------------+-------------------------+
|     2     | complete, no truncations | **lifetime** : 10       |
|           |                          |                         |
|           |                          | **left truncation** : 0 |
+-----------+--------------------------+-------------------------+
|     3     | complete, no truncations | **lifetime** : 10       |
|           |                          |                         |
|           |                          | **left truncation** : 0 |
+-----------+--------------------------+-------------------------+
|     4     | complete, no truncations | **lifetime** : 10       |
|           |                          |                         |
|           |                          | **left truncation** : 0 |
+-----------+--------------------------+-------------------------+
|     5     | complete, no truncations | **lifetime** : 10       |
|           |                          |                         |
|           |                          | **left truncation** : 0 |
+-----------+--------------------------+-------------------------+
|     6     | complete, no truncations | **lifetime** : 10       |
|           |                          |                         |
|           |                          | **left truncation** : 0 |
+-----------+--------------------------+-------------------------+

.. warning::

    Of course, one can specify ``left_censored_indicators`` too. In such case, indicators can't tag lifetimes as right censored and left censored too.
    This is one major limitation of the 1d lifetime format. It can't handle interval censorship. That's why one may want to use 2d lifetime format.


Lifetimes as 2d array
---------------------

.. code-block:: python

    import numpy as np

    observed_lifetimes = np.array(
        [[1, 2], [0, 4], [5, 5], [7, np.inf], [10, 10], [2, 10], [10, 11]]
    )
    entry = np.array([0, 0, 3, 5, 3, 1, 9])
    departure = np.array([4, 0, 7, 10, 0, 12, 0])


2d lifetimes is more complex but encodes more types of censorships. All lifetimes are interval encoded as follow :

+-------------------+-----------------+-----------------+
|   **Censorship**  | **Lower bound** | **Upper bound** |
+-------------------+-----------------+-----------------+
|        None       |      ``x``      | ``x``           |
+-------------------+-----------------+-----------------+
|   left censored   |      ``0``      | ``x_l``         |
+-------------------+-----------------+-----------------+
|   right censored  |     ``x_r``     | ``np.inf``      |
+-------------------+-----------------+-----------------+
| interval censored |     ``x_r``     | ``x_l``         |
+-------------------+-----------------+-----------------+

Here we would have the following information : 

+-----------+---------------------------------------+---------------------------------+
| **Index** |            **Information**            |            **Values**           |
+-----------+---------------------------------------+---------------------------------+
|     0     |   interval censored, right truncated  | **lifetime** between 1 and 2    |
|           |                                       |                                 |
|           |                                       | **right truncation** : 4        |
+-----------+---------------------------------------+---------------------------------+
|     1     |     left censored, no truncations     | **lifetime** less than 4        |
+-----------+---------------------------------------+---------------------------------+
|     2     |      complete, interval truncated     | **lifetime** is 5               |
|           |                                       |                                 |
|           |                                       | **left truncation** : 3         |
|           |                                       |                                 |
|           |                                       | **right truncation** : 7        |
+-----------+---------------------------------------+---------------------------------+
|     3     |   right censored, interval truncated  | **lifetime** more than 7        |
|           |                                       |                                 |
|           |                                       | **left truncation** : 5         |
|           |                                       |                                 |
|           |                                       | **right truncation** : 10       |
+-----------+---------------------------------------+---------------------------------+
|     4     |        complete, left truncated       | **lifetime** is 10              |
|           |                                       |                                 |
|           |                                       | **left truncation** : 3         |
+-----------+---------------------------------------+---------------------------------+
|     5     | interval censored, interval truncated | **lifetime**  between 2 and 10  |
|           |                                       |                                 |
|           |                                       | **left truncation** : 1         |
|           |                                       |                                 |
|           |                                       | **right truncation** : 12       |
|           |                                       |                                 |
+-----------+---------------------------------------+---------------------------------+
|     6     |   interval censored, left truncated   | **lifetime**  between 10 and 11 |
|           |                                       |                                 |
|           |                                       | **left truncation** : 9         |
+-----------+---------------------------------------+---------------------------------+

.. warning::

    When using 2d lifetimes, no censorship indicators arrays are required
    nor allowed in the next steps.