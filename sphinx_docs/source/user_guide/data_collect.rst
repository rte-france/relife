Lifetime data format
====================

.. role:: python(code)
   :language: python

.. image:: ../_img/main_process_bis_0.png
    :scale: 100 %
    :align: center

In ReLife, collected data must be loaded in `np.array <https://numpy.org/doc/stable/reference/generated/numpy.array.html>`_. ReLife can carry either 1d-array or 2d-array observed lifetimes. This section describes in-depth how to handle these two
formats. When one method expects lifetime data, its arguments are :

+-------------------------------+--------------+--------------+----------------+-------------------------------------------------------------------------------------------------------------------------+
|          **Variable**         | **Default**  |   **Value**  |    **Shape**   |                                                     **Description**                                                     |
+===============================+==============+==============+================+=========================================================================================================================+
|     ``observed_lifetimes``    |     unset    | ``np.float`` | (n,) or (n, 2) | Observed lifetime values, either 1d or 2d                                                                               |
+-------------------------------+--------------+--------------+----------------+-------------------------------------------------------------------------------------------------------------------------+
|    ``complete_indicators``    |   ``None``   | ``np.bool_`` |      (n,)      | Boolean indicators used to tag complete observations or not. Only used when ``observed_lifetimes`` shape is (n,)        |
+-------------------------------+--------------+--------------+----------------+-------------------------------------------------------------------------------------------------------------------------+
|  ``left_censored_indicators`` |   ``None``   | ``np.bool_`` |      (n,)      | Boolean indicators used to tag left censored observations or not. Only used when ``observed_lifetimes`` shape is (n,)   |
+-------------------------------+--------------+--------------+----------------+-------------------------------------------------------------------------------------------------------------------------+
| ``right_censored_indicators`` |   ``None``   | ``np.bool_`` |      (n,)      | Boolean indicators used to tag right censored observations or not. Only used when ``observed_lifetimes`` shape is (n,)  |
+-------------------------------+--------------+--------------+----------------+-------------------------------------------------------------------------------------------------------------------------+
|           ``entry``           |   ``None``   | ``np.float`` |      (n,)      | Left truncations values (value is 0 for untruncated lifetimes)                                                          |
+-------------------------------+--------------+--------------+----------------+-------------------------------------------------------------------------------------------------------------------------+
|         ``departure``         |   ``None``   | ``np.float`` |      (n,)      | Right truncations values (value is 0 for untruncated lifetimes)                                                         |
+-------------------------------+--------------+--------------+----------------+-------------------------------------------------------------------------------------------------------------------------+


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

+-----------+------------------+-----------------------------+
| **Index** | **Information**  |          **Values**         |
+===========+==================+=============================+
|     0     | - complete       | - **lifetime** is 10        |
|           | - no truncations |                             |
+-----------+------------------+-----------------------------+
|     1     | - right censored | - **lifetime** more than 11 |
|           | - no truncations |                             |
+-----------+------------------+-----------------------------+
|     2     | - complete       | - **lifetime** is 9         |
|           | - left truncated | - **left truncation** of 3  |
+-----------+------------------+-----------------------------+
|     3     | - right censored | - **lifetime** more than 10 |
|           | - left truncated | - **left truncation** of 5  |
+-----------+------------------+-----------------------------+
|     4     | - right censored | - **lifetime** more than 12 |
|           | - left truncated | - **left truncation** of 3  |
+-----------+------------------+-----------------------------+
|     5     | - right censored | - **lifetime** more than 13 |
|           | - left truncated | - **left truncation** of 1  |
+-----------+------------------+-----------------------------+
|     6     | - complete       | - **lifetime** is 11        |
|           | - left truncated | - **left truncation** of 9  |
+-----------+------------------+-----------------------------+

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


2d lifetimes encodes more types of censorships. All lifetimes are interval encoded as follow :

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

+-----------+----------------------+------------------------------------+
| **Index** |   **Information**    |             **Values**             |
+===========+======================+====================================+
|     0     | - interval censored  | - **lifetime** between 1 and 2     |
|           | - right truncated    | - **right truncation** of 4        |
+-----------+----------------------+------------------------------------+
|     1     | - left censored      | - **lifetime** less than 4         |
|           | - no truncations     |                                    |
+-----------+----------------------+------------------------------------+
|     2     | - complete           | - **lifetime** is 5                |
|           | - interval truncated | - **left truncation** of 3         |
|           |                      | - **right truncation** of 7        |
+-----------+----------------------+------------------------------------+
|     3     | - right censored     | - **lifetime** more than 7         |
|           | - interval truncated | - **left truncation** of 5         |
|           |                      | - **right truncation** of 10       |
+-----------+----------------------+------------------------------------+
|     4     | - complete           | - **lifetime** is 10               |
|           | - left truncated     | - **left truncation** of 3         |
+-----------+----------------------+------------------------------------+
|     5     | - interval censored  | - **lifetime** is between 2 and 10 |
|           | - interval truncated | - **left truncation** of 1         |
|           |                      | - **right truncation** of 12       |
+-----------+----------------------+------------------------------------+
|     6     | - interval censored  | - **lifetime** between 10 and 11   |
|           | - left truncated     | - **right truncation** of 9        |
+-----------+----------------------+------------------------------------+

.. warning::

    When using 2d lifetimes, no censorship indicators arrays are required
    nor allowed as this format contains all information of censorships inherently