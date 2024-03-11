Lifetime data format
====================

.. role:: python(code)
   :language: python

.. image:: ../img/main_process_bis_0.png
    :scale: 100 %
    :align: center

In ReLife, collected data must be loaded in `np.array <https://numpy.org/doc/stable/reference/generated/numpy.array.html>`_ 
ReLife can carry either 1d-array or 2d-array observed lifetimes. This section describes in-depth how to handle these two
format and when to use them




1d lifetimes
------------

.. code-block:: python
    
    import numpy as np

    observed_lifetimes = np.array([10, 11, 9, 10, 12, 13, 11])
    event = np.array([1, 0, 1, 0, 0, 0, 1])
    entry = np.array([0, 0, 3, 5, 3, 1, 9])



2d lifetimes
------------
.. code-block:: python

    import numpy as np

    observed_lifetimes = np.array(
        [[1, 2], [0, 4], [5, 5], [7, np.inf], [10, 10], [2, 10], [10, 11]]
    )
    entry = np.array([0, 0, 3, 5, 3, 1, 9])
    departure = np.array([4, 0, 7, 10, 0, 12, 0])


As you see :python:`first_data` and :python:`second_data` only differs in the shape of 
:python:`observed_lifetimes`. The advantage of 2d-array observed lifetimes is that this
data format inherently carries censorship information whereas with 1d-array, one must
add indicators of censorship (here it is handled by :python:`event` array). 

.. warning::
    When using 2d-array observed lifetimes, no censorship indicators arrays are required
    nor allowed in the next steps. Moreover, 1d-array observed lifetimes can't hold interval censorships.