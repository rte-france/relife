How to use databook
====================

.. role:: python(code)
   :language: python

When you start using ReLife, you first need to load your survival data in a databook. 

.. note::

    To understand more clearly how databook works, please refer to databook section. The 
    databook stores your data in a specific format and allows you to call specific data fields 
    depending on your needs.

Databook takes numpy arrays as inputs. Observed lifetimes can either be 1d-array or 
2d-array, see the examples below.

.. code-block:: python
    
    import numpy as np

    1d_data = {
        "observed_lifetimes": np.array([10, 11, 9, 10, 12, 13, 11]),
        "event": np.array([1, 0, 1, 0, 0, 0, 1]),
        "entry": np.array([0, 0, 3, 5, 3, 1, 9]),
    }

    2d_data = {
        "observed_lifetimes": np.array(
            [[1, 2], [0, 4], [5, 5], [7, np.inf], [10, 10], [2, 10], [10, 11]]
        ),
        "entry": np.array([0, 0, 3, 5, 3, 1, 9]),
        "departure": np.array([4, 0, 7, 10, 0, 12, 0]),
    }

As you see :python:`1d_data` and :python:`2d_data` only differs in the shape of 
:python:`observed_lifetimes`. The advantage of 2d-array observed lifetimes is that this
data format inherently carries censorship information whereas with 1d-array, one must
add indicators of censorship (here it is handled by :python:`1d_data` array). Then a
databook is created as follow:


Instanciate a databook
----------------------

.. code-block:: python

    from relife2.data import databook

    first_db = databook(
        observed_lifetimes = 1d_data["observed_lifetimes"],
        right_censored_indicators = 1d_data["event"] == 0,
        complete_indicators = 1d_data["event"] == 1,
        entry = 1d_data["entry"],
    )

    second_db = databook(
        observed_lifetimes = 2d_data["observed_lifetimes"],
        entry = 2d_data["entry"],
        departure = 2d_data["departure"],
    )

As mentionned before, with 1d-array lifetimes, censored lifetimes must be explicitly
tagged through indicators. Here :python:`event` can serve both :python:`right_censored_indicators`
and :python:`complete_indicators`.


Databook manipulations
----------------------

Now, lifetimes data can be explored very easily. For instance, one might get every
complete lifetimes. To do so just call:

.. code-block:: python

    first_db("complete").values

.. code-block:: python

    second_db("complete").values


These commands will return 1d-array containing complete lifetimes values. The first one
returns :python:`np.array([10, 9, 11])` and the latter returns :python:`np.array([5, 10])`.
One can also get corresponding data index. Just replace :python:`.values` by :python:`.index`.

Databook can do more. One might wants to access lifetimes being complete **and** left truncated.
To do so, one can use the "and" operator as follow : 

.. code-block:: python

    first_db("complete & left_truncated")

This command returns 2 objects which contain complete and left truncations values/index.
To access To get the values of complete lifetimes being left truncated, just call :

.. code-block:: python

    first_db("complete & left_truncated")[0].values

It must returns :python:`np.array([9, 11])`. Inversly, to get left truncations values of
complete lifetimes, call:

.. code-block:: python

    first_db("complete & left_truncated")[1].values

It must returns :python:`np.array([3, 9])`. The "or" operator can also be used. For instance :

.. code-block:: python

    first_db("complete | left_truncated")[0].values


It returns all complete lifetimes :python:`np.array([10, 9, 11])`.

Finally, a convenient method of databook is :python:`info`. It summarizes all the databook
content in one view :

.. code-block:: python

    first_db.info