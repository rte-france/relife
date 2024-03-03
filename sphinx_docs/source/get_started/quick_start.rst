Quick start
===================

.. role:: python(code)
   :language: python


.. image:: ../img/main_process.png
    :scale: 100 %
    :align: center


Data collection
---------------

First get lifetimes data in numpy array. For instance :

.. code-block:: python
    
    import numpy as np

    1d_data = {
        "observed_lifetimes": np.array([10, 11, 9, 10, 12, 13, 11]),
        "event": np.array([1, 0, 1, 0, 0, 0, 1]),
        "entry": np.array([0, 0, 3, 5, 3, 1, 9]),
    }

.. seealso::
    For more details, please read :doc:`data_collect`.

Data exploration
----------------

Then instanciate a :py:func:`~relife2.survival.data.databook` databook :

.. code-block:: python

    from relife2.data import databook

    db = databook(
        observed_lifetimes = 1d_data["observed_lifetimes"],
        right_censored_indicators = 1d_data["event"] == 0,
        complete_indicators = 1d_data["event"] == 1,
        entry = 1d_data["entry"],
    )

.. seealso::
    For more details, please read :doc:`data`.

Lifetime data analysis
----------------------
    
You're now ready to use ReLife models like this basic exponential distrubition :

.. code-block:: python

    from relife2.survival.parametric import exponential

    exponential_distri = exponential(db)

.. seealso::
    For more details, please see :doc:`survival`


Asset management policy
-----------------------
Coming soon


How to custom ReLife ?
----------------------

Some users may want to test their own implementations. We tried to make each ReLife
processes customizable. If you want to go deeper and test ReLife with your own data 
format and/or survival model please read : :doc:`../customization_guides/data` 
and :doc:`../customization_guides/survival`
