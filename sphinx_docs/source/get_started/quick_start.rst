Quick start
===================

.. role:: python(code)
   :language: python

First get lifetimes data in numpy array. For instance :


.. code-block:: python
    
    import numpy as np

    1d_data = {
        "observed_lifetimes": np.array([10, 11, 9, 10, 12, 13, 11]),
        "event": np.array([1, 0, 1, 0, 0, 0, 1]),
        "entry": np.array([0, 0, 3, 5, 3, 1, 9]),
    }

Then instanciate a databook :

.. code-block:: python

    from relife2.data import databook

    db = databook(
        observed_lifetimes = 1d_data["observed_lifetimes"],
        right_censored_indicators = 1d_data["event"] == 0,
        complete_indicators = 1d_data["event"] == 1,
        entry = 1d_data["entry"],
    )

You're now ready to use ReLife models. For instance :

.. code-block:: python

    from relife2.survival.parametric import exponential

    exponential_distri = exponential(db)

For more details, please see :
* How to use databook
* How to create survival model

If you want to go deeper and test ReLife with your own data format and/or survival model 
please see:
* How to use custom databook
* How to custom survival model
