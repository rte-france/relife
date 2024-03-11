Quick start
===================

.. role:: python(code)
   :language: python


.. image:: img/main_process_bis.png
    :scale: 100 %
    :align: center


Data collection
---------------

First get lifetime data in `np.array <https://numpy.org/doc/stable/reference/generated/numpy.array.html>`_.
For instance :

.. code-block:: python
    
    import numpy as np

    observed_lifetimes = np.array([10, 11, 9, 10, 12, 13, 11])
    event = np.array([1, 0, 1, 0, 0, 0, 1])
    entry = np.array([0, 0, 3, 5, 3, 1, 9])

Here, ``observed_lifetimes`` are collected lifetimes, ``event`` is an indicator whose value is 1 
when lifetimes are complete observations but 0 when they are right censored. ``entry`` are left
truncation values.

.. seealso::
    For more details, please read :doc:`user_guide/data_collect`.

Lifetime data analysis
----------------------
    
Several survival models are available in ReLife. A curated list can be found in ... Every ReLife's
survival models are in the ``relife2.survival`` module and must be imported from there.

.. code-block:: python
    
    from relife2.survival.parametric import *

This command imports all parametric survival models of ReLife. At model's instanciation, one can either :

1. specify model's parameters
2. let parameters initialization be random


.. code-block:: python
    
    exp_dist = exponential(rate = 0.00795203) # or just exponential(0.00795203)
    random_exp_dist = exponential()

One may wants to see model's parameter at this step. Just print ``params`` :

.. code-block:: python

    print(exp_dist.params)
    >>> Parameter
        rate = 0.00795203

.. code-block:: python

    print(random_exp_dist.params)
    >>> Parameter
        rate = 0.14797189320089466

In each case, models can be fitted to given data using the ``fit`` method. 

.. code-block:: python
    
    random_exp_dist.fit(
        observed_lifetimes,
        complete_indicators = event == 1,
        right_censored_indicators = event == 0,
        entry = entry,
    )

Then, one can print the fitting parameters :

.. code-block:: python

    print(random_exp_dist.fitting_params)
    >>> Parameter 
        rate = 0.054545454630883686

.. seealso::
    For more details, please see :doc:`user_guide/survival`

For inference, just call the desired function method. For instance : 

.. code-block:: python

    random_exp_dist.sf(np.linspace(1, 10, 5))
    >>> array([0.94691547, 0.83755133, 0.74081822, 0.65525731, 0.57957828])

Here, ``sf`` values are computed with fitting parameter because model has been fitted before.
One can still   override model's parameters by adding ``params`` key-word argument.

.. code-block:: python

    random_exp_dist.sf(np.linspace(1, 10, 5), params=0.005)
    >>> array([0.99501248, 0.98388132, 0.97287468, 0.96199118, 0.95122942])

Asset management policy
-----------------------
Coming soon


How to custom ReLife ?
----------------------

Some users may want to test their own implementations. We tried to make each ReLife
processes customizable. If you want to go deeper and test ReLife with your own data 
format and/or survival model please read : :doc:`contributor_guide/data` 
and :doc:`contributor_guide/survival`
