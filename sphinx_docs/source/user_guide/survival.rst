How to use survival model
==========================

.. role:: python(code)
   :language: python

.. image:: ../_img/main_process_bis_1.png
    :scale: 100 %
    :align: center

Once you've loaded your data in the correct `np.array <https://numpy.org/doc/stable/reference/generated/numpy.array.html>`_ format, you're ready to use ReLife's models
available in the survival module. In this section, we will describe every usage of a ReLife model.


Model importation
-------------------

Every models can be imported from ``relife2.survival`` module. For instance, one can catch the exponential distribution with :

.. code-block:: python

    from relife2.survival import exponential

Model instanciation
-------------------

Every models have named parameters. When you instanciate a model, one can either give parameter
values through ``*args`` or ``**kwargs``. Obviously, number of given arguments must corresponds
to parameters' model number.

.. code-block:: python
    
    exp_dist = exponential(rate = 0.00795203)

Here ``exp_dist`` has one fixed parameter that is equal to 0.00795203. Only one parameter exists as the
exponential distribution expects one parameter. The following instanciation would have been equivalent but
is less explicit

.. code-block:: python

    exp_dist = exponential(0.00795203)

In the previous commands, parameters values are given by the user. If no parameters values are given, then model parameters are initialized
at random

.. code-block:: python

    random_exp_dist = exponential()


If one wants to inspect model's parameters, print its ``params`` attribute :

.. code-block:: python

    print(exp_dist.params)
    >>> Parameter
        rate = 0.00795203

``params`` stores model's parameters in :python:`Parameter` instance. 

.. seealso::
    For more details, please read :doc:`../contributor_guide/survival`.


To catch parameters' values in one variable, just get the ``values`` attribute of ``params``:

.. code-block:: python

    exp_dist.params.values
    >>> np.array([0.00795203])


Parameters estimations
----------------------

If you want to estimate model's parameters, you have to call the :python:`fit` method. The ``fit`` method
expects lifetime data in its arguments that respect a specific format.

.. seealso::

    For more details, please read :doc:`data_collect`.


.. code-block:: python
    
    import numpy as np

    observed_lifetimes = np.array([10, 11, 9, 10, 12, 13, 11])
    event = np.array([1, 0, 1, 0, 0, 0, 1])
    entry = np.array([0, 0, 3, 5, 3, 1, 9])

    random_exp_dist.fit(
        observed_lifetimes,
        complete_indicators = event == 1,
        right_censored_indicators = event == 0,
        entry = entry,
    )

After that, the model instance holds a :python:`fitting_params` and a :python:`fitting_results`
attribute. The former gives the values of fitting parameters. The latter stores information
about the estimations like the standard error derived from the information matrix. One can see
the fitting parameters values with a print : 

.. code-block:: python

    print(random_exp_dist.fitting_params)
    >>> Parameter 
        rate = 0.054545454630883686


As before, if one wants to catch the values of fitting parameters in one variable, just get the
``values`` of ``fitting_params``

.. code-block:: python

    exp_dist.fitting_params.values
    >>> np.array([0.054545454630883686])


Inference
---------

One can call model's functions to obtain their corresponding values.
For instance : 

.. code-block:: python

    random_exp_dist.sf(np.linspace(1, 10, 5))
    >>> array([0.94691547, 0.83755133, 0.74081822, 0.65525731, 0.57957828])

Here, ``sf`` values are computed with fitting parameter because model has been fitted before.
One can still   override model's parameters by adding ``params`` key-word argument.

.. code-block:: python

    random_exp_dist.sf(np.linspace(1, 10, 5), params=0.005)
    >>> array([0.99501248, 0.98388132, 0.97287468, 0.96199118, 0.95122942])

.. warning::

    If model's parameters are initialized at random and model has not been fitted yet, calling
    a function without specifying ``params`` will raise an error encouraging you to fit the model first 
    or to specify parameters as above. 
