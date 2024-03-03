How to use survival model
==========================

.. role:: python(code)
   :language: python

.. image:: ../img/main_process_2.png
    :scale: 100 %
    :align: center

Once you've instanciated your databook, you're ready to use ReLife's models. ReLife models
are grouped in three modules :

* :python:`relife2.survival.parametric`
* :python:`relife2.survival.semiparametric`
* :python:`relife2.survival.nonparametric`

Every models can be imported from one of the three modules with :python:`from <module> import <model>`
For instance, one can instanciated an exponential distribution with :

.. code-block:: python

    from relife2.survival.parametric import exponential

    exponential_distri = exponential(db)

Every ReLife's model shares the same structure. They are basically a model object composed of
a :python:`functions` and an :python:`optmizer` instance.

.. seealso::
    For more details, please read :doc:`../customization_guides/survival`.

For basic a usage, one can only remember that these model objects allow the user to quickly
estimate its parameters and make inferences from its functions.

.. note::
    Avaible functions might differ from one model to another. To see which function is
    available per model.


Parameters estimations
----------------------

If you want to estimate model's parameters, you have to call the :python:`fit` method.

.. code-block:: python

    exponential_distri.fit(**kwargs)

After that, the model instance holds a :python:`fitting_params` and a :python:`fitting_results`
attribute. The former gives the values of fitting parameters. The latter stores information
about the estimations like the standard error derived from the information matrix.

* :python:`exponential_distri.fitting_params.values` returns values of fittings parameters in a numpy array
* :python:`print(exponential_distri.fitting_params)` prints the parameters and its fitting values
* :python:`exponential_distri.fitting_results.se` returns the standard error of the estimations
* :python:`exponential_distri.fitting_results.AIC` returns the AIC

Inference
---------

Once parameters have been estimated, one can call functions to obtain their corresponding values.
For instance : 

.. code-block:: python

    t = np.linspace(0, 10)
    sf_values = exponential_distri.sf(t)

It will return the :python:`sf` values of :python:`t`, here an array of shape :python:`(50,)`

Sometimes, one might wants to access function values without having to fit model's parameters.
To do so, just add :python:`params` key-word argument in the desired function. :python:`params`
has to be a 1d-array whose size corresponds to number of model parameters. For instance :

.. code-block:: python

    sf_values = exponential_distri.sf(t, params = np.array([0.00795203]))

It will return the :python:`sf` values of :python:`t` for an exponential rate of 0.00795203.

.. warning::

    Before fitting the model, its parameters values are initialized at random. In such case, calling
    a function without specifying ``params`` will raise a warning encouraging you to fit the model first 
    or to specify parameters as above. 
