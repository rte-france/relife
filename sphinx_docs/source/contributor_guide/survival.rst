How to contribute to ReLife's survival analysis ?
=================================================

.. role:: python(code)
   :language: python

In ReLife, 4 objects are generally used in models :


* ``Parameter``: this object contains parameters values and names
* ``Function`` : this object contains a ``Parameter`` object and several methods that implements expected model's functions (hazard function, survival function, mean, etc.)
* ``Likelihood``: this object contains likelihood functions computed from ``DataBook`` object
* ``Optimizer``: this object contains a ``Likelihood`` object instanciated with a ``DataBook``. It holds the ``fit`` method

If you want to define your own model, you have to customize these objects.

Function object
---------------

Function objects have the following pattern. They all inherit from a super class defining an interface
and expecting either ``param_names`` or ``nb_params`` at initialization.

.. code-block:: python

    class Function(FunctionSuperClass):
        def __init__(self, param_names = ["toto", "titi"]):
            super().__init__(self, param_names)

        def hf(self, *args) -> np.ndarray:
            # self.params.titi * args[0] + self.params.toto
            # self.params[1] * args[0] + self.params[0]
            pass


.. code-block:: python

    class Function(FunctionSuperClass):
        def __init__(self, nb_params = 3):
            super().__init__(self, nb_params)

        def hf(self, *args) -> np.ndarray:
            # self.params[0] * args[0]
            # self.params.param_0 * args[0] 
            pass

As you may have noticed, parameter values can be accessed very easily in ``Function`` methods by calling them
with their names  ``self.params.<param_name>`` or slicing ``self.params[i]``.

.. warning::
    Outputs of function methods must be 1d-array or float


Likelihood object
-----------------

Likelihood objects have the following pattern. They all inherit from a super class defining an interface
and expecting a ``databook`` object at initialization. Briefly, ``databook`` is an object that stores 
lifetime data given by the user after verifying their integrity. To know more about ``databook``, read :doc:`data`

.. code-block:: python

    class Likelihood(LikelihoodSuperClass):
        def __init__(self, databook : DataBook):
            super().__init__(self, databook)

        def negative_log_likelihood(self, functions : Function, *args) -> float: 
            pass

        def jac_hf(self, functions : Function, *args) -> np.ndarray: 
            pass


In ``Likelihood`` methods, ``functions`` object is used as an argument to access function definitions of the model.

.. warning::
    Outputs of likelihood methods must be float or 2d-array for derivates


Optimizer object
----------------

Likelihood objects have the following pattern. They all inherit from a super class defining an interface
and expecting a ``Likelihood`` object at initialization.

.. code-block:: python

    class Optimizer(OptimizerSuperClass):
        def __init__(self, likelihood : Likelihood):
            super().__init__(self, likelihood)

        def fit(functions : Function, *args, **kwargs) -> Function:
            pass


The ``fit`` method transforms ``Function`` object by modifying its parameters. 


Models' factories
-----------------

Contributions are easier with factories. In :python:`relife2.survival` module, every models are made from factories expecting
previous object definitions as arguments. The following factories are used :

* ``dist`` to create survival distribution
* etc.


For instance, in the back-end, ``exponential`` is created by calling ``dist`` like this :

.. code-block:: python

    exponential = dist(
        ExponentialDistFunction,
        ExponentialDistLikelihood,
        DistOptimizer,
    )

Here ``ExponentialDistFunction`` is the ``Function`` object of the exponential distribution, ``ExponentialDistLikelihood``
the likelihood and ``DistOptimizer`` the optimizer. If you change one of these arguments, you will create a new distribution.
