How to contribute to ReLife's survival analysis ?
=================================================

.. role:: python(code)
   :language: python


In ReLife, every models are created using factory functions. We expect contributors to use them when creating
a new model to make sure that the rest of the code will run properly with this new piece. A list of available
factories is presented below. They expects model object as input. In this section, you will also find one guide for each model object.


Models' factories
-----------------

In :python:`relife2.survival` module, every models are made from factories expecting object definitions as arguments. Factories are written
in ``relife2.survival.model``. Models are built from factories in ``relife2.survival.build`` 




Models' objects
-----------------


In ReLife, 4 objects are generally used in models :


* ``Parameter``: this object contains parameters values and names
* ``Function`` : this object contains a ``Parameter`` object and several methods that implements expected model's functions (hazard function, survival function, mean, etc.)
* ``Likelihood``: this object contains likelihood functions computed from ``DataBook`` object
* ``Optimizer``: this object contains a ``Likelihood`` object instanciated with a ``DataBook``. It holds the ``fit`` method

If you want to define your own model, you have to customize these objects and use factories listed in the previous section.

Function object
^^^^^^^^^^^^^^^

Function object are written in ``function.py`` modules. 

+---------------+--------------------------------------------+
| ReLife models |               Function module              |
+===============+============================================+
| Distributions | ``relife2.survival.distribution.function`` |
|               |                                            |
+---------------+--------------------------------------------+
|  Regressions  | ``relife2.survival.regression.function``   |
+---------------+--------------------------------------------+
|               |                                            |
+---------------+--------------------------------------------+
|               |                                            |
+---------------+--------------------------------------------+
|               |                                            |
+---------------+--------------------------------------------+
|               |                                            |
+---------------+--------------------------------------------+
|               |                                            |
+---------------+--------------------------------------------+

Function objects use a specific pattern. They all inherit from a super class defining an interface
and expecting either ``param_names`` or ``nb_params`` at initialization. When you specify the ``param_names``,
the general form of ``Function`` is :

.. code-block:: python

    class Function(FunctionSuperClass):
        def __init__(self, param_names = ["toto", "titi"]):
            super().__init__(self, param_names)

        def hf(self, *args) -> np.ndarray:
            # self.params.titi * args[0] + self.params.toto
            # self.params[1] * args[0] + self.params[0]
            pass

As you may have noticed, parameter values can be accessed very easily in ``Function`` methods by calling them
with their names  ``self.params.<param_name>``. Otherwise, if you don't specify ``param_names``, you must specify
``nb_params``. 

.. code-block:: python

    class Function(FunctionSuperClass):
        def __init__(self, nb_params = 3):
            super().__init__(self, nb_params)

        def hf(self, *args) -> np.ndarray:
            # self.params[0] * args[0]
            # self.params.param_0 * args[0] 
            pass

Here, parameter values can be accessed with slicing ``self.params[i]``.

.. warning::
    Outputs of function methods must be 1d-array or float



Likelihood object
^^^^^^^^^^^^^^^^^

Function object are written in ``likelihood.py`` modules. 

+---------------+--------------------------------------------+
| ReLife models |               Likelihood module            |
+===============+============================================+
| Distributions |``relife2.survival.distribution.likelihood``|
|               |                                            |
+---------------+--------------------------------------------+
|  Regressions  | ``relife2.survival.regression.likelihood`` |
+---------------+--------------------------------------------+
|               |                                            |
+---------------+--------------------------------------------+
|               |                                            |
+---------------+--------------------------------------------+
|               |                                            |
+---------------+--------------------------------------------+
|               |                                            |
+---------------+--------------------------------------------+
|               |                                            |
+---------------+--------------------------------------------+


Likelihood objects have the following pattern. They all inherit from a super class defining an interface
and expecting a ``databook`` object at initialization. Briefly, ``databook`` is an object that stores 
lifetime data given by the user after verifying their integrity. 

.. seealso::
    To know more about ``databook``, read :doc:`data`

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
^^^^^^^^^^^^^^^^

Optimizer object are written in ``optimizer.py`` modules. 

+---------------+--------------------------------------------+
| ReLife models |               Optimizer module             |
+===============+============================================+
| Distributions |``relife2.survival.distribution.optimizer`` |
|               |                                            |
+---------------+--------------------------------------------+
|  Regressions  | ``relife2.survival.regression.optimizer``  |
+---------------+--------------------------------------------+
|               |                                            |
+---------------+--------------------------------------------+
|               |                                            |
+---------------+--------------------------------------------+
|               |                                            |
+---------------+--------------------------------------------+
|               |                                            |
+---------------+--------------------------------------------+
|               |                                            |
+---------------+--------------------------------------------+



Optimizer objects have the following pattern. They all inherit from a super class defining an interface
and expecting a ``Likelihood`` object at initialization.

.. code-block:: python

    class Optimizer(OptimizerSuperClass):
        def __init__(self, likelihood : Likelihood):
            super().__init__(self, likelihood)

        def fit(functions : Function, *args, **kwargs) -> Function:
            pass


The ``fit`` method transforms ``Function`` object by modifying its parameters. 


