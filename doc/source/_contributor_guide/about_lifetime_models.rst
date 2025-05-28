About lifetime models
=====================

When a model is composed of several parametric parts, which can
themselves be other ``ParametricModel`` objects, it can be challenging
to create a new model without disrupting the overall logic of the ReLife
implementations. To assist with the implementation of new statistical
models, we have implemented *framework* objects called ``Parameters``
and ``ParametricComponent``. These objects provide helpful properties
and operations to simplify the creation of new parametric models.

The ``Parameters`` object is used to encode the parameters of a whole
model while preserving the composition structure of models. It can be
seen as a tree of parameter sets that follows the tree of model
composition. Additionally, it provides a set of helper properties and
operations that make it easier to define and work with the parameters of
a parametric model.

The ``ParametricComponent`` object, on the other hand, is used to
represent a single parametric part of a model. It is composed of a
``Parameters`` instance and provides a unified interface for adding new
components to existing ones, similar to function composition. It can be
seen as a tree of parametric functions.

By using these framework objects, it is possible to more easily create
new parametric models that are composed of multiple parts, without
having to worry about disrupting the overall logic of the ReLife
implementations.

``Parameters``
--------------

In previous versions of ReLife, model parameters were encapsulated in a
varying-sized list of floats, depending on the number of parameters. The
``Parameters`` object is a more powerful alternative to this simple
list, and it has been implemented to encapsulate the structure and
values of the parameters. This object is responsible for automatically
adapting the storage of parameters in a ``ParametricComponent`` and
ensuring that parameters follow the structure of model composition. We
have adapted the `composite
pattern <https://en.wikipedia.org/wiki/Composite_pattern>`__ to encode
parameters in a tree structure that follows the composition of the
model. Our ``Parameters`` objects can be seen as a tree of dictionaries.

Each node in the tree has:

-  A dictionary of parameter names and values
-  All parameter names and values stored in lists (including those from
   the current node and all leaf nodes)

This allows for multiple nodes to have the same parameter names.
Separating the dictionary of current node parameters and lists of all
parameters is mainly due to computation constraints. When operations
modify all parameters of a model, it avoids the need to read the entire
parameter tree each time.

Each node can answer several requests, including:

-  Getting/setting all parameter values from a node (updating the
   current node’s parameter values and those of its leaves)
-  Getting/setting all parameter names from a node (updating the current
   node’s parameter names and those of its leaves)
-  Getting/setting a parameter from its name at the node level (avoiding
   the ability to call other node parameters by name to prevent naming
   conflicts)
-  Modifying the entire set of parameter names and values, including
   their number (updating the current node’s dictionary of parameters)

I hope this revised text is clear and helpful. Let me know if you have
any further questions or concerns.

**improvements :** - replace composite by
`namedtuple <https://docs.python.org/fr/3/library/collections.html#collections.namedtuple>`__
and do not store all params and all names (not really used)

``ParametricComponent``
-----------------------

The ``ParametricComponent`` object also follows a tree-like structure,
as it stores other ``ParametricComponent`` objects in its ``leaves``
attribute. It benefits from the ``__getattr__`` and ``__setattr__``
Python magic methods, which create a sort of bridge with the parameters
data in order to call parameters by their names inside methods.

A ``ParametricComponent`` object has the following attributes:

-  ``params``: This is composed of a ``Composite`` object and contains
   the parameters for the current node.
-  ``leaves``: This is a list of other ``ParametricComponent`` objects.

The ``ParametricComponent`` object has the following methods:

-  ``compose_with``: This method adds a ``ParametricComponent`` object
   as a leaf that can be called from the current node.
-  ``new_params``: This method changes the local parameters structure at
   the node level.

``LifetimeModel`` abstract baseclass : implementation control
-------------------------------------------------------------

The ``LifetimeModel`` class provides an abstract base interface consisting of survival probability functions.
At this level, the methods ``sf``, ``hf``, ``chf``, and ``pdf`` are marked as abstract methods, meaning
they must be implemented in any derived class. It is worth noting that these abstract methods have a default implementation
that is conditionally based on the existence of other methods. For example, if a derived class implements ``hf`` and ``pdf``,
it can use the ``super`` mechanism to call the default implementation of ``sf`` instead of providing a concrete formula.

For those who are new to Python, it may seem cumbersome to rewrite a method that already has a default implementation and
simply call ``super`` within it. This is partially true. It is also important to recognize that a new contributor only needs
to read the derived class to understand the entire interface of its instances. In object-oriented programming (OOP), this explicitness
is particularly valuable. Additionally, in this case, ``LifetimeModel`` is a **variadic** generic abstract class,
where concrete methods can have variadic parameterization. Therefore, the docstring used to document the object interface
must be specified in each case.

A potential workaround for implementing a derived class of ``LifetimeModel`` could involve calling ``super`` for each abstract method.
This approach works at compile time since all abstract methods are implemented in the code; however, it will result
in a ``RecursionError`` at runtime if any of these methods are called.
One might consider using another Python feature, such as metaclasses, to exert more control over these classes.
However, we believe that using metaclasses would complicate the code significantly, while the `abc` module is a well-known
and established solution.

Finally, when examining a derived class of ``LifetimeModel``, one may notice that some methods are tagged with the
`override` decorator. This decorator is used solely for static type checking tools (such as mypy) to ensure that the overridden
method does not alter the base signature and simply provides an alternative implementation.
Thus, it is applied whenever a non-abstract method is overridden, either for documentation purposes
or to offer a more suitable or straightforward implementation.

**improvements** :

-  with ``__init_subclass__`` read methods signature recursively in
   order to to parse \*args names and to fill args_names and nb_args

Variadic model ``args`` : ``LifetimeModel`` is ``Generic``
----------------------------------------------------------

In previous versions of ReLife, the unpacking operator ``*`` was used to
create an infinite number of arguments that could be passed to a
function. This allowed the ``LifetimeModel`` interface to be responsive
to a variadic number of extra arguments in methods signatures when the
model was composed of other models. The following piece of code
illustrates this idea in the case of a regression model:

.. code:: python

   class LifetimeModel:
       ...
       def hf(self, time: NDArray[np.float64], *args: NDArray[np.float64]):...

   class ProportionalHazard(LifetimeModel):
       baseline : LifetimeModel
       ...
       def hf(self, time: NDArray[np.float64], covar : NDArray[np.float64], *args: NDArray[np.float64]):...
           return self.covar_effect.g(covar) * self.baseline.hf(time, *args)

In this example, ``ProportionalHazard`` objects are composed of any
other ``LifetimeModel`` instance and inherit the ``LifetimeModel``
interface in order to reuse the base implementation of probability
functions if needed. However, ``ProportionalHazard`` extends the ``hf``
signature with one extra argument named ``covar`` to explicitly tell
users that in its case ``*args`` must have at least one ``covar``
object. The ``*args`` parameter also allows
``model = ProportionalHazard(AFT(AFT(...(Weibull())`` to run, because if
one wants to request ``model.hf``, the number of arguments that must be
passed varies and is spread recursively in the chain of ``baseline``
composition.

However, typing rules can be easily fooled or misrespected if one is not
careful. In the previous example, strictly speaking,
``ProportionalHazard`` overrides the ``hf`` signature and violates the
Liskov Substitution Principle (LSP): ``hf`` expects
``[float, tuple[float, ...]]`` in ``LifetimeModel``, but
``[float, float, tuple[float, ...]]`` in ``ProportionalHazard``.

To handle correct type hinting and avoid issues related to the problem
explained above, ReLife uses ``TypeVarTuple`` introduced in Python 3.11.
This allows ``LifetimeModel`` to act as a
`template <https://en.wikipedia.org/wiki/Template_(C%2B%2B)>`__,
enabling parametric polymorphism and variadic args.

Here is an example of how this can be implemented using
``TypeVarTuple``:

.. code:: python

   VariadicArgs = TypeVarTuple("VariadicArgs")

   class LifetimeModel(Generic[*VariadicArgs]):
       ...
       def hf(self, time: NDArray[np.float64], *args: *VariadicArgs):...

   ModelArgs = tuple[NDArray[np.float64], ...]

   class ProportionalHazard(LifetimeModel[NDArray[np.float64], *ModelArgs]):
       baseline : LifetimeModel[*ModelArgs]
       ...
       def hf(self, time: NDArray[np.float64], covar : NDArray[np.float64], *args: *ModelArgs):...
           return self.covar_effect.g(covar) * self.baseline.hf(time, *args)

In this example, ``VariadicArgs`` is a type variable that can be any
*tuple* of types. Concrete implementation, like ``ProportionalHazard``
can specify the expected *tuple* of types while still maintaining
correct type hinting. Here, ``ProportionalHazard`` expects this tuple of
types as extra arguments :
``tuple[NDArray[np.float64], *ModelArgs] = tuple[NDArray[np.float64], *tuple[NDArray[np.float64], ...]]``
meaning a tuple consisting of at least one ``NDArray[np.float64]`` as
first element followed by zero or more ``NDArray[np.float64]``. Note
that ``tuple[NDArray[np.float64], *tuple[NDArray[np.float64], ...]]``
cannot be rewritten as ``tuple[NDArray[np.float64], ...]`` as it would
mean a tuple consisting of zero or more ``NDArray[np.float64]``.

``LifetimeData`` factory
------------------------

The ``ParametricLifetimeModel`` fitting process uses a ``Likelihood``
object to estimate model parameters. In survival analysis, the
contribution of each observation to the likelihood depends on the type
of lifetime observation (complete, right censored, etc.) and any
truncations. Therefore, it is necessary to parse the data provided by
users and categorize each observation.

To accomplish this task, we use ``LifetimeReader`` objects, which are
responsible for parsing lifetime data. These objects are then used in a
factory object called ``lifetime_data_factory`` to construct a
``LifetimeData`` object. This object encapsulates each group of lifetime
data in an ``IndexedData`` object, which keeps track of the index of the
original data.

``IndexedData`` can be thought of as a simplified version of
``pandas.Series`` that only allows for the intersection or union of data
based. For example, you can use: - ``intersection(*others)`` to get
observations that are left truncated and complete. - ``union(*others)``
to get observations that are complete or right censored.

Additionally, all values of lifetime data are stored as 2D arrays, which
makes probability computations more homogeneous in cases where there are
covariates.

**Why a factory ?** The advantage of using a factory is that it
decouples the process of reading data and creating ``LifetimeData``
objects. This makes it much easier to create variations of the reader
process if needed and isolate code in a cleaner way.

Other considerations
--------------------

There are a few constraints that must be followed when using the
``ParametricModel`` object:

-  At the model level, a user cannot request methods of a model if one
   of the ``params`` values is ``np.nan``. All parameter values must be
   passed at the instantiation or the empty model must be fit before any
   requests are made.
-  At the model level, ``params`` cannot be set individually or by name.
   The user can only set all param values at once using a single setter.
   If a user wants to control ``params`` names, they can use the
   ``params_names`` getter or the string representation of the instance.
