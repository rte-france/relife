Stochastic process sampling
===========================

Suppose we want to sample lifetimes given an ``end_time`` and a sampling
``size``. The first and easiest way to visualize the sampling process is
to consider one asset :

::

   0 1 2 -> samples_index
   -----
   4 2 4 -> it.1
   1 5 2 -> it.2
   2 4 3 -> it.3
   2 . 2 -> it.4
   3 . . -> it.5
   . . . -> StopIteration

As you can see, the sampling generates a sequence of lifetime values per
sample index (here ``size`` = 3). The sequences generated vary in length
depending on whether the cumulative sum of the durations has reached the
time limit (here ``end_time``\ =10).

Sometimes, one wants to generate lifetimes for different assets. In that
case, the number of sequences equals the ``size * nb_assets``

::

   0 0 0 1 1 1 2 2 2 -> samples_index
   0 1 2 0 1 2 0 1 2 -> assets_index
   -----
   4 2 4 2 5 1 2 4 7 -> it.1
   1 5 2 3 6 1 1 4 5 -> it.2
   2 4 3 4 . 8 2 2 . -> it.3
   2 . 2 3 . 4 3 . . -> it.4
   3 . . . . . 1 . . -> it.5
   . . . . . . 5 . . -> it.6
   . . . . . . . . . -> StopIteration

A simple storage of the generated data would be to translate the array
structure shown above in 2d-array, where missing elements are encoded by
``np.nan`` or masked in ``MaskArray``-object. The disadvantage of this
approach is that it can severely overload memory if the number of masked
elements generated becomes very large, as in very large sampling. A
better approach is to store the elements in a compacted 1d-array like
this :

::

   0 0 0 0 0 1 1 1 2 2 2 2 -> samples_index
   -----------------------
   4 1 2 2 3 2 5 4 4 2 3 2

::

   0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 ... -> samples_index
   0 0 0 0 0 1 1 1 2 2 2 2 0 0 0 0 1 1 2 2 2 2 ... -> assets_index
   -----------------------
   4 1 2 2 3 2 5 4 4 2 3 2 2 3 4 3 5 6 1 1 8 4 ...

This format is lighter, but requires some index manipulation to easily
slice on generated data.

Advantages of generator approach
--------------------------------

At a first glimps, used generator approach in ReLife2 only encapsulates
lifetime generation routine in one objet that keeps in memory previous
states without recomputing them many times. It is exactly what basic
while loop did. But, it offers to advantages in comparison to a while
loop :

-  It provides a convenient way to pass other generator routine without
   creating another RenewalProcess class
-  It avoids code duplication (see init_loop and main_loop) : same
   generation, only model changes. First generation has only to send its
   results to main generator

**Solutions**

Lifetime generators are first parametrized with : ``nb_assets``,
``nb_samples``, ``args``. It allows to keep in memory the expected
rvs_size depending on ``nb_assets``,\ ``nb_samples``, ``args``.
Generators also knows ``end_time`` in order to slice uppon valid
lifetime values.

Generators receive an 1d of times then : - it yields variable number of
computed data - update data in an object

one stacks results :

::

   def stacker(*args):
       init = list(args)
       while True:
           new = yield init
           for i, x in enumerate(new):
               init[i] = np.concatenate((init[i], x))


   def generator(..., stacker):
       while True:
           try :
               lifetime = rvs ...
               assets = ...
               stacker.send(lifetime, ...)
           except:
               yield stacker
               stacker.close()
               return


``events`` and ``a0``
---------------------

The current implementation has ``events`` and ``a0`` providing data for
``ReplacementPolicyData``, which are used to construct lifetime data in
``to_lifetime_data``. However, this introduces cumbersome code in the
``sample`` functionalities.

-  if model is ``LeftTruncated``, ``a0`` must be catch for
   ``delayed_model`` only and added to generated lifetimes as a
   rectification
-  if model is ``AgeReplacementModel``, ``events`` that represents right
   censoring indicators, is conditionned on ``ar`` values

So, type checking on ``model`` is made combined with ugly numpy slicing
to retrieve correct sampled elements. One can propose easier approach
with generators : why not just writting those data inside the generation
process and not after it was made ?

**Solution :**

Generation process relies on ``rvs`` functionnality of ``LifetimeModel``
objects and ``a0`` is an ``args`` of those model type. We can modify the
``rvs`` function to directly generate rectified lifetimes by
incorporating ``a0``: ``self.baseline(*args, size=size) + a0``. This
way, we no longer need to check for the ``LeftTruncated`` model in the
``sample`` function, as the lifetimes will be correctly generated with
their final values.

Next, we can handle ``events`` more straightforwardly. In the lifetime
generator routine, we can add a check for the model type and generate
``events`` alongside lifetimes, given the ``ar`` values. The
``CountData`` can be updated to include ``events`` data, which is
consistent with the ``ReplacementPolicyData`` interface.

With these modifications, the ``to_lifetime_data`` function no longer
needs to be specific for ``ReplacementPolicyData`` subtypes. Every
``RenewalData`` can have a ``to_lifetime_data`` method, enhancing
coherence and consistency. This approach ensures that every ``sample``
method of both ``RenewalProcess`` and ``Policy`` returns objects that
can be converted to lifetime data.

``sample`` signature and ``args``
---------------------------------

The ``sample`` methods in both ``RenewalProcess`` objects and
``Policy``-like objects (see next example) result in a varying interface
due to the inclusion of ``args``-like parameters. These parameters are
necessary to customize the associated model, reward, and/or discount. I
have identified two possible solutions to this problem:

1. Keep ``sample`` as part of the interface, but encapsulate ``args``
   values in a dictionary of type ``Dict[str, Any]`` during object
   instantiation. The downside of this approach is that users must
   provide each argument value during instantiation, along with
   ``model``, ``reward``, and/or ``discount`` instances.
2. Remove ``sample`` from the interface and make it a standalone
   function (``sample(obj, nb_sample=10, ...)``) or a method within
   another object, such as ``Simulator``.

The second solution, however, still requires varying ``sample``
parametrization depending on the type of object (``obj``) passed as the
first argument. If ``obj`` is a ``RenewalProcess``, ``args`` would
correspond to ``model_args`` and optionally ``delayed_model_args``. If
``obj`` is a ``RunToFailure``, ``args`` would include ``cp``, ``cf``,
``rate``, ``cp1``, and so on. Although this approach could be
implemented using single dispatch from ``functools``, it may not be
user-friendly, as understanding the various parametrization options
would require consulting the documentation.

The first approach could be implemented using a ``Protocol`` to define a
clear and concise ``Policy`` type.\`

.. code:: python

   class Policy(Protocol):
       model: LifetimeModel[*ModelArgs],
       model_args: tuple[*ModelArgs] | tuple[()] = (),
       reward_args : Dict[str, Any],
       nb_assets: int = 1,
       a0: Optional[NDArray[np.float64]] = None,
       delayed_model: LifetimeModel[*DelayedModelArgs],
       delayed_model_args: tuple[*DelayedModelArgs] | tuple[()] = (),

       def expected_total_cost(self, timeline : NDArray[np.float64]) -> NDArray[np.float64]: ...

       def asymptotic_expected_total_cost(self) -> NDArray[np.float64]: ...

       def expected_equivalent_annual_cost(self, timeline : NDArray[np.float64]) -> NDArray[np.float64]: ....

       def asymptotic_expected_equivalent_annual_cost(self) -> NDArray[np.float64]: ...

       def sample(self, nb_samples : int, : float, random_state = None)

       def fit(self): ...

The issue of ``args`` in ``sample`` has been addressed by storing them
as a dictionary of values. Every method will retrieve the required arg
values from this dictionary. From a user’s perspective, every concrete
``Policy`` will explicitly state the names of the ``args`` needed in
``reward_args``. Only the core of the constructor will fill the
dictionary. This attribute could even be a descriptor to automatically
control and convert filled ``args`` values with respect to
``nb_assets``.

One drawback of solution 2 is that it is more aligned with the
object-oriented paradigm and may be less appealing to users who prefer
functional programming. It is true that this approach requires users to
reinstantiate the ``Policy``-like object each time they want to change
``args``. However, this only adds one additional line of code compared
to calling ``sample`` with different arguments. Furthermore, the number
of given args is significant, and it is likely that users would have
already stored them in variables. It is merely a matter of copying and
pasting the relevant variables when reinstantiating the object.

NB : ``Policy`` objects do not need nor ``reward`` or ``discount``
attribute. Discount is always exponential and ``reward`` is implicit.
