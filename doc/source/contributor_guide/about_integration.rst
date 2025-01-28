About integrations
==================

Many code blocks depend upon ``ls_integrate``, especially in the renewal
package. This method relies on ``support_upper_bound`` and
``support_lower_bound`` properties of model. Because these properties
only exist for ``ls_instagrate`` operations, the ISP and SRP principles
tend to delete them from the ``Model`` interface and delegate their
usage in ``ls_integrate``. As a consequence, ``mrl`` must also be
overridden in derived class where ``support_upper_bound`` is not
``np.inf``.

``ls_integrate`` implementation might vary from one concrete ``Model``
to another. The obvious question is : should ``Model`` interfaces
contain ``ls_integrate`` method. One can say that this operation is only
used to make other operations (moment computation, etc.) and would not
be used by “normal” users. Then, it may be good thing to decouple
``Model`` from ``ls_integrate`` and make ``Model``-objects use
``ls_integrate``. One can also consider ``ls_integrate`` as an usefull
request for advanced mathematical users and no seperate it from
``Model`` interface.

For now, ``ls_integrate`` won’t be seperated from ``Model`` interface
and its base implementation might be overriden in concrete class.

**``func`` argument is a callable that only expects one ``np.ndarray``
as input and return ``np.ndarray`` as output. If one wants to add args,
he must use ``functools.partial``.**

Another problem is that ``ls_integrate`` relied on ``ndim`` argument
which was basically the maximum number of dimension of all array
variables used in the integrated function. It mainly looks at ``*args``
variables but sometimes ``time`` is also a variable in the integrand
(see ``mrl``). To avoid having to specify ndim depending on the variable
shapes at run time, now ``ls_integrate`` automatically convert all
variables in 2d and squeeze the result. This feature is permitted
because variables can’t have more than 2d. Concretely, it uses
``np.atleast_2d`` for both ``args`` and ``integrand`` result.
