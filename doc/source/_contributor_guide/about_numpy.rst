About numpy
===========

User should know numpy
----------------------

We expect user to know ``numpy`` and to understand basics of
`broadcasting <https://numpy.org/doc/stable/user/basics.broadcasting.html>`__.
As a consequence, most methods inputs are supposed to be ``np.array``
object and we **did not add control on user inputs nor user inputs
automatic conversion**

Numpy objects dim
-----------------

All numpy objects inputs never have more than 2 dimensions.

-  ``0d-array`` : one point of measure
-  ``1d-array`` : :math:`n` points of measure on 1 unit, shape
   :math:`(n,)`
-  ``2d-array`` : :math:`n` points of measure on :math:`m` units, shape
   :math:`(m,)`

Subsystems operations are written in numpy meaning *broadcastable*. So
we expect user to write correctly its inputs and to know in advance what
kind of shape is expecting.
