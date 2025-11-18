"""
``relife.typing``
=================

The ReLife typing module exposes various types to type hint codes
using relife.

"""

from ._models import AnyLifetimeDistribution, AnyParametricLifetimeModel
from ._random import Seed
from ._scalars import AnyFloat, NumpyBool, NumpyFloat
from ._scipy import ScipyMinimizeOptions

__all__ = [
    "AnyParametricLifetimeModel",
    "AnyLifetimeDistribution",
    "Seed",
    "AnyFloat",
    "NumpyBool",
    "NumpyFloat",
    "ScipyMinimizeOptions",
]
