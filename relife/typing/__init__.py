"""
``relife.typing``
=================

The ReLife typing module exposes various types to type hint codes
using relife.

"""

from ._models import AnyParametricLifetimeModel
from ._random import Seed
from ._scalars import AnyFloat, NumpyBool, NumpyFloat
from ._scipy import MethodMinimize, MaximumLikelihoodOptimizerOptions

__all__ = [
    "AnyParametricLifetimeModel",
    "Seed",
    "AnyFloat",
    "NumpyBool",
    "NumpyFloat",
    "MaximumLikelihoodOptimizerOptions",
    "MethodMinimize",
]
