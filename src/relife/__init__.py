from contextlib import suppress
from importlib.metadata import PackageNotFoundError, version

from . import (
    datasets,
    economic,
    lifetime_model,
    policy,
    quadrature,
    stochastic_process,
    utils,
)

with suppress(PackageNotFoundError):
    __version__ = version("relife")

__all__ = [
    "datasets",
    "lifetime_model",
    "policy",
    "stochastic_process",
    "quadrature",
    "economic",
    "utils",
]
