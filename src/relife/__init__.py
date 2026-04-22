from contextlib import suppress
from importlib.metadata import PackageNotFoundError, version

from . import (
    datasets,
    lifetime_models,
    policies,
    quadratures,
    rewards,
    stochastic_processes,
    utils,
)

with suppress(PackageNotFoundError):
    __version__ = version("relife")

__all__ = [
    "datasets",
    "lifetime_models",
    "policies",
    "stochastic_processes",
    "quadratures",
    "rewards",
    "utils",
]
