from functools import partial
from typing import TYPE_CHECKING, Callable, NewType, Optional

import numpy as np
from numpy.typing import NDArray

from relife.economic.discounting import Discounting

if TYPE_CHECKING:
    from relife.model import FrozenLifetimeModel


Rewards = NewType(
    "Rewards",
    Callable[[NDArray[np.float64]], NDArray[np.float64]],
)


def _age_replacement_rewards(
    durations: NDArray[np.float64],
    ar: float | NDArray[np.float64],
    cf: float | NDArray[np.float64],
    cp: float | NDArray[np.float64],
) -> NDArray[np.float64]:
    return np.where(durations < ar, cf, cp)


def age_replacement_rewards(
    ar: float | NDArray[np.float64],
    cf: float | NDArray[np.float64],
    cp: float | NDArray[np.float64],
) -> Rewards:
    return partial(_age_replacement_rewards, ar=ar, cf=cf, cp=cp)


def _run_to_failure_rewards(
    durations: NDArray[np.float64],
    cf: float | NDArray[np.float64],
) -> NDArray[np.float64]:
    return np.ones_like(durations) * cf


def run_to_failure_rewards(cf: float | NDArray[np.float64]) -> Rewards:
    return partial(_run_to_failure_rewards, cf=cf)


def reward_partial_expectation(
    timeline: NDArray[np.float64],
    model: FrozenLifetimeModel,
    rewards: Rewards,
    *,
    discounting: Optional[Discounting] = None,
) -> NDArray[np.float64]:
    def func(x):
        return rewards(x) * discounting.factor(x)

    ls = model.ls_integrate(func, np.zeros_like(timeline), timeline)
    # reshape 2d -> final_dim
    ndim = max(map(np.ndim, (timeline, *model.args)), default=0)
    if ndim < 2:
        ls = np.squeeze(ls)
    return ls
