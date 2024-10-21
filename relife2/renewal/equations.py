from typing import Callable, Optional, TypeVar

import numpy as np
from numpy.typing import NDArray

from relife2.model import LifetimeModel
from relife2.renewal.discounts import Discount

M = TypeVar("M", tuple[NDArray[np.float64], ...], tuple[()])
M1 = TypeVar("M1", tuple[NDArray[np.float64], ...], tuple[()])
R = TypeVar("R", tuple[NDArray[np.float64], ...], tuple[()])
R1 = TypeVar("R1", tuple[NDArray[np.float64], ...], tuple[()])
D = TypeVar("D", tuple[NDArray[np.float64], ...], tuple[()])


# make evaluated func  Callable[[array], array] only and do partial elsewhere
def renewal_equation_solver(
    timeline: NDArray[np.float64],
    model: LifetimeModel[*M],
    evaluated_func: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    *,
    model_args: M = (),
    discount: Optional[Discount[*D]] = None,
    discount_args: D = (),
) -> NDArray[np.float64]:

    tm = 0.5 * (timeline[1:] + timeline[:-1])
    f = model.cdf(timeline, *model_args)
    fm = model.cdf(tm, *model_args)
    y = evaluated_func(timeline)
    if discount is not None:
        d = discount.factor(timeline, *discount_args)
    else:
        d = np.ones_like(f)
    z = np.empty(y.shape)
    u = d * np.insert(f[:, 1:] - fm, 0, 1, axis=-1)
    v = d[:, :-1] * np.insert(np.diff(fm), 0, 1, axis=-1)
    q0 = 1 / (1 - d[:, 0] * fm[:, 0])
    z[:, 0] = y[:, 0]
    z[:, 1] = q0 * (y[:, 1] + z[:, 0] * u[:, 1])
    for n in range(2, f.shape[-1]):
        z[:, n] = q0 * (
            y[:, n]
            + z[:, 0] * u[:, n]
            + np.sum(z[:, 1:n][:, ::-1] * v[:, 1:n], axis=-1)
        )
    return z


def delayed_renewal_equation_solver(
    timeline: NDArray[np.float64],
    z: NDArray[np.float64],
    model1: LifetimeModel[*M1],
    evaluated_func: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    delayed_model_args: M1 = (),
    discount: Optional[Discount[*D]] = None,
    discount_args: D = (),
) -> NDArray[np.float64]:

    tm = 0.5 * (timeline[1:] + timeline[:-1])
    f1 = model1.cdf(timeline, *delayed_model_args)
    f1m = model1.cdf(tm, *delayed_model_args)
    y1 = evaluated_func(timeline)

    if discount is not None:
        d = discount.factor(timeline, *discount_args)
    else:
        d = np.ones_like(f1)
    z1 = np.empty(y1.shape)
    u1 = d * np.insert(f1[:, 1:] - f1m, 0, 1, axis=-1)
    v1 = d[:, :-1] * np.insert(np.diff(f1m), 0, 1, axis=-1)
    z1[:, 0] = y1[:, 0]
    z1[:, 1] = y1[:, 1] + z[:, 0] * u1[:, 1] + z[:, 1] * d[:, 0] * f1m[:, 0]
    for n in range(2, f1.shape[-1]):
        z1[:, n] = (
            y1[:, n]
            + z[:, 0] * u1[:, n]
            + z[:, n] * d[:, 0] * f1m[:, 0]
            + np.sum(z[:, 1:n][:, ::-1] * v1[:, 1:n], axis=-1)
        )
    return z1
