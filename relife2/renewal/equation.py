from typing import Callable, Optional, TypeVarTuple

import numpy as np
from numpy.typing import NDArray

from relife2.model import LifetimeModel
from relife2.renewal.discounting import Discounting

Ts = TypeVarTuple("Ts")


def renewal_equation_solver(
    timeline: NDArray[np.float64],
    model: LifetimeModel,
    evaluated_func: Callable[
        [NDArray[np.float64], *tuple[NDArray[np.float64], ...]], NDArray[np.float64]
    ],
    model_args: tuple[NDArray[np.float64], ...] = (),
    evaluated_func_args: tuple[NDArray[np.float64], ...] = (),
    discounting: Optional[Discounting] = None,
    discounting_args: tuple[NDArray[np.float64], ...] = (),
) -> NDArray[np.float64]:

    tm = 0.5 * (timeline[1:] + timeline[:-1])
    f = model.cdf(timeline, *model_args)
    fm = model.cdf(tm, *model_args)
    y = evaluated_func(timeline, *evaluated_func_args)
    if discounting is not None:
        d = discounting.factor(timeline, *discounting_args)
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
    delayed_model: LifetimeModel,
    evaluated_func: Callable[
        [NDArray[np.float64], *tuple[NDArray[np.float64], ...]], NDArray[np.float64]
    ],
    delayed_model_args: tuple[NDArray[np.float64], ...] = (),
    evaluated_func_args: tuple[NDArray[np.float64], ...] = (),
    discounting: Optional[Discounting] = None,
    discounting_args: tuple[float, ...] = (),
) -> NDArray[np.float64]:

    tm = 0.5 * (timeline[1:] + timeline[:-1])
    f1 = delayed_model.cdf(timeline, *delayed_model_args)
    f1m = delayed_model.cdf(tm, *delayed_model_args)
    y1 = evaluated_func(timeline, *evaluated_func_args)

    if discounting is not None:
        d = discounting.factor(timeline, *discounting_args)
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
