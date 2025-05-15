from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from relife.economic import Discounting
    from relife.lifetime_model import LifetimeDistribution, FrozenParametricLifetimeModel


def renewal_equation_solver(
    tf: float,
    nb_steps: int,
    model: LifetimeDistribution | FrozenParametricLifetimeModel,
    evaluated_func: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    *,
    discounting: Optional[Discounting] = None,
) -> NDArray[np.float64]:

    t = np.linspace(0, tf, nb_steps, dtype=np.float64)  # (nb_steps,)
    tm = 0.5 * (t[1:] + t[:-1])  # (nb_steps - 1,)
    f = model.cdf(t)  # (nb_steps,) or (m, nb_steps)
    fm = model.cdf(tm)  # (nb_steps - 1,) or (m, nb_steps - 1)
    y = evaluated_func(t)  # (nb_steps,)

    if y.shape != f.shape:
        raise ValueError("Invalid shape between model and evaluated_func")

    if discounting is not None:
        d = discounting.factor(t)
    else:
        d = np.ones_like(f)
    z = np.empty(y.shape)
    u = d * np.insert(f[..., 1:] - fm, 0, 1, axis=-1)
    v = d[..., :-1] * np.insert(np.diff(fm), 0, 1, axis=-1)
    q0 = 1 / (1 - d[..., 0] * fm[..., 0])
    z[..., 0] = y[..., 0]
    z[..., 1] = q0 * (y[..., 1] + z[..., 0] * u[..., 1])
    for n in range(2, f.shape[-1]):
        z[..., n] = q0 * (y[..., n] + z[..., 0] * u[..., n] + np.sum(z[..., 1:n][..., ::-1] * v[..., 1:n], axis=-1))
    return z


def delayed_renewal_equation_solver(
    tf: float,
    nb_steps: int,
    z: NDArray[np.float64],
    model1: LifetimeDistribution | FrozenParametricLifetimeModel,
    evaluated_func: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    discounting: Optional[Discounting] = None,
) -> NDArray[np.float64]:

    t = np.linspace(0, tf, nb_steps, dtype=np.float64)  # (nb_steps,)
    tm = 0.5 * (t[1:] + t[:-1])  # (nb_steps - 1,)
    f1 = model1.cdf(t)  # (nb_steps,) or (m, nb_steps)
    f1m = model1.cdf(tm)  # (nb_steps - 1,) or (m, nb_steps - 1)
    y1 = evaluated_func(t)  # (nb_steps,)

    if discounting is not None:
        d = discounting.factor(t)
    else:
        d = np.ones_like(f1)
    z1 = np.empty(y1.shape)
    u1 = d * np.insert(f1[..., 1:] - f1m, 0, 1, axis=-1)
    v1 = d[..., :-1] * np.insert(np.diff(f1m), 0, 1, axis=-1)
    z1[..., 0] = y1[..., 0]
    z1[..., 1] = y1[..., 1] + z[..., 0] * u1[..., 1] + z[..., 1] * d[..., 0] * f1m[..., 0]
    for n in range(2, f1.shape[-1]):
        z1[..., n] = (
            y1[..., n]
            + z[..., 0] * u1[..., n]
            + z[..., n] * d[..., 0] * f1m[..., 0]
            + np.sum(z[..., 1:n][..., ::-1] * v1[..., 1:n], axis=-1)
        )
    return z1
