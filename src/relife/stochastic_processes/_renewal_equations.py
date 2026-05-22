from collections.abc import Callable
from typing import TypeAlias

import numpy as np
from optype.numpy import Array1D, Array2D, ArrayND

from relife.lifetime_models._base import ParametricLifetimeModel
from relife.rewards import ExponentialDiscounting

__all__ = ["RenewalEquationSolver"]

ST: TypeAlias = int | float
NumpyST: TypeAlias = np.floating | np.uint


FT: TypeAlias = Callable[
    [ST | NumpyST | ArrayND[NumpyST]],
    np.float64 | ArrayND[np.float64],
]


class RenewalEquationSolver:
    lifetime_model: ParametricLifetimeModel[()]
    first_lifetime_model: ParametricLifetimeModel[()] | None
    func: FT
    func1: FT | None

    def __init__(
        self,
        lifetime_model: ParametricLifetimeModel[()],
        func: FT,
        first_lifetime_model: ParametricLifetimeModel[()] | None = None,
        func1: FT | None = None,
    ) -> None:
        self.lifetime_model = lifetime_model
        self.func = func
        if first_lifetime_model:
            assert func1 is not None
        self.first_lifetime_model = first_lifetime_model
        self.func1 = func1

    def solve(
        self, tf: float, nb_steps: int, discounting_rate: float = 0.0
    ) -> tuple[Array1D[np.float64], Array1D[np.float64] | Array2D[np.float64]]:

        discounting = ExponentialDiscounting(discounting_rate)
        timeline = np.atleast_2d(np.linspace(0, tf, nb_steps, dtype=np.float64))
        tm = 0.5 * (timeline[:, 1:] + timeline[:, :-1])  # (1, nb_steps - 1)
        f = np.atleast_2d(self.lifetime_model.cdf(timeline))  # (m, nb_steps)
        fm = np.atleast_2d(self.lifetime_model.cdf(tm))  # (m, nb_steps - 1)
        y = np.atleast_2d(self.func(timeline))  # (1, nb_steps)
        d = np.asarray(discounting.factor(timeline))  # (m, nb_steps)
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

        if self.first_lifetime_model is not None and self.func1 is not None:
            f1 = np.atleast_2d(self.first_lifetime_model.cdf(timeline))  # (m, nb_steps)
            f1m = np.atleast_2d(self.first_lifetime_model.cdf(tm))  # (m, nb_steps - 1)
            y1 = np.atleast_2d(self.func1(timeline))  # (m, nb_steps - 1)
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
            return np.squeeze(timeline), np.squeeze(z1)
        return np.squeeze(timeline), np.squeeze(z)
