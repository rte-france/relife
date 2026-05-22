from abc import ABC
from typing import Generic, Literal, TypeAlias, TypeVar

import numpy as np
from optype.numpy import (
    Array,
    Array1D,
    Array2D,
    ArrayND,
    is_array_1d,
    is_array_2d,
)

from relife.lifetime_models._base import (
    ParametricLifetimeModel,
)
from relife.lifetime_models._conditional_models import get_conditional_lifetime_model
from relife.rewards import ExponentialDiscounting, Reward
from relife.stochastic_processes._renewal_processes import (
    make_timeline,
    reshape_a0_ar,
)

__all__ = ["OneCycleExpectedCosts", "BaseReplacementPolicy"]

ST: TypeAlias = int | float
NumpyST: TypeAlias = np.floating | np.uint


class OneCycleExpectedCosts:
    lifetime_model: ParametricLifetimeModel[()]
    reward: Reward
    discounting: ExponentialDiscounting
    period_before_discounting: float

    def __init__(
        self,
        lifetime_model: ParametricLifetimeModel[()],
        reward: Reward,
        discounting_rate: float = 0.0,
        period_before_discounting: float = 1.0,
    ) -> None:
        self.reward = reward
        self.discounting = ExponentialDiscounting(discounting_rate)
        if period_before_discounting <= 0:
            raise ValueError("The period_before_discounting must be greater than 0")
        self.period_before_discounting = period_before_discounting
        self.lifetime_model = lifetime_model

    @reshape_a0_ar
    def expected_net_present_value(
        self,
        tf: float,
        nb_steps: int,
        a0: ST | NumpyST | Array1D[NumpyST] = 0.0,
        ar: ST | NumpyST | Array1D[NumpyST] = np.inf,
    ) -> tuple[Array1D[np.float64], Array1D[np.float64] | Array2D[np.float64]]:
        timeline = make_timeline(tf, nb_steps)
        etc = np.asarray(
            get_conditional_lifetime_model(
                self.lifetime_model, a0=a0, ar=ar
            ).ls_integrate(
                lambda x: (
                    self.reward.conditional_expectation(x, a0)
                    * self.discounting.factor(x)
                ),
                np.zeros_like(timeline),
                timeline,
                deg=15,
            ),
            dtype=float,
        )  # (nb_steps,) or (m, nb_steps)
        if timeline.ndim == 2:
            timeline = timeline[0, :]
        assert is_array_1d(timeline)  # typeguard
        return timeline, etc  # (nb_steps,) and (nb_steps,)/(m, nb_steps)

    @reshape_a0_ar
    def expected_equivalent_annual_cost(
        self,
        tf: float,
        nb_steps: int,
        a0: ST | NumpyST | Array1D[NumpyST] = 0.0,
        ar: ST | NumpyST | Array1D[NumpyST] = np.inf,
    ) -> tuple[Array1D[np.float64], Array1D[np.float64] | Array2D[np.float64]]:
        timeline = make_timeline(tf, nb_steps)  # (nb_steps,) or (m, nb_steps)
        value = self._expected_equivalent_annual_cost(timeline, ar=ar, a0=a0)
        if timeline.ndim == 2:
            timeline = timeline[0, :]  # (nb_steps,)
        assert is_array_1d(timeline)  # typeguard
        assert is_array_1d(value) or is_array_2d(value)
        return timeline, value  # (nb_steps,) or (m, nb_steps)

    @reshape_a0_ar
    def asymptotic_expected_net_present_value(
        self,
        a0: ST | NumpyST | Array1D[NumpyST] = 0.0,
        ar: ST | NumpyST | Array1D[NumpyST] = np.inf,
    ) -> np.float64 | Array1D[np.float64]:
        # reward partial expectation
        return np.squeeze(
            get_conditional_lifetime_model(
                self.lifetime_model, a0=a0, ar=ar
            ).ls_integrate(
                lambda x: (
                    self.reward.conditional_expectation(x, a0)
                    * self.discounting.factor(x)
                ),
                np.float64(0.0),
                np.asarray(np.inf),
                deg=15,
            )
        )

    @reshape_a0_ar
    def asymptotic_expected_equivalent_annual_cost(
        self,
        a0: ST | NumpyST | Array1D[NumpyST] = 0.0,
        ar: ST | NumpyST | Array1D[NumpyST] = np.inf,
    ) -> np.float64 | Array1D[np.float64]:
        timeline = np.atleast_2d(np.array(np.inf))  # (1, 1) to ensure broadcasting
        value = self._expected_equivalent_annual_cost(timeline, a0=a0, ar=ar)
        assert is_array_1d(value) or isinstance(value, np.float64)  # typeguard
        return value

    @reshape_a0_ar
    def _expected_equivalent_annual_cost(
        self,
        timeline: ArrayND[np.float64],
        a0: ST | NumpyST | Array1D[NumpyST] = 0.0,
        ar: ST | NumpyST | Array1D[NumpyST] = np.inf,
    ) -> np.float64 | Array1D[np.float64] | Array2D[np.float64]:
        # timeline : (nb_steps,) or (m, nb_steps)
        def f(x: ST | NumpyST | ArrayND[NumpyST]) -> np.float64 | ArrayND[np.float64]:
            # avoid zero division + 1e-6
            return (
                self.reward.conditional_expectation(x, a0)
                * self.discounting.factor(x)
                / (self.discounting.annuity_factor(x) + 1e-6)
            )

        conditional_model = get_conditional_lifetime_model(
            self.lifetime_model, a0=a0, ar=ar
        )
        q0 = conditional_model.cdf(self.period_before_discounting) * f(
            np.asarray(self.period_before_discounting, dtype=float)
        )  # () or (m, 1)
        a = np.full_like(
            timeline, self.period_before_discounting
        )  # (nb_steps,) or (m, nb_steps)

        # change first value of lower bound to compute the integral
        a[timeline < self.period_before_discounting] = 0.0  # (nb_steps,)
        # a = np.where(timeline < self.period_before_discounting, 0., a)  # (nb_steps,)
        integral = conditional_model.ls_integrate(
            f, a, timeline, deg=20
        )  # (nb_steps,) or (m, nb_steps) if q0: (), or (m, nb_steps) if q0 : (m, 1)
        mask = np.broadcast_to(
            timeline < self.period_before_discounting, integral.shape
        )  # (), (nb_steps,) or (m, nb_steps)
        q0 = np.broadcast_to(q0, integral.shape)  # (nb_steps,) or (m, nb_steps)
        integral = np.squeeze(np.where(mask, q0, q0 + integral))
        if integral.ndim == 0:
            return np.float64(integral)
        return integral


M = TypeVar("M")


class BaseReplacementPolicy(Generic[M], ABC):
    baseline_model: M
    discounting_rate: float
    _cost_structure: dict[
        str,
        np.float64 | Array[tuple[int, Literal[1]], np.float64],
    ]

    def __init__(
        self,
        baseline_model: M,
        cost_structure: dict[
            str,
            np.float64 | Array[tuple[int, Literal[1]], np.float64],
        ],
        discounting_rate: float = 0.0,
    ):
        self.baseline_model = baseline_model
        self.discounting_rate = discounting_rate
        self._cost_structure = cost_structure  # hidden, contains reshaped cost arrays
