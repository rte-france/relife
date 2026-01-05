# pyright: basic
from abc import ABC, abstractmethod
from typing import Generic, Literal, TypeVar, overload

import numpy as np
from numpy.typing import NDArray

from relife.economic import ExponentialDiscounting, Reward
from relife.typing import (
    AnyParametricLifetimeModel,
    NumpyFloat,
)

__all__ = ["OneCycleExpectedCosts", "ReplacementPolicy"]


def _make_timeline(tf: float, nb_steps: int) -> NDArray[np.float64]:
    timeline = np.linspace(0, tf, nb_steps, dtype=np.float64)  # (nb_steps,)
    return np.atleast_2d(timeline)  # (1, nb_steps) to ensure broadcasting


class ExpectedCostsABC(ABC):
    @abstractmethod
    def expected_net_present_value(
        self, tf: float, nb_steps: int, total_sum: bool = False
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        r"""
        The expected net present value.

        .. math::

            z(t) = \mathbb{E}(Z_t) = \int_{0}^{\infty}\mathbb{E}(Z_t~|~X_1 = x)dF(x)

        where :

        - :math:`t` is the time
        - :math:`X_1 \sim F` is the random lifetime of the first asset
        - :math:`Z_t` are the random costs at each time :math:`t`
        - :math:`\delta` is the discounting rate

        It is computed by solving the renewal equation.

        Parameters
        ----------
        tf : float
            The final time.
        nb_steps : int
            The number of steps used to discretized the time.
        total_sum : bool, default False
            If True, returns the total sum over the first axis of the result. If the policy data encodes several
            assets, this option allows to return the sum result on the flit rather than calling ``np.sum`` afterwards.

        Returns
        -------
        tuple of two ndarrays
            A tuple containing the timeline and the computed values.
        """

    @overload
    def asymptotic_expected_net_present_value(self, total_sum: Literal[False]) -> NDArray[np.float64]: ...
    @overload
    def asymptotic_expected_net_present_value(self, total_sum: Literal[True]) -> np.float64: ...
    @overload
    def asymptotic_expected_net_present_value(self, total_sum: bool = False) -> np.float64 | NDArray[np.float64]: ...
    @abstractmethod
    def asymptotic_expected_net_present_value(self, total_sum: bool = False) -> np.float64 | NDArray[np.float64]:
        r"""
        The asymtotic expected net present value.

        .. math::

            \lim_{t\to\infty} z(t)

        Parameters
        ----------
        total_sum : bool, default False
            If True, returns the total sum over the first axis of the result. If the policy data encodes several
            assets, this option allows to return the sum result on the flit rather than calling ``np.sum`` afterwards.

        Returns
        -------
        ndarray
            The asymptotic expected values.
        """

    @abstractmethod
    def expected_equivalent_annual_cost(
        self, tf: float, nb_steps: int, total_sum: bool = False
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        r"""
        The expected equivalent annual cost.

        .. math::

            q(t) = \dfrac{\delta z(t)}{1 - e^{-\delta t}}

        where :

        - :math:`t` is the time.
        - :math:`z(t)` is the expected net present value at time :math:`t`.
        - :math:`\delta` is the discounting rate.

        Parameters
        ----------
        tf : float
            The final time.
        nb_steps : int
            The number of steps used to discretized the time.
        total_sum : bool, default False
            If True, returns the total sum over the first axis of the result. If the policy data encodes several
            assets, this option allows to return the sum result on the flit rather than calling ``np.sum`` afterwards.

        Returns
        -------
        tuple of two ndarrays
            A tuple containing the timeline and the computed values.
        """

    @overload
    def asymptotic_expected_equivalent_annual_cost(self, total_sum: Literal[False]) -> NDArray[np.float64]: ...
    @overload
    def asymptotic_expected_equivalent_annual_cost(self, total_sum: Literal[True]) -> np.float64: ...
    @overload
    def asymptotic_expected_equivalent_annual_cost(
        self, total_sum: bool = False
    ) -> np.float64 | NDArray[np.float64]: ...
    @abstractmethod
    def asymptotic_expected_equivalent_annual_cost(self, total_sum: bool = False) -> np.float64 | NDArray[np.float64]:
        r"""
        The asymtotic expected equivalent annual cost.

        .. math::

            \lim_{t\to\infty} q(t)

        Parameters
        ----------
        total_sum : bool, default False
            If True, returns the total sum over the first axis of the result. If the policy data encodes several
            assets, this option allows to return the sum result on the flit rather than calling ``np.sum`` afterwards.

        Returns
        -------
        ndarray
            The asymptotic expected values.
        """


class OneCycleExpectedCosts(ExpectedCostsABC):
    lifetime_model: AnyParametricLifetimeModel[()]
    reward: Reward
    discounting: ExponentialDiscounting
    period_before_discounting: float

    def __init__(
        self,
        lifetime_model: AnyParametricLifetimeModel[()],
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

    def expected_net_present_value(
        self, tf: float, nb_steps: int, total_sum: bool = False
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        timeline = _make_timeline(tf, nb_steps)
        etc = np.asarray(
            self.lifetime_model.ls_integrate(
                lambda x: self.reward.conditional_expectation(x) * self.discounting.factor(x),
                np.zeros_like(timeline),
                timeline,
                deg=15,
            ),
            dtype=float,
        )  # (nb_steps,) or (m, nb_steps)
        if total_sum and etc.ndim == 2:
            etc = np.sum(etc, axis=0)
        if timeline.ndim == 2:
            timeline = timeline[0, :]
        return timeline, etc  # (nb_steps,) and (nb_steps,)/(m, nb_steps)

    @overload
    def asymptotic_expected_net_present_value(self, total_sum: Literal[False]) -> NDArray[np.float64]: ...
    @overload
    def asymptotic_expected_net_present_value(self, total_sum: Literal[True]) -> np.float64: ...
    @overload
    def asymptotic_expected_net_present_value(self, total_sum: bool = False) -> np.float64 | NDArray[np.float64]: ...
    def asymptotic_expected_net_present_value(self, total_sum: bool = False) -> np.float64 | NDArray[np.float64]:
        # reward partial expectation
        value = np.squeeze(
            self.lifetime_model.ls_integrate(
                lambda x: self.reward.conditional_expectation(x) * self.discounting.factor(x),
                np.float64(0.0),
                np.asarray(np.inf),
                deg=15,
            )
        )
        if total_sum:
            value = np.sum(value)
        return value  # () or (m,)

    def _expected_equivalent_annual_cost(self, timeline: NDArray[np.float64]) -> NDArray[np.float64]:

        # timeline : (nb_steps,) or (m, nb_steps)
        def f(x: NDArray[np.float64]) -> NDArray[np.float64]:
            # avoid zero division + 1e-6
            return (
                self.reward.conditional_expectation(x)
                * self.discounting.factor(x)
                / (self.discounting.annuity_factor(x) + 1e-6)
            )

        q0 = self.lifetime_model.cdf(self.period_before_discounting) * f(
            np.asarray(self.period_before_discounting, dtype=float)
        )  # () or (m, 1)
        a = np.full_like(timeline, self.period_before_discounting)  # (nb_steps,) or (m, nb_steps)

        # change first value of lower bound to compute the integral
        a[timeline < self.period_before_discounting] = 0.0  # (nb_steps,)
        # a = np.where(timeline < self.period_before_discounting, 0., a)  # (nb_steps,)
        integral = self.lifetime_model.ls_integrate(
            f, a, timeline, deg=15
        )  # (nb_steps,) or (m, nb_steps) if q0: (), or (m, nb_steps) if q0 : (m, 1)
        mask = np.broadcast_to(
            timeline < self.period_before_discounting, integral.shape
        )  # (), (nb_steps,) or (m, nb_steps)
        q0 = np.broadcast_to(q0, integral.shape)  # (nb_steps,) or (m, nb_steps)
        integral = np.where(mask, q0, q0 + integral)
        return np.squeeze(integral)  # (nb_steps,)/(m, nb_steps)

    def expected_equivalent_annual_cost(
        self, tf: float, nb_steps: int, total_sum: bool = False
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        timeline = _make_timeline(tf, nb_steps)  # (nb_steps,) or (m, nb_steps)
        value = self._expected_equivalent_annual_cost(timeline)
        if timeline.ndim == 2:
            timeline = timeline[0, :]  # (nb_steps,)
        if total_sum and value.ndim == 2:
            value = np.sum(value, axis=0)
        return timeline, value  # (nb_steps,) or (m, nb_steps)

    @overload
    def asymptotic_expected_equivalent_annual_cost(self, total_sum: Literal[False]) -> NDArray[np.float64]: ...
    @overload
    def asymptotic_expected_equivalent_annual_cost(self, total_sum: Literal[True]) -> np.float64: ...
    @overload
    def asymptotic_expected_equivalent_annual_cost(
        self, total_sum: bool = False
    ) -> np.float64 | NDArray[np.float64]: ...
    def asymptotic_expected_equivalent_annual_cost(self, total_sum: bool = False) -> np.float64 | NDArray[np.float64]:
        timeline = np.atleast_2d(np.array(np.inf))  # (1, 1) to ensure broadcasting
        value = self._expected_equivalent_annual_cost(timeline)
        if total_sum:
            value = np.sum(value)
        return value  # () or (m,)


M = TypeVar("M")


class ReplacementPolicy(ExpectedCostsABC, Generic[M], ABC):
    baseline_model: M
    discounting_rate: float
    _cost_structure: dict[str, NumpyFloat]

    def __init__(
        self,
        baseline_model: M,
        cost_structure: dict[str, NumpyFloat],
        discounting_rate: float = 0.0,
    ):
        self.baseline_model = baseline_model
        self.discounting_rate = discounting_rate
        self._cost_structure = cost_structure  # hidden, contains reshaped cost arrays
