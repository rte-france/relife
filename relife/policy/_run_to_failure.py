# pyright: basic
from __future__ import annotations

from abc import ABC
from typing import Any, Literal, Optional, overload

import numpy as np
from numpy.typing import NDArray

from relife.economic import RunToFailureReward
from relife.lifetime_model import (
    LeftTruncatedModel,
)
from relife.stochastic_process import RenewalRewardProcess
from relife.typing import (
    AnyFloat,
    AnyParametricLifetimeModel,
    NumpyFloat,
)
from relife.utils import flatten_if_possible, reshape_1d_arg

from ._base import OneCycleExpectedCosts, ReplacementPolicy


@overload
def run_to_failure_policy(
    baseline_model: AnyParametricLifetimeModel[()],
    cf: AnyFloat,
    one_cycle: Literal[True],
    **kwargs: Any,
) -> OneCycleRunToFailurePolicy: ...
@overload
def run_to_failure_policy(
    baseline_model: AnyParametricLifetimeModel[()],
    cf: AnyFloat,
    one_cycle: Literal[False],
    **kwargs: Any,
) -> RunToFailurePolicy: ...
@overload
def run_to_failure_policy(
    baseline_model: AnyParametricLifetimeModel[()],
    cf: AnyFloat,
    one_cycle: bool = False,
    **kwargs: Any,
) -> OneCycleRunToFailurePolicy | RunToFailurePolicy: ...
def run_to_failure_policy(
    baseline_model: AnyParametricLifetimeModel[()],
    cf: AnyFloat,
    one_cycle: bool = False,
    **kwargs: Any,
) -> OneCycleRunToFailurePolicy | RunToFailurePolicy:
    """
    Creates a run-to-failure policy.

    Parameters
    ----------
    baseline_model : parametric model
        Parametric model required by the policy.
    cf : float or 1d-array
        Cost of failure.
    one_cycle : bool, default False
        If True, returns the one cycle variation of the policy.
    **kwargs
        Extra arguments required by the policy (a0, discounting_rate, etc.)

    Returns
    -------
    Policy
        Policy corresponding to the ``baseline_model`` and the ``cost_structure``.

    Raises
    ------
    ValueError
        If ``baseline_model`` or ``cost_structure`` does not have a corresponding policy.
    """
    if one_cycle:
        return OneCycleRunToFailurePolicy(baseline_model, cf, **kwargs)
    return RunToFailurePolicy(baseline_model, cf, **kwargs)


class BaseRunToFailure(ReplacementPolicy[AnyParametricLifetimeModel[()]], ABC):
    _cost_structure: dict[str, NumpyFloat]
    _a0: Optional[NumpyFloat]
    discounting_rate: float

    def __init__(
        self,
        lifetime_model: AnyParametricLifetimeModel[()],
        cf: AnyFloat,
        discounting_rate: float = 0.0,
        a0: Optional[AnyFloat] = None,
    ):
        super().__init__(
            lifetime_model,
            {"cf": reshape_1d_arg(cf)},
            discounting_rate=discounting_rate,
        )
        self._a0 = reshape_1d_arg(a0) if a0 is not None else a0

    @property
    def a0(self) -> Optional[NumpyFloat]:
        """Current ages of the assets.

        Returns
        -------
        np.ndarray
        """
        # _a0 is (m, 1) but exposed cf is (m,)
        if self._a0 is None:
            return self._a0
        return flatten_if_possible(self._a0)

    @property
    def cf(self) -> NumpyFloat:
        """Cost of failure.

        Returns
        -------
        np.ndarray
        """
        # _cf is (m, 1) but exposed cf is (m,)
        return flatten_if_possible(self._cost_structure["cf"])

    @cf.setter
    def cf(self, value: AnyFloat) -> None:
        self._cost_structure["cf"] = reshape_1d_arg(value)


class OneCycleRunToFailurePolicy(BaseRunToFailure):
    r"""One cyle run-to-failure policy.

    Asset is replaced upon failure with cost :math:`c_f`.

    .. note::

        ``OneCycleRunToFailurePolicy`` differs from ``RunToFailurePolicy``
        because only one cycle of replacement is considered.

    Parameters
    ----------
    lifetime_model : any lifetime distribution or frozen lifetime model
        A lifetime model representing the durations between events.
    cf : float or 1darray
        Costs of failures
    discounting_rate : float, default is 0.
        The discounting rate value used in the exponential discounting function
    a0 : float or 1darray, optional
        Current ages of the assets, by default 0 for each asset. If it is given, left truncations of ``a0`` will
        be take into account for the first cycle.

    Attributes
    ----------
    cf
    """

    def __init__(
        self,
        lifetime_model: AnyParametricLifetimeModel[()],
        cf: AnyFloat,
        discounting_rate: float = 0.0,
        a0: Optional[AnyFloat] = None,
        period_before_discounting: float = 1.0,
    ) -> None:
        super().__init__(lifetime_model, cf, discounting_rate=discounting_rate, a0=a0)
        self.period_before_discounting = period_before_discounting

    @property
    def _expected_costs(self) -> OneCycleExpectedCosts:
        if self.a0 is None:
            return OneCycleExpectedCosts(
                self.baseline_model,
                RunToFailureReward(self.cf),
                discounting_rate=self.discounting_rate,
                period_before_discounting=self.period_before_discounting,
            )
        return OneCycleExpectedCosts(
            LeftTruncatedModel(self.baseline_model).freeze(self.a0),
            RunToFailureReward(self.cf),
            discounting_rate=self.discounting_rate,
            period_before_discounting=self.period_before_discounting,
        )

    def expected_net_present_value(
        self, tf: float, nb_steps: int, total_sum: bool = False
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        timeline, npv = self._expected_costs.expected_net_present_value(tf, nb_steps)
        if total_sum and npv.ndim == 2:
            return timeline, np.sum(npv, axis=0)
        return timeline, npv

    @overload
    def asymptotic_expected_net_present_value(self, total_sum: Literal[False]) -> NDArray[np.float64]: ...
    @overload
    def asymptotic_expected_net_present_value(self, total_sum: Literal[True]) -> np.float64: ...
    @overload
    def asymptotic_expected_net_present_value(self, total_sum: bool = False) -> np.float64 | NDArray[np.float64]: ...
    def asymptotic_expected_net_present_value(self, total_sum: bool = False) -> np.float64 | NDArray[np.float64]:

        asymptotic_npv = self._expected_costs.asymptotic_expected_net_present_value()
        if total_sum:
            return np.sum(asymptotic_npv)
        return asymptotic_npv

    def expected_equivalent_annual_cost(
        self, tf: float, nb_steps: int, total_sum: bool = False
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:

        timeline, eeac = self._expected_costs.expected_equivalent_annual_cost(tf, nb_steps)
        if total_sum and eeac.ndim == 2:
            return timeline, np.sum(eeac, axis=0)
        return timeline, eeac

    @overload
    def asymptotic_expected_equivalent_annual_cost(self, total_sum: Literal[False]) -> NDArray[np.float64]: ...
    @overload
    def asymptotic_expected_equivalent_annual_cost(self, total_sum: Literal[True]) -> np.float64: ...
    @overload
    def asymptotic_expected_equivalent_annual_cost(
        self, total_sum: bool = False
    ) -> np.float64 | NDArray[np.float64]: ...
    def asymptotic_expected_equivalent_annual_cost(self, total_sum: bool = False) -> np.float64 | NDArray[np.float64]:
        asymptotic_eeac = self._expected_costs.asymptotic_expected_equivalent_annual_cost()
        if total_sum:
            return np.sum(asymptotic_eeac)
        return asymptotic_eeac


class RunToFailurePolicy(BaseRunToFailure):
    r"""Run-to-failure renewal policy.

    Asset is replaced upon failure with cost :math:`c_f`.

    Parameters
    ----------
    lifetime_model : any lifetime distribution or frozen lifetime model
        A lifetime model representing the durations between events.
    cf : float or 1darray
        Costs of failures
    discounting_rate : float, default is 0.
        The discounting rate value used in the exponential discounting function
    a0 : float or 1darray, optional
        Current ages of the assets, by default 0 for each asset. If it is given, left truncations of ``a0`` will
        be take into account for the first cycle.

    Attributes
    ----------
    cf

    References
    ----------
    .. [1] Van der Weide, J. A. M., & Van Noortwijk, J. M. (2008). Renewal
        theory with exponential and hyperbolic discounting. Probability in
        the Engineering and Informational Sciences, 22(1), 53-74.
    """

    @property
    def _stochastic_process(self) -> RenewalRewardProcess:
        if self.a0 is None:
            return RenewalRewardProcess(
                self.baseline_model,
                RunToFailureReward(self.cf),
                discounting_rate=self.discounting_rate,
            )
        return RenewalRewardProcess(
            self.baseline_model,
            RunToFailureReward(self.cf),
            discounting_rate=self.discounting_rate,
            first_lifetime_model=LeftTruncatedModel(self.baseline_model).freeze(self.a0),
        )

    def expected_net_present_value(
        self, tf: float, nb_steps: int, total_sum: bool = False
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:

        timeline, npv = self._stochastic_process.expected_total_reward(tf, nb_steps)
        if total_sum and npv.ndim == 2:
            npv = np.sum(npv, axis=0)
        return timeline, npv

    @overload
    def asymptotic_expected_net_present_value(self, total_sum: Literal[False]) -> NDArray[np.float64]: ...
    @overload
    def asymptotic_expected_net_present_value(self, total_sum: Literal[True]) -> np.float64: ...
    @overload
    def asymptotic_expected_net_present_value(self, total_sum: bool = False) -> np.float64 | NDArray[np.float64]: ...
    def asymptotic_expected_net_present_value(self, total_sum: bool = False) -> np.float64 | NDArray[np.float64]:

        asymptotic_npv = self._stochastic_process.asymptotic_expected_total_reward()
        if total_sum:
            return np.sum(asymptotic_npv)
        return asymptotic_npv

    def expected_equivalent_annual_cost(
        self, tf: float, nb_steps: int, total_sum: bool = False
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:

        timeline, eeac = self._stochastic_process.expected_equivalent_annual_worth(tf, nb_steps)
        if total_sum and eeac.ndim == 2:
            eeac = np.sum(eeac, axis=0)
        return timeline, eeac

    @overload
    def asymptotic_expected_equivalent_annual_cost(self, total_sum: Literal[False]) -> NDArray[np.float64]: ...
    @overload
    def asymptotic_expected_equivalent_annual_cost(self, total_sum: Literal[True]) -> np.float64: ...
    @overload
    def asymptotic_expected_equivalent_annual_cost(
        self, total_sum: bool = False
    ) -> np.float64 | NDArray[np.float64]: ...
    def asymptotic_expected_equivalent_annual_cost(self, total_sum: bool = False) -> np.float64 | NDArray[np.float64]:
        asymptotic_eeac = self._stochastic_process.asymptotic_expected_equivalent_annual_worth()
        if total_sum:
            return np.sum(asymptotic_eeac)
        return asymptotic_eeac

    def sample(self, size, tf, t0=0.0, seed=None):
        """Renewal data sampling.

        This function will sample data and encapsulate them in an object.

        Parameters
        ----------
        size : int
            The size of the desired sample.
        tf : float
            Time at the end of the observation.
        t0 : float, default 0
            Time at the beginning of the observation.
        size : int or tuple of 2 int
            Size of the sample
        seed : int, optional
            Random seed, by default None.

        """
        return self._stochastic_process.sample(tf, t0, size, seed)
