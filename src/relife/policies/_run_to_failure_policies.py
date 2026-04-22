from __future__ import annotations

from abc import ABC
from typing import Any, Literal, TypeAlias, overload

import numpy as np
from optype.numpy import Array, Array1D, Array2D
from typing_extensions import override

from relife.lifetime_models import LeftTruncatedModel
from relife.lifetime_models._base import ParametricLifetimeModel
from relife.rewards import RunToFailureReward
from relife.stochastic_processes._renewal_processes import RenewalRewardProcess
from relife.stochastic_processes._sample import StochasticSampleMapping
from relife.utils import flatten_if_at_least_2d, to_column_2d_if_1d

from ._base import OneCycleExpectedCosts, ReplacementPolicy

ST: TypeAlias = int | float
NumpyST: TypeAlias = np.floating | np.uint


@overload
def run_to_failure_policy(
    baseline_model: ParametricLifetimeModel[()],
    cf: int | float | Array1D[np.float64],
    one_cycle: Literal[True],
    **kwargs: Any,
) -> OneCycleRunToFailurePolicy: ...
@overload
def run_to_failure_policy(
    baseline_model: ParametricLifetimeModel[()],
    cf: int | float | Array1D[np.float64],
    one_cycle: Literal[False],
    **kwargs: Any,
) -> RunToFailurePolicy: ...
def run_to_failure_policy(
    baseline_model: ParametricLifetimeModel[()],
    cf: int | float | Array1D[np.float64],
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
    """  # noqa: E501
    if one_cycle:
        return OneCycleRunToFailurePolicy(baseline_model, cf, **kwargs)
    return RunToFailurePolicy(baseline_model, cf, **kwargs)


class BaseRunToFailure(ReplacementPolicy[ParametricLifetimeModel[()]], ABC):
    _a0: np.float64 | Array[tuple[int, Literal[1]], np.float64] | None
    discounting_rate: float

    def __init__(
        self,
        lifetime_model: ParametricLifetimeModel[()],
        cf: ST | NumpyST | Array1D[NumpyST],
        discounting_rate: float = 0.0,
        a0: ST | NumpyST | Array1D[NumpyST] | None = None,
    ):
        super().__init__(
            lifetime_model,
            {"cf": to_column_2d_if_1d(cf)},
            discounting_rate=discounting_rate,
        )
        self._a0 = to_column_2d_if_1d(a0) if a0 is not None else a0

    @property
    def a0(self) -> np.float64 | Array1D[np.float64] | None:
        """Current ages of the assets.

        Returns
        -------
        np.ndarray
        """
        if self._a0 is None:
            return self._a0
        return flatten_if_at_least_2d(self._a0)

    def get_cf(self) -> np.float64 | Array1D[np.float64]:
        """Cost of failure.

        Returns
        -------
        np.ndarray
        """
        return flatten_if_at_least_2d(self._cost_structure["cf"])

    def set_cf(self, value: ST | NumpyST | Array1D[NumpyST]) -> None:
        self._cost_structure["cf"] = to_column_2d_if_1d(value)


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
    """  # noqa: E501

    period_before_discounting: float

    def __init__(
        self,
        lifetime_model: ParametricLifetimeModel[()],
        cf: ST | NumpyST | Array1D[NumpyST],
        discounting_rate: float = 0.0,
        a0: ST | NumpyST | Array1D[NumpyST] | None = None,
        period_before_discounting: float = 1.0,
    ) -> None:
        super().__init__(lifetime_model, cf, discounting_rate=discounting_rate, a0=a0)
        self.period_before_discounting = period_before_discounting

    @property
    def _expected_costs(self) -> OneCycleExpectedCosts:
        if self.a0 is None:
            return OneCycleExpectedCosts(
                self.baseline_model,
                RunToFailureReward(self.get_cf()),
                discounting_rate=self.discounting_rate,
                period_before_discounting=self.period_before_discounting,
            )
        return OneCycleExpectedCosts(
            LeftTruncatedModel(self.baseline_model).freeze(self.a0),
            RunToFailureReward(self.get_cf()),
            discounting_rate=self.discounting_rate,
            period_before_discounting=self.period_before_discounting,
        )

    @override
    def expected_net_present_value(
        self, tf: float, nb_steps: int, total_sum: bool = False
    ) -> tuple[Array1D[np.float64], Array1D[np.float64] | Array2D[np.float64]]:
        timeline, npv = self._expected_costs.expected_net_present_value(tf, nb_steps)
        if total_sum and npv.ndim == 2:
            return timeline, np.sum(npv, axis=0)
        return timeline, npv

    @overload
    def asymptotic_expected_net_present_value(
        self, total_sum: Literal[False]
    ) -> np.float64 | Array1D[np.float64]: ...
    @overload
    def asymptotic_expected_net_present_value(
        self, total_sum: Literal[True]
    ) -> np.float64: ...
    @override
    def asymptotic_expected_net_present_value(
        self, total_sum: bool = False
    ) -> np.float64 | Array1D[np.float64]:
        asymptotic_npv = self._expected_costs.asymptotic_expected_net_present_value(
            total_sum=total_sum
        )
        if total_sum:
            return np.sum(asymptotic_npv)
        return asymptotic_npv

    @override
    def expected_equivalent_annual_cost(
        self, tf: float, nb_steps: int, total_sum: bool = False
    ) -> tuple[Array1D[np.float64], Array1D[np.float64] | Array2D[np.float64]]:
        timeline, eeac = self._expected_costs.expected_equivalent_annual_cost(
            tf, nb_steps
        )
        if total_sum and eeac.ndim == 2:
            return timeline, np.sum(eeac, axis=0)
        return timeline, eeac

    @overload
    def asymptotic_expected_equivalent_annual_cost(
        self, total_sum: Literal[False]
    ) -> np.float64 | Array1D[np.float64]: ...
    @overload
    def asymptotic_expected_equivalent_annual_cost(
        self, total_sum: Literal[True]
    ) -> np.float64: ...
    @override
    def asymptotic_expected_equivalent_annual_cost(
        self, total_sum: bool = False
    ) -> np.float64 | Array1D[np.float64]:
        asymptotic_eeac = (
            self._expected_costs.asymptotic_expected_equivalent_annual_cost(
                total_sum=total_sum
            )
        )
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
    """  # noqa: E501

    @property
    def _stochastic_process(self) -> RenewalRewardProcess:
        if self.a0 is None:
            return RenewalRewardProcess(
                self.baseline_model,
                RunToFailureReward(self.get_cf()),
                discounting_rate=self.discounting_rate,
            )
        return RenewalRewardProcess(
            self.baseline_model,
            RunToFailureReward(self.get_cf()),
            discounting_rate=self.discounting_rate,
            first_lifetime_model=LeftTruncatedModel(self.baseline_model).freeze(
                self.a0
            ),
        )

    @override
    def expected_net_present_value(
        self, tf: float, nb_steps: int, total_sum: bool = False
    ) -> tuple[Array1D[np.float64], Array1D[np.float64] | Array2D[np.float64]]:
        timeline, npv = self._stochastic_process.expected_total_reward(tf, nb_steps)
        if total_sum and npv.ndim == 2:
            npv = np.sum(npv, axis=0)
        return timeline, npv

    @overload
    def asymptotic_expected_net_present_value(
        self, total_sum: Literal[False]
    ) -> np.float64 | Array1D[np.float64]: ...
    @overload
    def asymptotic_expected_net_present_value(
        self, total_sum: Literal[True]
    ) -> np.float64: ...
    @override
    def asymptotic_expected_net_present_value(
        self, total_sum: bool = False
    ) -> np.float64 | Array1D[np.float64]:
        asymptotic_npv = self._stochastic_process.asymptotic_expected_total_reward()
        if total_sum:
            return np.sum(asymptotic_npv)
        return asymptotic_npv

    @override
    def expected_equivalent_annual_cost(
        self, tf: float, nb_steps: int, total_sum: bool = False
    ) -> tuple[Array1D[np.float64], Array1D[np.float64] | Array2D[np.float64]]:
        timeline, eeac = self._stochastic_process.expected_equivalent_annual_worth(
            tf, nb_steps
        )
        if total_sum and eeac.ndim == 2:
            eeac = np.sum(eeac, axis=0)
        return timeline, eeac

    @overload
    def asymptotic_expected_equivalent_annual_cost(
        self, total_sum: Literal[False]
    ) -> Array1D[np.float64]: ...
    @overload
    def asymptotic_expected_equivalent_annual_cost(
        self, total_sum: Literal[True]
    ) -> np.float64: ...
    @override
    def asymptotic_expected_equivalent_annual_cost(
        self, total_sum: bool = False
    ) -> np.float64 | Array1D[np.float64]:
        asymptotic_eeac = (
            self._stochastic_process.asymptotic_expected_equivalent_annual_worth()
        )
        if total_sum:
            return np.sum(asymptotic_eeac)
        return asymptotic_eeac

    def sample(
        self,
        nb_samples: int,
        time_window: tuple[float, float],
        seed: int
        | np.random.Generator
        | np.random.BitGenerator
        | np.random.RandomState
        | None = None,
    ) -> StochasticSampleMapping:
        """Renewal data sampling.

        This function will sample data and encapsulate them in an object.

        Parameters
        ----------
        nb_samples : int
            The size of the desired sample
        time_window : tuple of two floats
            Time window in which data are sampled
        seed : int, optional
            Random seed, by default None.

        """
        return self._stochastic_process.sample(
            nb_samples, time_window, a0=self.a0, seed=seed
        )
