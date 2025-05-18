from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Self

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import newton

from relife.economic import AgeReplacementReward, reward_partial_expectation, Discounting, Cost
from relife.lifetime_model import AgeReplacementModel, LeftTruncatedModel, FrozenLeftTruncatedModel, \
    FrozenAgeReplacementModel
from relife.stochastic_process import RenewalRewardProcess

from .. import freeze
from ..lifetime_model._base import legendre_quadrature, FrozenLifetimeRegression
from ..stochastic_process.frozen_process import FrozenNonHomogeneousPoissonProcess
from ._base import AgeRenewalPolicy
from ._decorator import get_if_none

if TYPE_CHECKING:
    from relife.lifetime_model import (
        FrozenParametricLifetimeModel,
        LifetimeDistribution,
    )
    from relife.stochastic_process import NonHomogeneousPoissonProcess


class OneCycleAgeReplacementPolicy:
    r"""One-cyle age replacement policy.

    The asset is disposed at a fixed age :math:`a_r` with costs :math:`c_p` or upon failure
    with costs :math:`c_f` if earlier.

    Parameters
    ----------
    model : Lifetimemodel[()]
        The lifetime model of the assets.
    cf : np.ndarray
        The cost of failure for each asset.
    cp : np.ndarray
        The cost of preventive replacements for each asset.
    discounting_rate : float, default is 0.
        The discounting rate.
    period_before_discounting: float, default is 1.
         The length of the first period before discounting.
    ar : np.ndarray, optional
        Times until preventive replacements. This parameter can be optimized
        with ``optimize``
    a0 : ndarray, optional
        Current ages of the assets (default is None). Setting ``a0`` will add
        left truncations.


    References
    ----------
    .. [1] Coolen-Schrijner, P., & Coolen, F. P. A. (2006). On optimality
        criteria for age replacement. Proceedings of the Institution of
        Mechanical Engineers, Part O: Journal of Risk and Reliability,
        220(1), 21-29
    """
    ar : float | NDArray[np.float64]
    lifetime_model : FrozenAgeReplacementModel

    def __init__(
        self,
        lifetime_model : LifetimeDistribution | FrozenLifetimeRegression | FrozenLeftTruncatedModel | FrozenAgeReplacementModel,
        cost : Cost,
        discounting : Discounting,
        period_before_discounting: float = 1.0,
    ) -> None:
        if not isinstance(lifetime_model, FrozenAgeReplacementModel):
            lifetime_model = freeze(AgeReplacementModel(lifetime_model), np.nan)
        self.lifetime_model = lifetime_model
        self.cost = cost
        self.reward = AgeReplacementReward(cost["cf"], cost["cp"], self.ar)
        self.discounting = discounting
        if period_before_discounting <= 0:
            raise ValueError("The period_before_discounting must be greater than 0")
        self.period_before_discounting = period_before_discounting

    @property
    def ar(self) -> float | NDArray[np.float64]:
        return self.lifetime_model.ar

    @ar.setter
    def ar(self, value : float | NDArray[np.float64]) -> None:
        self.lifetime_model.ar = value

    @property
    def cp(self) -> NDArray[np.float64]:
        return self.cost["cp"]

    @property
    def cf(self) -> NDArray[np.float64]:
        return self.cost["cf"]

    @property
    def discounting_rate(self) -> float:
        return self.discounting.rate

    def expected_total_cost(
        self,
        tf: float,
        nb_steps: int,
    ) -> Optional[NDArray[np.float64]]:
        timeline = np.linspace(0, tf, nb_steps)
        if np.isnan(self.ar):
            raise ValueError

        return reward_partial_expectation(
            timeline,
            AgeReplacementModel(self.model).freeze(ar),
            age_replacement_rewards(ar, self.cp, self.cf),
            discounting=self.discounting,
        )

    def asymptotic_expected_total_cost(self) -> NDArray[np.float64]:
        if np.isnan(self.ar):
            raise ValueError
        return self.expected_total_cost(np.array(np.inf), ar=ar)

    @get_if_none("ar")
    def expected_equivalent_annual_cost(
        self,
        timeline: NDArray[np.float64],
        ar: Optional[float | NDArray[np.float64]] = None,
    ) -> NDArray[np.float64]:
        f = (
            lambda x: age_replacement_rewards(ar, self.cf, self.cp)(x)
            * self.discounting.factor(x)
            / self.discounting.annuity_factor(x)
        )
        mask = timeline < self.period_before_discounting
        q0 = self.model.cdf(self.period_before_discounting) * f(self.period_before_discounting)
        model = AgeReplacementModel(self.model).freeze(ar)
        return np.squeeze(
            q0
            + np.where(
                mask,
                0,
                model.ls_integrate(
                    f,
                    np.array(self.period_before_discounting),
                    timeline,
                ),
            )
        )

    def asymptotic_expected_equivalent_annual_cost(
        self, ar: Optional[float | NDArray[np.float64]] = None
    ) -> NDArray[np.float64]:
        if np.isnan(self.ar):
            raise ValueError
        return self.expected_equivalent_annual_cost(np.array(np.inf), ar=ar)

    def optimize(
        self,
    ) -> Self:
        """
        Computes the optimal age(s) of replacement and updates the internal ``ar`̀` value(s) and, optionally ``ar1``.

        Returns
        -------
        Self
             Same instance with optimized ``ar`` (optionnaly ``ar1``).
        """

        cf_3d, cp_3d = np.array(self.cf, ndmin=3), np.array(self.cp, ndmin=3)
        x0 = np.minimum(np.sum(cp_3d, axis=0) / np.sum(cf_3d - cp_3d, axis=0), 1)

        def eq(a):
            return np.sum(
                self.discounting.factor(a)
                / self.discounting.annuity_factor(a)
                * ((cf_3d - cp_3d) * self.lifetime_model.unfrozen_model.baseline.hf(a) - cp_3d / self.discounting.annuity_factor(a)),
                axis=0,
            )

        ar = np.asarray(newton(eq, x0), dtype=np.float64)
        self.ar = ar
        return self


class AgeReplacementPolicy:
    r"""Time based replacement policy.

    Renewal reward stochastic_process where assets are replaced at a fixed age :math:`a_r`
    with costs :math:`c_p` or upon failure with costs :math:`c_f` if earlier [1]_.

    .. [1] Mazzuchi, T. A., Van Noortwijk, J. M., & Kallen, M. J. (2007).
        Maintenance optimization. Encyclopedia of Statistics in Quality and
        Reliability, 1000-1008.

    |

    Parameters
    ----------
    model : BaseLifetimeModel
        The lifetime core of the assets.
    cf : np.ndarray
        The cost of failure for each asset.
    cp : np.ndarray
        The cost of preventive replacements for each asset.
    ar : np.ndarray, optional
        Times until preventive replacements. This parameter can be optimized
        with ``optimize``
    ar1 : np.ndarray, optional
        Times until preventive replacements for the first cycle. This parameter can be optimized
        with ``optimize``
    discounting_rate : float, default is 0.
        The discounting rate.
    a0 : ndarray, optional
        Current ages of the assets (default is None). Setting ``a0`` will add
        left truncations.

    Attributes
    ----------
    ar : np.ndarray or None
        Times until preventive replacements. This parameter can be optimized
        with ``optimize``
    ar1 : np.ndarray or None
        Times until preventive replacements for the first cycle. This parameter can be optimized
        with ``optimize``
    """
    lifetime_model : FrozenAgeReplacementModel
    first_lifetime_model : Optional[FrozenAgeReplacementModel]
    reward : AgeReplacementReward
    first_reward : Optional[AgeReplacementReward]
    cost : Cost
    discounting : Discounting

    def __init__(
        self,
        lifetime_model: LifetimeDistribution | FrozenLifetimeRegression | FrozenLeftTruncatedModel | FrozenAgeReplacementModel,
        cost: Cost,
        discounting: Discounting,
        first_lifetime_model : Optional[LifetimeDistribution | FrozenLifetimeRegression | FrozenLeftTruncatedModel | FrozenAgeReplacementModel] = None,
    ) -> None:
        if not isinstance(lifetime_model, FrozenAgeReplacementModel):
            lifetime_model = freeze(AgeReplacementModel(lifetime_model), np.nan)
        if first_lifetime_model is not None and not isinstance(first_lifetime_model, FrozenAgeReplacementModel):
            first_lifetime_model = freeze(AgeReplacementModel(first_lifetime_model), np.nan)
        first_reward = None
        if first_lifetime_model is not None:
            first_reward = AgeReplacementReward(cost["cf"], cost["cp"], first_lifetime_model.ar)
        self.lifetime_model = lifetime_model
        self.first_lifetime_model = first_lifetime_model
        self.cost = cost
        self.reward = AgeReplacementReward(cost["cf"], cost["cp"], lifetime_model.ar)
        self.first_reward = first_reward
        self.discounting = discounting

    @property
    def underlying_process(self) -> RenewalRewardProcess:
        return RenewalRewardProcess(
            FrozenAgeReplacementModel,
            AgeReplacementReward(self.cost["cf"], self.cost["cp"], self.ar),
            self.discounting,
            first_lifetime_model=self.first_lifetime_model,
            first_reward=self.first_reward,
        )
    @property
    def ar(self) -> float | NDArray[np.float64]:
        return self.lifetime_model.ar

    @ar.setter
    def ar(self, value : float | NDArray[np.float64]) -> None:
        self.lifetime_model.ar = value

    @property
    def ar1(self) -> Optional[float | NDArray[np.float64]]:
        if self.first_lifetime_model is not None:
            return self.lifetime_model.ar
        return None

    @ar1.setter
    def ar1(self, value : float | NDArray[np.float64]) -> None:
        if self.first_lifetime_model is not None:
            self.first_lifetime_model.ar = value
        raise ValueError

    @property
    def discounting_rate(self) -> float:
        return self.discounting.rate

    @property
    def cp(self) -> float | NDArray[np.float64]:
        return self.reward.cost["cp"]

    @property
    def cf(self) -> float | NDArray[np.float64]:
        return self.reward.cost["cf"]

    def expected_total_cost(
        self,
        tf: float,
        nb_steps: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        if np.isnan(self.ar):
            raise ValueError
        if self.ar1 is not None and np.isnan(self.ar):
            raise ValueError
        return self.underlying_process.expected_total_reward(tf, nb_steps)

    def expected_equivalent_annual_cost(
        self,
        tf: float,
        nb_steps: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        if np.isnan(self.ar):
            raise ValueError
        if self.ar1 is not None and np.isnan(self.ar):
            raise ValueError
        return self.underlying_process.expected_equivalent_annual_worth(tf, nb_steps)

    def asymptotic_expected_total_cost(self) -> NDArray[np.float64]:
        if np.isnan(self.ar):
            raise ValueError
        if self.ar1 is not None and np.isnan(self.ar):
            raise ValueError
        return self.underlying_process.asymptotic_expected_total_reward()

    def asymptotic_expected_equivalent_annual_cost(self) -> NDArray[np.float64]:
        if np.isnan(self.ar):
            raise ValueError
        if self.ar1 is not None and np.isnan(self.ar):
            raise ValueError
        return self.underlying_process.asymptotic_expected_equivalent_annual_worth()

    def optimize(
        self,
    ) -> tuple[NDArray[np.float64], Optional[NDArray[np.float64]]]:
        """
        Computes the optimal age(s) of replacement and updates the internal ``ar`̀` value(s) and, optionally ``ar1``.

        Returns
        -------
        Self
             Same instance with optimized ``ar`` (optionnaly ``ar1``).
        """
        cf_3d, cp_3d = np.array(self.cf, ndmin=3), np.array(self.cp, ndmin=3)
        x0 = np.minimum(np.sum(cp_3d, axis=0) / np.sum(cf_3d - cp_3d, axis=0), 1)

        def eq(a):
            f = legendre_quadrature(
                lambda x: self.discounting.factor(x) * self.lifetime_model.unfrozen_model.baseline.sf(x),
                0,
                a,
            )
            g = legendre_quadrature(
                lambda x: self.discounting.factor(x) * self.lifetime_model.unfrozen_model.baseline.pdf(x),
                0,
                a,
            )
            return np.sum(
                self.discounting.factor(a) * ((cf_3d - cp_3d) * (self.lifetime_model.unfrozen_model.baseline.hf(a) * f - g) - cp_3d) / f**2,
                axis=0,
            )
        self.ar = newton(eq, x0)
        if self.ar1 is not None:
            one_cycle_ar_policy = OneCycleAgeReplacementPolicy(
                self.lifetime_model,
                self.cost,
                self.discounting,
            ).optimize()
            self.ar1 = one_cycle_ar_policy.ar
        return self


class NonHomogeneousPoissonAgeReplacementPolicy(AgeRenewalPolicy):
    """
    Implements a Non-Homogeneous Poisson Process (NHPP) age-replacement policy..

    Attributes
    ----------
    ar : np.ndarray or None
        Optimized replacement age (optimized policy parameter).
    cp : np.ndarray
        The cost of failure for each asset.
    cr : np.ndarray
        The cost of repair for each asset.
    """

    def __init__(
        self,
        process: NonHomogeneousPoissonProcess[()],
        cp: float | NDArray[np.float64],
        cr: float | NDArray[np.float64],
        *,
        discounting_rate: Optional[float] = None,
        ar: Optional[float | NDArray[np.float64]] = None,
    ) -> None:
        super().__init__(process.model, discounting_rate=discounting_rate)
        self.ar = ar
        self.cp = cp
        self.cr = cr
        if not isinstance(process, FrozenNonHomogeneousPoissonProcess):
            raise ValueError
        self._process = process

    @property
    def underlying_process(self) -> FrozenNonHomogeneousPoissonProcess:
        return self._process

    @property
    def discounting_rate(self):
        return self.discounting.rate

    @get_if_none("ar")
    def expected_total_cost(
        self,
        timeline: NDArray[np.float64],
        ar: Optional[float | NDArray[np.float64]] = None,
    ) -> NDArray[np.float64]:
        pass

    @get_if_none("ar")
    def asymptotic_expected_total_cost(self, ar: Optional[float | NDArray[np.float64]] = None) -> NDArray[np.float64]:
        pass

    @get_if_none("ar")
    def expected_equivalent_annual_cost(
        self,
        timeline: NDArray[np.float64],
        ar: Optional[float | NDArray[np.float64]] = None,
    ) -> NDArray[np.float64]:
        pass

    def number_of_replacements(self, timeline: NDArray[np.float64]) -> NDArray[np.float64]:
        pass

    def expected_number_of_repairs(self, timeline: NDArray[np.float64]) -> NDArray[np.float64]:
        pass

    def asymptotic_expected_equivalent_annual_cost(self) -> NDArray[np.float64]:
        pass

    def optimize(
        self,
    ) -> NDArray[np.float64]:

        x0 = self.model.mean()

        if self.discounting.rate != 0:

            def dcost(a):
                a = np.atleast_2d(a)
                return (
                    (1 - self.discounting.factor(a)) / self.discounting.rate * self.underlying_process.intensity(a)
                    - legendre_quadrature(
                        lambda t: self.discounting.factor(t) * self.underlying_process.intensity(t),
                        np.array(0.0),
                        a,
                        ndim=2,
                    )
                    - self.cp / self.cr
                )

        else:

            def dcost(a):
                a = np.atleast_2d(a)
                return (
                    a * self.underlying_process.intensity(a)
                    - self.underlying_process.cumulative_intensity(a)
                    - self.cp / self.cr
                )

        ar = np.squeeze(newton(dcost, x0))
        self.ar = ar
        return ar


from ._docstring import (
    ASYMPTOTIC_EEAC_DOCSTRING,
    ASYMPTOTIC_ETC_DOCSTRING,
    EEAC_DOCSTRING,
    ETC_DOCSTRING,
)

WARNING = r"""

.. warning::

    This method requires the ``ar`` attribute to be set either at initialization
    or with the ``optimize`` method.
"""

OneCycleAgeReplacementPolicy.expected_total_cost.__doc__ = ETC_DOCSTRING + WARNING
OneCycleAgeReplacementPolicy.expected_equivalent_annual_cost.__doc__ = EEAC_DOCSTRING + WARNING
OneCycleAgeReplacementPolicy.asymptotic_expected_total_cost.__doc__ = ASYMPTOTIC_ETC_DOCSTRING + WARNING
OneCycleAgeReplacementPolicy.asymptotic_expected_equivalent_annual_cost.__doc__ = ASYMPTOTIC_EEAC_DOCSTRING + WARNING

DefaultAgeReplacementPolicy.expected_total_cost.__doc__ = ETC_DOCSTRING + WARNING
DefaultAgeReplacementPolicy.expected_equivalent_annual_cost.__doc__ = EEAC_DOCSTRING + WARNING
DefaultAgeReplacementPolicy.asymptotic_expected_total_cost.__doc__ = ASYMPTOTIC_ETC_DOCSTRING + WARNING

DefaultAgeReplacementPolicy.asymptotic_expected_equivalent_annual_cost.__doc__ = ASYMPTOTIC_EEAC_DOCSTRING + WARNING
