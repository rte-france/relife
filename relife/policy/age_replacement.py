from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import newton

from relife.economic import age_replacement_rewards, reward_partial_expectation
from relife.lifetime_model import AgeReplacementModel, LeftTruncatedModel
from ..lifetime_model._base import legendre_quadrature
from relife.stochastic_process import RenewalRewardProcess

from .._args import broadcast_args
from ..stochastic_process.frozen_process import FrozenNonHomogeneousPoissonProcess
from ._base import RenewalPolicy
from ._decorator import get_if_none

if TYPE_CHECKING:
    from relife.lifetime_model import ParametricLifetimeModel
    from relife.stochastic_process import NonHomogeneousPoissonProcess


class OneCycleAgeReplacementPolicy(RenewalPolicy):
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

    model1 = None

    def __init__(
        self,
        model: ParametricLifetimeModel[()],
        cf: float | NDArray[np.float64],
        cp: float | NDArray[np.float64],
        *,
        discounting_rate: Optional[float] = None,
        period_before_discounting: float = 1.0,
        ar: Optional[float | NDArray[np.float64]] = None,
        a0: Optional[float | NDArray[np.float64]] = None,
    ) -> None:
        super().__init__(model, discounting_rate=discounting_rate, cf=cf, cp=cp)

        if period_before_discounting == 0:
            raise ValueError("The period_before_discounting must be greater than 0")
        self.period_before_discounting = period_before_discounting

        if a0 is not None:
            self.model = LeftTruncatedModel(self.model).freeze(a0=a0)

        self.ar = self._reshape_ar(ar)

    def _reshape_ar(
        self, ar: Optional[float | NDArray[np.float64]]
    ) -> Optional[float | NDArray[np.float64]]:
        if ar is not None:
            ar = np.asarray(ar)
            if ar.size != 1:
                ar = ar.reshape(-1, 1)
                if ar.shape[0] != self.nb_assets:
                    raise ValueError
            else:
                ar = ar.item()
        return ar

    @property
    def cp(self):
        return self.cost_structure["cp"]

    @property
    def cf(self):
        return self.cost_structure["cf"]

    @property
    def discounting_rate(self):
        return self.discounting.rate

    @get_if_none("ar")
    def expected_total_cost(
        self,
        timeline: NDArray[np.float64],
        ar: Optional[float | NDArray[np.float64]] = None,
    ) -> NDArray[np.float64]:
        return reward_partial_expectation(
            timeline,
            AgeReplacementModel(self.model).freeze(ar),
            age_replacement_rewards(ar, self.cp, self.cf),
            discounting=self.discounting,
        )

    @get_if_none("ar")
    def asymptotic_expected_total_cost(
        self, ar: Optional[float | NDArray[np.float64]] = None
    ) -> NDArray[np.float64]:
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
        q0 = self.model.cdf(self.period_before_discounting) * f(
            self.period_before_discounting
        )
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

    @get_if_none("ar")
    def asymptotic_expected_equivalent_annual_cost(
        self, ar: Optional[float | NDArray[np.float64]] = None
    ) -> NDArray[np.float64]:
        return self.expected_equivalent_annual_cost(np.array(np.inf), ar=ar)

    def optimize(
        self,
    ) -> NDArray[np.float64]:
        """
        Computes the optimal age(s) of replacement and updates the internal ``ar`̀` value(s) and, optionally ``ar1``.

        Returns
        -------
        Self
             Same instance with optimized ``ar`` (optionnaly ``ar1``).
        """

        cf_3d, cp_3d = np.array(self.cf, ndmin=3), np.array(self.cp, ndmin=3)
        x0 = np.minimum(np.sum(cp_3d, axis=0) / np.sum(cf_3d - cp_3d, axis=0), 1)
        if np.size(x0) == 1:
            x0 = np.tile(x0, (self.nb_assets, 1))

        def eq(a):
            return np.sum(
                self.discounting.factor(a)
                / self.discounting.annuity_factor(a)
                * (
                    (cf_3d - cp_3d) * self.model.hf(a)
                    - cp_3d / self.discounting.annuity_factor(a)
                ),
                axis=0,
            )

        ar = np.asarray(newton(eq, x0), dtype=np.float64)
        self.ar = ar
        return ar


class DefaultAgeReplacementPolicy(RenewalPolicy):
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

    def __init__(
        self,
        model: ParametricLifetimeModel[()],
        cf: float | NDArray[np.float64],
        cp: float | NDArray[np.float64],
        *,
        discounting_rate: Optional[float] = None,
        ar: Optional[float | NDArray[np.float64]] = None,
        ar1: Optional[float | NDArray[np.float64]] = None,
        a0: Optional[float | NDArray[np.float64]] = None,
        model1: Optional[ParametricLifetimeModel[()]] = None,
    ) -> None:
        super().__init__(model, model1, discounting_rate, cf=cf, cp=cp)

        if a0 is not None:
            if model1 is not None:
                raise ValueError("model1 and a0 can't be set together")
            self.model1 = LeftTruncatedModel(self.model).freeze(a0)
        elif model1 is None and ar1 is not None:
            raise ValueError("model1 is not set, ar1 is useless")

        broadcast_args(ar=ar)

        self.ar = self._reshape_ar(ar)
        self.ar1 = self._reshape_ar(ar1)

    def _reshape_ar(
        self, ar: Optional[float | NDArray[np.float64]]
    ) -> Optional[float | NDArray[np.float64]]:
        if ar is not None:
            ar = np.asarray(ar)
            if ar.size != 1:
                ar = ar.reshape(-1, 1)
                if ar.shape[0] != self.nb_assets:
                    raise ValueError
            else:
                ar = ar.item()
        return ar

    @property
    def discounting_rate(self):
        return self.discounting.rate

    @property
    def cp(self):
        return self.cost_structure["cp"]

    @property
    def cf(self):
        return self.cost_structure["cf"]

    def underlying_process(
        self, ar: float | NDArray[np.float64], ar1: float | NDArray[np.float64]
    ) -> RenewalRewardProcess:
        return RenewalRewardProcess(
            AgeReplacementModel(self.model).freeze(ar),
            age_replacement_rewards(ar, self.cf, self.cp),
            discounting_rate=self.discounting_rate,
            model1=(
                AgeReplacementModel(self.model1).freeze(ar1)
                if self.model1 is not None
                else None
            ),
            reward1=age_replacement_rewards(ar1, self.cf, self.cp) if ar1 else None,
        )

    @get_if_none("ar", "ar1")
    def expected_total_cost(
        self,
        tf: float,
        nb_steps: int,
        ar: Optional[float | NDArray[np.float64]] = None,
        ar1: Optional[float | NDArray[np.float64]] = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        timeline = np.linspace(0, tf, nb_steps)
        return timeline, self.underlying_process(ar, ar1).expected_total_reward(
            timeline
        )

    @get_if_none("ar", "ar1")
    def expected_equivalent_annual_cost(
        self,
        tf: float,
        nb_steps: int,
        ar: Optional[float | NDArray[np.float64]] = None,
        ar1: Optional[float | NDArray[np.float64]] = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        timeline = np.linspace(0, tf, nb_steps)
        return timeline, self.underlying_process(
            ar, ar1
        ).expected_equivalent_annual_worth(timeline)

    @get_if_none("ar", "ar1")
    def asymptotic_expected_total_cost(
        self,
        ar: Optional[float | NDArray[np.float64]] = None,
        ar1: Optional[float | NDArray[np.float64]] = None,
    ) -> NDArray[np.float64]:
        return self.underlying_process(ar, ar1).asymptotic_expected_total_reward()

    @get_if_none("ar", "ar1")
    def asymptotic_expected_equivalent_annual_cost(
        self,
        ar: Optional[float | NDArray[np.float64]] = None,
        ar1: Optional[float | NDArray[np.float64]] = None,
    ) -> NDArray[np.float64]:

        return self.underlying_process(
            ar, ar1
        ).asymptotic_expected_equivalent_annual_worth()

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
        if np.size(x0) == 1:
            x0 = np.tile(x0, (self.nb_assets, 1))

        ndim = max(
            map(np.ndim, (cf_3d, cp_3d, *self.model.args)),
            default=0,
        )

        def eq(a):
            f = legendre_quadrature(
                lambda x: self.discounting.factor(x) * self.model.sf(x),
                0,
                a,
                ndim=ndim,
            )
            g = legendre_quadrature(
                lambda x: self.discounting.factor(x) * self.model.pdf(x),
                0,
                a,
                ndim=ndim,
            )
            return np.sum(
                self.discounting.factor(a)
                * ((cf_3d - cp_3d) * (self.model.hf(a) * f - g) - cp_3d)
                / f**2,
                axis=0,
            )

        ar = newton(eq, x0)
        if np.size(ar) == 1:
            ar = np.squeeze(ar)

        ar1 = None
        if self.model1 is not None:
            ar1 = OneCycleAgeReplacementPolicy(
                self.model1,
                self.cf,
                self.cp,
                discounting_rate=self.discounting_rate,
            ).optimize()

        return ar, ar1


class NonHomogeneousPoissonAgeReplacementPolicy(RenewalPolicy):
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
    def asymptotic_expected_total_cost(
        self, ar: Optional[float | NDArray[np.float64]] = None
    ) -> NDArray[np.float64]:
        pass

    @get_if_none("ar")
    def expected_equivalent_annual_cost(
        self,
        timeline: NDArray[np.float64],
        ar: Optional[float | NDArray[np.float64]] = None,
    ) -> NDArray[np.float64]:
        pass

    def number_of_replacements(
        self, timeline: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        pass

    def expected_number_of_repairs(
        self, timeline: NDArray[np.float64]
    ) -> NDArray[np.float64]:
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
                    (1 - self.discounting.factor(a))
                    / self.discounting.rate
                    * self.underlying_process.intensity(a)
                    - legendre_quadrature(
                        lambda t: self.discounting.factor(t)
                        * self.underlying_process.intensity(t),
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
OneCycleAgeReplacementPolicy.expected_equivalent_annual_cost.__doc__ = (
    EEAC_DOCSTRING + WARNING
)
OneCycleAgeReplacementPolicy.asymptotic_expected_total_cost.__doc__ = (
    ASYMPTOTIC_ETC_DOCSTRING + WARNING
)
OneCycleAgeReplacementPolicy.asymptotic_expected_equivalent_annual_cost.__doc__ = (
    ASYMPTOTIC_EEAC_DOCSTRING + WARNING
)

DefaultAgeReplacementPolicy.expected_total_cost.__doc__ = ETC_DOCSTRING + WARNING
DefaultAgeReplacementPolicy.expected_equivalent_annual_cost.__doc__ = (
    EEAC_DOCSTRING + WARNING
)
DefaultAgeReplacementPolicy.asymptotic_expected_total_cost.__doc__ = (
    ASYMPTOTIC_ETC_DOCSTRING + WARNING
)

DefaultAgeReplacementPolicy.asymptotic_expected_equivalent_annual_cost.__doc__ = (
    ASYMPTOTIC_EEAC_DOCSTRING + WARNING
)
