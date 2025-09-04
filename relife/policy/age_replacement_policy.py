from __future__ import annotations

import copy
import warnings
from typing import Optional, Self

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import newton
from typing_extensions import override

from relife.economic import AgeReplacementReward, ExponentialDiscounting, cost
from relife.lifetime_model import (
    AgeReplacementModel,
    FrozenAgeReplacementModel,
    FrozenLeftTruncatedModel,
    LeftTruncatedModel,
)
from relife.quadrature import legendre_quadrature
from relife import is_frozen

from relife.lifetime_model.distribution import LifetimeDistribution
from relife.lifetime_model.regression import FrozenLifetimeRegression
from relife.stochastic_process import (
    FrozenNonHomogeneousPoissonProcess,
    RenewalRewardProcess, NonHomogeneousPoissonProcess,
)
from ._base import BaseAgeReplacementPolicy, BaseOneCycleAgeReplacementPolicy


class OneCycleAgeReplacementPolicy(BaseOneCycleAgeReplacementPolicy[FrozenAgeReplacementModel, AgeReplacementReward]):
    # noinspection PyUnresolvedReferences
    r"""One-cyle age replacement policy.

    The asset is disposed at a fixed age :math:`a_r` with costs :math:`c_p` or upon failure
    with costs :math:`c_f` if earlier.

    Parameters
    ----------
    lifetime_model : any lifetime distribution or frozen lifetime model
        A lifetime model representing the durations between events.
    cf : float or 1darray
        Costs of failures
    cp : float or 1darray
        Costs of preventive replacements
    discounting_rate : float, default is 0.
        The discounting rate value used in the exponential discounting function
    a0 : float or 1darray, optional
        Current ages of the assets. If it is given, left truncations of ``a0`` will
        be take into account for the first cycle.
    ar : float or 1darray, optional
        Ages of preventive replacements, by default None. If not given, one must call ``optimize`` to set ``ar`` values
        and access to the rest of the object interface.

    Attributes
    ----------
    a0 : float or 1darray, optional
        Current ages of the assets. If it is given, left truncations of ``a0`` will
        be take into account for the first cycle.
    ar
    cf
    cp
    tr


    References
    ----------
    .. [1] Coolen-Schrijner, P., & Coolen, F. P. A. (2006). On optimality
        criteria for age replacement. Proceedings of the Institution of
        Mechanical Engineers, Part O: Journal of Risk and Reliability,
        220(1), 21-29
    """

    lifetime_model: FrozenAgeReplacementModel
    reward: AgeReplacementReward

    def __init__(
        self,
        lifetime_model: LifetimeDistribution | FrozenLifetimeRegression | FrozenLeftTruncatedModel,
        cf: float | NDArray[np.float64],
        cp: float | NDArray[np.float64],
        discounting_rate: float = 0.0,
        period_before_discounting: float = 1.0,
        a0: Optional[float | NDArray[np.float64]] = None,
        ar: Optional[float | NDArray[np.float64]] = None,
    ) -> None:
        self.a0 = np.float64(a0) if isinstance(a0, (float, int)) else a0  # None, arr () or (m,) or (m, 1)
        self._ar = ar if ar is not None else np.nan
        if a0 is not None:
            lifetime_model: FrozenAgeReplacementModel = AgeReplacementModel(LeftTruncatedModel(lifetime_model)).freeze(
                self._ar, a0
            )
        else:
            lifetime_model: FrozenAgeReplacementModel = AgeReplacementModel(lifetime_model).freeze(self._ar)
        reward = AgeReplacementReward(cf, cp, ar)
        super().__init__(lifetime_model, reward, discounting_rate, period_before_discounting)

    @property
    def cf(self) -> NDArray[np.float64]:
        """
        Cost of failures

        Returns
        -------
        ndarray
        """
        return self.reward.cf

    @property
    def cp(self) -> NDArray[np.float64]:
        """
        Cost of preventive replacements

        Returns
        -------
        ndarray
        """
        return self.reward.cp

    @property
    def ar(self) -> float | NDArray[np.float64]:
        """
        Ages of the preventive replacements

        Returns
        -------
        ndarray
        """
        return np.squeeze(self._ar)

    @ar.setter
    def ar(self, value: float | NDArray[np.float64]) -> None:
        self._ar = value
        if self.a0 is None:
            a0 = np.zeros_like(value)
        else:
            a0 = self.a0
        time_before_replacement = np.maximum(value - a0, 0)
        self.lifetime_model.ar = time_before_replacement
        self.reward.ar = time_before_replacement

    @property
    def tr(self) -> float | NDArray[np.float64]:
        """
        Time before the replacement

        Returns
        -------
        ndarray
        """
        return np.squeeze(self.lifetime_model.ar)

    @override
    def expected_total_cost(
        self,
        tf: float,
        nb_steps: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        if np.any(np.isnan(self.ar)):
            raise ValueError
        return super().expected_total_cost(tf, nb_steps)  # (nb_steps,), (nb_steps,) or (nb_steps,), (m, nb_steps)

    @override
    def asymptotic_expected_total_cost(self) -> NDArray[np.float64]:
        if np.any(np.isnan(self.ar)):
            raise ValueError
        return super().asymptotic_expected_total_cost()  # () or (m, nb_steps)

    @override
    def expected_equivalent_annual_cost(
        self, tf: float, nb_steps: int
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        if np.any(np.isnan(self.ar)):
            raise ValueError
        # because b = np.minimum(ar, b) in ls_integrate, b can be lower than a depending on period before discounting
        if np.any(self.period_before_discounting >= self.ar):
            raise ValueError("The period before discounting must be lower than ar values")

        if np.any(self.tr == 0):
            warnings.warn(
                "Some assets has already been replaced for the first cycle (where tr is 0). For these assets, consider adjusting ar values to be greater than a0"
            )

        ar = self.ar.copy()
        #  change ar temporarly to enable computation of eeac (if not, AgeReplacementModel.ls_integrate bounds will be problematic)
        self.ar = np.where(self.tr == 0, np.inf, self.ar)
        timeline, eeac = super().expected_equivalent_annual_cost(tf, nb_steps)
        if eeac.ndim == 2:  # more than one asset
            eeac[np.where(np.atleast_1d(self.ar) == np.inf)[0], :] = np.nan
        if eeac.ndim == 1 and self.ar == np.inf:
            eeac.fill(np.nan)
        self.ar = ar
        return timeline, eeac  # (nb_steps,), (nb_steps,) or (nb_steps,), (m, nb_steps)

    @override
    def asymptotic_expected_equivalent_annual_cost(self) -> NDArray[np.float64]:
        if np.any(np.isnan(self.ar)):
            raise ValueError
        # because b = np.minimum(ar, b) in ls_integrate, b can be lower than a depending on period before discounting
        if np.any(self.period_before_discounting >= self.ar):
            raise ValueError("The period before discounting must be lower than ar values")

        if np.any(self.tr == 0):
            warnings.warn(
                "Some assets has already been replaced for the first cycle (where tr is 0). For these assets, consider adjusting ar values to be greater than a0"
            )

        ar = self.ar.copy()
        #  change ar temporarly to enable computation of eeac (if not, AgeReplacementModel.ls_integrate bounds are not in right order)
        self.ar = np.where(self.tr == 0, np.inf, self.ar)
        asymptotic_eeac = super().asymptotic_expected_equivalent_annual_cost()

        if asymptotic_eeac.ndim == 1:  # more than one asset
            asymptotic_eeac[np.where(self.ar == np.inf)[0]] = np.nan
        if asymptotic_eeac.ndim == 0 and self.ar == np.inf:
            asymptotic_eeac = np.nan
        self.ar = ar
        return asymptotic_eeac  # () or (m, nb_steps)

    def optimize(
        self,
    ) -> OneCycleAgeReplacementPolicy:
        """
        Computes the optimal age(s) of replacement and updates the internal ``ar`` value(s) and, optionally ``ar1``.

        Returns
        -------
        Self
             Same instance with optimized ``ar`` (optionnaly ``ar1``).
        """

        x0 = np.minimum(self.cp / (self.cf - self.cp), 1)  # ()
        #  TODO : LeftTruncatedModel.hf (ne pas utiliser LeftTruncatedModel),
        #  retourner ar parfois < a0 (ou ar = 0), mais garder LeftTruncated pour la projection des conséquences
        if self.a0 is not None:
            # FrozenAgeReplacementModel(LeftTruncatedModel(...))
            hf = self.lifetime_model.unfrozen_model.baseline.baseline.hf
        else:
            # FrozenAgeReplacementModel(...)
            hf = self.lifetime_model.unfrozen_model.baseline.hf
        x0 = np.broadcast_to(x0, hf(x0).shape).copy()  # () or (m, 1)

        def eq(a):  # () or (m, 1)
            return (
                self.discounting.factor(a)
                / self.discounting.annuity_factor(a)
                * ((self.cf - self.cp) * hf(a) - self.cp / self.discounting.annuity_factor(a))
            )
            # return ((self.cf - self.cp) / self.cp) * hf(a) * self.discounting.annuity_factor(a)

        ar = np.squeeze(newton(eq, x0))  # () or (m,)

        if self.a0 is not None and self.a0.ndim > 0:
            ar = np.broadcast_to(ar, self.a0.shape).copy()

        self.ar = ar  # setter is called

        return self


class AgeReplacementPolicy(BaseAgeReplacementPolicy[FrozenAgeReplacementModel, AgeReplacementReward]):
    # noinspection PyUnresolvedReferences
    r"""Age replacement policy.

    Behind the scene, a renewal reward stochastic process is used where assets are replaced at a fixed age :math:`a_r`
    with costs :math:`c_p` or upon failure with costs :math:`c_f` if earlier [1]_.

    Parameters
    ----------
    lifetime_model : any lifetime distribution or frozen lifetime model
        A lifetime model representing the durations between events.
    cf : float or 1darray
        Costs of failures
    cp : float or 1darray
        Costs of preventive replacements
    discounting_rate : float, default is 0.
        The discounting rate value used in the exponential discounting function
    a0 : float or 1darray, optional
        Current ages of the assets. If it is given, left truncations of ``a0`` will
        be take into account for the first cycle.
    ar : float or 1darray, optional
        Ages of preventive replacements, by default None. If not given, one must call ``optimize`` to set ``ar`` values
        and access to the rest of the object interface.

    Attributes
    ----------
    a0 : float or 1darray, optional
        Current ages of the assets. If it is given, left truncations of ``a0`` will
        be take into account for the first cycle.
    ar
    first_cycle_tr
    cf
    cp

    References
    ----------
    .. [1] Mazzuchi, T. A., Van Noortwijk, J. M., & Kallen, M. J. (2007).
        Maintenance optimization. Encyclopedia of Statistics in Quality and
        Reliability, 1000-1008.
    """

    stochastic_process: RenewalRewardProcess[FrozenAgeReplacementModel, AgeReplacementReward]

    def __init__(
        self,
        lifetime_model: LifetimeDistribution | FrozenLifetimeRegression,
        cf: float | NDArray[np.float64],
        cp: float | NDArray[np.float64],
        discounting_rate: float = 0.0,
        a0: Optional[float | NDArray[np.float64]] = None,
        ar: Optional[float | NDArray[np.float64]] = None,
    ) -> None:

        self.a0 = None
        first_lifetime_model: Optional[FrozenAgeReplacementModel] = None
        if a0 is not None:
            self.a0 = np.float64(a0) if isinstance(a0, (float, int)) else a0  # None, arr () or (m,) or (m, 1)
            first_lifetime_model: FrozenAgeReplacementModel = AgeReplacementModel(LeftTruncatedModel(lifetime_model)).freeze(ar if ar is not None else np.nan, a0)
        lifetime_model: FrozenAgeReplacementModel = AgeReplacementModel(lifetime_model).freeze(
            ar if ar is not None else np.nan
        )
        reward = AgeReplacementReward(cf, cp, ar if ar is not None else np.nan)

        stochastic_process = RenewalRewardProcess(
            lifetime_model,
            reward,
            discounting_rate,
            first_lifetime_model=first_lifetime_model,
        )
        super().__init__(stochastic_process)
        if ar is not None:
            self.ar = ar

    @property
    def ar(self) -> float | NDArray[np.float64]:
        """
        Ages of the preventive replacements

        Returns
        -------
        ndarray
        """
        return np.squeeze(self.stochastic_process.lifetime_model.ar)  # may be (m, 1) so squeeze it

    @ar.setter
    def ar(self, value: float | NDArray[np.float64]) -> None:

        if self.a0 is not None:
            value = np.broadcast_to(value, self.a0.shape).copy()

        self.stochastic_process.lifetime_model.ar = value
        self.stochastic_process.reward.ar = value

        if self.a0 is not None:
            first_cycle_tr = np.maximum(value - self.a0, 0)
            self.stochastic_process.first_lifetime_model.ar = first_cycle_tr
            self.stochastic_process.first_reward.ar = first_cycle_tr

    @property
    def first_cycle_tr(self) -> float | NDArray[np.float64]:
        """
        Time before the first replacement

        Returns
        -------
        ndarray
        """
        if self.a0 is not None:
            return np.squeeze(self.stochastic_process.first_lifetime_model.ar)
        return self.ar

    @property
    def cf(self) -> float | NDArray[np.float64]:
        """
        Cost of failures

        Returns
        -------
        ndarray
        """
        return self.stochastic_process.reward.cf

    @cf.setter
    def cf(self, value: float | NDArray[np.float64]) -> None:
        self.stochastic_process.reward.cf = value
        self.stochastic_process.first_reward.cf = value

    @property
    def cp(self) -> float | NDArray[np.float64]:
        """
        Cost of preventive replacements

        Returns
        -------
        ndarray
        """
        return self.stochastic_process.reward.cp

    @cp.setter
    def cp(self, value: float | NDArray[np.float64]) -> None:
        self.stochastic_process.reward.cp = value
        self.stochastic_process.first_reward.cp = value

    @override
    def expected_total_cost(
        self,
        tf: float,
        nb_steps: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        if np.any(np.isnan(self.ar)):
            raise ValueError
        return self.stochastic_process.expected_total_reward(
            tf, nb_steps
        )  # (nb_steps,), (nb_steps,) or (nb_steps,), (m, nb_steps)

    @override
    def asymptotic_expected_total_cost(self) -> NDArray[np.float64]:
        if np.any(np.isnan(self.ar)):
            raise ValueError
        return self.stochastic_process.asymptotic_expected_total_reward()  # () or (m,)

    @override
    def expected_equivalent_annual_cost(
        self,
        tf: float,
        nb_steps: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        if np.any(np.isnan(self.ar)):
            raise ValueError
        timeline, eeac = self.stochastic_process.expected_equivalent_annual_worth(tf, nb_steps)
        if np.any(self.first_cycle_tr == 0):
            warnings.warn(
                "Some assets has already been replaced for the first cycle (where first_cycle_tr is 0). For these assets, consider adjusting ar values to be greater than a0"
            )
        if eeac.ndim == 2:
            eeac[np.where(np.atleast_1d(self.first_cycle_tr) == 0)] = np.nan
        if eeac.ndim == 1 and self.ar == np.inf:
            eeac.fill(np.nan)
        return timeline, eeac  # (nb_steps,), (nb_steps,) or (nb_steps,), (m, nb_steps)

    @override
    def asymptotic_expected_equivalent_annual_cost(self) -> NDArray[np.float64]:
        if np.any(np.isnan(self.ar)):
            raise ValueError
        asymptotic_eeac = self.stochastic_process.asymptotic_expected_equivalent_annual_worth()
        if np.any(self.first_cycle_tr == 0):
            warnings.warn(
                "Some assets has already been replaced for the first cycle (where first_cycle_tr is 0). For these assets, consider adjusting ar values to be greater than a0"
            )
        if asymptotic_eeac.ndim == 1:
            asymptotic_eeac[np.where(np.atleast_1d(self.first_cycle_tr) == 0)] = np.nan
        if asymptotic_eeac.ndim == 0 and self.first_cycle_tr == 0:
            asymptotic_eeac = np.nan
        return asymptotic_eeac  # () or (m,)

    def annual_number_of_replacements(
        self,
        nb_years: int,
        upon_failure: bool = False,
        total: bool = True,
    ):
        """
        The expected number of annual replacements.

        Parameters
        ----------
        nb_years : int
            The number of years on which the annual number of replacements are projected
        upon_failure : bool, default is False
            If True, it also returns the annual number of replacements due to unexpected failures
        total : bool, default is True
            If True, the given numbers of replacements are the sum of all replacements without distinction between assets
        """
        if np.any(np.isnan(self.ar)):
            raise ValueError

        copied_policy = copy.deepcopy(self)
        copied_policy.cf = 1.0
        copied_policy.cp = 1.0
        copied_policy.discounting_rate = 0.0
        timeline, total_cost = copied_policy.expected_total_cost(nb_years, nb_years + 1)  # equiv to np.arange
        if total:
            mt = np.sum(np.atleast_2d(total_cost), axis=0)
        else:
            mt = total_cost
        nb_replacements = np.diff(mt)
        if upon_failure:
            copied_policy.cp = 0.0
            _, total_cost = copied_policy.expected_total_cost(nb_years, nb_years + 1)  # equiv to np.arange
            if total:
                mf = np.sum(np.atleast_2d(total_cost), axis=0)
            else:
                mf = total_cost
            nb_failures = np.diff(mf)
            return timeline[1:], nb_replacements, nb_failures
        return timeline[1:], nb_replacements

    def optimize(
        self,
    ) -> AgeReplacementPolicy:
        """
        Computes the optimal age(s) of replacement and updates the internal ``ar`` value(s).

        Returns
        -------
        Self
             Same instance with optimized ``ar``.
        """
        x0 = np.minimum(self.cp / (self.cf - self.cp), 1)  # ()
        discounting = ExponentialDiscounting(self.stochastic_process.discounting_rate)

        sf = self.stochastic_process.lifetime_model.unfrozen_model.baseline.sf
        pdf = self.stochastic_process.lifetime_model.unfrozen_model.baseline.pdf
        hf = self.stochastic_process.lifetime_model.unfrozen_model.baseline.hf
        x0 = np.broadcast_to(x0, hf(x0).shape).copy()  # () or (m, 1)

        def eq(a):  # () or (m, 1)
            f = legendre_quadrature(
                lambda x: discounting.factor(x) * sf(x),
                0,
                a,
            )
            g = legendre_quadrature(
                lambda x: discounting.factor(x) * pdf(x),
                0,
                a,
            )
            return discounting.factor(a) * ((self.cf - self.cp) * (hf(a) * f - g) - self.cp) / f**2

        self.ar = np.squeeze(newton(eq, x0))  # () or (m,) setter is called

        return self


class NonHomogeneousPoissonAgeReplacementPolicy:
    # noinspection PyUnresolvedReferences
    r"""Age replacement policy for non-Homogeneous Poisson process.

    Parameters
    ----------
    process : non-Homogeneous Poisson process
        The underlying process. If the process expects covars, it must be frozen before.
    cr : float or 1darray
        The cost of repair.
    cp : float or 1darray
        The cost of failure.
    discounting_rate : float, default is 0.
        The discounting rate value used in the exponential discounting function
    ar : float or 1darray, optional
        Ages of preventive replacements, by default None. If not given, one must call ``optimize`` to set ``ar`` values
        and access to the rest of the object interface.

    Attributes
    ----------
    ar
    cr
    cp
    discounting_rate
    """

    def __init__(
        self,
        process: FrozenNonHomogeneousPoissonProcess | NonHomogeneousPoissonProcess[()],
        cr: float | NDArray[np.float64],
        cp: float | NDArray[np.float64],
        discounting_rate: float = 0.0,
        ar: Optional[float | NDArray[np.float64]] = None,
    ) -> None:

        self.ar = ar if ar is not None else np.nan
        self.cost = cost(cr=cr, cp=cp)
        self.discounting = ExponentialDiscounting(discounting_rate)
        self.process = process

    @property
    def discounting_rate(self):
        """
        The discounting rate

        Returns
        -------
        float
        """
        return self.discounting.rate

    @property
    def cr(self) -> float | NDArray[np.float64]:
        """
        Cost of repair

        Returns
        -------
        ndarray
        """
        return self.cost["cr"]

    @cr.setter
    def cr(self, value: float | NDArray[np.float64]) -> None:
        self.cost["cr"] = value

    @property
    def cp(self) -> float | NDArray[np.float64]:
        """
        Cost of preventive replacements

        Returns
        -------
        ndarray
        """
        return self.cost["cp"]

    @cp.setter
    def cp(self, value: float | NDArray[np.float64]) -> None:
        self.cost["cp"] = value

    @property
    def ar(self):
        """
        Ages of the preventive replacements

        Returns
        -------
        ndarray
        """
        return np.squeeze(self._ar)

    @ar.setter
    def ar(self, value):
        ar = np.asarray(value, dtype=np.float64)
        shape = () if ar.ndim == 0 else (ar.size, 1)
        self._ar = ar.reshape(shape)

    def asymptotic_expected_equivalent_annual_cost(self) -> NDArray[np.float64]:
        r"""
        The asymtotic expected equivalent annual cost

        .. math::

            \lim_{t\to\infty} \text{EEAC}(t)

        where :math:`\text{EEAC}(t)` is the expected equivalent annual cost at :math:`t`.

        Returns
        -------
        ndarray
            The asymptotic expected equivalent annual cost
        """
        if np.any(np.isnan(self.ar)):
            raise ValueError
        if self.discounting_rate == 0.0:
            asymptotic_eeac = (
                self.cp + self.cr * legendre_quadrature(lambda t: self.process.intensity(t), 0, self._ar)
            ) / self._ar
        else:
            asymptotic_eeac = (
                self.discounting_rate
                * (
                    self.cp * self.discounting.factor(self._ar)
                    + self.cr
                    * legendre_quadrature(lambda t: self.discounting.factor(t) * self.process.intensity(t), 0, self._ar)
                )
                / (1 - self.discounting.factor(self._ar))
            )
        return np.squeeze(asymptotic_eeac)  # () or (m,)

    def optimize(
        self,
    ) -> NDArray[np.float64]:
        """
        Computes the optimal age(s) of replacement and updates the internal ``ar`` value(s).

        Returns
        -------
        Self
             Same instance with optimized ``ar``.
        """
        if is_frozen(self.process):
            x0 = self.process.unfreeze().lifetime_model.mean(*self.process.args)
        else: # lifetime_model : LifetimeDistribution
            x0 = self.process.lifetime_model.mean()
        x0 = np.broadcast_to(
            x0, np.broadcast_shapes(self.process.intensity(x0).shape, self.cp.shape, self.cr.shape)
        ).copy()

        def eq(ar):
            ar = np.atleast_2d(ar)
            if self.discounting.rate != 0:
                return (
                    (1 - self.discounting.factor(ar)) / self.discounting_rate * self.process.intensity(ar)
                    - legendre_quadrature(
                        lambda t: self.discounting.factor(t) * self.process.intensity(t),
                        np.array(0.0),
                        ar,
                    )
                    - self.cp / self.cr
                )
            return ar * self.process.intensity(ar) - self.process.cumulative_intensity(ar) - self.cp / self.cr

        self.ar = np.squeeze(newton(eq, x0))
        return self


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
