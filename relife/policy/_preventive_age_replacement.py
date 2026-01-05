# pyright: basic
from __future__ import annotations

import warnings
from abc import ABC
from typing import Any, Literal, Optional, Self, overload

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import newton

from relife.economic import AgeReplacementReward, ExponentialDiscounting
from relife.lifetime_model import (
    AgeReplacementModel,
    LeftTruncatedModel,
)
from relife.stochastic_process import RenewalRewardProcess
from relife.stochastic_process.non_homogeneous_poisson_process import (
    FrozenNonHomogeneousPoissonProcess,
)
from relife.typing import (
    AnyFloat,
    AnyParametricLifetimeModel,
    NumpyFloat,
)
from relife.utils import (
    flatten_if_possible,
    reshape_1d_arg,
)
from relife.utils.quadrature import legendre_quadrature

from ._base import OneCycleExpectedCosts, ReplacementPolicy

__all__ = ["OneCycleAgeReplacementPolicy", "AgeReplacementModel", "age_replacement_policy"]


@overload
def age_replacement_policy(
    baseline_model: AnyParametricLifetimeModel[()],
    cost_structure: dict[str, AnyFloat],
    one_cycle: Literal[True],
    **kwargs: Any,
) -> OneCycleAgeReplacementPolicy: ...
@overload
def age_replacement_policy(
    baseline_model: AnyParametricLifetimeModel[()],
    cost_structure: dict[str, AnyFloat],
    one_cycle: Literal[False],
    **kwargs: Any,
) -> AgeReplacementPolicy: ...
@overload
def age_replacement_policy(
    baseline_model: FrozenNonHomogeneousPoissonProcess[*tuple[Any, ...]],
    cost_structure: dict[str, AnyFloat],
    one_cycle: bool = False,
    **kwargs: Any,
) -> NonHomogeneousPoissonAgeReplacementPolicy: ...
@overload
def age_replacement_policy(
    baseline_model: AnyParametricLifetimeModel[()] | FrozenNonHomogeneousPoissonProcess[*tuple[Any, ...]],
    cost_structure: dict[str, AnyFloat],
    one_cycle: bool = False,
    **kwargs: Any,
) -> OneCycleAgeReplacementPolicy | AgeReplacementPolicy | NonHomogeneousPoissonAgeReplacementPolicy: ...
def age_replacement_policy(
    baseline_model: AnyParametricLifetimeModel[()] | FrozenNonHomogeneousPoissonProcess[*tuple[Any, ...]],
    cost_structure: dict[str, AnyFloat],
    one_cycle: bool = False,
    **kwargs: Any,
) -> OneCycleAgeReplacementPolicy | AgeReplacementPolicy | NonHomogeneousPoissonAgeReplacementPolicy:
    """
    Creates a preventive age replacement policy.

    Parameters
    ----------
    baseline_model : parametric model
        Parametric model required by the policy.
    cost_structure : dict of costs
        Dictionnary containing the cost values (float or 1d-array) and their corresponding names (either cf, cp or cr).
    one_cycle : bool, default False
        If True, returns the one cycle variation of the policy.
    **kwargs
        Extra arguments required by the policy (a0, ar, discounting_rate, etc.)

    Returns
    -------
    Policy
        Policy corresponding to the ``baseline_model`` and the ``cost_structure``.

    Raises
    ------
    ValueError
        If ``baseline_model`` or ``cost_structure`` does not have a corresponding policy.
    """
    if isinstance(baseline_model, FrozenNonHomogeneousPoissonProcess):
        try:
            cr = cost_structure["cr"]
            cp = cost_structure["cp"]
        except KeyError:
            raise ValueError("costs must contain 'cr' and 'cp'")
        return NonHomogeneousPoissonAgeReplacementPolicy(baseline_model, cr, cp, **kwargs)
    try:
        cf = cost_structure["cf"]
        cp = cost_structure["cp"]
    except KeyError:
        raise ValueError("costs must contain 'cf' and 'cp'")
    if one_cycle:
        return OneCycleAgeReplacementPolicy(baseline_model, cf, cp, **kwargs)
    return AgeReplacementPolicy(baseline_model, cf, cp, **kwargs)


class BaseAgeReplacementPolicy(ReplacementPolicy[AnyParametricLifetimeModel[()]], ABC):
    _cost_structure: dict[str, NumpyFloat]
    _ar: Optional[NumpyFloat]
    _a0: Optional[NumpyFloat]
    discounting_rate: float

    def __init__(
        self,
        lifetime_model: AnyParametricLifetimeModel[()],
        cf: AnyFloat,
        cp: AnyFloat,
        discounting_rate: float = 0.0,
        a0: Optional[AnyFloat] = None,
        ar: Optional[AnyFloat] = None,
    ):
        super().__init__(
            lifetime_model,
            {"cf": reshape_1d_arg(cf), "cp": reshape_1d_arg(cp)},
            discounting_rate=discounting_rate,
        )
        self._a0 = reshape_1d_arg(a0) if a0 is not None else a0
        self._ar = reshape_1d_arg(ar) if ar is not None else ar

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

    @property
    def cp(self) -> NumpyFloat:
        """Costs of preventive replacements.

        Returns
        -------
        np.ndarray
        """
        # _cp is (m, 1) but exposed cp is (m,)
        return flatten_if_possible(self._cost_structure["cp"])

    @cp.setter
    def cp(self, value: AnyFloat) -> None:
        self._cost_structure["cp"] = reshape_1d_arg(value)

    @property
    def ar(self) -> Optional[NumpyFloat]:
        """Preventive ages of replacement.

        Returns
        -------
        np.ndarray
        """
        # _ar is (m, 1) but exposed ar is (m,)
        if self._ar is None:
            return self._ar
        return flatten_if_possible(self._ar)

    @ar.setter
    def ar(self, value: Optional[AnyFloat]) -> None:
        if value is not None:
            self._ar = reshape_1d_arg(value)
        else:
            self._ar = None

    @property
    def tr1(self) -> Optional[NumpyFloat]:
        """Times before the first replacement.

        Returns
        -------
        np.ndarray
        """
        if self._a0 is not None and self._ar is not None:
            tr = np.maximum(self._ar - self._a0, 0)
            return flatten_if_possible(tr)
        return self.ar


class OneCycleAgeReplacementPolicy(BaseAgeReplacementPolicy):
    r"""One-cyle age replacement policy.

    Asset is replaced at a fixed age :math:`a_r` with cost :math:`c_p` or it is replaced
    upon failure with cost :math:`c_f`.

    .. note::

        ``OneCycleAgeReplacementPolicy`` differs from ``AgeReplacementPolicy``
        because only one cycle of replacement is considered.

    The object's methods require the ``ar`` attribute to be set either at the instanciation
    or by calling the ``optimize`` method. Otherwise, an error will be raised.

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
    cf
    cp
    ar

    References
    ----------
    .. [1] Coolen-Schrijner, P., & Coolen, F. P. A. (2006). On optimality
        criteria for age replacement. Proceedings of the Institution of
        Mechanical Engineers, Part O: Journal of Risk and Reliability,
        220(1), 21-29
    """

    period_before_discounting: float

    def __init__(
        self,
        lifetime_model: AnyParametricLifetimeModel[()],
        cf: AnyFloat,
        cp: AnyFloat,
        discounting_rate: float = 0.0,
        period_before_discounting: float = 1.0,
        a0: Optional[AnyFloat] = None,
        ar: Optional[AnyFloat] = None,
    ):
        super().__init__(lifetime_model, cf, cp, discounting_rate=discounting_rate, a0=a0, ar=ar)
        self.period_before_discounting = period_before_discounting

    @property
    def _expected_costs(self) -> OneCycleExpectedCosts:
        if self.ar is None or self.tr1 is None:
            raise ValueError("ar must be set or optimized")
        else:
            if self.a0 is None:
                return OneCycleExpectedCosts(
                    AgeReplacementModel(self.baseline_model).freeze(self.tr1),
                    AgeReplacementReward(self.cf, self.cp, self.tr1),
                    discounting_rate=self.discounting_rate,
                    period_before_discounting=self.period_before_discounting,
                )
            return OneCycleExpectedCosts(
                AgeReplacementModel(LeftTruncatedModel(self.baseline_model)).freeze(self.tr1, self.a0),
                AgeReplacementReward(self.cf, self.cp, self.tr1),
                discounting_rate=self.discounting_rate,
                period_before_discounting=self.period_before_discounting,
            )

    def expected_net_present_value(
        self, tf: float, nb_steps: int, total_sum: bool = False
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:

        if self.ar is None:
            raise ValueError
        return self._expected_costs.expected_net_present_value(
            tf, nb_steps, total_sum=total_sum
        )  # (nb_steps,), (nb_steps,) or (nb_steps,), (m, nb_steps)

    @overload
    def asymptotic_expected_net_present_value(self, total_sum: Literal[False]) -> NDArray[np.float64]: ...
    @overload
    def asymptotic_expected_net_present_value(self, total_sum: Literal[True]) -> np.float64: ...
    @overload
    def asymptotic_expected_net_present_value(self, total_sum: bool = False) -> np.float64 | NDArray[np.float64]: ...
    def asymptotic_expected_net_present_value(self, total_sum: bool = False) -> np.float64 | NDArray[np.float64]:
        if self.ar is None:
            raise ValueError
        return self._expected_costs.asymptotic_expected_net_present_value()  # () or (m, nb_steps)

    def expected_equivalent_annual_cost(
        self, tf: float, nb_steps: int, total_sum: bool = False
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:

        if self.ar is None or self.tr1 is None:
            raise ValueError
        else:
            # because b = np.minimum(ar, b) in ls_integrate, b can be lower than a depending on period before discounting
            if np.any(self.period_before_discounting >= self.ar):
                raise ValueError("The period before discounting must be lower than ar values")

            if np.any(self.tr1 == 0):
                warnings.warn(
                    "Some assets has already been replaced for the first cycle (where tr is 0). For these assets, consider adjusting ar values to be greater than a0"
                )

            ar = self.ar.copy()
            #  change ar temporarly to enable computation of eeac (if not, AgeReplacementModel.ls_integrate bounds will be problematic)
            self.ar = np.where(self.tr1 == 0, np.inf, self.ar)
            timeline, eeac = self._expected_costs.expected_equivalent_annual_cost(tf, nb_steps)
            if eeac.ndim == 2:  # more than one asset
                eeac[np.where(self.ar == np.inf)[0], :] = np.nan
            if eeac.ndim == 1 and self.ar == np.inf:
                eeac.fill(np.nan)
            self.ar = ar
            return timeline, eeac  # (nb_steps,), (nb_steps,) or (nb_steps,), (m, nb_steps)

    @overload
    def asymptotic_expected_equivalent_annual_cost(self, total_sum: Literal[False]) -> NDArray[np.float64]: ...
    @overload
    def asymptotic_expected_equivalent_annual_cost(self, total_sum: Literal[True]) -> np.float64: ...
    @overload
    def asymptotic_expected_equivalent_annual_cost(
        self, total_sum: bool = False
    ) -> np.float64 | NDArray[np.float64]: ...
    def asymptotic_expected_equivalent_annual_cost(self, total_sum: bool = False) -> np.float64 | NDArray[np.float64]:

        if self.ar is None:
            raise ValueError
        # because b = np.minimum(ar, b) in ls_integrate, b can be lower than a depending on period before discounting
        if np.any(self.period_before_discounting >= self.ar):
            raise ValueError("The period before discounting must be lower than ar values")

        if np.any(self.tr1 == 0):
            warnings.warn(
                "Some assets has already been replaced for the first cycle (where tr is 0). For these assets, consider adjusting ar values to be greater than a0"
            )

        ar = self.ar.copy()
        #  change ar temporarly to enable computation of eeac (if not, AgeReplacementModel.ls_integrate bounds are not in right order)
        self.ar = np.where(self.tr1 == 0, np.inf, self.ar)
        asymptotic_eeac = self._expected_costs.asymptotic_expected_equivalent_annual_cost()

        if asymptotic_eeac.ndim == 1:  # more than one asset
            asymptotic_eeac[np.where(self.ar == np.inf)[0]] = np.nan
        if asymptotic_eeac.ndim == 0 and self.ar == np.inf:
            asymptotic_eeac = np.array(np.nan)
        self.ar = ar
        return asymptotic_eeac  # () or (m, nb_steps)

    def optimize(self) -> Self:
        """
        Optimize the policy according the costs, the discounting rate and the underlying lifetime model.

        Returns
        -------
        Self
             Optimized policy.
        """

        discounting = ExponentialDiscounting(self.discounting_rate)

        x0 = np.minimum(
            self._cost_structure["cp"] / (self._cost_structure["cf"] - self._cost_structure["cp"]), 1
        )  # () or (m, 1)

        # if costs are floats, x0 is float. BUT, baseline_model can have many assets
        # x0 must have the same shape than eq(x0) (see scipy.newton doc)
        _sf_x0 = self.baseline_model.sf(x0)  # gives us the shape, thus nb_assets
        if _sf_x0.ndim == 2 and x0.ndim == 0:
            x0 = np.tile(x0, (_sf_x0.shape[0], 1))

        def eq(a: NDArray[np.float64]) -> NDArray[np.float64]:
            return (
                discounting.factor(a)
                / discounting.annuity_factor(a)
                * (
                    (self._cost_structure["cf"] - self._cost_structure["cp"]) * self.baseline_model.hf(a)
                    - self._cost_structure["cp"] / discounting.annuity_factor(a)
                )
            )

        # no idea on how to type eq
        self.ar = newton(eq, x0)  # () or (m, 1) # pyright:ignore
        return self


class AgeReplacementPolicy(BaseAgeReplacementPolicy):
    r"""Age replacement policy.

    Asset is replaced at a fixed age :math:`a_r` with cost :math:`c_p` or it is replaced
    upon failure with cost :math:`c_f`.

    The object's methods require the ``ar`` attribute to be set either at the instanciation
    or by calling the ``optimize`` method. Otherwise, an error will be raised.

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
    cf
    cp
    ar

    References
    ----------
    .. [1] Mazzuchi, T. A., Van Noortwijk, J. M., & Kallen, M. J. (2007).
        Maintenance optimization. Encyclopedia of Statistics in Quality and
        Reliability, 1000-1008.
    """

    @property
    def _stochastic_process(self) -> RenewalRewardProcess:
        if self.ar is None or self.tr1 is None:
            raise ValueError("ar must be set or optimized")
        if self.a0 is None:
            return RenewalRewardProcess(
                AgeReplacementModel(self.baseline_model).freeze(self.ar),
                AgeReplacementReward(self.cf, self.cp, self.ar),
                discounting_rate=self.discounting_rate,
            )
        return RenewalRewardProcess(
            AgeReplacementModel(self.baseline_model).freeze(self.ar),
            AgeReplacementReward(self.cf, self.cp, self.ar),
            discounting_rate=self.discounting_rate,
            first_lifetime_model=AgeReplacementModel(LeftTruncatedModel(self.baseline_model)).freeze(self.tr1, self.a0),
            first_reward=AgeReplacementReward(self.cf, self.cp, self.tr1),
        )

    def expected_net_present_value(
        self, tf: float, nb_steps: int, total_sum: bool = False
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        if self.ar is None:
            raise ValueError("ar must be set or optimized")
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
        asymptotic_npv = self._stochastic_process.asymptotic_expected_total_reward()  # () or (m,)
        if total_sum:
            asymptotic_npv = np.sum(asymptotic_npv)
        return asymptotic_npv

    def expected_equivalent_annual_cost(
        self, tf: float, nb_steps: int, total_sum: bool = False
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:

        if self.ar is None or self.tr1 is None:
            raise ValueError

        timeline, eeac = self._stochastic_process.expected_equivalent_annual_worth(tf, nb_steps)
        if np.any(self.tr1 == 0):
            warnings.warn(
                "Some assets has already been replaced for the first cycle (where tr1 is 0). For these assets, consider adjusting ar values to be greater than a0"
            )
        if eeac.ndim == 2:
            eeac[np.where(np.atleast_1d(self.tr1) == 0)] = np.nan
        if eeac.ndim == 1 and self.ar == np.inf:
            eeac.fill(np.nan)
        if total_sum and eeac.ndim == 2:
            eeac = np.sum(eeac, axis=0)
        return timeline, eeac  # (nb_steps,), (nb_steps,) or (nb_steps,), (m, nb_steps)

    @overload
    def asymptotic_expected_equivalent_annual_cost(self, total_sum: Literal[False]) -> NDArray[np.float64]: ...
    @overload
    def asymptotic_expected_equivalent_annual_cost(self, total_sum: Literal[True]) -> np.float64: ...
    @overload
    def asymptotic_expected_equivalent_annual_cost(
        self, total_sum: bool = False
    ) -> np.float64 | NDArray[np.float64]: ...
    def asymptotic_expected_equivalent_annual_cost(self, total_sum: bool = False) -> np.float64 | NDArray[np.float64]:

        if self.ar is None or self.tr1 is None:
            raise ValueError

        asymptotic_eeac = self._stochastic_process.asymptotic_expected_equivalent_annual_worth()
        if np.any(self.tr1 == 0):
            warnings.warn(
                "Some assets has already been replaced for the first cycle (where tr1 is 0). For these assets, consider adjusting ar values to be greater than a0"
            )
        if asymptotic_eeac.ndim == 1:
            asymptotic_eeac[np.where(self.tr1 == 0)] = np.nan
        if asymptotic_eeac.ndim == 0 and self.tr1 == 0:
            asymptotic_eeac = np.array(np.nan)
        if total_sum:
            asymptotic_eeac = np.sum(asymptotic_eeac)
        return asymptotic_eeac  # () or (m,)

    def annual_number_of_replacements(self, nb_years, upon_failure=False, total=True):
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
        copied_policy = self.__class__(
            self.baseline_model,
            1.0,
            1.0,
            ar=self.ar,
            a0=self.a0,
            discounting_rate=0.0,
        )
        timeline, total_cost = copied_policy.expected_net_present_value(nb_years, nb_years + 1)  # equiv to np.arange
        if total:
            mt = np.sum(np.atleast_2d(total_cost), axis=0)
        else:
            mt = total_cost
        nb_replacements = np.diff(mt)
        if upon_failure:
            copied_policy = self.__class__(
                self.baseline_model,
                1.0,
                0.0,
                ar=self.ar,
                a0=self.a0,
                discounting_rate=0.0,
            )
            _, total_cost = copied_policy.expected_net_present_value(nb_years, nb_years + 1)  # equiv to np.arange
            if total:
                mf = np.sum(np.atleast_2d(total_cost), axis=0)
            else:
                mf = total_cost
            nb_failures = np.diff(mf)
            return timeline[1:], nb_replacements, nb_failures
        return timeline[1:], nb_replacements

    def optimize(self) -> Self:
        """
        Optimize the policy according the costs, the discounting rate and the underlying lifetime model.

        Returns
        -------
        Self
             Optimized policy.
        """
        discounting = ExponentialDiscounting(self.discounting_rate)
        x0 = np.minimum(self._cost_structure["cp"] / (self._cost_structure["cf"] - self._cost_structure["cp"]), 1)

        # if costs are floats, x0 is float. BUT, baseline_model can have many assets
        # x0 must have the same shape than eq(x0) (see scipy.newton doc)
        _sf_x0 = self.baseline_model.sf(x0)  # gives us the shape, thus nb_assets
        if _sf_x0.ndim == 2 and x0.ndim == 0:
            x0 = np.tile(x0, (_sf_x0.shape[0], 1))

        def eq(a: NDArray[np.float64]) -> NDArray[np.float64]:  # () or (m, 1)
            f = legendre_quadrature(
                lambda x: discounting.factor(x) * self.baseline_model.sf(x),
                0,
                a,
            )
            g = legendre_quadrature(
                lambda x: discounting.factor(x) * self.baseline_model.pdf(x),
                0,
                a,
            )
            return (
                discounting.factor(a)
                * (
                    (self._cost_structure["cf"] - self._cost_structure["cp"]) * (self.baseline_model.hf(a) * f - g)
                    - self._cost_structure["cp"]
                )
                / f**2
            )

        # no idea on how to type eq properly
        self.ar = newton(eq, x0)  # () or (m, 1)  # pyright: ignore
        return self

    def generate_failure_data(self, size, tf, t0=0.0, seed=None):
        """Generate failure data

        This function will generate failure data that can be used to fit a lifetime model.

        Parameters
        ----------
        size : int
            The size of the desired sample.
        tf : float
            Time at the end of the observation.
        t0 : float, default 0
            Time at the beginning of the observation.
        seed : int, optional
            Random seed, by default None.

        Returns
        -------
        A dict of time, event, entry and args (covariates)

        """
        return self._stochastic_process.generate_failure_data(tf, t0, size, seed)

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


class NonHomogeneousPoissonAgeReplacementPolicy(ReplacementPolicy):
    r"""Age replacement policy for non-Homogeneous Poisson process.

    Parameters
    ----------
    nhpp : non-homogeneous Poisson process
        The underlying non homogeneous poisson process. If the process expects covars, it must be frozen before.
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
    cp
    cr
    ar
    """

    def __init__(self, nhpp, cr, cp, discounting_rate=0.0, ar=None):
        super().__init__(
            nhpp, cost_structure={"cr": reshape_1d_arg(cr), "cp": reshape_1d_arg(cp)}, discounting_rate=discounting_rate
        )
        self.ar = ar

    @property
    def cp(self):
        """Costs of preventive replacement.

        Returns
        -------
        np.ndarray
        """
        return flatten_if_possible(self._cost_structure["cp"])

    @cp.setter
    def cp(self, value):
        self._cost_structure["cp"] = reshape_1d_arg(value)

    @property
    def cr(self):
        """Cost of minimal repair.

        Returns
        -------
        np.ndarray
        """
        return flatten_if_possible(self._cost_structure["cr"])

    @cr.setter
    def cr(self, value):
        self._cost_structure["cr"] = reshape_1d_arg(value)

    @property
    def ar(self):
        """Preventive ages of replacement.

        Returns
        -------
        np.ndarray
        """
        if self._ar is None:
            return self._ar
        return flatten_if_possible(self._ar)

    @ar.setter
    def ar(self, value):
        if value is not None:
            value = reshape_1d_arg(value)
            self._ar = value
        else:
            self._ar = None

    def expected_net_present_value(self, tf, nb_steps, total_sum=False):
        raise NotImplemented("implementation will come in a future release")

    def asymptotic_expected_net_present_value(self, total_sum=False):
        raise NotImplemented("implementation will come in a future release")

    def expected_equivalent_annual_cost(self, tf, nb_steps, total_sum=False):
        raise NotImplemented("implementation will come in a future release")

    def asymptotic_expected_equivalent_annual_cost(self):
        discounting = ExponentialDiscounting(self.discounting_rate)

        if self.ar is None:
            raise ValueError

        if self.discounting_rate == 0.0:
            asymptotic_eeac = (
                self.cp + self.cr * legendre_quadrature(lambda t: self.baseline_model.intensity(t), 0, self._ar)
            ) / self._ar
        else:
            asymptotic_eeac = (
                self.discounting_rate
                * (
                    self.cp * discounting.factor(self._ar)
                    + self.cr
                    * legendre_quadrature(
                        lambda t: discounting.factor(t) * self.baseline_model.intensity(t), 0, self._ar
                    )
                )
                / (1 - discounting.factor(self._ar))
            )
        return np.squeeze(asymptotic_eeac)  # () or (m,)

    def optimize(self):
        """
        Optimize the policy according the costs, the discounting rate and the underlying non-homogeneous Poisson process.

        Returns
        -------
        Self
             Optimized policy.
        """

        discounting = ExponentialDiscounting(self.discounting_rate)
        x0 = np.atleast_2d(self.baseline_model.lifetime_model.mean())

        def eq(a):
            if discounting.rate != 0:
                return (
                    (1 - discounting.factor(a)) / self.discounting_rate * self.baseline_model.intensity(a)
                    - legendre_quadrature(
                        lambda t: discounting.factor(t) * self.baseline_model.intensity(t),
                        np.array(0.0),
                        a,
                    )
                    - self._cost_structure["cp"] / self._cost_structure["cr"]
                )
            return (
                a * self.baseline_model.intensity(a)
                - self.baseline_model.cumulative_intensity(a)
                - self._cost_structure["cp"] / self._cost_structure["cr"]
            )

        self.ar = newton(eq, x0)
        return self
