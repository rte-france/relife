# pyright: basic
from __future__ import annotations

import functools
from abc import ABC
from typing import Any, Literal, overload

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import newton

from relife.economic import AgeReplacementReward, ExponentialDiscounting
from relife.lifetime_model import (
    AgeReplacementModel,
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
from relife.utils.observation_bias import with_reshape_a0_ar
from relife.utils.quadrature import legendre_quadrature

from ._base import OneCycleExpectedCosts, ReplacementPolicy

__all__ = [
    "OneCycleAgeReplacementPolicy",
    "AgeReplacementModel",
    "age_replacement_policy",
]


def check_impossible_replacements(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Get ar and a0
        import inspect

        sig = inspect.signature(func)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        params = bound.arguments

        ar = params.get("ar")
        a0 = params.get("a0", None)

        # check ar is greater than a0 if a0 is provided
        if a0 is not None:
            if not (np.atleast_1d(ar) >= np.atleast_1d(a0)).all():
                raise ValueError(
                    "Some assets are using an optimal age of replacement inferior to their current age. Please consider changing the age of replacement."
                )

        return func(*args, **kwargs)

    return wrapper


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
    baseline_model: AnyParametricLifetimeModel[()]
    | FrozenNonHomogeneousPoissonProcess[*tuple[Any, ...]],
    cost_structure: dict[str, AnyFloat],
    one_cycle: bool = False,
    **kwargs: Any,
) -> (
    OneCycleAgeReplacementPolicy
    | AgeReplacementPolicy
    | NonHomogeneousPoissonAgeReplacementPolicy
): ...
def age_replacement_policy(
    baseline_model: AnyParametricLifetimeModel[()]
    | FrozenNonHomogeneousPoissonProcess[*tuple[Any, ...]],
    cost_structure: dict[str, AnyFloat],
    one_cycle: bool = False,
    **kwargs: Any,
) -> (
    OneCycleAgeReplacementPolicy
    | AgeReplacementPolicy
    | NonHomogeneousPoissonAgeReplacementPolicy
):
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
        return NonHomogeneousPoissonAgeReplacementPolicy(
            baseline_model, cr, cp, **kwargs
        )
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
    discounting_rate: float

    def __init__(
        self,
        lifetime_model: AnyParametricLifetimeModel[()],
        cf: AnyFloat,
        cp: AnyFloat,
        discounting_rate: float = 0.0,
    ):
        super().__init__(
            lifetime_model,
            {"cf": reshape_1d_arg(cf), "cp": reshape_1d_arg(cp)},
            discounting_rate=discounting_rate,
        )

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
    ):
        super().__init__(lifetime_model, cf, cp, discounting_rate=discounting_rate)
        self.period_before_discounting = period_before_discounting

    def _expected_costs(self, ar: NumpyFloat) -> OneCycleExpectedCosts:
        return OneCycleExpectedCosts(
            self.baseline_model,
            AgeReplacementReward(self.cf, self.cp, ar),
            discounting_rate=self.discounting_rate,
            period_before_discounting=self.period_before_discounting,
        )

    @check_impossible_replacements
    @with_reshape_a0_ar
    def expected_net_present_value(
        self,
        ar: NumpyFloat,
        tf: float,
        nb_steps: int,
        total_sum: bool = False,
        a0: NumpyFloat | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return self._expected_costs(ar=ar).expected_net_present_value(
            tf, nb_steps, total_sum=total_sum, ar=ar, a0=a0
        )  # (nb_steps,), (nb_steps,) or (nb_steps,), (m, nb_steps)

    @overload
    def asymptotic_expected_net_present_value(
        self, ar: NumpyFloat, total_sum: Literal[False], a0: NumpyFloat | None = None
    ) -> NDArray[np.float64]: ...
    @overload
    def asymptotic_expected_net_present_value(
        self, ar: NumpyFloat, total_sum: Literal[True], a0: NumpyFloat | None = None
    ) -> np.float64: ...
    @overload
    def asymptotic_expected_net_present_value(
        self, ar: NumpyFloat, total_sum: bool = False, a0: NumpyFloat | None = None
    ) -> np.float64 | NDArray[np.float64]: ...

    @check_impossible_replacements
    def asymptotic_expected_net_present_value(
        self, ar: NumpyFloat, total_sum: bool = False, a0: NumpyFloat | None = None
    ) -> np.float64 | NDArray[np.float64]:
        return self._expected_costs(ar=ar).asymptotic_expected_net_present_value(
            total_sum, ar=ar, a0=a0
        )  # () or (m, nb_steps)

    @check_impossible_replacements
    @with_reshape_a0_ar
    def expected_equivalent_annual_cost(
        self,
        ar: NumpyFloat,
        tf: float,
        nb_steps: int,
        total_sum: bool = False,
        a0: NumpyFloat | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:

        timeline, eeac = self._expected_costs(ar=ar).expected_equivalent_annual_cost(
            tf, nb_steps, total_sum=total_sum, ar=ar, a0=a0
        )
        return (
            timeline,
            eeac,
        )  # (nb_steps,), (nb_steps,) or (nb_steps,), (m, nb_steps)

    @overload
    def asymptotic_expected_equivalent_annual_cost(
        self, ar: NumpyFloat, total_sum: Literal[False], a0: NumpyFloat | None = None
    ) -> NDArray[np.float64]: ...
    @overload
    def asymptotic_expected_equivalent_annual_cost(
        self, ar: NumpyFloat, total_sum: Literal[True], a0: NumpyFloat | None = None
    ) -> np.float64: ...
    @overload
    def asymptotic_expected_equivalent_annual_cost(
        self, ar: NumpyFloat, total_sum: bool = False, a0: NumpyFloat | None = None
    ) -> np.float64 | NDArray[np.float64]: ...

    @check_impossible_replacements
    @with_reshape_a0_ar
    def asymptotic_expected_equivalent_annual_cost(
        self, ar: NumpyFloat, total_sum: bool = False, a0: NumpyFloat | None = None
    ) -> np.float64 | NDArray[np.float64]:

        asymptotic_eeac = self._expected_costs(
            ar=ar
        ).asymptotic_expected_equivalent_annual_cost(total_sum, ar=ar, a0=a0)
        return asymptotic_eeac  # () or (m, nb_steps)

    def compute_optimal_ar(self) -> NumpyFloat:
        """
        Optimize the policy according the costs, the discounting rate and the underlying lifetime model.

        Returns
        -------
        Self
             Optimized policy.
        """

        discounting = ExponentialDiscounting(self.discounting_rate)

        x0 = np.minimum(
            self._cost_structure["cp"]
            / (self._cost_structure["cf"] - self._cost_structure["cp"]),
            1,
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
                    (self._cost_structure["cf"] - self._cost_structure["cp"])
                    * self.baseline_model.hf(a)
                    - self._cost_structure["cp"] / discounting.annuity_factor(a)
                )
            )

        # no idea on how to type eq
        return newton(eq, x0)  # () or (m, 1) # pyright:ignore


class AgeReplacementPolicy(BaseAgeReplacementPolicy):
    r"""Age replacement policy.

    Asset is replaced at a fixed age :math:`a_r` with cost :math:`c_p` or it is replaced
    upon failure with cost :math:`c_f`.

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

    Attributes
    ----------
    cf
    cp

    References
    ----------
    .. [1] Mazzuchi, T. A., Van Noortwijk, J. M., & Kallen, M. J. (2007).
        Maintenance optimization. Encyclopedia of Statistics in Quality and
        Reliability, 1000-1008.
    """

    def _stochastic_reward_process(self, ar: NumpyFloat) -> RenewalRewardProcess:
        return RenewalRewardProcess(
            self.baseline_model,
            AgeReplacementReward(self.cf, self.cp, ar),
            discounting_rate=self.discounting_rate,
        )

    @check_impossible_replacements
    @with_reshape_a0_ar
    def expected_net_present_value(
        self,
        ar: NumpyFloat,
        tf: float,
        nb_steps: int,
        total_sum: bool = False,
        a0: NumpyFloat | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        timeline, npv = self._stochastic_reward_process(ar=ar).expected_total_reward(
            tf, nb_steps, a0=a0, ar=ar
        )
        if total_sum and npv.ndim == 2:
            npv = np.sum(npv, axis=0)
        return timeline, npv

    @overload
    def asymptotic_expected_net_present_value(
        self, ar: NumpyFloat, total_sum: Literal[False], a0: NumpyFloat | None = None
    ) -> NDArray[np.float64]: ...
    @overload
    def asymptotic_expected_net_present_value(
        self, ar: NumpyFloat, total_sum: Literal[True], a0: NumpyFloat | None = None
    ) -> np.float64: ...
    @overload
    def asymptotic_expected_net_present_value(
        self, ar: NumpyFloat, total_sum: bool = False, a0: NumpyFloat | None = None
    ) -> np.float64 | NDArray[np.float64]: ...

    @check_impossible_replacements
    @with_reshape_a0_ar
    def asymptotic_expected_net_present_value(
        self, ar: NumpyFloat, total_sum: bool = False, a0: NumpyFloat | None = None
    ) -> np.float64 | NDArray[np.float64]:
        asymptotic_npv = self._stochastic_reward_process(
            ar=ar
        ).asymptotic_expected_total_reward(a0=a0, ar=ar)  # () or (m,)
        if total_sum:
            asymptotic_npv = np.sum(asymptotic_npv)
        return asymptotic_npv

    @check_impossible_replacements
    @with_reshape_a0_ar
    def expected_equivalent_annual_cost(
        self,
        ar: NumpyFloat,
        tf: float,
        nb_steps: int,
        total_sum: bool = False,
        a0: NumpyFloat | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:

        timeline, eeac = self._stochastic_reward_process(
            ar=ar
        ).expected_equivalent_annual_worth(tf, nb_steps, a0=a0, ar=ar)
        if total_sum and eeac.ndim == 2:
            eeac = np.sum(eeac, axis=0)
        return timeline, eeac  # (nb_steps,), (nb_steps,) or (nb_steps,), (m, nb_steps)

    @overload
    def asymptotic_expected_equivalent_annual_cost(
        self, ar: NumpyFloat, total_sum: Literal[False], a0: NumpyFloat | None = None
    ) -> NDArray[np.float64]: ...
    @overload
    def asymptotic_expected_equivalent_annual_cost(
        self, ar: NumpyFloat, total_sum: Literal[True], a0: NumpyFloat | None = None
    ) -> np.float64: ...
    @overload
    def asymptotic_expected_equivalent_annual_cost(
        self, ar: NumpyFloat, total_sum: bool = False, a0: NumpyFloat | None = None
    ) -> np.float64 | NDArray[np.float64]: ...

    @check_impossible_replacements
    @with_reshape_a0_ar
    def asymptotic_expected_equivalent_annual_cost(
        self, ar: NumpyFloat, total_sum: bool = False, a0: NumpyFloat | None = None
    ) -> np.float64 | NDArray[np.float64]:

        asymptotic_eeac = self._stochastic_reward_process(
            ar=ar
        ).asymptotic_expected_equivalent_annual_worth(a0=a0, ar=ar)
        if total_sum:
            asymptotic_eeac = np.sum(asymptotic_eeac)
        return asymptotic_eeac  # () or (m,)

    @check_impossible_replacements
    @with_reshape_a0_ar
    def annual_number_of_replacements(
        self,
        ar: NumpyFloat,
        nb_years: int,
        upon_failure=False,
        total=True,
        a0: NumpyFloat | None = None,
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

        timeline, total_renewals = self._stochastic_reward_process(
            ar=ar
        ).renewal_function(nb_years, nb_years + 1, a0=a0, ar=ar)
        if total:
            mt = np.sum(np.atleast_2d(total_renewals), axis=0)
        else:
            mt = total_renewals
        nb_replacements = np.diff(mt)
        if upon_failure:
            _, failures_only = self._stochastic_reward_process(
                ar=ar
            ).expected_number_of_events(nb_years, nb_years + 1, a0=a0, ar=ar)
            if total:
                mf = np.sum(np.atleast_2d(failures_only), axis=0)
            else:
                mf = failures_only
            nb_failures = np.diff(mf)
            return timeline[1:], nb_replacements, nb_failures
        return timeline[1:], nb_replacements

    def compute_optimal_ar(self) -> NumpyFloat:
        """
        Optimize the policy according the costs, the discounting rate and the underlying lifetime model.

        Returns
        -------
        Self
             Optimized policy.
        """
        discounting = ExponentialDiscounting(self.discounting_rate)
        x0 = np.minimum(
            self._cost_structure["cp"]
            / (self._cost_structure["cf"] - self._cost_structure["cp"]),
            1,
        )

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
                    (self._cost_structure["cf"] - self._cost_structure["cp"])
                    * (self.baseline_model.hf(a) * f - g)
                    - self._cost_structure["cp"]
                )
                / f**2
            )

        return newton(eq, x0)  # pyright: ignore

    @check_impossible_replacements
    @with_reshape_a0_ar
    def generate_failure_data(
        self,
        ar: NumpyFloat,
        nb_samples: int,
        time_window: tuple[float, float],
        a0: NumpyFloat | None = None,
        seed=None,
    ):
        """Generate failure data

        This function will generate failure data that can be used to fit a lifetime model.

        Parameters
        ----------
        nb_samples : int
            The number of samples.
        time_window : tuple of two floats
            Time window in which data are sampled.
        seed : int, optional
            Random seed, by default None.

        Returns
        -------
        A dict of time, event, entry and args (covariates)

        """
        return self._stochastic_reward_process(ar=ar).generate_failure_data(
            nb_samples, time_window, ar=ar, a0=a0, seed=seed
        )

    @check_impossible_replacements
    @with_reshape_a0_ar
    def sample(
        self,
        ar: NumpyFloat,
        nb_samples: int,
        time_window: tuple[float, float],
        a0: NumpyFloat | None = None,
        seed=None,
    ):
        """Renewal data sampling.

        This function will sample data and encapsulate them in an object.

        Parameters
        ----------
        nb_samples : int
            The number of samples.
        time_window : tuple of two floats
            Time window in which data are sampled.
        seed : int, optional
            Random seed, by default None.

        """

        return self._stochastic_reward_process(ar=ar).sample(
            nb_samples, time_window, ar=ar, a0=a0, seed=seed
        )


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

    def __init__(self, nhpp, cr, cp, discounting_rate=0.0):
        super().__init__(
            nhpp,
            cost_structure={"cr": reshape_1d_arg(cr), "cp": reshape_1d_arg(cp)},
            discounting_rate=discounting_rate,
        )

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

    @with_reshape_a0_ar
    def expected_net_present_value(
        self,
        tf,
        ar: NumpyFloat,
        nb_steps,
        total_sum=False,
        a0: NumpyFloat | None = None,
    ):
        raise NotImplementedError("implementation will come in a future release")

    @with_reshape_a0_ar
    def asymptotic_expected_net_present_value(
        self, ar: NumpyFloat, total_sum=False, a0: NumpyFloat | None = None
    ):
        raise NotImplementedError("implementation will come in a future release")

    @with_reshape_a0_ar
    def expected_equivalent_annual_cost(
        self,
        ar: NumpyFloat,
        tf,
        nb_steps,
        total_sum=False,
        a0: NumpyFloat | None = None,
    ):
        raise NotImplementedError("implementation will come in a future release")

    @with_reshape_a0_ar
    def asymptotic_expected_equivalent_annual_cost(
        self, ar: NumpyFloat, a0: NumpyFloat | None = None
    ):
        if a0 is not None:
            raise ValueError("NHPP policies with initial ages will be covered in a future release")

        discounting = ExponentialDiscounting(self.discounting_rate)

        if self.discounting_rate == 0.0:
            asymptotic_eeac = (
                self.cp
                + self.cr
                * legendre_quadrature(lambda t: self.baseline_model.intensity(t), 0, ar)
            ) / ar
        else:
            asymptotic_eeac = (
                self.discounting_rate
                * (
                    self.cp * discounting.factor(ar)
                    + self.cr
                    * legendre_quadrature(
                        lambda t: (
                            discounting.factor(t) * self.baseline_model.intensity(t)
                        ),
                        0,
                        ar,
                    )
                )
                / (1 - discounting.factor(ar))
            )
        return np.squeeze(asymptotic_eeac)  # () or (m,)

    def compute_optimal_ar(self) -> NumpyFloat:
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
                    (1 - discounting.factor(a))
                    / self.discounting_rate
                    * self.baseline_model.intensity(a)
                    - legendre_quadrature(
                        lambda t: (
                            discounting.factor(t) * self.baseline_model.intensity(t)
                        ),
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

        return newton(eq, x0)
