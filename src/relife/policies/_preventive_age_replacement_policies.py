from __future__ import annotations

import functools
import warnings
from abc import ABC, abstractmethod
from typing import Any, Literal, TypeAlias, overload

import numpy as np
from optype.numpy import Array1D, Array2D, ArrayND
from scipy.optimize import newton
from typing_extensions import override

from relife.lifetime_models import (
    AgeReplacementModel,
)
from relife.lifetime_models._base import (
    FittableParametricLifetimeModel,
    ParametricLifetimeModel,
)
from relife.quadratures import legendre_quadrature
from relife.rewards import AgeReplacementReward, ExponentialDiscounting
from relife.stochastic_processes._non_homogeneous_poisson_process import (
    FrozenNonHomogeneousPoissonProcess,
)
from relife.stochastic_processes._renewal_processes import (
    RenewalRewardProcess,
    reshape_a0_ar,
)
from relife.utils import (
    flatten_if_at_least_2d,
    to_column_2d_if_1d,
    to_numpy_float64,
)

from ._base import BaseReplacementPolicy, OneCycleExpectedCosts

__all__ = [
    "OneCycleAgeReplacementPolicy",
    "AgeReplacementModel",
    "age_replacement_policy",
]

ST: TypeAlias = int | float
NumpyST: TypeAlias = np.floating | np.uint


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
            if not np.all(np.atleast_1d(ar) >= np.atleast_1d(a0)):
                warnings.warn(
                    """ Some assets are using an optimal age of replacement
                    inferior to their current age. Please consider changing the
                    age of replacement.
                    """,
                    stacklevel=2,
                )

        return func(*args, **kwargs)

    return wrapper


@overload
def age_replacement_policy(
    baseline_model: ParametricLifetimeModel[()],
    cost_structure: dict[str, ST | NumpyST | Array1D[NumpyST]],
    one_cycle: Literal[True],
    **kwargs: Any,
) -> OneCycleAgeReplacementPolicy: ...
@overload
def age_replacement_policy(
    baseline_model: ParametricLifetimeModel[()],
    cost_structure: dict[str, ST | NumpyST | Array1D[NumpyST]],
    one_cycle: Literal[False],
    **kwargs: Any,
) -> AgeReplacementPolicy: ...
@overload
def age_replacement_policy(
    baseline_model: FrozenNonHomogeneousPoissonProcess[
        FittableParametricLifetimeModel[*tuple[ST | NumpyST | ArrayND[NumpyST], ...]]
    ],
    cost_structure: dict[str, ST | NumpyST | Array1D[NumpyST]],
    one_cycle: bool = False,
    **kwargs: Any,
) -> NonHomogeneousPoissonAgeReplacementPolicy: ...
def age_replacement_policy(
    baseline_model: ParametricLifetimeModel[()]
    | FrozenNonHomogeneousPoissonProcess[
        FittableParametricLifetimeModel[*tuple[ST | NumpyST | ArrayND[NumpyST], ...]]
    ],
    cost_structure: dict[str, ST | NumpyST | Array1D[NumpyST]],
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
    """  # noqa: E501
    if isinstance(baseline_model, FrozenNonHomogeneousPoissonProcess):
        try:
            cr = to_numpy_float64(cost_structure["cr"])
            cp = to_numpy_float64(cost_structure["cp"])
        except KeyError as err:
            raise ValueError("costs must contain 'cr' and 'cp'") from err
        return NonHomogeneousPoissonAgeReplacementPolicy(
            baseline_model, cr, cp, **kwargs
        )
    try:
        cf = to_numpy_float64(cost_structure["cf"])
        cp = to_numpy_float64(cost_structure["cp"])
    except KeyError as err:
        raise ValueError("costs must contain 'cf' and 'cp'") from err
    if one_cycle:
        return OneCycleAgeReplacementPolicy(baseline_model, cf, cp, **kwargs)
    return AgeReplacementPolicy(baseline_model, cf, cp, **kwargs)


class BaseAgeReplacementPolicy(BaseReplacementPolicy[ParametricLifetimeModel[()]], ABC):
    discounting_rate: float

    def __init__(
        self,
        lifetime_model: ParametricLifetimeModel[()],
        cf: ST | NumpyST | Array1D[NumpyST],
        cp: ST | NumpyST | Array1D[NumpyST],
        discounting_rate: float = 0.0,
    ):
        super().__init__(
            lifetime_model,
            {"cf": to_column_2d_if_1d(cf), "cp": to_column_2d_if_1d(cp)},
            discounting_rate=discounting_rate,
        )

    def get_cf(self) -> np.float64 | Array1D[np.float64]:
        """Cost of failure.

        Returns
        -------
        np.ndarray
        """
        # _cf is (m, 1) but exposed cf is (m,)
        return flatten_if_at_least_2d(self._cost_structure["cf"])

    def set_cf(self, value: ST | NumpyST | Array1D[NumpyST]) -> None:
        self._cost_structure["cf"] = to_column_2d_if_1d(value)

    def get_cp(self) -> np.float64 | Array1D[np.float64]:
        """Costs of preventive replacements.

        Returns
        -------
        np.ndarray
        """
        # _cp is (m, 1) but exposed cp is (m,)
        return flatten_if_at_least_2d(self._cost_structure["cp"])

    def set_cp(self, value: ST | NumpyST | Array1D[NumpyST]) -> None:
        self._cost_structure["cp"] = to_column_2d_if_1d(value)

    @abstractmethod
    def expected_net_present_value(
        self,
        tf: float,
        nb_steps: int,
        ar: ST | NumpyST | Array1D[NumpyST],
        a0: ST | NumpyST | Array1D[NumpyST] = 0.0,
    ) -> tuple[Array1D[np.float64], Array1D[np.float64] | Array2D[np.float64]]:
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

        Returns
        -------
        tuple of two ndarrays
            A tuple containing the timeline and the computed values.
        """

    @abstractmethod
    def expected_equivalent_annual_cost(
        self,
        tf: float,
        nb_steps: int,
        ar: ST | NumpyST | Array1D[NumpyST],
        a0: ST | NumpyST | Array1D[NumpyST] = 0.0,
    ) -> tuple[Array1D[np.float64], Array1D[np.float64] | Array2D[np.float64]]:
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

        Returns
        -------
        tuple of two ndarrays
            A tuple containing the timeline and the computed values.
        """

    @abstractmethod
    def asymptotic_expected_net_present_value(
        self,
        ar: ST | NumpyST | Array1D[NumpyST],
        a0: ST | NumpyST | Array1D[NumpyST] = 0.0,
    ) -> np.float64 | Array1D[np.float64]:
        r"""
        The asymtotic expected net present value.

        .. math::

            \lim_{t\to\infty} z(t)

        Parameters
        ----------
        total_sum : bool, default False
            If True, returns the total sum over the first axis of the result.
            If the policy data encodes several assets, this option allows to
            return the sum result on the fleet rather than calling
            ``np.sum`` afterwards.

        Returns
        -------
        ndarray
            The asymptotic expected values.
        """

    @abstractmethod
    def asymptotic_expected_equivalent_annual_cost(
        self,
        ar: ST | NumpyST | Array1D[NumpyST],
        a0: ST | NumpyST | Array1D[NumpyST] = 0.0,
    ) -> np.float64 | Array1D[np.float64]:
        r"""
        The asymtotic expected equivalent annual cost.

        .. math::

            \lim_{t\to\infty} q(t)

        Parameters
        ----------

        Returns
        -------
        ndarray
            The asymptotic expected values.
        """


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

    Attributes
    ----------
    cf
    cp

    References
    ----------
    .. [1] Coolen-Schrijner, P., & Coolen, F. P. A. (2006). On optimality
        criteria for age replacement. Proceedings of the Institution of
        Mechanical Engineers, Part O: Journal of Risk and Reliability,
        220(1), 21-29
    """  # noqa: E501

    period_before_discounting: float

    def __init__(
        self,
        lifetime_model: ParametricLifetimeModel[()],
        cf: ST | NumpyST | Array1D[NumpyST],
        cp: ST | NumpyST | Array1D[NumpyST],
        discounting_rate: float = 0.0,
        period_before_discounting: float = 1.0,
    ):
        super().__init__(lifetime_model, cf, cp, discounting_rate=discounting_rate)
        self.period_before_discounting = period_before_discounting

    def _expected_costs(
        self, ar: ST | NumpyST | Array1D[NumpyST]
    ) -> OneCycleExpectedCosts:
        return OneCycleExpectedCosts(
            self.baseline_model,
            AgeReplacementReward(self.get_cf(), self.get_cp(), ar),
            discounting_rate=self.discounting_rate,
            period_before_discounting=self.period_before_discounting,
        )

    @override
    @check_impossible_replacements
    @reshape_a0_ar
    def expected_net_present_value(
        self,
        tf: float,
        nb_steps: int,
        ar: ST | NumpyST | Array1D[NumpyST],
        a0: ST | NumpyST | Array1D[NumpyST] = 0.0,
        total_sum: bool = False,
    ) -> tuple[Array1D[np.float64], Array1D[np.float64] | Array2D[np.float64]]:
        return self._expected_costs(ar=ar).expected_net_present_value(
            tf, nb_steps, a0, ar
        )

    @override
    @check_impossible_replacements
    def asymptotic_expected_net_present_value(
        self,
        ar: ST | NumpyST | Array1D[NumpyST],
        a0: ST | NumpyST | Array1D[NumpyST] = 0.0,
    ) -> np.float64 | Array1D[np.float64]:
        return self._expected_costs(ar=ar).asymptotic_expected_net_present_value(a0, ar)

    @override
    @reshape_a0_ar
    def expected_equivalent_annual_cost(
        self,
        tf: float,
        nb_steps: int,
        ar: ST | NumpyST | Array1D[NumpyST],
        a0: ST | NumpyST | Array1D[NumpyST] = 0.0,
    ) -> tuple[Array1D[np.float64], Array1D[np.float64] | Array2D[np.float64]]:

        return self._expected_costs(ar=ar).expected_equivalent_annual_cost(
            tf, nb_steps, a0, ar
        )

    @override
    @check_impossible_replacements
    @reshape_a0_ar
    def asymptotic_expected_equivalent_annual_cost(
        self,
        ar: ST | NumpyST | Array1D[NumpyST],
        a0: ST | NumpyST | Array1D[NumpyST] = 0.0,
    ) -> np.float64 | Array1D[np.float64]:

        return self._expected_costs(ar=ar).asymptotic_expected_equivalent_annual_cost(
            a0, ar
        )

    def get_optimal_ar(self) -> ST | Array1D[np.float64]:
        """
        Optimize the policy according to the costs, the discounting rate and the underlying lifetime model.

        Returns
        -------
        ar : float or np.ndarray
            Optimal ages of replacements.
        """  # noqa: E501

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

        def eq(a: ArrayND[np.float64]) -> np.float64 | ArrayND[np.float64]:
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
    """  # noqa: E501

    def _stochastic_reward_process(
        self,
        ar: ST | NumpyST | Array1D[NumpyST],
    ) -> RenewalRewardProcess:
        return RenewalRewardProcess(
            self.baseline_model,
            AgeReplacementReward(self.get_cf(), self.get_cp(), ar),
            discounting_rate=self.discounting_rate,
        )

    @override
    @check_impossible_replacements
    @reshape_a0_ar
    def expected_net_present_value(
        self,
        tf: float,
        nb_steps: int,
        ar: ST | NumpyST | Array1D[NumpyST],
        a0: ST | NumpyST | Array1D[NumpyST] = 0.0,
    ) -> tuple[Array1D[np.float64], Array1D[np.float64] | Array2D[np.float64]]:
        return self._stochastic_reward_process(ar=ar).expected_total_reward(
            tf, nb_steps, a0, ar
        )

    @override
    @check_impossible_replacements
    @reshape_a0_ar
    def expected_equivalent_annual_cost(
        self,
        tf: float,
        nb_steps: int,
        ar: ST | NumpyST | Array1D[NumpyST],
        a0: ST | NumpyST | Array1D[NumpyST] = 0.0,
    ) -> tuple[Array1D[np.float64], Array1D[np.float64] | Array2D[np.float64]]:

        return self._stochastic_reward_process(ar).expected_equivalent_annual_worth(
            tf, nb_steps, a0, ar
        )

    @override
    @check_impossible_replacements
    @reshape_a0_ar
    def asymptotic_expected_net_present_value(
        self,
        ar: ST | NumpyST | Array1D[NumpyST],
        a0: ST | NumpyST | Array1D[NumpyST] = 0.0,
    ) -> np.float64 | Array1D[np.float64]:
        return self._stochastic_reward_process(ar=ar).asymptotic_expected_total_reward(
            a0, ar
        )

    @override
    @check_impossible_replacements
    @reshape_a0_ar
    def asymptotic_expected_equivalent_annual_cost(
        self,
        ar: ST | NumpyST | Array1D[NumpyST],
        a0: ST | NumpyST | Array1D[NumpyST] = 0.0,
    ) -> np.float64 | Array1D[np.float64]:

        return self._stochastic_reward_process(
            ar=ar
        ).asymptotic_expected_equivalent_annual_worth(a0, ar)

    @check_impossible_replacements
    @reshape_a0_ar
    def annual_number_of_replacements(
        self,
        ar: ST | NumpyST | Array1D[NumpyST],
        nb_years: int,
        upon_failure: bool = False,
        total: bool = True,
        a0: ST | NumpyST | Array1D[NumpyST] = 0.0,
    ):
        """
        The expected number of annual replacements.

        Parameters
        ----------
        ar : float or np.ndarray
            Ages of replacements
        nb_years : int
            The number of years on which the annual number of replacements are projected
        upon_failure : bool, default is False
            If True, it also returns the annual number of replacements due to unexpected failures
        a0 : float or np.ndarray or None
            Optional, initial ages
        """  # noqa: E501

        timeline, total_renewals = self._stochastic_reward_process(
            ar=ar
        ).renewal_function(nb_years, nb_years + 1, a0=a0, ar=ar)
        nb_replacements = np.diff(total_renewals)
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

    def get_optimal_ar(self) -> ST | Array1D[np.float64]:
        """
        Optimize the policy according to the costs, the discounting rate and the underlying lifetime model.

        Returns
        -------
        ar : float or np.ndarray
            Optimal ages of replacements.
        """  # noqa: E501

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

        def eq(
            a: ArrayND[np.float64],
        ) -> np.float64 | ArrayND[np.float64]:  # () or (m, 1)
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
    @reshape_a0_ar
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
        ar : float or np.ndarray
            Ages of replacements
        nb_samples : int
            The number of samples.
        time_window : tuple of two floats
            Time window in which data are sampled.
        seed : int, optional
            Random seed, by default None.
        a0 : float or np.ndarray or None
            Optional, initial ages

        Returns
        -------
        A dict of time, event, entry and args (covariates)

        """
        return self._stochastic_reward_process(ar=ar).generate_failure_data(
            nb_samples, time_window, ar=ar, a0=a0, seed=seed
        )

    @check_impossible_replacements
    @reshape_a0_ar
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


class NonHomogeneousPoissonAgeReplacementPolicy(BaseReplacementPolicy):
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

    Attributes
    ----------
    cp
    cr
    """

    def __init__(self, nhpp, cr, cp, discounting_rate=0.0):
        super().__init__(
            nhpp,
            cost_structure={"cr": to_column_2d_if_1d(cr), "cp": to_column_2d_if_1d(cp)},
            discounting_rate=discounting_rate,
        )

    @property
    def cp(self):
        """Costs of preventive replacement.

        Returns
        -------
        np.ndarray
        """
        return flatten_if_at_least_2d(self._cost_structure["cp"])

    @cp.setter
    def cp(self, value):
        self._cost_structure["cp"] = to_column_2d_if_1d(value)

    @property
    def cr(self):
        """Cost of minimal repair.

        Returns
        -------
        np.ndarray
        """
        return flatten_if_at_least_2d(self._cost_structure["cr"])

    @cr.setter
    def cr(self, value):
        self._cost_structure["cr"] = to_column_2d_if_1d(value)

    @reshape_a0_ar
    def expected_net_present_value(
        self,
        tf,
        ar: NumpyFloat,
        nb_steps,
        total_sum=False,
        a0: NumpyFloat | None = None,
    ):
        raise NotImplementedError("implementation will come in a future release")

    @reshape_a0_ar
    def asymptotic_expected_net_present_value(
        self, ar: NumpyFloat, total_sum=False, a0: NumpyFloat | None = None
    ):
        raise NotImplementedError("implementation will come in a future release")

    @reshape_a0_ar
    def expected_equivalent_annual_cost(
        self,
        ar: NumpyFloat,
        tf,
        nb_steps,
        total_sum=False,
        a0: NumpyFloat | None = None,
    ):
        raise NotImplementedError("implementation will come in a future release")

    @reshape_a0_ar
    def asymptotic_expected_equivalent_annual_cost(
        self, ar: NumpyFloat, a0: NumpyFloat | None = None
    ):
        if a0 is not None:
            raise ValueError(
                "NHPP policies with initial ages will be covered in a future release"
            )

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
        Optimize the policy according to the costs, the discounting rate and the underlying lifetime model.

        Returns
        -------
        ar : float or np.ndarray
            Optimal ages of replacements.
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
