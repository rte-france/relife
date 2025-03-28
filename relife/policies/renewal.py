import functools
from typing import Optional, Self, NewType, Union

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import newton

from relife.data import CountData
from relife.economics.rewards import (
    age_replacement_rewards,
    exp_discounting,
    run_to_failure_rewards,
)
from relife.processes import NonHomogeneousPoissonProcess, RenewalRewardProcess
from relife.processes.renewal import reward_partial_expectation
from relife.quadratures import gauss_legendre
from ..distributions.mixins import FrozenLifetimeDistribution
from ..distributions.protocols import LifetimeDistribution

NumericalArrayLike = NewType(
    "NumericalArrayLike",
    Union[NDArray[np.floating], NDArray[np.integer], float, int],
)


# RenewalPolicy
class RenewalPolicy:

    def __init__(
        self,
        distribution: LifetimeDistribution[()],
        distribution1: Optional[LifetimeDistribution[()]] = None,
        discounting_rate: Optional[float] = None,
    ):

        self.distribution = distribution
        self.distribution1 = distribution1
        self.discounting = exp_discounting(discounting_rate)
        self.nb_assets = None
        if isinstance(distribution, FrozenLifetimeDistribution):
            self.nb_assets = distribution.nb_assets

    @property
    def discounting_rate(self):
        return self.discounting.rate

    def sample(
        self,
        size: int,
        tf: float,
        t0: float = 0.0,
        maxsample: int = 1e5,
        seed: Optional[int] = None,
    ) -> CountData:
        from relife.sampling import sample_count_data

        return sample_count_data(self, size, tf, t0=t0, maxsample=maxsample, seed=seed)

    def failure_data_sample(
        self,
        size: int,
        tf: float,
        t0: float = 0.0,
        maxsample: int = 1e5,
        seed: Optional[int] = None,
        use: str = "model",
    ) -> tuple[NDArray[np.float64], ...]:
        from relife.sampling import failure_data_sample

        return failure_data_sample(
            self, size, tf, t0=t0, maxsample=maxsample, seed=seed, use=use
        )


def _reshape_like(arr: NumericalArrayLike, nb_assets: Optional[int] = None):
    """
    Reshape NumericalArrayLike object that only contains 1 value per asset.
    """
    arr = np.asarray(arr)
    if arr.ndim > 2:
        raise ValueError
    if arr.size == 1:
        if nb_assets > 1 and nb_assets is not None:
            return np.tile(arr, (nb_assets, 1))
        return arr
    else:  # more than 1 value
        if arr.ndim == 1 and nb_assets is not None:
            arr = arr.reshape(-1, 1)
        if arr.shape[0] != nb_assets and nb_assets is not None:
            raise ValueError
        return arr


class Cost:
    def __set_name__(self, owner, name):
        self.private_name = "_" + name
        self.public_name = name

    def __get__(self, obj, objtype=None):
        return getattr(obj, self.private_name)

    def __set__(
        self,
        obj,
        value: Optional[NumericalArrayLike],
    ) -> NDArray[np.float64]:
        if value is not None:
            nb_assets = obj.distribution.nb_assets
            setattr(obj, self.private_name, _reshape_like(value, nb_assets))
        else:
            setattr(obj, self.private_name, value)


class OneCycleRunToFailurePolicy(RenewalPolicy):
    r"""One cyle run-to-failure policy

    A policy for running assets to failure within one cycle.

    Parameters
    ----------
    distribution : LifetimeDistribution
        The lifetime distribution of the assets.
    cf : np.ndarray
        The cost of failure for each asset.
    discounting_rate : float, default is 0.
        The discounting rate.
    period_before_discounting: float, default is 1.
        The length of the first period before discounting.
    a0 : ndarray, optional
        Current ages of the assets (default is None). Setting ``a0`` will add
        left truncations.
    """

    distribution1 = None
    cf = Cost()

    def __init__(
        self,
        distribution: LifetimeDistribution,
        cf: NumericalArrayLike,
        *,
        discounting_rate: Optional[float] = None,
        period_before_discounting: float = 1.0,
        a0: Optional[NumericalArrayLike] = None,
    ) -> None:
        super().__init__(distribution, discounting_rate=discounting_rate)
        self.a0 = a0  # no _reshape_like_nb_assets because if is passed to FrozenDistribution ShapedArgs
        self.cf = cf
        if period_before_discounting == 0:
            raise ValueError("The period_before_discounting must be greater than 0")
        self.period_before_discounting = period_before_discounting

        if self.a0 is not None:
            self.distribution = left_truncated(self.distribution, self.a0)

    @property
    def discounting_rate(self):
        return self.discounting.rate

    def expected_total_cost(self, timeline: NDArray[np.float64]) -> NDArray[np.float64]:
        return reward_partial_expectation(
            timeline,
            self.distribution,
            run_to_failure_rewards(self.cf),
            discounting=self.discounting,
        )

    def asymptotic_expected_total_cost(self) -> NDArray[np.float64]:
        return self.expected_total_cost(np.array(np.inf))

    def expected_equivalent_annual_cost(
        self, timeline: NDArray[np.float64]
    ) -> NDArray[np.float64]:

        f = (
            lambda x: run_to_failure_rewards(self.cf)(x)
            * self.discounting.factor(x)
            / self.discounting.annuity_factor(x)
        )
        mask = timeline < self.period_before_discounting
        q0 = self.distribution.cdf(self.period_before_discounting) * f(
            self.period_before_discounting
        )
        return np.squeeze(
            q0
            + np.where(
                mask,
                0,
                self.distribution.ls_integrate(
                    f, self.period_before_discounting, timeline
                ),
            )
        )

    def asymptotic_expected_equivalent_annual_cost(
        self,
    ) -> NDArray[np.float64]:
        return self.expected_equivalent_annual_cost(np.array(np.inf))


class DefaultRunToFailurePolicy(RenewalPolicy):
    r"""Run-to-failure renewal policy.

    Renewal reward processes where assets are replaced on failure with costs
    :math:`c_f`.

    Parameters
    ----------
    distribution : LifetimeDistribution
        The lifetime distribution of the assets.
    cf : np.ndarray
        The cost of failure for each asset.
    discounting_rate : float, default is 0.
        The discounting rate.
    a0 : ndarray, optional
        Current ages of the assets (default is None). Setting ``a0`` will add
        left truncations.
    distribution1 : LifetimeDistribution, optional
        The lifetime distribution used for the first cycle of replacements. When one adds
        `distribution1`, we assume that `distribution1` is different from `distribution` meaning
        the underlying survival probabilities behave differently for the first
        cycle.


    References
    ----------
    .. [1] Van der Weide, J. A. M., & Van Noortwijk, J. M. (2008). Renewal
        theory with exponential and hyperbolic discounting. Probability in
        the Engineering and Informational Sciences, 22(1), 53-74.
    """

    cf = Cost()

    def __init__(
        self,
        distribution: LifetimeDistribution[()],
        cf: NumericalArrayLike,
        *,
        discounting_rate: Optional[float] = None,
        a0: Optional[NumericalArrayLike] = None,
        distribution1: Optional[LifetimeDistribution] = None,
    ) -> None:
        super().__init__(distribution, distribution1, discounting_rate)
        self.a0 = a0  # no _reshape_like_nb_assets because if is passed to FrozenDistribution ShapedArgs
        self.cf = cf

        if self.a0 is not None:
            if self.distribution1 is not None:
                raise ValueError("distribution1 and a0 can't be set together")
            self.distribution1 = left_truncated(self.distribution, self.a0)

    @property
    def discounting_rate(self):
        return self.discounting.rate

    @property
    def underlying_process(
        self,
    ) -> RenewalRewardProcess:
        return RenewalRewardProcess(
            self.distribution,
            run_to_failure_rewards(self.cf),
            discounting_rate=self.discounting_rate,
            distribution1=self.distribution1,
            rewards1=run_to_failure_rewards(self.cf) if self.distribution1 else None,
        )

    def expected_total_cost(self, timeline: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.underlying_process.expected_total_reward(timeline)

    def asymptotic_expected_total_cost(self) -> NDArray[np.float64]:
        return self.underlying_process.asymptotic_expected_total_reward()

    def expected_equivalent_annual_cost(
        self, timeline: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return self.underlying_process.expected_equivalent_annual_cost(timeline)

    def asymptotic_expected_equivalent_annual_cost(self) -> NDArray[np.float64]:
        return self.underlying_process.asymptotic_expected_equivalent_annual_cost()


def get_if_not_given(*args_names: str):
    """
    Decorators that get the attribute value if argument value is None
    Reshape depending on number of assets.
    If both are None, an error is raised.
    Priority is always given to the attribute value

    Parameters
    ----------
    args_names

    Returns
    -------

    """

    def decorator(method):
        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            for name in args_names:
                attr_value = getattr(self, name)
                arg_value = kwargs.pop(name, None)
                if attr_value is None and arg_value is None:
                    raise ValueError(
                        f"{name} is not set. If fit exists, you may need to fit the object first or instanciate the object with {name}"
                    )
                elif attr_value is not None and arg_value is not None:
                    # priority on arg
                    kwargs[name] = _reshape_like(arg_value, self.nb_assets)
                elif attr_value is not None and arg_value is None:
                    kwargs[name] = _reshape_like(attr_value, self.nb_assets)
                else:
                    kwargs[name] = _reshape_like(arg_value, self.nb_assets)

            return method(self, *args, **kwargs)

        return wrapper

    return decorator


class OneCycleAgeReplacementPolicy(RenewalPolicy):
    r"""One-cyle age replacement policy.

    The asset is disposed at a fixed age :math:`a_r` with costs :math:`c_p` or upon failure
    with costs :math:`c_f` if earlier.

    Parameters
    ----------
    distribution : LifetimeDistribution[()]
        The lifetime distribution of the assets.
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

    distribution1 = None
    cp = Cost()
    cf = Cost()

    def __init__(
        self,
        distribution: LifetimeDistribution[()],
        cf: NumericalArrayLike,
        cp: NumericalArrayLike,
        *,
        discounting_rate: Optional[float] = None,
        period_before_discounting: float = 1.0,
        ar: Optional[NumericalArrayLike] = None,
        a0: Optional[NumericalArrayLike] = None,
    ) -> None:
        super().__init__(distribution, discounting_rate=discounting_rate)
        self.a0 = a0  # no _reshape_like_nb_assets because if is passed to FrozenDistribution ShapedArgs
        self.ar = ar
        self.cf = cf
        self.cp = cp

        if period_before_discounting == 0:
            raise ValueError("The period_before_discounting must be greater than 0")
        self.period_before_discounting = period_before_discounting

        if self.a0 is not None:
            self.distribution = left_truncated(self.distribution, self.a0)

    @property
    def discounting_rate(self):
        return self.discounting.rate

    @get_if_not_given("ar")
    def expected_total_cost(
        self, timeline: NDArray[np.float64], ar: Optional[NumericalArrayLike] = None
    ) -> NDArray[np.float64]:
        return reward_partial_expectation(
            timeline,
            replace_at_age(self.distribution, ar),
            age_replacement_rewards(ar, self.cp, self.cf),
            discounting=self.discounting,
        )

    @get_if_not_given("ar")
    def asymptotic_expected_total_cost(
        self, ar: Optional[NumericalArrayLike] = None
    ) -> NDArray[np.float64]:
        return self.expected_total_cost(np.array(np.inf), ar=ar)

    @get_if_not_given("ar")
    def expected_equivalent_annual_cost(
        self, timeline: NDArray[np.float64], ar: Optional[NumericalArrayLike] = None
    ) -> NDArray[np.float64]:
        f = (
            lambda x: age_replacement_rewards(ar, self.cf, self.cp)(x)
            * self.discounting.factor(x)
            / self.discounting.annuity_factor(x)
        )
        mask = timeline < self.period_before_discounting
        q0 = self.distribution.cdf(self.period_before_discounting) * f(
            self.period_before_discounting
        )
        distribution = replace_at_age(self.distribution, ar)
        return np.squeeze(
            q0
            + np.where(
                mask,
                0,
                distribution.ls_integrate(
                    f,
                    np.array(self.period_before_discounting),
                    timeline,
                ),
            )
        )

    @get_if_not_given("ar")
    def asymptotic_expected_equivalent_annual_cost(
        self, ar: Optional[NumericalArrayLike] = None
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
                    (cf_3d - cp_3d) * self.distribution.hf(a)
                    - cp_3d / self.discounting.annuity_factor(a)
                ),
                axis=0,
            )

        ar = np.asarray(newton(eq, x0), dtype=np.float64)
        self.ar = ar
        return ar


class DefaultAgeReplacementPolicy(RenewalPolicy):
    r"""Time based replacement policy.

    Renewal reward processes where assets are replaced at a fixed age :math:`a_r`
    with costs :math:`c_p` or upon failure with costs :math:`c_f` if earlier [1]_.

    .. [1] Mazzuchi, T. A., Van Noortwijk, J. M., & Kallen, M. J. (2007).
        Maintenance optimization. Encyclopedia of Statistics in Quality and
        Reliability, 1000-1008.

    |

    Parameters
    ----------
    distribution : LifetimeModel
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
    cf : np.ndarray
        The cost of failure for each asset.
    cp : np.ndarray
        The cost of preventive replacements for each asset.
    ar : np.ndarray or None
        Times until preventive replacements. This parameter can be optimized
        with ``optimize``
    ar1 : np.ndarray or None
        Times until preventive replacements for the first cycle. This parameter can be optimized
        with ``optimize``
    """

    cp = Cost()
    cf = Cost()

    def __init__(
        self,
        distribution: LifetimeDistribution[()],
        cf: float | NDArray[np.float64],
        cp: float | NDArray[np.float64],
        *,
        discounting_rate: Optional[float] = None,
        ar: float | NDArray[np.float64] = None,
        ar1: float | NDArray[np.float64] = None,
        a0: Optional[float | NDArray[np.float64]] = None,
        distribution1: Optional[LifetimeDistribution[()]] = None,
    ) -> None:
        super().__init__(distribution, distribution1, discounting_rate)

        self.a0 = a0  # no _reshape_like_nb_assets because if is passed to FrozenDistribution ShapedArgs
        self.cf = cf
        self.cp = cp
        self.ar = ar
        self.ar1 = ar1

        if self.a0 is not None:
            if distribution1 is not None:
                raise ValueError("distribution1 and a0 can't be set together")
            self.distribution1 = left_truncated(self.distribution, a0)
        elif distribution1 is None and ar1 is not None:
            raise ValueError("model1 is not set, ar1 is useless")

    @property
    def discounting_rate(self):
        return self.discounting.rate

    def underlying_process(
        self, ar: NumericalArrayLike, ar1: NumericalArrayLike
    ) -> RenewalRewardProcess:
        return RenewalRewardProcess(
            self.distribution,
            age_replacement_rewards(ar, self.cf, self.cp),
            discounting_rate=self.discounting_rate,
            distribution1=self.distribution1,
            rewards1=age_replacement_rewards(ar1, self.cf, self.cp) if ar1 else None,
        )

    @get_if_not_given("ar", "ar1")
    def expected_total_cost(
        self,
        timeline: NDArray[np.float64],
        ar: Optional[NumericalArrayLike] = None,
        ar1: Optional[NumericalArrayLike] = None,
    ) -> NDArray[np.float64]:
        return self.underlying_process(ar, ar1).expected_total_reward(timeline)

    @get_if_not_given("ar", "ar1")
    def expected_equivalent_annual_cost(
        self,
        timeline: NDArray[np.float64],
        ar: Optional[NumericalArrayLike] = None,
        ar1: Optional[NumericalArrayLike] = None,
    ) -> NDArray[np.float64]:
        return self.underlying_process(ar, ar1).expected_equivalent_annual_cost(
            timeline
        )

    @get_if_not_given("ar", "ar1")
    def asymptotic_expected_total_cost(
        self,
        ar: Optional[NumericalArrayLike] = None,
        ar1: Optional[NumericalArrayLike] = None,
    ) -> NDArray[np.float64]:
        return self.underlying_process(ar, ar1).asymptotic_expected_total_reward()

    @get_if_not_given("ar", "ar1")
    def asymptotic_expected_equivalent_annual_cost(
        self,
        ar: Optional[NumericalArrayLike] = None,
        ar1: Optional[NumericalArrayLike] = None,
    ) -> NDArray[np.float64]:
        return self.underlying_process(
            ar, ar1
        ).asymptotic_expected_equivalent_annual_cost()

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
            map(np.ndim, (cf_3d, cp_3d, *self.distribution.args)),
            default=0,
        )

        def eq(a):
            f = gauss_legendre(
                lambda x: self.discounting.factor(x) * self.distribution.sf(x),
                0,
                a,
                ndim=ndim,
            )
            g = gauss_legendre(
                lambda x: self.discounting.factor(x) * self.distribution.pdf(x),
                0,
                a,
                ndim=ndim,
            )
            return np.sum(
                self.discounting.factor(a)
                * ((cf_3d - cp_3d) * (self.distribution.hf(a) * f - g) - cp_3d)
                / f**2,
                axis=0,
            )

        ar = newton(eq, x0)
        if np.size(ar) == 1:
            ar = np.squeeze(ar)

        ar1 = None
        if self.distribution1 is not None:
            ar1 = OneCycleAgeReplacementPolicy(
                self.distribution1,
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

    cp = Cost()
    cr = Cost()

    def __init__(
        self,
        process: NonHomogeneousPoissonProcess,
        cp: NDArray[np.float64],
        cr: NDArray[np.float64],
        *,
        discounting_rate: Optional[float] = None,
        ar: Optional[NDArray[np.float64]] = None,
    ) -> None:
        super().__init__(process.distribution, discounting_rate=discounting_rate)
        self.ar = ar
        self.cp = cp
        self.cr = cr
        self._underlying_process = process

    @property
    def underlying_process(self):
        return self._underlying_process

    @property
    def discounting_rate(self):
        return self.discounting.rate

    @get_if_not_given("ar")
    def expected_total_cost(
        self, timeline: NDArray[np.float64], ar: Optional[NumericalArrayLike] = None
    ) -> NDArray[np.float64]:
        pass

    @get_if_not_given("ar")
    def asymptotic_expected_total_cost(
        self, ar: Optional[NumericalArrayLike] = None
    ) -> NDArray[np.float64]:
        pass

    @get_if_not_given("ar")
    def expected_equivalent_annual_cost(
        self, timeline: NDArray[np.float64], ar: Optional[NumericalArrayLike] = None
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

        x0 = self.distribution.mean()

        if self.discounting.rate != 0:

            def dcost(a):
                a = np.atleast_2d(a)
                return (
                    (1 - self.discounting.factor(a))
                    / self.discounting.rate
                    * self.underlying_process.intensity(a)
                    - gauss_legendre(
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


from .docstrings import (
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


OneCycleRunToFailurePolicy.expected_total_cost.__doc__ = ETC_DOCSTRING
OneCycleRunToFailurePolicy.expected_equivalent_annual_cost.__doc__ = EEAC_DOCSTRING
OneCycleRunToFailurePolicy.asymptotic_expected_total_cost.__doc__ = (
    ASYMPTOTIC_ETC_DOCSTRING
)
OneCycleRunToFailurePolicy.asymptotic_expected_equivalent_annual_cost.__doc__ = (
    ASYMPTOTIC_EEAC_DOCSTRING
)

DefaultRunToFailurePolicy.expected_total_cost.__doc__ = ETC_DOCSTRING
DefaultRunToFailurePolicy.expected_equivalent_annual_cost.__doc__ = EEAC_DOCSTRING
DefaultRunToFailurePolicy.asymptotic_expected_total_cost.__doc__ = (
    ASYMPTOTIC_ETC_DOCSTRING
)
DefaultRunToFailurePolicy.asymptotic_expected_equivalent_annual_cost.__doc__ = (
    ASYMPTOTIC_EEAC_DOCSTRING
)
