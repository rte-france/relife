from typing import Optional, Self, NewType, Union

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import newton

from relife.core import LifetimeModel, LifetimeDistribution
from relife.core.decorators import require_attributes
from relife.core.descriptors import ShapedArgs
from relife.data import CountData
from relife.models import AgeReplacementModel, LeftTruncatedModel
from relife.processes import NonHomogeneousPoissonProcess, RenewalRewardProcess
from relife.processes.renewal import reward_partial_expectation
from relife.quadratures import gauss_legendre
from relife.rewards import (
    age_replacement_rewards,
    exp_discounting,
    run_to_failure_rewards,
)

NumericalArrayLike = NewType(
    "NumericalArrayLike",
    Union[NDArray[np.floating], NDArray[np.integer], float, int],
)


# RenewalPolicy
class RenewalPolicy:
    distribution: LifetimeDistribution
    distribution1: Optional[LifetimeDistribution]

    def __init__(self, discounting_rate: Optional[float] = None):
        self.discounting = exp_discounting(discounting_rate)

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


def _reshape_like_nb_assets(arr: NumericalArrayLike, nb_assets: int):
    arr = np.asarray(arr)
    if arr.ndim > 2:
        raise ValueError
    if arr.size == 1:
        if nb_assets > 1:
            return np.tile(arr, (nb_assets, 1))
        return arr
    else:
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.shape[0] != nb_assets:
            raise ValueError
        return arr


class OneCycleRunToFailurePolicy(RenewalPolicy):
    r"""One cyle run-to-failure policy

    A policy for running assets to failure within one cycle.

    Parameters
    ----------
    model : LifetimeModel
        The lifetime core of the assets.
    cf : np.ndarray
        The cost of failure for each asset.
    discounting_rate : float, default is 0.
        The discounting rate.
    period_before_discounting: float, default is 1.
        The length of the first period before discounting.
    model_args : ModelArgs, optional
        ModelArgs is a tuple of zero or more ndarray required by the underlying
        lifetime core of the processes.
    nb_assets : int, optional
        Number of assets (default is 1).
    a0 : ndarray, optional
        Current ages of the assets (default is None). Setting ``a0`` will add
        left truncations.
    """

    distribution1 = None

    def __init__(
        self,
        distribution: LifetimeDistribution,
        cf: NumericalArrayLike,
        *,
        discounting_rate: Optional[float] = None,
        period_before_discounting: float = 1.0,
        a0: Optional[NumericalArrayLike] = None,
    ) -> None:
        super().__init__(discounting_rate)
        nb_assets = 1 if distribution.nb_assets is None else distribution.nb_assets
        self.a0 = a0  # no _reshape_like_nb_assets because if is passed to FrozenDistribution ShapedArgs
        self.cf = _reshape_like_nb_assets(cf, nb_assets) if cf is not None else None
        self.rewards = run_to_failure_rewards(self.cf)
        if period_before_discounting == 0:
            raise ValueError("The period_before_discounting must be greater than 0")
        self.period_before_discounting = period_before_discounting

        if self.a0 is not None:
            distribution = LeftTruncatedModel(distribution).get_distribution(self.a0)
        self.distribution = distribution

    @property
    def discounting_rate(self):
        return self.discounting.rate

    def expected_total_cost(self, timeline: NDArray[np.float64]) -> NDArray[np.float64]:
        return reward_partial_expectation(
            timeline,
            self.distribution,
            self.rewards,
            discounting=self.discounting,
        )

    def asymptotic_expected_total_cost(self) -> NDArray[np.float64]:
        return self.expected_total_cost(np.array(np.inf))

    def expected_equivalent_annual_cost(
        self, timeline: NDArray[np.float64]
    ) -> NDArray[np.float64]:

        f = (
            lambda x: self.rewards(x)
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
    model : LifetimeModel
        The lifetime core of the assets.
    cf : np.ndarray
        The cost of failure for each asset.
    discounting_rate : float, default is 0.
        The discounting rate.
    model_args : ModelArgs, optional
        ModelArgs is a tuple of zero or more ndarray required by the underlying
        lifetime core of the processes.
    nb_assets : int, optional
        Number of assets (default is 1).
    a0 : ndarray, optional
        Current ages of the assets (default is None). Setting ``a0`` will add
        left truncations.
    model1 : LifetimeModel, optional
        The lifetime core used for the cycle of replacements. When one adds
        `model1`, we assume that `model1` is different from `core` meaning
        the underlying survival probabilities behave differently for the first
        cycle
    model1_args : ModelArgs, optional
        ModelArgs is a tuple of zero or more ndarray required by the lifetime
        core of the first cycle of replacements.

    References
    ----------
    .. [1] Van der Weide, J. A. M., & Van Noortwijk, J. M. (2008). Renewal
        theory with exponential and hyperbolic discounting. Probability in
        the Engineering and Informational Sciences, 22(1), 53-74.
    """

    def __init__(
        self,
        distribution: LifetimeDistribution,
        cf: NumericalArrayLike,
        *,
        discounting_rate: Optional[float] = None,
        a0: Optional[NumericalArrayLike] = None,
        distribution1: Optional[LifetimeDistribution] = None,
    ) -> None:
        super().__init__(discounting_rate)
        nb_assets = 1 if distribution.nb_assets is None else distribution.nb_assets
        self.a0 = a0  # no _reshape_like_nb_assets because if is passed to FrozenDistribution ShapedArgs
        self.cf = _reshape_like_nb_assets(cf, nb_assets) if cf is not None else None
        self.rewards = run_to_failure_rewards(self.cf)

        self.distribution1 = None
        if self.a0 is not None:
            if distribution1 is not None:
                raise ValueError("distribution1 and a0 can't be set together")
            self.distribution1 = LeftTruncatedModel(distribution).get_distribution(
                self.a0
            )
        self.distribution = distribution

        # if Policy is parametrized, set the underlying renewal reward processes
        # note the rewards are the same for the first cycle and the rest of the processes
        self.process = RenewalRewardProcess(
            self.distribution,
            self.rewards,
            discounting_rate=discounting_rate,
            distribution1=self.distribution1,
            rewards1=self.rewards,
        )

    @property
    def discounting_rate(self):
        return self.discounting.rate

    def expected_total_cost(self, timeline: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.process.expected_total_reward(timeline)

    def asymptotic_expected_total_cost(self) -> NDArray[np.float64]:
        return self.process.asymptotic_expected_total_reward()

    def expected_equivalent_annual_cost(
        self, timeline: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return self.process.expected_equivalent_annual_cost(timeline)

    def asymptotic_expected_equivalent_annual_cost(self) -> NDArray[np.float64]:
        return self.process.asymptotic_expected_equivalent_annual_cost()


class OneCycleAgeReplacementPolicy(RenewalPolicy):
    r"""One-cyle age replacement policy.

    The asset is disposed at a fixed age :math:`a_r` with costs :math:`c_p` or upon failure
    with costs :math:`c_f` if earlier.

    Parameters
    ----------
    model : LifetimeModel
        The lifetime core of the assets.
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
    model_args : ModelArgs, optional
        ModelArgs is a tuple of zero or more ndarray required by the underlying
        lifetime core of the processes.
    nb_assets : int, optional
        Number of assets (default is 1).
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
    ar1 = None

    def __init__(
        self,
        distribution: LifetimeDistribution,
        cf: NumericalArrayLike,
        cp: NumericalArrayLike,
        *,
        discounting_rate: Optional[float] = None,
        period_before_discounting: float = 1.0,
        ar: Optional[NumericalArrayLike] = None,
        a0: Optional[NumericalArrayLike] = None,
    ) -> None:
        super().__init__(discounting_rate)
        nb_assets = 1 if distribution.nb_assets is None else distribution.nb_assets
        self.a0 = a0  # no _reshape_like_nb_assets because if is passed to FrozenDistribution ShapedArgs
        self.ar = ar
        self.cf = _reshape_like_nb_assets(cf, nb_assets) if cf is not None else None
        self.cp = _reshape_like_nb_assets(cp, nb_assets) if cf is not None else None
        self.rewards = run_to_failure_rewards(self.cf)
        if period_before_discounting == 0:
            raise ValueError("The period_before_discounting must be greater than 0")
        self.period_before_discounting = period_before_discounting

        if self.a0 is not None:
            distribution = LeftTruncatedModel(distribution).get_distribution(self.a0)
        self.distribution = distribution

        if self.ar is not None:
            self.distribution = AgeReplacementModel(distribution).get_distribution(
                self.ar
            )

    @property
    def discounting_rate(self):
        return self.discounting.rate

    @require_attributes("ar")
    def expected_total_cost(self, timeline: NDArray[np.float64]) -> NDArray[np.float64]:
        return reward_partial_expectation(
            timeline,
            self.distribution,
            self.rewards,
            discounting=self.discounting,
        )

    def asymptotic_expected_total_cost(self) -> NDArray[np.float64]:
        return self.expected_total_cost(np.array(np.inf))

    @require_attributes("ar")
    def expected_equivalent_annual_cost(
        self, timeline: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        f = (
            lambda x: self.rewards(x)
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
                    f,
                    np.array(self.period_before_discounting),
                    timeline,
                ),
            )
        )

    def asymptotic_expected_equivalent_annual_cost(
        self,
    ) -> NDArray[np.float64]:
        return self.expected_equivalent_annual_cost(np.array(np.inf))

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
        if np.size(x0) == 1:
            x0 = np.tile(x0, (self.distribution.nb_assets, 1))

        def eq(a):
            return np.sum(
                self.discounting.factor(a)
                / self.discounting.annuity_factor(a)
                * (
                    (cf_3d - cp_3d) * self.model.baseline.hf(a, *self.model_args[1:])
                    - cp_3d / self.discounting.annuity_factor(a)
                ),
                axis=0,
            )

        ar = np.asarray(newton(eq, x0), dtype=np.float64)

        self.model_args = (ar,) + self.model_args[1:]
        self.ar = ar
        self.rewards = age_replacement_rewards(ar=ar, cf=self.cf, cp=self.cp)
        return OneCycleAgeReplacementPolicy(
            self.model.baseline,
            self.cf,
            self.cp,
            discounting_rate=self.discounting_rate,
            period_before_discounting=self.period_before_discounting,
            ar=ar,
            model_args=self.model_args[1:],
            nb_assets=self.nb_assets,
            a0=self.a0,
        )


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
    model_args : ModelArgs, optional
        ModelArgs is a tuple of zero or more ndarray required by the underlying
        lifetime core of the processes.
    nb_assets : int, optional
        Number of assets (default is 1).
    a0 : ndarray, optional
        Current ages of the assets (default is None). Setting ``a0`` will add
        left truncations.
    model1 : LifetimeModel, optional
        The lifetime core used for the cycle of replacements. When one adds
        `model1`, we assume that ``model1`` is different from ``core`` meaning
        the underlying survival probabilities behave differently for the first
        cycle
    model1_args : ModelArgs, optional
        ModelArgs is a tuple of zero or more ndarray required by the lifetime
        core of the first cycle of replacements.

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

    distribution: AgeReplacementModel
    distribution1: AgeReplacementModel

    def __init__(
        self,
        distribution: LifetimeDistribution,
        cf: float | NDArray[np.float64],
        cp: float | NDArray[np.float64],
        *,
        discounting_rate: Optional[float] = None,
        ar: float | NDArray[np.float64] = None,
        ar1: float | NDArray[np.float64] = None,
        model_args: tuple[NumericalArrayLike, ...] = (),
        nb_assets: int = 1,
        a0: Optional[float | NDArray[np.float64]] = None,
        model1: Optional[LifetimeModel[*tuple[NumericalArrayLike, ...]]] = None,
        model1_args: tuple[NumericalArrayLike, ...] = (),
    ) -> None:

        self.nb_assets = nb_assets
        self.a0 = a0
        if a0 is not None:
            if model1 is not None:
                raise ValueError("model1 and a0 can't be set together")
            model1 = AgeReplacementModel(LeftTruncatedModel(distribution))
            model1_args = (a0, *model_args)
        elif model1 is not None:
            model1 = AgeReplacementModel(model1)
        elif model1 is None and ar1 is not None:
            raise ValueError("model1 is not set, ar1 is useless")

        self.distribution = AgeReplacementModel(distribution)
        self.distribution1 = model1

        self.cf = cf
        self.cp = cp
        self.ar = ar
        self.ar1 = ar1
        self.discounting = exp_discounting(discounting_rate)
        self.rewards = age_replacement_rewards(ar=self.ar, cf=self.cf, cp=self.cp)
        self.rewards1 = age_replacement_rewards(ar=self.ar1, cf=self.cf, cp=self.cp)

        self.model_args = (ar,) + model_args

        # (None, ...) or (ar1, ...)
        self.model1_args = (ar1,) + model1_args if model1_args else None

        self.process = None
        parametrized = False
        if self.ar is not None:
            parametrized = True
            if self.distribution1 is not None:
                if self.ar1 is None:
                    parametrized = False

        # if Policy is parametrized, set the underlying renewal reward processes
        # note the rewards are the same for the first cycle and the rest of the processes
        if parametrized:
            self.process = RenewalRewardProcess(
                self.distribution,
                self.rewards,
                nb_assets=self.nb_assets,
                model_args=self.model_args,
                discounting_rate=discounting_rate,
                model1=self.distribution1,
                model1_args=self.model1_args,
                rewards1=self.rewards1,
            )

    @property
    def discounting_rate(self):
        return self.discounting.rate

    @require_attributes("ar")
    def expected_total_cost(self, timeline: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.process.expected_total_reward(timeline)

    @require_attributes("ar")
    def expected_equivalent_annual_cost(
        self, timeline: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return self.process.expected_equivalent_annual_cost(timeline)

    @require_attributes("ar")
    def asymptotic_expected_total_cost(self) -> NDArray[np.float64]:
        return self.process.asymptotic_expected_total_reward()

    @require_attributes("ar")
    def asymptotic_expected_equivalent_annual_cost(self) -> NDArray[np.float64]:
        return self.process.asymptotic_expected_equivalent_annual_cost()

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
        if np.size(x0) == 1:
            x0 = np.tile(x0, (self.nb_assets, 1))

        ndim = max(
            map(np.ndim, (cf_3d, cp_3d, *self.model_args[1:])),
            default=0,
        )

        def eq(a):
            f = gauss_legendre(
                lambda x: self.discounting.factor(x)
                * self.distribution.baseline.sf(x, *self.model_args[1:]),
                0,
                a,
                ndim=ndim,
            )
            g = gauss_legendre(
                lambda x: self.discounting.factor(x)
                * self.distribution.baseline.pdf(x, *self.model_args[1:]),
                0,
                a,
                ndim=ndim,
            )
            return np.sum(
                self.discounting.factor(a)
                * (
                    (cf_3d - cp_3d)
                    * (self.distribution.baseline.hf(a, *self.model_args[1:]) * f - g)
                    - cp_3d
                )
                / f**2,
                axis=0,
            )

        ar = newton(eq, x0)
        if np.size(ar) == 1:
            ar = np.squeeze(ar)

        ar1 = None
        if self.distribution1 is not None:
            onecycle = OneCycleAgeReplacementPolicy(
                self.distribution1.baseline,
                self.cf,
                self.cp,
                nb_assets=self.nb_assets,
                discounting_rate=self.discounting_rate,
                model_args=self.model1_args[1:],
            ).optimize()
            ar1 = onecycle.ar

        return DefaultAgeReplacementPolicy(
            self.distribution.baseline,
            self.cf,
            self.cp,
            discounting_rate=self.discounting_rate,
            ar=ar,
            ar1=ar1,
            model_args=self.model_args[1:],
            model1=self.distribution1.baseline if self.distribution1 else None,
            model1_args=self.model1_args[1:] if self.distribution1 else None,
            nb_assets=self.nb_assets,
            a0=self.a0 if self.distribution1 is None else None,
        )


class NonHomogeneousPoissonAgeReplacementPolicy(RenewalPolicy):
    """
    Implements a Non-Homogeneous Poisson Process (NHPP) age-replacement policy..

    Attributes
    ----------
    nb_assets : int
        Number of assets involved in the age-replacement policy.
    model : LifetimeModel
        The lifetime model defining the underlying processes.
    process : NonHomogeneousPoissonProcess
        NHPP instance modeling the intensity and cumulative intensity.
    model_args : ModelArgs
        Additional arguments required by the lifetime model.
    ar : np.ndarray or None
        Optimized replacement age (optimized policy parameter).
    cp : np.ndarray
        The cost of failure for each asset.
    cr : np.ndarray
        The cost of repair for each asset.
    """

    cp = ShapedArgs()
    cr = ShapedArgs()
    ar = ShapedArgs()

    def __init__(
        self,
        process: NonHomogeneousPoissonProcess,
        cp: NDArray[np.float64],
        cr: NDArray[np.float64],
        *,
        discounting_rate: Optional[float] = None,
        ar: Optional[NDArray[np.float64]] = None,
        nb_assets: int = 1,
    ) -> None:

        self.nb_assets = nb_assets

        self.process = process
        self.model = process.model
        self.model_args = process.model_args
        self.ar = ar
        self.cp = cp
        self.cr = cr
        self.discounting = exp_discounting(discounting_rate)
        self.rewards = age_replacement_rewards(self.ar, self.cr, self.cp)

    @property
    def discounting_rate(self):
        return self.discounting.rate

    @require_attributes("ar")
    def expected_total_cost(self, timeline: NDArray[np.float64]) -> NDArray[np.float64]:
        pass

    def asymptotic_expected_total_cost(self) -> NDArray[np.float64]:
        pass

    @require_attributes("ar")
    def expected_equivalent_annual_cost(
        self, timeline: NDArray[np.float64]
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
    ) -> Self:

        x0 = self.model.mean(*self.model_args)

        cr_2d, cp_2d, *model_args_2d = np.atleast_2d(self.cr, self.cp, *self.model_args)
        if isinstance(model_args_2d, np.ndarray):
            model_args_2d = (model_args_2d,)

        if self.discounting.rate != 0:

            def dcost(a):
                a = np.atleast_2d(a)
                return (
                    (1 - self.discounting.factor(a))
                    / self.discounting.rate
                    * self.process.intensity(a, *model_args_2d)
                    - gauss_legendre(
                        lambda t: self.discounting.factor(t)
                        * self.process.intensity(t, *model_args_2d),
                        np.array(0.0),
                        a,
                        ndim=2,
                    )
                    - cp_2d / cr_2d
                )

        else:

            def dcost(a):
                a = np.atleast_2d(a)
                return (
                    a * self.process.intensity(a, *model_args_2d)
                    - self.process.cumulative_intensity(a, *model_args_2d)
                    - cp_2d / cr_2d
                )

        ar = newton(dcost, x0)

        ndim = max(map(np.ndim, (self.cp, self.cr, *self.model_args)), default=0)
        if ndim < 2:
            ar = np.squeeze(ar)
        self.ar = ar
        self.rewards = age_replacement_rewards(ar, self.cr, self.cp)
        return self


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
