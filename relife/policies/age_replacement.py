from typing import Optional, Self, Union

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import newton

from relife.data import RenewalRewardData
from relife.discounting import exponential_discounting
from relife.generator import lifetimes_rewards_generator
from relife.model import AgeReplacementModel, LeftTruncatedModel, LifetimeModel
from relife.quadratures import gauss_legendre
from relife.renewal import RenewalRewardProcess, reward_partial_expectation
from relife.typing import Model1Args, ModelArgs, Policy

from .decorators import ifset
from relife.descriptors import ShapedArgs


def age_replacement_cost(
    lifetimes: NDArray[np.float64],
    ar: NDArray[np.float64],
    cf: NDArray[np.float64],
    cp: NDArray[np.float64],
) -> NDArray[np.float64]:
    return np.where(lifetimes < ar, cf, cp)


class OneCycleAgeReplacementPolicy(Policy):
    r"""One-cyle age replacement policy.

    The asset is disposed at a fixed age :math:`a_r` with costs :math:`c_p` or upon failure
    with costs :math:`c_f` if earlier.

    Parameters
    ----------
    model : LifetimeModel
        The lifetime model of the assets.
    cf : np.ndarray
        The cost of failure for each asset.
    cp : np.ndarray
        The cost of preventive replacements for each asset.
    discounting_rate : float, default is 0.
        The discounting rate.
    ar : np.ndarray, optional
        Times until preventive replacements. This parameter can be optimized
        with ``fit``
    model_args : ModelArgs, optional
        ModelArgs is a tuple of zero or more ndarray required by the underlying
        lifetime model of the process.
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

    model1 = None

    def __init__(
        self,
        model: LifetimeModel[*ModelArgs],
        cf: float | NDArray[np.float64],
        cp: float | NDArray[np.float64],
        *,
        discounting_rate: float = 0.0,
        ar: Optional[float | NDArray[np.float64]] = None,
        model_args: ModelArgs = (),
        nb_assets: int = 1,
        a0: Optional[float | NDArray[np.float64]] = None,
    ) -> None:
        if a0 is not None:
            model = LeftTruncatedModel(model)
            model_args = (a0, *model_args)
        self.model = AgeReplacementModel(model)
        self.nb_assets = nb_assets

        self.ar = ar
        self.cf = cf
        self.cp = cp
        self.discounting_rate = discounting_rate
        self.model_args = (ar,) + model_args

    @ifset("ar")
    def expected_total_cost(self, timeline: NDArray[np.float64]) -> NDArray[np.float64]:
        """The expected total cost.

        Parameters
        ----------
        timeline : ndarray
            Timeline of points where the function is evaluated

        Returns
        -------
        ndarray
            The expected total cost for each asset along the timeline
        """
        return reward_partial_expectation(
            timeline,
            self.model,
            age_replacement_cost,
            model_args=self.model_args,
            reward_args=(self.ar, self.cf, self.cp),
            discounting_rate=self.discounting_rate,
        )

    def asymptotic_expected_total_cost(self) -> NDArray[np.float64]:
        """
        The asymptotic expected total cost.

        Returns
        -------
        ndarray
            The asymptotic expected total cost for each asset.
        """
        return self.expected_total_cost(np.array(np.inf))

    @ifset("ar")
    def expected_equivalent_annual_cost(
        self, timeline: NDArray[np.float64], dt: float = 1.0
    ) -> NDArray[np.float64]:
        """The expected equivalent annual cost.

        Parameters
        ----------
        timeline : ndarray
            Timeline of points where the function is evaluated
        dt : float, default 1.0
            The length of the first period before discounting

        Returns
        -------
        ndarray
            The expected equivalent annual cost until each time point
        """

        f = (
            lambda x: age_replacement_cost(x, self.ar, self.cf, self.cp)
            * exponential_discounting.factor(x, self.discounting_rate)
            / exponential_discounting.annuity_factor(x, self.discounting_rate)
        )
        mask = timeline < dt
        q0 = self.model.cdf(dt, *self.model_args) * f(dt)
        return q0 + np.where(
            mask,
            0,
            self.model.ls_integrate(f, np.array(dt), timeline, *self.model_args),
        )

    def asymptotic_expected_equivalent_annual_cost(
        self, dt: float = 1.0
    ) -> NDArray[np.float64]:
        """
        The asymptotic expected equivalent annual cost.

        Parameters
        ----------
        dt : float, default 1.0
            The length of the first period before discounting

        Returns
        -------
        ndarray
            The asymptotic expected equivalent annual cost.
        """

        return self.expected_equivalent_annual_cost(np.array(np.inf), dt)

    @ifset("ar")
    def sample(
        self,
        nb_samples: int,
        seed: Optional[int] = None,
    ) -> RenewalRewardData:
        """Sample simulation .

        Parameters
        ----------
        nb_samples : int
            Number of samples generated
        seed : int, optional
            Sample seed. Usefull to fix random generation and reproduce results

        Returns
        -------
        RenewalRewardData
            Iterable object that encapsulates results with additional functions
        """
        generator = lifetimes_rewards_generator(
            self.model,
            age_replacement_cost,
            nb_samples,
            self.nb_assets,
            np.inf,
            model_args=self.model_args,
            reward_args=(self.ar, self.cf, self.cp),
            discounting_rate=self.discounting_rate,
            seed=seed,
        )
        _lifetimes, _event_times, _total_rewards, _events, still_valid = next(generator)
        assets_index, samples_index = np.where(still_valid)
        assets_index.astype(np.int64)
        samples_index.astype(np.int64)
        lifetimes = _lifetimes[still_valid]
        event_times = _event_times[still_valid]
        total_rewards = _total_rewards[still_valid]
        events = _events[still_valid]
        order = np.zeros_like(lifetimes)

        return RenewalRewardData(
            samples_index,
            assets_index,
            order,
            event_times,
            lifetimes,
            events,
            self.model_args,
            False,
            total_rewards,
        )

    def fit(
        self,
        inplace: Optional[bool] = False,
    ) -> Union[Self, None]:
        """
        Computes the optimal age of replacement for each asset.

        Parameters
        ----------
        inplace : bool, default is False
            If True, it sets the optimal age of replacement inplace.

        Returns
        -------
        OneCycleAgeReplacementPolicy (inplace is False) object or None (inplace is True)
        """

        cf_3d, cp_3d = np.array(self.cf, ndmin=3), np.array(self.cp, ndmin=3)
        x0 = np.minimum(np.sum(cp_3d, axis=0) / np.sum(cf_3d - cp_3d, axis=0), 1)
        if np.size(x0) == 1:
            x0 = np.tile(x0, (self.nb_assets, 1))

        def eq(a):
            return np.sum(
                exponential_discounting.factor(a, self.discounting_rate)
                / exponential_discounting.annuity_factor(a, self.discounting_rate)
                * (
                    (cf_3d - cp_3d) * self.model.baseline.hf(a, *self.model_args[1:])
                    - cp_3d
                    / exponential_discounting.annuity_factor(a, self.discounting_rate)
                ),
                axis=0,
            )

        ar = np.asarray(newton(eq, x0), dtype=np.float64)

        if inplace:
            self.model_args = (ar,) + self.model_args[1:]
            self.ar = ar
        else:
            return OneCycleAgeReplacementPolicy(
                self.model.baseline,
                self.cf,
                self.cp,
                discounting_rate=self.discounting_rate,
                ar=ar,
                model_args=self.model_args[1:],
                nb_assets=self.nb_assets,
            )


class AgeReplacementPolicy(Policy):
    r"""Time based replacement policy.

    Renewal reward process where assets are replaced at a fixed age :math:`a_r`
    with costs :math:`c_p` or upon failure with costs :math:`c_f` if earlier.

    Parameters
    ----------
    model : LifetimeModel
        The lifetime model of the assets.
    cf : np.ndarray
        The cost of failure for each asset.
    cp : np.ndarray
        The cost of preventive replacements for each asset.
    ar : np.ndarray, optional
        Times until preventive replacements. This parameter can be optimized
        with ``fit``
    ar1 : np.ndarray, optional
        Times until preventive replacements for the first cycle. This parameter can be optimized
        with ``fit``
    discounting_rate : float, default is 0.
        The discounting rate.
    model_args : ModelArgs, optional
        ModelArgs is a tuple of zero or more ndarray required by the underlying
        lifetime model of the process.
    nb_assets : int, optional
        Number of assets (default is 1).
    a0 : ndarray, optional
        Current ages of the assets (default is None). Setting ``a0`` will add
        left truncations.
    model1 : LifetimeModel, optional
        The lifetime model used for the cycle of replacements. When one adds
        `model1`, we assume that ``model1`` is different from ``model`` meaning
        the underlying survival probabilities behave differently for the first
        cycle
    model1_args : ModelArgs, optional
        ModelArgs is a tuple of zero or more ndarray required by the lifetime
        model of the first cycle of replacements.


    References
    ----------
    .. [1] Mazzuchi, T. A., Van Noortwijk, J. M., & Kallen, M. J. (2007).
        Maintenance optimization. Encyclopedia of Statistics in Quality and
        Reliability, 1000-1008.
    """

    model_args = ShapedArgs()
    model1_args = ShapedArgs()
    cf = ShapedArgs()
    cp = ShapedArgs()
    ar = ShapedArgs()
    ar1 = ShapedArgs()
    a0 = ShapedArgs()

    def __init__(
        self,
        model: LifetimeModel[*ModelArgs],
        cf: float | NDArray[np.float64],
        cp: float | NDArray[np.float64],
        *,
        discounting_rate: float = 0.0,
        ar: float | NDArray[np.float64] = None,
        ar1: float | NDArray[np.float64] = None,
        model_args: ModelArgs = (),
        nb_assets: int = 1,
        a0: Optional[float | NDArray[np.float64]] = None,
        model1: Optional[LifetimeModel[*Model1Args]] = None,
        model1_args: Model1Args = (),
    ) -> None:

        self.nb_assets = nb_assets

        if a0 is not None:
            if model1 is not None:
                raise ValueError("model1 and a0 can't be set together")
            model1 = AgeReplacementModel(LeftTruncatedModel(model))
            model1_args = (a0, *model_args)
        elif model1 is not None:
            model1 = AgeReplacementModel(model1)
        elif model1 is None and ar1 is not None:
            raise ValueError("model1 is not set, ar1 is useless")

        self.model = AgeReplacementModel(model)
        self.model1 = model1

        self.cf = cf
        self.cp = cp
        self.ar = ar
        self.ar1 = ar1
        self.discounting_rate = discounting_rate

        self.model_args = (ar,) + model_args

        # (None, ...) or (ar1, ...)
        self.model1_args = (ar1,) + model1_args if model1_args else None

        self.rrp = None
        parametrized = False
        if self.ar is not None:
            parametrized = True
            if self.model1 is not None:
                if self.ar1 is None:
                    parametrized = False

        # if Policy is parametrized, set the underlying renewal reward process
        # note the rewards are the same for the first cycle and the rest of the process
        if parametrized:
            self.rrp = RenewalRewardProcess(
                self.model,
                age_replacement_cost,
                nb_assets=self.nb_assets,
                model_args=self.model_args,
                reward_args=(self.ar, self.cf, self.cp),
                discounting_rate=self.discounting_rate,
                model1=self.model1,
                model1_args=self.model1_args,
                reward1=age_replacement_cost,
                reward1_args=(self.ar1, self.cf, self.cp),
            )

    @ifset("rrp")
    def expected_total_cost(self, timeline: NDArray[np.float64]) -> NDArray[np.float64]:
        """The expected total cost.

        Parameters
        ----------
        timeline : ndarray
            Timeline of points where the function is evaluated

        Returns
        -------
        ndarray
            The expected total cost for each asset along the timeline
        """
        return self.rrp.expected_total_reward(timeline)

    @ifset("rrp")
    def expected_equivalent_annual_cost(
        self, timeline: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """The expected equivalent annual cost.

        Parameters
        ----------
        timeline : ndarray
            Timeline of points where the function is evaluated

        Returns
        -------
        ndarray
            The expected equivalent annual cost until each time point
        """

        return self.rrp.expected_equivalent_annual_cost(timeline)

    @ifset("rrp")
    def asymptotic_expected_total_cost(self) -> NDArray[np.float64]:
        """
        The asymptotic expected total cost.

        Returns
        -------
        ndarray
            The asymptotic expected total cost for each asset.
        """
        return self.rrp.asymptotic_expected_total_reward()

    @ifset("rrp")
    def asymptotic_expected_equivalent_annual_cost(self) -> NDArray[np.float64]:
        """
        The asymptotic expected equivalent annual cost.

        Returns
        -------
        ndarray
            The asymptotic expected equivalent annual cost.
        """
        return self.rrp.asymptotic_expected_equivalent_annual_cost()

    def expected_number_of_replacements(
        self, timeline: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        pass

    def expected_number_of_failures(
        self, timeline: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        pass

    def expected_number_of_preventive_replacements(
        self, timeline: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        pass

    def fit(
        self,
        inplace: Optional[bool] = False,
    ) -> Union[Self, None]:
        """
        Computes the optimal age of replacement for each asset.

        Parameters
        ----------
        inplace : bool, default is False
            If True, it sets the optimal age of replacement inplace.

        Returns
        -------
        AgeReplacementPolicy (inplace is False) object or None (inplace is True)
        """

        cf_3d, cp_3d = np.array(self.cf, ndmin=3), np.array(self.cp, ndmin=3)
        x0 = np.minimum(np.sum(cp_3d, axis=0) / np.sum(cf_3d - cp_3d, axis=0), 1)
        if np.size(x0) == 1:
            x0 = np.tile(x0, (self.nb_assets, 1))

        ndim = max(
            map(np.ndim, (cf_3d, cp_3d, self.discounting_rate, *self.model_args[1:])),
            default=0,
        )

        def eq(a):
            f = gauss_legendre(
                lambda x: exponential_discounting.factor(x, self.discounting_rate)
                * self.model.baseline.sf(x, *self.model_args[1:]),
                0,
                a,
                ndim=ndim,
            )
            g = gauss_legendre(
                lambda x: exponential_discounting.factor(x, self.discounting_rate)
                * self.model.baseline.pdf(x, *self.model_args[1:]),
                0,
                a,
                ndim=ndim,
            )
            return np.sum(
                exponential_discounting.factor(a, self.discounting_rate)
                * (
                    (cf_3d - cp_3d)
                    * (self.model.baseline.hf(a, *self.model_args[1:]) * f - g)
                    - cp_3d
                )
                / f**2,
                axis=0,
            )

        ar = newton(eq, x0)
        if np.size(ar) == 1:
            ar = np.squeeze(ar)

        ar1 = None
        if self.model1 is not None:
            onecycle = OneCycleAgeReplacementPolicy(
                self.model1.baseline,
                self.cf,
                self.cp,
                discounting_rate=self.discounting_rate,
                model_args=self.model1_args[1:],
            ).fit()
            ar1 = onecycle.ar

        if inplace:
            self.ar = ar
            self.ar1 = ar1
            self.model_args = (ar,) + self.model_args[1:]
            self.model1_args = (ar1,) + self.model1_args[1:] if self.model1 else None
            self.rrp = RenewalRewardProcess(
                self.model,
                age_replacement_cost,
                nb_assets=self.nb_assets,
                model_args=self.model_args,
                reward_args=(self.ar, self.cf, self.cp),
                discounting_rate=self.discounting_rate,
                model1=self.model1,
                model1_args=self.model1_args,
                reward1=age_replacement_cost,
                reward1_args=(self.ar1, self.cf, self.cp),
            )
        else:
            return AgeReplacementPolicy(
                self.model.baseline,
                self.cf,
                self.cp,
                discounting_rate=self.discounting_rate,
                ar=ar,
                ar1=ar1,
                model_args=self.model_args[1:],
                model1=self.model1.baseline if self.model1 else None,
                model1_args=self.model1_args[1:] if self.model1 else None,
            )

    @ifset("rrp")
    def sample(
        self,
        nb_samples: int,
        end_time: float,
        seed: Optional[int] = None,
    ) -> RenewalRewardData:
        """Sample simulation .

        Parameters
        ----------
        nb_samples : int
            Number of samples generated
        end_time : float
            End of the observation period. It is the upper bound of the cumulative generated lifetimes.
        seed : int, optional
            Sample seed. Usefull to fix random generation and reproduce results

        Returns
        -------
        RenewalRewardData
            Iterable object that encapsulates results with additional functions
        """
        return self.rrp.sample(nb_samples, end_time, seed=seed)
